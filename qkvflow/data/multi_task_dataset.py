# qkvflow/data/multi_task_dataset.py
"""
Multi-task dataset for meta-learning policy training.
Supports: Winogrande, SciQ, ARC-Easy, HellaSwag, PIQA
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator, Tuple
import numpy as np
import jax.numpy as jnp
import haliax as hax
from datasets import load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task dataset."""
    
    # Task weights for sampling
    task_weights: Dict[str, float] = None
    
    # Maximum samples per task (for balancing)
    max_samples_per_task: Optional[int] = None
    
    # Sequence length
    seq_len: int = 512
    
    # Whether to add task prefixes
    add_task_prefix: bool = True
    
    # Cache directory
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.task_weights is None:
            # Equal weights by default
            self.task_weights = {
                "winogrande": 1.0,
                "sciq": 1.0,
                "arc_easy": 1.0,
                "hellaswag": 1.0,
                "piqa": 1.0,
            }


class MultiTaskDataset:
    """Multi-task dataset for meta-learning."""
    
    TASK_CONFIGS = {
        "winogrande": {
            "hf_name": "winogrande",
            "hf_config": "winogrande_xl",
            "text_key": "sentence",
            "split": "train",
        },
        "sciq": {
            "hf_name": "sciq",
            "hf_config": None,
            "text_key": "question",
            "split": "train",
        },
        "arc_easy": {
            "hf_name": "ai2_arc",
            "hf_config": "ARC-Easy",
            "text_key": "question",
            "split": "train",
        },
        "hellaswag": {
            "hf_name": "hellaswag",
            "hf_config": None,
            "text_key": "ctx",
            "split": "train",
        },
        "piqa": {
            "hf_name": "piqa",
            "hf_config": None,
            "text_key": "goal",
            "split": "train",
        },
    }
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: MultiTaskConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.datasets = {}
        self.task_sizes = {}
        
        logger.info("Loading multi-task datasets...")
        self._load_all_datasets()
        
    def _load_all_datasets(self):
        """Load all task datasets."""
        for task_name, task_config in self.TASK_CONFIGS.items():
            if task_name not in self.config.task_weights:
                continue
                
            logger.info(f"Loading {task_name}...")
            
            # Load from HuggingFace
            dataset = load_dataset(
                task_config["hf_name"],
                task_config["hf_config"],
                split=task_config["split"],
                cache_dir=self.config.cache_dir,
            )
            
            # Limit samples if needed
            if self.config.max_samples_per_task:
                max_samples = min(len(dataset), self.config.max_samples_per_task)
                dataset = dataset.shuffle(seed=42).select(range(max_samples))
            
            self.datasets[task_name] = dataset
            self.task_sizes[task_name] = len(dataset)
            
            logger.info(f"  ✓ {task_name}: {len(dataset)} samples")
        
        total_samples = sum(self.task_sizes.values())
        logger.info(f"Total samples across all tasks: {total_samples}")
    
    def _format_example(self, example: dict, task_name: str) -> str:
        """Format example as text with optional task prefix."""
        task_config = self.TASK_CONFIGS[task_name]
        text_key = task_config["text_key"]
        
        # Get main text
        text = example[task_key]
        
        # Add task prefix if enabled
        if self.config.add_task_prefix:
            task_prefix = f"[Task: {task_name.upper()}] "
            text = task_prefix + text
        
        # Add context for certain tasks
        if task_name == "sciq" and "support" in example and example["support"]:
            text = f"Context: {example['support']}\nQuestion: {text}"
        elif task_name == "hellaswag" and "activity_label" in example:
            text = f"Activity: {example['activity_label']}\n{text}"
        
        return text
    
    def _tokenize_example(self, text: str) -> np.ndarray:
        """Tokenize and pad/truncate to seq_len."""
        tokens = self.tokenizer(
            text,
            max_length=self.config.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        return tokens["input_ids"][0]  # [seq_len]
    
    def create_iterator(
        self, 
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Create iterator that yields batches with task labels.
        
        Yields:
            (input_ids, task_ids)
            - input_ids: [batch_size, seq_len]
            - task_ids: [batch_size] (integer task IDs)
        """
        # Create task ID mapping
        task_to_id = {name: idx for idx, name in enumerate(self.datasets.keys())}
        
        # Create weighted sampler
        rng = np.random.RandomState(seed)
        
        # Normalize weights
        total_weight = sum(self.config.task_weights.values())
        task_probs = {
            task: weight / total_weight 
            for task, weight in self.config.task_weights.items()
            if task in self.datasets
        }
        
        # Create infinite iterator
        task_names = list(self.datasets.keys())
        task_probabilities = [task_probs[name] for name in task_names]
        
        # Pre-shuffle all datasets if needed
        if shuffle:
            for task_name in self.datasets:
                self.datasets[task_name] = self.datasets[task_name].shuffle(seed=seed)
        
        # Create iterators for each task
        task_iterators = {
            task_name: iter(dataset)
            for task_name, dataset in self.datasets.items()
        }
        
        # Infinite loop
        while True:
            batch_tokens = []
            batch_task_ids = []
            
            for _ in range(batch_size):
                # Sample task
                task_name = rng.choice(task_names, p=task_probabilities)
                task_id = task_to_id[task_name]
                
                # Get next example from task
                try:
                    example = next(task_iterators[task_name])
                except StopIteration:
                    # Reset iterator for this task
                    dataset = self.datasets[task_name]
                    if shuffle:
                        dataset = dataset.shuffle(seed=rng.randint(0, 1000000))
                    task_iterators[task_name] = iter(dataset)
                    example = next(task_iterators[task_name])
                
                # Format and tokenize
                text = self._format_example(example, task_name)
                tokens = self._tokenize_example(text)
                
                batch_tokens.append(tokens)
                batch_task_ids.append(task_id)
            
            # Stack into arrays
            batch_tokens = np.stack(batch_tokens, axis=0)  # [batch_size, seq_len]
            batch_task_ids = np.array(batch_task_ids)      # [batch_size]
            
            yield batch_tokens, batch_task_ids
    
    def get_task_info(self) -> Dict[str, any]:
        """Get information about loaded tasks."""
        return {
            "tasks": list(self.datasets.keys()),
            "task_sizes": self.task_sizes,
            "total_samples": sum(self.task_sizes.values()),
            "task_weights": self.config.task_weights,
        }


def create_haliax_batch(
    tokens: np.ndarray,
    task_ids: np.ndarray,
    Batch: hax.Axis,
    Pos: hax.Axis,
) -> Tuple[hax.NamedArray, hax.NamedArray]:
    """
    Convert numpy arrays to Haliax NamedArrays.
    
    Args:
        tokens: [batch_size, seq_len]
        task_ids: [batch_size]
        Batch: Batch axis
        Pos: Position axis
        
    Returns:
        (input_ids, task_ids) as NamedArrays
    """
    input_ids = hax.named(jnp.array(tokens), (Batch, Pos))
    task_ids_named = hax.named(jnp.array(task_ids), (Batch,))
    
    return input_ids, task_ids_named


if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    print("Testing MultiTaskDataset...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create config
    config = MultiTaskConfig(
        task_weights={
            "winogrande": 1.0,
            "sciq": 1.0,
            "arc_easy": 1.0,
        },
        max_samples_per_task=1000,  # Limit for testing
        seq_len=128,
        add_task_prefix=True,
    )
    
    # Create dataset
    dataset = MultiTaskDataset(tokenizer, config)
    
    # Print info
    info = dataset.get_task_info()
    print("\nDataset Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    # Test iterator
    print("\nTesting iterator...")
    iterator = dataset.create_iterator(batch_size=4, shuffle=True)
    
    for i, (tokens, task_ids) in enumerate(iterator):
        print(f"\nBatch {i}:")
        print(f"  Tokens shape: {tokens.shape}")
        print(f"  Task IDs: {task_ids}")
        
        # Decode first example
        decoded = tokenizer.decode(tokens[0], skip_special_tokens=True)
        print(f"  Example: {decoded[:100]}...")
        
        if i >= 2:
            break
    
    print("\n✓ All tests passed!")