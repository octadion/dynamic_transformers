# qkvflow/train_multitask_svd_llama.py
"""
Meta-learning training script for SVD Llama with multi-task dataset.
Policy learns to adapt based on task patterns.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.random as jrandom
import jax.numpy as jnp
import levanter
import equinox as eqx
import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter import callbacks
from levanter.trainer import Trainer
from levanter.utils.jax_utils import parameter_count
from levanter.models.lm_model import LmExample

import wandb
from qkvflow.train_pretrained_svd_llama import (
    PretrainedSVDConfig, create_model_from_pretrained
)
from qkvflow.data.multi_task_dataset import (
    MultiTaskDataset, MultiTaskConfig, create_haliax_batch
)
from qkvflow.train_svd_llama_lm import log_diagnostics

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskTrainingConfig(PretrainedSVDConfig):
    """Config for multi-task meta-learning training."""
    
    # Multi-task dataset config
    multi_task: MultiTaskConfig = field(default_factory=MultiTaskConfig)
    
    # Task mixing strategy
    task_batch_mixing: str = "random"  # 'random', 'round_robin', 'curriculum'
    
    # Logging
    log_task_performance: bool = True
    log_task_every: int = 100


class MultiTaskTrainer:
    """Wrapper for multi-task meta-learning training."""
    
    def __init__(
        self,
        config: MultiTaskTrainingConfig,
        trainer: Trainer,
        dataset: MultiTaskDataset,
    ):
        self.config = config
        self.trainer = trainer
        self.dataset = dataset
        
        # Track per-task statistics
        self.task_losses = {task: [] for task in dataset.datasets.keys()}
        self.task_steps = {task: 0 for task in dataset.datasets.keys()}
        
    def create_train_loader(self):
        """Create training data loader."""
        Batch = self.config.trainer.TrainBatch
        Pos = self.config.model.Pos
        
        # Create iterator
        base_iterator = self.dataset.create_iterator(
            batch_size=Batch.size,
            shuffle=True,
            seed=self.config.trainer.seed,
        )
        
        # Wrap with Haliax conversion
        def data_iterator():
            for tokens, task_ids in base_iterator:
                input_ids, task_ids_named = create_haliax_batch(tokens, task_ids, Batch, Pos)
                
                # Create attention mask (1 for real tokens, 0 for padding)
                attn_mask = hax.named(
                    jnp.array(tokens != self.dataset.tokenizer.pad_token_id),
                    (Batch, Pos)
                )
                
                # Create loss mask (same as attn_mask for now)
                loss_mask = attn_mask
                
                # Create LmExample
                example = LmExample(
                    tokens=input_ids,
                    attn_mask=attn_mask,
                    loss_mask=loss_mask,
                )
                
                yield example, task_ids_named
        
        return data_iterator()
    
    def compute_loss_with_task_tracking(self, model, example_and_task, key=None):
        """Compute loss and track per-task statistics."""
        example, task_ids = example_and_task
        
        # Compute standard loss
        loss = model.compute_loss(
            example,
            key=key,
            policy_reg_strength=self.config.svd_config.policy_reg_strength,
        )
        
        # Track per-task losses (for logging only, not used in training)
        if self.config.log_task_performance and wandb.run:
            # Get task-wise losses
            task_id_array = task_ids.array
            loss_per_token = loss  # This is already reduced
            
            # Log to wandb (simplified - just log overall loss)
            # In practice, you'd want to compute per-task metrics separately
            pass
        
        return loss.scalar()
    
    def train(self, initial_state):
        """Run meta-learning training."""
        logger.info("="*60)
        logger.info("Starting Meta-Learning Training")
        logger.info("="*60)
        
        # Print dataset info
        info = self.dataset.get_task_info()
        logger.info("Multi-task dataset:")
        for task_name, size in info["task_sizes"].items():
            weight = info["task_weights"].get(task_name, 0)
            logger.info(f"  - {task_name}: {size} samples (weight: {weight:.2f})")
        
        # Create data loader
        train_loader = self.create_train_loader()
        
        # Override trainer's compute_loss
        original_loss_fn = self.trainer.loss_fn
        self.trainer.loss_fn = self.compute_loss_with_task_tracking
        
        # Add task performance logging hook
        if self.config.log_task_performance:
            def log_task_stats(step_info):
                # Log task distribution statistics
                stats = {
                    "tasks/total_steps": step_info.step,
                }
                
                # Log per-task counters
                for task_name, count in self.task_steps.items():
                    stats[f"tasks/{task_name}_steps"] = count
                
                if wandb.run:
                    wandb.log(stats, step=step_info.step)
            
            self.trainer.add_hook(log_task_stats, every=self.config.log_task_every)
        
        # Resume from checkpoint if needed
        if initial_state.step > 0:
            import tqdm
            logger.info(f"Resuming from step {initial_state.step}")
            for _ in tqdm.tqdm(range(initial_state.step + 1), desc="Finding resume point"):
                next(train_loader)
        
        # Train!
        logger.info(f"Training for {self.config.trainer.num_train_steps} steps...")
        final_state = self.trainer.train(initial_state, train_loader)
        
        # Restore original loss function
        self.trainer.loss_fn = original_loss_fn
        
        return final_state


def main(config: MultiTaskTrainingConfig):
    logger.info("="*60)
    logger.info("Multi-Task Meta-Learning for SVD Llama")
    logger.info("="*60)
    
    # Validate config
    assert config.freeze_backbone, "Must freeze backbone for meta-learning!"
    assert config.train_policy_only, "Must train policy only!"
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        cache_dir=config.pretrained_cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize trainer config
    config.trainer.initialize(config)
    
    seed = config.trainer.seed
    model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)
    
    Pos = config.model.Pos
    parameter_axis_mapping = config.trainer.parameter_axis_mapping
    
    # Setup optimizer
    from dataclasses import replace
    new_optimizer = replace(
        config.optimizer,
        weight_decay_modules=r"policy\.policy_net.*"
    )
    config = replace(config, optimizer=new_optimizer)
    
    optimizer = config.optimizer.build(config.trainer.num_train_steps)
    
    # Create multi-task dataset
    logger.info("Creating multi-task dataset...")
    dataset = MultiTaskDataset(tokenizer, config.multi_task)
    
    # Dummy loss function (will be overridden by MultiTaskTrainer)
    def dummy_loss(model, example, key=None):
        return model.compute_loss(
            example,
            key=key,
            policy_reg_strength=config.svd_config.policy_reg_strength,
        ).scalar()
    
    # Policy-only training filter
    def is_policy_param(node):
        def check_path(path, value):
            return "policy.policy_net" in path
        return eqx.tree_util.tree_map(check_path, node, is_leaf=eqx.is_array)
    
    trainer = Trainer(
        config.trainer,
        optimizer,
        dummy_loss,
        is_trainable_param=is_policy_param,
    )
    
    # Setup eval loader (use first task for validation)
    from levanter.data.text import CausalLmDataset
    eval_dataset = CausalLmDataset(
        config.data.token_seq_dataset("validation", Pos.size),
        Pos,
        config.model.KeyPos,
    )
    eval_loader = trainer.replicated_loader(eval_dataset, config.trainer.EvalBatch)
    
    with trainer.device_mesh:
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(
            Axis("vocab", vocab_size), parameter_axis_mapping
        )
        
        # Create model with pretrained weights
        logger.info("Creating model with pretrained weights...")
        model = create_model_from_pretrained(
            pretrained_model_name=config.pretrained_model_name,
            config=config,
            Vocab=Vocab,
            key=model_key,
        )
        
        # Initialize state
        state = trainer.initial_state(
            training_key=train_key,
            model=model,
        )
        
        # Log parameter counts
        total_params = parameter_count(state.model)
        trainable_params = parameter_count(trainer.trainable_params_only(state.model))
        
        logger.info(f"Parameter counts:")
        logger.info(f"  Total: {total_params:,}")
        logger.info(f"  Trainable (policy): {trainable_params:,}")
        logger.info(f"  Frozen (backbone): {total_params - trainable_params:,}")
        
        wandb.summary["total_parameters"] = total_params
        wandb.summary["trainable_parameters"] = trainable_params
        
        # Add standard hooks
        trainer.add_default_hooks(eval_loader)
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, config.trainer.train_batch_size),
            every=1,
        )
        trainer.add_hook(log_diagnostics, every=config.trainer.steps_per_eval)
        
        # Create multi-task trainer and run
        mt_trainer = MultiTaskTrainer(config, trainer, dataset)
        final_state = mt_trainer.train(state)
        
        logger.info("✓ Training complete!")
        
        return final_state


if __name__ == "__main__":
    import sys
    
    # Default to multi-task mode
    if "--model_choice" not in " ".join(sys.argv):
        sys.argv.extend(["--model_choice", "llamaode-svd"])
    
    levanter.config.main(main)()