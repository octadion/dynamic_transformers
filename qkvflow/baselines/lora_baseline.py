# qkvflow/baselines/lora_baseline.py
"""
Fair LoRA baseline: Train separate LoRA adapter per task.
This is the main baseline to compare against PCSVM.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Dict
import os

import jax
import jax.random as jrandom
import levanter
import equinox as eqx
import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter import callbacks
from levanter.data.text import CausalLmDataset
from levanter.models.lm_model import LmExample
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.checkpoint import save_checkpoint

from qkvflow.lora import loraize, LoraConfig, is_lora_param
from qkvflow.train_pretrained_svd_llama import PretrainedSVDConfig
from qkvflow.utils.hf_loader import create_levanter_config_from_pretrained
from qkvflow.nn.dynamic_svd_llama import SVDLlamaOdeLMHeadModel

logger = logging.getLogger(__name__)


@dataclass
class LoRABaselineConfig(PretrainedSVDConfig):
    """Config for training individual LoRA adapters."""
    
    # LoRA-specific settings
    lora_r: int = 8  # Rank
    lora_alpha: float = 16.0  # Scaling
    lora_dropout: float = 0.05
    lora_target_modules: str = "mlp.*_proj"  # Target MLP projections
    
    # Task to train on
    task_name: str = "winogrande"  # Single task per training run
    
    # Output path for this specific task
    task_output_dir: Optional[str] = None


def create_lora_model_from_pretrained(
    config: LoRABaselineConfig,
    Vocab: hax.Axis,
    *,
    key: jax.random.PRNGKey,
):
    """
    Create Llama model with LoRA adapters.
    
    Unlike PCSVM which uses SVD, this uses standard LoRA.
    """
    from levanter.models.llama import LlamaLMHeadModel
    
    logger.info(f"Creating LoRA model for task: {config.task_name}")
    
    # Create base Llama config
    levanter_config, tokenizer = create_levanter_config_from_pretrained(
        model_name=config.pretrained_model_name,
        seq_len=config.model.seq_len,
        cache_dir=config.pretrained_cache_dir,
    )
    
    # Initialize base model
    base_model = LlamaLMHeadModel.init(
        Vocab=Vocab,
        config=levanter_config,
        key=key,
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        target_modules=config.lora_target_modules,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    
    lora_model = loraize(base_model, config=lora_config, key=key)
    
    # Count parameters
    total_params = parameter_count(lora_model)
    lora_params = parameter_count(eqx.filter(lora_model, is_lora_param, is_leaf=is_lora_param))
    
    logger.info(f"✓ LoRA model created:")
    logger.info(f"  - Total params: {total_params:,}")
    logger.info(f"  - LoRA params: {lora_params:,} ({lora_params/total_params*100:.2f}%)")
    logger.info(f"  - Rank: {config.lora_r}")
    logger.info(f"  - Alpha: {config.lora_alpha}")
    
    return lora_model, tokenizer


def train_lora_for_task(config: LoRABaselineConfig):
    """Train LoRA adapter for a specific task."""
    logger.info("="*60)
    logger.info(f"Training LoRA Baseline for Task: {config.task_name.upper()}")
    logger.info("="*60)
    
    # Initialize trainer config
    config.trainer.initialize(config)
    
    seed = config.trainer.seed
    model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)
    
    Batch = config.trainer.TrainBatch
    EvalBatch = config.trainer.EvalBatch
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos
    
    parameter_axis_mapping = config.trainer.parameter_axis_mapping
    
    # Loss function
    def compute_loss(model, example: LmExample, key=None):
        return model.compute_loss(example, key=key).scalar()
    
    # Optimizer with weight decay on LoRA params only
    from dataclasses import replace
    new_optimizer = replace(
        config.optimizer,
        weight_decay_modules=r"lora.*"
    )
    config = replace(config, optimizer=new_optimizer)
    
    optimizer = config.optimizer.build(config.trainer.num_train_steps)
    
    # Create trainer with LoRA filter
    def is_lora_trainable(node):
        """Only LoRA parameters are trainable."""
        return eqx.tree_util.tree_map(
            is_lora_param,
            node,
            is_leaf=is_lora_param,
        )
    
    trainer = Trainer(
        config.trainer,
        optimizer,
        compute_loss,
        is_trainable_param=is_lora_trainable,
    )
    
    # Setup data loaders for specific task
    # Note: You'll need to implement task-specific data loading
    # For now, using placeholder from config.data
    eval_dataset = CausalLmDataset(
        config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos
    )
    eval_loader = trainer.replicated_loader(eval_dataset, EvalBatch)
    train_dataset = CausalLmDataset(
        config.data.token_seq_dataset("train", Pos.size), Pos, KeyPos
    )
    train_loader = iter(trainer.sharded_loader(train_dataset, Batch))
    
    with trainer.device_mesh:
        # Get tokenizer and create Vocab axis
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(
            Axis("vocab", vocab_size), parameter_axis_mapping
        )
        
        # Create LoRA model
        model, _ = create_lora_model_from_pretrained(
            config=config,
            Vocab=Vocab,
            key=model_key,
        )
        
        # Initialize state
        state = trainer.initial_state(
            training_key=train_key,
            model=model,
        )
        
        # Log to wandb
        total_params = parameter_count(state.model)
        trainable_params = parameter_count(trainer.trainable_params_only(state.model))
        
        import wandb
        wandb.summary["total_parameters"] = total_params
        wandb.summary["trainable_parameters"] = trainable_params
        wandb.summary["task"] = config.task_name
        
        # Add hooks
        trainer.add_default_hooks(eval_loader)
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, config.trainer.train_batch_size),
            every=1,
        )
        
        # Resume if needed
        if state.step > 0:
            import tqdm
            for _ in tqdm.tqdm(range(state.step + 1), desc="Finding resume point"):
                next(train_loader)
        
        # Train
        logger.info(f"Training LoRA for {config.task_name}...")
        final_state = trainer.train(state, train_loader)
        
        # Save final adapter
        if config.task_output_dir:
            output_path = os.path.join(config.task_output_dir, f"lora_{config.task_name}")
            os.makedirs(output_path, exist_ok=True)
            save_checkpoint(final_state.model, path=output_path, step=final_state.step)
            logger.info(f"✓ Saved LoRA adapter to {output_path}")
        
        return final_state


if __name__ == "__main__":
    import sys
    
    # Ensure task_name is provided
    if "--task_name" not in " ".join(sys.argv):
        print("ERROR: Must provide --task_name argument")
        print("Usage: python lora_baseline.py --task_name winogrande")
        sys.exit(1)
    
    levanter.config.main(train_lora_for_task)()