# qkvflow/train_pretrained_svd_llama.py
"""
Training script for SVD Llama with pretrained backbone (FROZEN).
Only policy network is trained.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

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
from levanter.trainer import Trainer
from levanter.utils.jax_utils import parameter_count
from levanter.utils.tree_utils import inference_mode

import wandb
from qkvflow.nn.dynamic_svd_llama import SVDLlamaOdeLMHeadModel
from qkvflow.train_svd_llama_lm import (
    TrainSVDLlamaLmConfig, SVDModelConfig, log_diagnostics
)
from qkvflow.utils.hf_loader import (
    load_llama_from_hf, create_levanter_config_from_pretrained
)
from qkvflow.nn.svd_from_pretrained import initialize_svd_mlp_from_pretrained

logger = logging.getLogger(__name__)


@dataclass
class PretrainedSVDConfig(TrainSVDLlamaLmConfig):
    """Config for pretrained SVD Llama training."""
    
    # Pretrained model settings
    pretrained_model_name: str = "meta-llama/Llama-3.1-8B"
    pretrained_cache_dir: Optional[str] = "/content/cache/hf_models"
    
    # CRITICAL: Freeze pretrained weights
    freeze_backbone: bool = True  # Always True for pretrained
    train_policy_only: bool = True  # Train ONLY policy
    
    # Override to ensure correct behavior
    train_svd_from_scratch: bool = False  # Use pretrained weights
    load_pretrained_ode: Optional[str] = None  # Not needed


def create_model_from_pretrained(
    pretrained_model_name: str,
    config: PretrainedSVDConfig,
    Vocab: hax.Axis,
    *,
    key: jax.random.PRNGKey,
) -> SVDLlamaOdeLMHeadModel:
    """
    Create SVD Llama model with pretrained weights (FROZEN).
    """
    logger.info(f"Loading pretrained model: {pretrained_model_name}")
    
    # Load pretrained PyTorch model
    pt_model, tokenizer, model_info = load_llama_from_hf(
        model_name=pretrained_model_name,
        cache_dir=config.pretrained_cache_dir,
    )
    
    # Create Levanter config matching pretrained
    levanter_config = convert_llama_config_from_hf(model_info)
    levanter_config = levanter_config.replace(seq_len=config.model.seq_len)
    
    # Initialize model structure
    model = SVDLlamaOdeLMHeadModel.init(
        Vocab=Vocab,
        config=levanter_config,
        time_embed_dim=config.time_embed_dim,
        sinusodial_dim=config.sinusodial_dim,
        rank_ratio=config.svd_config.rank_ratio,
        policy_init_scale=config.svd_config.policy_init_scale,
        key=key,
    )
    
    # Replace MLP weights with pretrained SVD decomposition
    logger.info("Replacing MLP weights with pretrained SVD...")
    
    for layer_idx in range(levanter_config.num_layers):
        svd_mlp = initialize_svd_mlp_from_pretrained(
            pt_model=pt_model,
            layer_idx=layer_idx,
            Embed=levanter_config.Embed,
            Mlp=levanter_config.Mlp,
            rank_ratio=config.svd_config.rank_ratio,
        )
        
        # Update model's MLP SVD layers
        # This replaces the randomly initialized SVD with pretrained
        model = eqx.tree_at(
            lambda m: m.transformer.block.mlp.adaptive_mlp.gate_proj,
            model,
            svd_mlp["gate_proj"],
            is_leaf=lambda x: isinstance(x, type(svd_mlp["gate_proj"])),
        )
        model = eqx.tree_at(
            lambda m: m.transformer.block.mlp.adaptive_mlp.up_proj,
            model,
            svd_mlp["up_proj"],
            is_leaf=lambda x: isinstance(x, type(svd_mlp["up_proj"])),
        )
        model = eqx.tree_at(
            lambda m: m.transformer.block.mlp.adaptive_mlp.down_proj,
            model,
            svd_mlp["down_proj"],
            is_leaf=lambda x: isinstance(x, type(svd_mlp["down_proj"])),
        )
    
    logger.info("✓ Pretrained weights loaded and decomposed")
    
    # Set model to inference mode (FROZEN)
    model = inference_mode(model, True)
    
    return model


def main(config: PretrainedSVDConfig):
    logger.info("="*60)
    logger.info("Training SVD Llama with PRETRAINED + FROZEN backbone")
    logger.info("="*60)
    
    # Validate config
    assert config.freeze_backbone, "Must freeze backbone for pretrained!"
    assert config.train_policy_only, "Must train policy only for pretrained!"
    
    tokenizer = config.data.the_tokenizer
    config.trainer.initialize(config)
    
    seed = config.trainer.seed
    model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)
    
    Batch = config.trainer.TrainBatch
    EvalBatch = config.trainer.EvalBatch
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos
    
    parameter_axis_mapping = config.trainer.parameter_axis_mapping
    
    def compute_loss(model, example: LmExample, key=None):
        return model.compute_loss(
            example, 
            key=key, 
            policy_reg_strength=config.svd_config.policy_reg_strength
        ).scalar()
    
    # Configure optimizer (weight decay only on policy)
    from dataclasses import replace
    new_optimizer = replace(
        config.optimizer,
        weight_decay_modules=r"policy\.policy_net.*"
    )
    config = replace(config, optimizer=new_optimizer)
    
    optimizer = config.optimizer.build(config.trainer.num_train_steps)
    
    # CRITICAL: Filter for policy-only training
    def is_policy_param(node):
        """Only policy.policy_net parameters are trainable."""
        # Check if node is part of policy network
        def check_path(path, value):
            return "policy.policy_net" in path
        
        return eqx.tree_util.tree_map(
            check_path,
            node,
            is_leaf=eqx.is_array,
        )
    
    trainer = Trainer(
        config.trainer, 
        optimizer, 
        compute_loss,
        is_trainable_param=is_policy_param  # ONLY train policy
    )
    
    # Setup data loaders
    eval_dataset = CausalLmDataset(
        config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos
    )
    eval_loader = trainer.replicated_loader(eval_dataset, EvalBatch)
    train_dataset = CausalLmDataset(
        config.data.token_seq_dataset("train", Pos.size), Pos, KeyPos
    )
    train_loader = iter(trainer.sharded_loader(train_dataset, Batch))
    
    with trainer.device_mesh:
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(
            Axis("vocab", vocab_size), parameter_axis_mapping
        )
        
        # Create model with pretrained weights
        model = create_model_from_pretrained(
            pretrained_model_name=config.pretrained_model_name,
            config=config,
            Vocab=Vocab,
            key=model_key,
        )
        
        state = trainer.initial_state(
            training_key=train_key,
            model=model,
        )
        
        # Log parameter counts
        total_params = parameter_count(state.model)
        trainable_params = parameter_count(trainer.trainable_params_only(state.model))
        frozen_params = total_params - trainable_params
        
        logger.info(f"Parameter counts:")
        logger.info(f"  Total: {total_params:,}")
        logger.info(f"  Trainable (policy): {trainable_params:,}")
        logger.info(f"  Frozen (backbone): {frozen_params:,}")
        logger.info(f"  Trainable %: {trainable_params/total_params*100:.2f}%")
        
        wandb.summary["total_parameters"] = total_params
        wandb.summary["trainable_parameters"] = trainable_params
        wandb.summary["frozen_parameters"] = frozen_params
        
        # Add hooks
        trainer.add_default_hooks(eval_loader)
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size),
            every=1,
        )
        trainer.add_hook(log_diagnostics, every=config.trainer.steps_per_eval)
        
        # Resume from checkpoint if needed
        if state.step > 0:
            import tqdm
            for _ in tqdm.tqdm(range(state.step + 1), desc="Finding resume point"):
                next(train_loader)
        
        # Train!
        logger.info("Starting training...")
        trainer.train(state, train_loader)


if __name__ == "__main__":
    levanter.config.main(main)()