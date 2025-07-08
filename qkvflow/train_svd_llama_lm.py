import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import jax
import jax.random as jrandom
import jax.numpy as jnp
import jax.nn as jnn
import levanter
import equinox as eqx
import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter import callbacks
from levanter.data.text import CausalLmDataset
from levanter.models.lm_model import LmExample, LmHeadModel, LmConfig
from levanter.models.llama import LlamaConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.trainer import Trainer
from levanter.utils.jax_utils import parameter_count

import wandb
from qkvflow.nn.dynamic import NeuralOdeLMHeadModel
from qkvflow.nn.dynamic_svd import SVDNeuralOdeLMHeadModel
from qkvflow.nn.dynamic_svd_llama import SVDLlamaOdeLMHeadModel
from qkvflow.train_lm import (
    DatasetConfig, OptimizerConfigWithWeightDecay, TrainLmConfig
)

logger = logging.getLogger(__name__)


@dataclass
class SVDModelConfig:
    """Configuration for SVD adaptive model."""
    rank_ratio: float = 0.5 
    policy_init_scale: float = 0.1
    policy_reg_strength: float = 0.01


@dataclass
class TrainSVDLlamaLmConfig(TrainLmConfig):
    """Extended config for SVD Llama model training."""
    
    svd_config: SVDModelConfig = field(default_factory=SVDModelConfig)
    
    load_pretrained_ode: Optional[str] = None
    
    train_policy_only: bool = False 
    train_svd_from_scratch: bool = True 

def log_diagnostics(step_info):
    if not hasattr(step_info.model.transformer, "policy"):
        return

    model = step_info.model
    stats = {}

    mlp_block = getattr(getattr(model.transformer, "block", None), "mlp", None)
    if not mlp_block:
        return

    if hasattr(mlp_block, "gate_logit_fc"):
        gate_fc_raw = jnn.sigmoid(mlp_block.gate_logit_fc.astype(jnp.float32).array)
        stats["gates/mlp_fc"] = float(gate_fc_raw)
    
    if hasattr(mlp_block, "gate_logit_proj"):
        gate_proj_raw = jnn.sigmoid(mlp_block.gate_logit_proj.astype(jnp.float32).array)
        stats["gates/mlp_proj"] = float(gate_proj_raw)

    projection_names = []
    if hasattr(mlp_block, "adaptive_mlp"):
        if hasattr(mlp_block.adaptive_mlp, "c_fc"):
            projection_names = ["c_fc", "c_proj"]
        elif hasattr(mlp_block.adaptive_mlp, "gate_proj"):
            projection_names = ["gate_proj", "up_proj", "down_proj"]

    if not projection_names:
        if wandb.run and stats:
            wandb.log(stats, step=step_info.step)
        return

    t = hax.named(jnp.array(0.5, dtype=jnp.float32), ())
    time_embed = model.transformer.time_embedding(t)

    Batch = hax.Axis("batch", 1)
    Embed = model.config.Embed
    task_vector = hax.ones((Batch, Embed))
    multipliers = model.transformer.policy(task_vector)

    for proj_name in projection_names:
        temporal_proj = getattr(mlp_block, f"{proj_name}_temporal", None)
        adaptive_mlp = getattr(mlp_block, "adaptive_mlp", None)
        
        if temporal_proj and adaptive_mlp:
            adaptive_proj = getattr(adaptive_mlp, proj_name, None)
            if not adaptive_proj: continue

            w_temporal, _ = temporal_proj.evaluate_at_components(time_embed)
            multiplier_key = f"layer_0_{proj_name}"
            if multiplier_key in multipliers:
                s_multiplier = multipliers[multiplier_key]
                w_svd = adaptive_proj.get_effective_weight(s_multiplier=s_multiplier)

                norm_temporal = jnp.linalg.norm(w_temporal.array)
                norm_svd = jnp.linalg.norm(w_svd.array[0])

                stats[f"weight_norms/{proj_name}/temporal"] = float(norm_temporal)
                stats[f"weight_norms/{proj_name}/svd"] = float(norm_svd)
                ratio = norm_svd / (norm_temporal + 1e-8)
                stats[f"weight_norms/{proj_name}/svd_vs_temporal_ratio"] = float(ratio)

    if wandb.run and stats:
        wandb.log(stats, step=step_info.step)


def main(config: TrainSVDLlamaLmConfig):
    logger.info(f"Training SVD Neural ODE Llama model {config.model_choice}")
    
    tokenizer = config.data.the_tokenizer
    config.trainer.initialize(config)
    
    seed = config.trainer.seed
    model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)
    
    Batch = config.trainer.TrainBatch
    EvalBatch = config.trainer.EvalBatch
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos
    
    parameter_axis_mapping = config.trainer.parameter_axis_mapping
    
    def compute_loss(model: LmHeadModel, example: LmExample, key=None):
        if hasattr(model, 'compute_loss') and hasattr(model.compute_loss, '__code__'):
            if 'policy_reg_strength' in model.compute_loss.__code__.co_varnames:
                return model.compute_loss(
                    example, 
                    key=key, 
                    policy_reg_strength=config.svd_config.policy_reg_strength
                ).scalar()
        return model.compute_loss(example, key=key).scalar()
    
    # Configure weight decay patterns based on model type
    if config.model_choice == "llamaode-svd":
        from dataclasses import replace
        new_optimizer = replace(
            config.optimizer,
            weight_decay_modules=r".*\.(weight|U|V)$"
        )
        config = replace(config, optimizer=new_optimizer)
    elif config.model_choice == "neuralode-svd":
        from dataclasses import replace
        new_optimizer = replace(
            config.optimizer,
            weight_decay_modules=r".*\.(weight|U|V)$"
        )
        config = replace(config, optimizer=new_optimizer)
    elif config.model_choice == "neuralode":
        from dataclasses import replace
        new_optimizer = replace(
            config.optimizer,
            weight_decay_modules=r".*time_embedding|.*token_embeddings|.*position_embeddings",
        )
        config = replace(config, optimizer=new_optimizer)
    
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    if config.train_policy_only:
        def is_policy_param(path, value):
            return "policy.policy_net" in path
        
        trainer = Trainer(
            config.trainer, 
            optimizer, 
            compute_loss,
            is_trainable_param=is_policy_param
        )
    else:
        trainer = Trainer(config.trainer, optimizer, compute_loss)
    
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
        
        if vocab_size != Vocab.size:
            logger.info(
                f"Round vocab size from {vocab_size} to {Vocab.size} for partitioning"
            )

        if config.model_choice == "llamaode-svd":
            def model_init():
                if config.load_pretrained_ode and not config.train_svd_from_scratch:
                    logger.info("Loading pretrained Neural ODE Llama model for conversion...")
                    pass
                
                return SVDLlamaOdeLMHeadModel.init(
                    Vocab,
                    config=config.model,
                    time_embed_dim=config.time_embed_dim,
                    sinusodial_dim=config.sinusodial_dim,
                    rank_ratio=config.svd_config.rank_ratio,
                    policy_init_scale=config.svd_config.policy_init_scale,
                    key=model_key,
                )
        elif config.model_choice == "neuralode-svd":
            def model_init():
                if config.load_pretrained_ode and not config.train_svd_from_scratch:
                    logger.info("Loading pretrained Neural ODE model for conversion...")
                    pass
                
                return SVDNeuralOdeLMHeadModel.init(
                    Vocab,
                    config=config.model,
                    time_embed_dim=config.time_embed_dim,
                    sinusodial_dim=config.sinusodial_dim,
                    rank_ratio=config.svd_config.rank_ratio,
                    policy_init_scale=config.svd_config.policy_init_scale,
                    key=model_key,
                )
        elif config.model_choice == "neuralode":
            def model_init():
                return NeuralOdeLMHeadModel.init(
                    Vocab,
                    config=config.model,
                    time_embed_dim=config.time_embed_dim,
                    sinusodial_dim=config.sinusodial_dim,
                    key=model_key,
                )
        else:
            raise ValueError(f"Unknown model_choice {config.model_choice}")
        
        state = trainer.initial_state(
            training_key=train_key,
            model_init=model_init,
        )
        
        # Log parameter counts
        total_params = parameter_count(state.model)
        if hasattr(state.model, 'get_policy_params'):
            policy_params = parameter_count(state.model.get_policy_params())
            logger.info(f"Total parameters: {total_params}")
            logger.info(f"Policy parameters: {policy_params}")
            logger.info(f"Policy percentage: {policy_params/total_params*100:.2f}%")
            
            wandb.summary["total_parameter_count"] = total_params
            wandb.summary["policy_parameter_count"] = policy_params
        else:
            wandb.summary["parameter_count"] = total_params
        
        trainer.add_default_hooks(eval_loader)
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size),
            every=1,
        )
        
        # Add policy statistics logging for SVD models
        if config.model_choice in ["llamaode-svd", "neuralode-svd"]:

          def log_policy_stats(step_info):
              if hasattr(step_info.model, 'get_policy_params'):
                  policy_params_pytree = step_info.model.get_policy_params()
                  
                  arrays_only_pytree = eqx.filter(policy_params_pytree, eqx.is_array)

                  leaf_arrays = jax.tree_util.tree_leaves(arrays_only_pytree)
                  
                  if not leaf_arrays:
                      return

                  flat_params = jnp.concatenate([jnp.ravel(p.astype(jnp.float32)) for p in leaf_arrays])
                  
                  stats = {
                      "policy/mean": float(jnp.mean(flat_params)),
                      "policy/std": float(jnp.std(flat_params)),
                      "policy/max": float(jnp.max(flat_params)),
                      "policy/min": float(jnp.min(flat_params)),
                  }

                  if config.trainer.wandb.mode != "offline":
                      import wandb
                      wandb.log(stats, step=step_info.step)
          
          trainer.add_hook(log_policy_stats, every=config.trainer.steps_per_eval)

        if "svd" in config.model_choice:
            trainer.add_hook(log_diagnostics, every=config.trainer.steps_per_eval)

        # Add multiplier statistics for Llama SVD model
        if config.model_choice == "llamaode-svd":
            def log_multiplier_stats(step_info):
                if hasattr(step_info.model.transformer, 'policy'):

                    task_vector = jnp.ones((step_info.model.config.Embed.size,))
 
                    task_vector_batched = task_vector[None, :] 

                    Batch = hax.Axis("batch", 1)
                    Embed = step_info.model.config.Embed
                    named_task_vector = hax.named(task_vector_batched, (Batch, Embed))
                    
                    multipliers = step_info.model.transformer.policy(named_task_vector)
                    
                    sample_stats = {}
                    for layer_idx in [0, step_info.model.config.num_layers // 2, step_info.model.config.num_layers - 1]:
                        for proj in ["gate_proj", "up_proj", "down_proj"]:
                            key = f"layer_{layer_idx}_{proj}"
                            if key in multipliers:
                                mult_sample = multipliers[key].take(Batch, 0)
                                sample_stats[f"multipliers/{key}/mean"] = float(jnp.mean(mult_sample.array))
                                sample_stats[f"multipliers/{key}/std"] = float(jnp.std(mult_sample.array))
                    
                    if config.trainer.wandb.mode != "offline" and sample_stats:
                        import wandb
                        wandb.log(sample_stats, step=step_info.step)
            
            trainer.add_hook(log_multiplier_stats, every=config.trainer.steps_per_eval * 2)
        
        if state.step > 0:
            import tqdm
            for _ in tqdm.tqdm(
                range(state.step + 1),
                desc="finding where to resume",
            ):
                next(train_loader)
        
        trainer.train(state, train_loader)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and "--model_choice" not in " ".join(sys.argv):
        sys.argv.extend(["--model_choice", "llamaode-svd"])
    
    levanter.config.main(main)()