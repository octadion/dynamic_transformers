import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.random as jrandom
import jax.numpy as jnp
import levanter
import equinox as eqx
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter import callbacks
from levanter.data.text import CausalLmDataset
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.trainer import Trainer
from levanter.utils.jax_utils import parameter_count

import wandb
from qkvflow.nn.dynamic import NeuralOdeLMHeadModel
from qkvflow.nn.dynamic_svd import SVDNeuralOdeLMHeadModel
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
class TrainSVDLmConfig(TrainLmConfig):
    """Extended config for SVD model training."""
    
    svd_config: SVDModelConfig = field(default_factory=SVDModelConfig)
    
    load_pretrained_ode: Optional[str] = None
    
    train_policy_only: bool = False 
    train_svd_from_scratch: bool = True 


def main(config: TrainSVDLmConfig):
    logger.info(f"Training SVD Neural ODE model {config.model_choice}")
    
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
    
    if config.model_choice == "neuralode-svd":
        from dataclasses import replace
        new_optimizer = replace(
            config.optimizer,
            weight_decay_modules=r".*(U|V)$"
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

        if config.model_choice == "neuralode-svd":
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
        
        if config.model_choice == "neuralode-svd":

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
        sys.argv.extend(["--model_choice", "neuralode-svd"])
    
    levanter.config.main(main)()