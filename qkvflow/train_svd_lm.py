import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, List

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
from levanter.models.lm_model import LmExample, LmHeadModel
from jax.tree_util import keystr
from levanter.trainer import Trainer
import itertools
from levanter.utils.jax_utils import parameter_count
from jaxtyping import PRNGKeyArray
import wandb
from qkvflow.nn.dynamic import NeuralOdeLMHeadModel
from qkvflow.nn.dynamic_svd import SVDNeuralOdeLMHeadModel
from qkvflow.train_lm import (
    DatasetConfig, OptimizerConfigWithWeightDecay, TrainLmConfig
)
from levanter.utils.tree_utils import inference_mode
import optax


logger = logging.getLogger(__name__)


@dataclass
class SVDModelConfig:
    """Configuration for SVD adaptive model."""
    rank_ratio: float = 0.5 
    policy_init_scale: float = 0.1
    policy_reg_strength: float = 0.01
    policy_hidden_dim_ratio: float = 4.0
    policy_activation_strength : float = 0.01
@dataclass
class TrainSVDLmConfig(TrainLmConfig):
    """Extended config for SVD model training."""
    svd_config: SVDModelConfig = field(default_factory=SVDModelConfig)
    load_pretrained_ode: Optional[str] = None
    train_policy_only: bool = False 
    train_svd_from_scratch: bool = True 

class GradientLogger:
    def __init__(self, trainer: Trainer, data_loader: itertools.islice, key: PRNGKeyArray):
        self.trainer = trainer
        self.loader = data_loader
        self.key = key
        loss_fn = self.trainer.loss_fn
        grad_fn = jax.value_and_grad(loss_fn, allow_int=True)
        self._grad_fn = eqx.filter_jit(grad_fn)

    def __call__(self, step_info):
        if wandb.run is None: return
        try:
            example = next(self.loader)
        except StopIteration:
            return
        model = step_info.model
        model_for_grad = inference_mode(model, False)
        self.key, key_for_grad = jax.random.split(self.key)
        loss, grads = self._grad_fn(model_for_grad, example, key=key_for_grad)
        filter_spec = jax.tree_util.tree_map_with_path(lambda path, leaf: "policy" in keystr(path), model)
        policy_grads, backbone_grads = eqx.partition(grads, filter_spec)
        def get_grad_norm(pytree):
            leaves = jax.tree_util.tree_leaves(pytree)
            if not leaves: return 0.0
            return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves if isinstance(g, jax.Array)))
        policy_grad_norm = get_grad_norm(policy_grads)
        backbone_grad_norm = get_grad_norm(backbone_grads)
        wandb.log({
            "gradients/policy_norm": float(policy_grad_norm),
            "gradients/backbone_norm": float(backbone_grad_norm),
            "gradients/ratio_policy_vs_backbone": float(policy_grad_norm / (backbone_grad_norm + 1e-8))
        }, step=step_info.step)

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
        return model.compute_loss(
            example, 
            key=key, 
            policy_activation_strength=config.svd_config.policy_activation_strength
        ).scalar()

    optimizer = config.optimizer.build(config.trainer.num_train_steps)
    trainer = Trainer(config.trainer, optimizer, compute_loss)
    
    with trainer.device_mesh:
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Membulatkan ukuran vocab dari {vocab_size} ke {Vocab.size} untuk partisi")

        if config.model_choice == "neuralode-svd":
            def model_init():
                return SVDNeuralOdeLMHeadModel.init(
                    Vocab=Vocab,
                    config=config.model,
                    time_embed_dim=config.time_embed_dim,
                    sinusodial_dim=config.sinusodial_dim,
                    rank_ratio=config.svd_config.rank_ratio,
                    policy_init_scale=config.svd_config.policy_init_scale,
                    policy_hidden_dim_ratio=config.svd_config.policy_hidden_dim_ratio,
                    key=model_key,
                )
        else:
            raise ValueError(f"Model choice '{config.model_choice}' tidak didukung di skrip ini.")

        eval_dataset = CausalLmDataset(config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos)
        eval_loader = trainer.replicated_loader(eval_dataset, EvalBatch)
        
        train_dataset = CausalLmDataset(config.data.token_seq_dataset("train", Pos.size), Pos, KeyPos)
        train_loader = iter(trainer.sharded_loader(train_dataset, Batch))

        state = trainer.initial_state(training_key=train_key, model_init=model_init)
        
        
        if wandb.run:
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
        trainer.add_hook(callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size), every=1)
        
        train_loader_for_hooks, train_loader = itertools.tee(train_loader)
        grad_logger = GradientLogger(trainer, train_loader_for_hooks, key=train_key)
        trainer.add_hook(grad_logger, every=10)
        
        if config.model_choice == "neuralode-svd":
            def log_policy_stats(step_info):
                if hasattr(step_info.model, 'get_policy_params'):
                    policy_params_pytree = step_info.model.get_policy_params()
                    arrays_only_pytree = eqx.filter(policy_params_pytree, eqx.is_array)
                    leaf_arrays = jax.tree_util.tree_leaves(arrays_only_pytree)
                    if not leaf_arrays: return
                    
                    flat_params = jnp.concatenate([jnp.ravel(p.astype(jnp.float32)) for p in leaf_arrays])
                    stats = {
                        "policy/mean": float(jnp.mean(flat_params)),
                        "policy/std": float(jnp.std(flat_params)),
                        "policy/max": float(jnp.max(flat_params)),
                        "policy/min": float(jnp.min(flat_params)),
                    }

                    if hasattr(config.svd_config, "policy_activation_strength"):
                        activation_strength = config.svd_config.policy_activation_strength
                        activation_loss = step_info.model.transformer.get_policy_loss(activation_strength)
                        stats["loss/activation_policy"] = float(activation_loss)

                    if wandb.run:
                        wandb.log(stats, step=step_info.step)
            trainer.add_hook(log_policy_stats, every=config.trainer.steps_per_eval)
        
        if state.step > 0:
            import tqdm
            for _ in tqdm.tqdm(range(state.step), desc="Mencari posisi resume data loader"):
                next(train_loader)
        
        if "svd" in config.model_choice:
            trainer.add_hook(log_diagnostics, every=config.trainer.steps_per_eval)
        
        trainer.train(state, train_loader)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and "--model_choice" not in " ".join(sys.argv):
        sys.argv.extend(["--model_choice", "neuralode-svd"])
    
    levanter.config.main(main)()