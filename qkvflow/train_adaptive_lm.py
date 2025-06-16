import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
import levanter
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter import callbacks
from levanter.data.text import CausalLmDataset
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count

import wandb
from qkvflow.nn.adaptive_transformer import AdaptiveNeuralOdeLMHeadModel
from qkvflow.train_lm import DatasetConfig, OptimizerConfigWithWeightDecay

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveTrainLmConfig:
    """Configuration for training adaptive Neural ODE language models."""
    
    model_choice: str = field(default="adaptive_neuralode")
    
    data: DatasetConfig = field(default_factory=DatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: Gpt2Config = field(default_factory=Gpt2Config)
    optimizer: OptimizerConfigWithWeightDecay = field(
        default_factory=OptimizerConfigWithWeightDecay
    )
    
    time_embed_dim: int = 100
    sinusodial_dim: int = 16
    num_experts: int = 4
    
    expert_types: list = field(default_factory=lambda: ["general", "math", "code", "reasoning"])
    use_difficulty_adaptation: bool = True
    use_task_prediction: bool = True
    
    use_expert_loss: bool = True
    expert_loss_weight: float = 0.1
    diversity_loss_weight: float = 0.05


def main(config: AdaptiveTrainLmConfig):
    logger.info(f"Training adaptive model {config.model_choice}")
    
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
        lm_loss = model.compute_loss(example, key=key).scalar()
        
        if not config.use_expert_loss:
            return lm_loss
        
        try:
            expert_analysis = model.get_expert_analysis(
                example.tokens, example.attn_mask, key=key
            )

            diversity_loss = 0.0
            num_layers = model.config.num_layers
            
            for layer_idx in range(num_layers):
                expert_key = f"layer_{layer_idx}_attn_experts"
                if expert_key in expert_analysis:
                    expert_weights = expert_analysis[expert_key]["expert_weights"]
                    entropy = -jnp.sum(expert_weights * jnp.log(expert_weights + 1e-8))
                    max_entropy = jnp.log(expert_weights.axis_size("expert"))
                    diversity_loss += (max_entropy - entropy) / num_layers
            
            total_loss = lm_loss + config.diversity_loss_weight * diversity_loss
            
        except Exception as e:
            logger.warning(f"Expert analysis failed: {e}, using standard loss")
            total_loss = lm_loss
        
        return total_loss
    
    from dataclasses import replace
    new_optimizer = replace(
        config.optimizer,
        weight_decay_modules=r".*expert_.*|.*time_embedding|.*token_embeddings",
    )
    config = replace(config, optimizer=new_optimizer)
    
    optimizer = config.optimizer.build(config.trainer.num_train_steps)
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
        
        def model_init():
            return AdaptiveNeuralOdeLMHeadModel.init(
                Vocab,
                config=config.model,
                time_embed_dim=config.time_embed_dim,
                sinusodial_dim=config.sinusodial_dim,
                num_experts=config.num_experts,
                key=model_key,
            )
        
        state = trainer.initial_state(
            training_key=train_key,
            model_init=model_init,
        )
        
        wandb.summary["parameter_count"] = parameter_count(state.model)
        wandb.summary["num_experts"] = config.num_experts
        
        trainer.add_default_hooks(eval_loader)
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size),
            every=1,
        )
        
        def log_expert_analysis(step_info):
            if step_info.step % 100 == 0:
                try:
                    model = step_info.model
                    example = next(iter(eval_loader))
                    tokens_sample = example.tokens.take("batch", 0, axis=0).add_axis("batch", 1)
                    mask_sample = example.attn_mask.take("batch", 0, axis=0).add_axis("batch", 1) if example.attn_mask is not None else None
                    
                    expert_analysis = model.get_expert_analysis(
                        tokens_sample, mask_sample, key=jrandom.PRNGKey(42)
                    )

                    expert_stats = {}
                    for layer_idx in range(min(model.config.num_layers, 4)):
                        expert_key = f"layer_{layer_idx}_attn_experts"
                        if expert_key in expert_analysis:
                            expert_weights = expert_analysis[expert_key]["expert_weights"]
                            for expert_idx in range(config.num_experts):
                                weight = expert_weights.take("expert", expert_idx).array.item()
                                expert_stats[f"expert_weight/layer_{layer_idx}_expert_{expert_idx}"] = weight
                    
                    if expert_stats:
                        wandb.log(expert_stats, step=step_info.step)
                        
                except Exception as e:
                    logger.warning(f"Expert analysis logging failed: {e}")
        
        trainer.add_hook(log_expert_analysis, every=100)
        
        if state.step > 0:
            import tqdm
            for _ in tqdm.tqdm(range(state.step + 1), desc="finding where to resume"):
                next(train_loader)
        
        trainer.train(state, train_loader)