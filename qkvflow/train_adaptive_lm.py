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
    """Main training function with better error handling."""
    
    logger.info(f"🚀 Starting training for adaptive model {config.model_choice}")
    logger.info(f"📊 Model config: {config.model.hidden_dim}d, {config.model.num_layers} layers")
    logger.info(f"🧠 Adaptive config: {config.num_experts} experts, {config.time_embed_dim}d time embed")
    
    try:
        tokenizer = config.data.the_tokenizer
        logger.info(f"✓ Tokenizer loaded: {len(tokenizer)} vocab size")
        
        config.trainer.initialize(config)
        logger.info(f"✓ Trainer initialized")
        
        seed = config.trainer.seed
        model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)
        
        Batch = config.trainer.TrainBatch
        EvalBatch = config.trainer.EvalBatch
        Pos = config.model.Pos
        KeyPos = config.model.KeyPos
        
        parameter_axis_mapping = config.trainer.parameter_axis_mapping
        
        def compute_loss(model: LmHeadModel, example: LmExample, key=None):
            """Compute loss with expert regularization."""
            
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
        
        logger.info("🔄 Loading datasets...")
        
        try:
            eval_dataset = CausalLmDataset(
                config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos
            )
            eval_loader = trainer.replicated_loader(eval_dataset, EvalBatch)
            logger.info(f"✓ Eval dataset loaded")
            
            train_dataset = CausalLmDataset(
                config.data.token_seq_dataset("train", Pos.size), Pos, KeyPos
            )
            train_loader = iter(trainer.sharded_loader(train_dataset, Batch))
            logger.info(f"✓ Train dataset loaded")
            
        except Exception as e:
            logger.error(f"❌ Dataset loading failed: {e}")
            raise
        
        with trainer.device_mesh:
            vocab_size = len(tokenizer)
            Vocab = round_axis_for_partitioning(
                Axis("vocab", vocab_size), parameter_axis_mapping
            )
            
            if vocab_size != Vocab.size:
                logger.info(
                    f"📏 Rounded vocab size from {vocab_size} to {Vocab.size} for partitioning"
                )
            
            logger.info("🏗️ Initializing model...")
            
            def model_init():
                return AdaptiveNeuralOdeLMHeadModel.init(
                    Vocab,
                    config=config.model,
                    time_embed_dim=config.time_embed_dim,
                    sinusodial_dim=config.sinusodial_dim,
                    num_experts=config.num_experts,
                    key=model_key,
                )
            
            try:
                state = trainer.initial_state(
                    training_key=train_key,
                    model_init=model_init,
                )
                logger.info(f"✓ Model initialized successfully")
                
                param_count = parameter_count(state.model)
                logger.info(f"📈 Parameter count: {param_count:,}")
                
                wandb.summary["parameter_count"] = param_count
                wandb.summary["num_experts"] = config.num_experts
                
            except Exception as e:
                logger.error(f"❌ Model initialization failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise

            trainer.add_default_hooks(eval_loader)
            trainer.add_hook(
                callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size),
                every=1,
            )
            
            def log_expert_analysis(step_info):
                """Log expert analysis periodically."""
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
                logger.info(f"📋 Resuming from step {state.step}")
                import tqdm
                for _ in tqdm.tqdm(range(state.step + 1), desc="finding where to resume"):
                    next(train_loader)
            
            logger.info("🎯 Starting training...")
            trainer.train(state, train_loader)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        raise
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        config = AdaptiveTrainLmConfig(
            data=DatasetConfig(id="Ankursingh/openwebtext_10K"),
            model=Gpt2Config(
                hidden_dim=64,
                num_heads=4,
                num_layers=2,
                seq_len=32,
            ),
            trainer=TrainerConfig(
                train_batch_size=2,
                num_train_steps=5,
            ),
            time_embed_dim=16,
            sinusodial_dim=8,
            num_experts=2,
            use_expert_loss=False,
        )
        logger.info("🧪 Running in test mode with minimal configuration")
    else:
        levanter.config.main(main)()