import logging

# GPU performance tips
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

import datasets
import equinox as eqx
import jax
import jax.random as jrandom
import levanter
import optax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.data.sharded_dataset import (
    ShardedDataset,
    TextUrlDataset,
    WrappedHFDataset,
)
from levanter.data.text import CausalLmDataset, LMDatasetConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import OptimizerConfig, Trainer, TrainerConfig
from levanter.utils.jax_utils import leaf_key_paths, parameter_count
from optax import GradientTransformation

import wandb
from qkvflow.nn.dynamic import NeuralOdeLMHeadModel
from qkvflow.nn.dynamic_llama import LlamaLMHeadModel as LlamaODELMHeadModel

try:
    from qkvflow.nn.svd_dynamic import SVDNeuralODELMHeadModel
    SVD_AVAILABLE = True
    print("SVD Neural ODE imported successfully!")
except ImportError as e:
    print(f"SVD Neural ODE not available: {e}")
    SVD_AVAILABLE = False
    SVDNeuralODELMHeadModel = None

try:
    from qkvflow.optimization.svd_ode_optimizer import SVDODEReinforce, SVDODETrainer
    SVD_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"SVD optimizers not available: {e}")
    SVD_OPTIMIZER_AVAILABLE = False
    SVDODEReinforce = None
    SVDODETrainer = None


os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)


logger = logging.getLogger(__name__)


class HFDataset(WrappedHFDataset):
    def __init__(self, id, val_ratio=0.0005, *, split, **kwargs):
        self.val_ratio = val_ratio
        super().__init__(id, split=split, **kwargs)

    def _load_dataset(self):
        raw_dataset = datasets.load_dataset(self.id, **self.kwargs)
        if "validation" not in raw_dataset:
            if not self.kwargs["streaming"]:
                # split train into subsets
                assert "train" in raw_dataset
                raw_dataset = raw_dataset["train"].train_test_split(
                    test_size=self.val_ratio,
                    seed=2357,  # same seed like flash-attention
                    shuffle=True,  # Otherwise test will be at the end of the dataset
                )
                raw_dataset["validation"] = raw_dataset["test"]
            else:
                raise NotImplementedError()

        return raw_dataset[self.split]


class DatasetConfig(LMDatasetConfig):

    val_ratio: float = 0.0005

    def get_shard_source(self, split) -> ShardedDataset[str]:
        if self.id is not None:
            hf_dataset = HFDataset(
                self.id,
                split=split,
                val_ratio=self.val_ratio,
                name=self.name,
                streaming=self.stream,
            )
            return hf_dataset.map(lambda x: x[self.text_key])
        else:
            return TextUrlDataset(self.urls_for_split(split), self.text_key)


@dataclass
class OptimizerConfigWithWeightDecay(OptimizerConfig):

    weight_decay_modules: Optional[Union[List[str], str]] = None

    def build(self, num_train_steps: int) -> GradientTransformation:
        """Creates the optimizer"""

        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))

            if self.weight_decay > 0:
                components.append(
                    optax.add_decayed_weights(
                        self.weight_decay, self.build_weight_decay_mask()
                    )
                )

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(
            learning_rate=self.lr_scheduler(num_train_steps)
        )

    def build_weight_decay_mask(self):
        if self.weight_decay_modules is None:
            return None
        else:
            # mask based on regex or module path
            def _apply_on(x, key_path):
                if isinstance(self.weight_decay_modules, str):
                    compiled_regex = re.compile(self.weight_decay_modules)
                    return compiled_regex.match(key_path) is not None
                else:
                    return any(
                        key_path.__contains__(target)
                        for target in self.weight_decay_modules
                    )

            def mask_fn(model):
                return jax.tree_util.tree_map(
                    _apply_on,
                    model,
                    leaf_key_paths(model, is_leaf=eqx.is_array),
                    is_leaf=eqx.is_array,
                )

            return mask_fn


@dataclass 
class SVDODEConfig:
    """Configuration for SVD Neural ODE specific parameters"""
    
    rank: int = 64
    num_experts: int = 4
    expert_init_scale: float = 0.05
    
    use_adaptive_mixing: bool = True
    policy_learning_rate: float = 1e-4
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    
    policy_update_frequency: int = 10
    warmup_steps: int = 1000
    expert_diversity_bonus: float = 0.1


@dataclass
class TrainLmConfig:

    model_choice: str = field(default="gpt2")

    data: DatasetConfig = field(default_factory=DatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=Gpt2Config)
    optimizer: OptimizerConfigWithWeightDecay = field(
        default_factory=OptimizerConfigWithWeightDecay
    )

    # config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    use_hf_model_config: bool = False

    fcm_prob: float = 0.0  # forgetful context masking prob. recommended 0.15

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000

    # additional config for Neural ODE
    time_embed_dim: int = 100
    sinusodial_dim: int = 16
    num_check_points: int = 2
    rank: int = 8
    alpha: float = 1.0
    num_blocks: int = 4
    multiplier: int = 2
    
    svd_ode: SVDODEConfig = field(default_factory=SVDODEConfig)


def create_svd_compute_loss_function(config: TrainLmConfig):
    """Create compute loss function for SVD Neural ODE"""

    step_counter = {"step": 0}
    
    def svd_compute_loss(model, example, key=None):
        step_counter["step"] += 1
        current_step = step_counter["step"]
        
        if hasattr(model, 'compute_policy_loss') and SVD_AVAILABLE:
            policy_losses = model.compute_policy_loss(example, key=key)

            if current_step > config.svd_ode.warmup_steps:
                total_loss = policy_losses["lm_loss"] + policy_losses["policy_loss"]
            else:
                total_loss = policy_losses["lm_loss"]

            if current_step % 100 == 0:
                try:
                    log_dict = {
                        "policy_loss": float(policy_losses["policy_loss"]),
                        "value_loss": float(policy_losses["value_loss"]), 
                        "entropy": float(policy_losses["entropy"]),
                        "reward": float(policy_losses["reward"]),
                        "training_step": current_step,
                        "warmup_phase": current_step <= config.svd_ode.warmup_steps,
                    }
                    wandb.log(log_dict)
                except Exception as e:
                    logger.warning(f"Failed to log wandb metrics: {e}")
            
            return total_loss
        else:
            return model.compute_loss(example, key=key).scalar()
    
    return svd_compute_loss


def main(config: TrainLmConfig):
    logger.info(f"Training model {config.model_choice}")

    tokenizer = config.data.the_tokenizer

    if isinstance(config.model, HFCompatConfig):
        converter = config.model.default_hf_checkpoint_converter
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    config.trainer.initialize(config)

    seed = config.trainer.seed
    model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)

    Batch = config.trainer.TrainBatch
    EvalBatch = config.trainer.EvalBatch
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    def compute_loss(model: LmHeadModel, example: LmExample, key=None):
        return model.compute_loss(example, key=key).scalar()

    if config.model_choice == "gpt2" or config.model_choice == "llama":
        from dataclasses import replace
        new_optimizer = replace(
            config.optimizer,
            weight_decay_modules=r".*attn.*weight|.*mlp.*weight|.*token_embeddings|.*position_embeddings",
        )
        config = replace(config, optimizer=new_optimizer)
    elif config.model_choice in ["neuralode", "llamaode"]:
        from dataclasses import replace
        new_optimizer = replace(
            config.optimizer,
            weight_decay_modules=r".*time_embedding|.*token_embeddings|.*position_embeddings",
        )
        config = replace(config, optimizer=new_optimizer)
    elif config.model_choice == "svd_neuralode":
        from dataclasses import replace
        new_optimizer = replace(
            config.optimizer,
            weight_decay_modules=r".*time_embedding|.*token_embeddings|.*position_embeddings|.*svd.*",
        )
        config = replace(config, optimizer=new_optimizer)
        
        # Use SVD-specific compute loss function
        if SVD_AVAILABLE:
            compute_loss = create_svd_compute_loss_function(config)
        else:
            logger.warning("SVD Neural ODE not available, falling back to standard loss")

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

        if config.model_choice == "gpt2" or config.model_choice == "llama":
            model_init = lambda: config.model.build(Vocab, key=model_key)
        elif config.model_choice == "neuralode":
            def model_init():
                return NeuralOdeLMHeadModel.init(
                    Vocab,
                    config=config.model,
                    time_embed_dim=config.time_embed_dim,
                    sinusodial_dim=config.sinusodial_dim,
                    key=model_key,
                )

        elif config.model_choice == "llamaode":
            def model_init():
                return LlamaODELMHeadModel.init(
                    Vocab,
                    config=config.model,
                    time_embed_dim=config.time_embed_dim,
                    sinusodial_dim=config.sinusodial_dim,
                    key=model_key,
                )
        
        elif config.model_choice == "svd_neuralode":
            if not SVD_AVAILABLE:
                raise ValueError("SVD Neural ODE not available. Please install required dependencies.")
                
            def model_init():
                # Create training configuration for SVD model
                training_config = {
                    "optimizer_type": "reinforce",
                    "learning_rate": config.svd_ode.policy_learning_rate,
                    "entropy_coeff": config.svd_ode.entropy_coeff,
                    "value_loss_coeff": config.svd_ode.value_loss_coeff,
                }
                
                return SVDNeuralODELMHeadModel.init(
                    Vocab=Vocab,
                    config=config.model,
                    rank=config.svd_ode.rank,
                    num_experts=config.svd_ode.num_experts,
                    time_embed_dim=config.time_embed_dim,
                    sinusodial_dim=config.sinusodial_dim,
                    use_adaptive_mixing=config.svd_ode.use_adaptive_mixing,
                    training_config=training_config,
                    key=model_key,
                )

        else:
            raise ValueError(f"Unknown model_choice {config.model_choice}")

        state = trainer.initial_state(
            training_key=train_key,
            model_init=model_init,
        )

        wandb.summary["parameter_count"] = parameter_count(state.model)
        
        if config.model_choice == "svd_neuralode":
            wandb.summary["svd_rank"] = config.svd_ode.rank
            wandb.summary["num_experts"] = config.svd_ode.num_experts
            wandb.summary["adaptive_mixing"] = config.svd_ode.use_adaptive_mixing

        trainer.add_default_hooks(eval_loader)
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size),
            every=1,
        )
        
        if config.model_choice == "svd_neuralode" and SVD_AVAILABLE:
            def log_svd_analysis(state):
                try:
                    dummy_tokens = hax.random.randint(
                        jrandom.PRNGKey(42), 
                        (min(64, Pos.size),), 
                        0, min(vocab_size, 1000)
                    )
                    dummy_example = type('Example', (), {
                        'tokens': dummy_tokens,
                        'attn_mask': None,
                        'loss_mask': hax.ones_like(dummy_tokens, dtype=bool)
                    })()
                    
                    # Analyze expert usage
                    expert_analysis = state.model.analyze_expert_usage(dummy_example)
                    
                    # Log expert diversity (simplified)
                    if "attention_experts" in expert_analysis:
                        attn_usage = expert_analysis["attention_experts"]
                        if hasattr(attn_usage, 'mean'):
                            attn_entropy = -jnp.sum(
                                attn_usage.mean(0) * jnp.log(attn_usage.mean(0) + 1e-8),
                                axis=-1
                            ).mean()
                        else:
                            attn_entropy = jnp.array(1.0)  # Default entropy
                    else:
                        attn_entropy = jnp.array(1.0)
                    
                    wandb.log({
                        "expert_diversity/attention_entropy": float(attn_entropy),
                        "training_step": state.step,
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to log SVD analysis: {e}")
            
            trainer.add_hook(log_svd_analysis, every=config.svd_ode.policy_update_frequency)

        if state.step > 0:
            import tqdm
            for _ in tqdm.tqdm(
                range(state.step + 1),
                desc="finding where to resume",
            ):
                next(train_loader)

        # Custom training loop for SVD models
        if config.model_choice == "svd_neuralode" and SVD_AVAILABLE:
            logger.info("Starting SVD Neural ODE training")
            
            # Initialize SVD-specific trainer components if available
            if SVD_OPTIMIZER_AVAILABLE:
                try:
                    svd_optimizer = SVDODEReinforce(
                        learning_rate=config.svd_ode.policy_learning_rate,
                        entropy_coeff=config.svd_ode.entropy_coeff,
                        value_loss_coeff=config.svd_ode.value_loss_coeff,
                    )
                    
                    svd_trainer = SVDODETrainer(
                        model=state.model,
                        policy=state.model.policy,
                        optimizer=svd_optimizer,
                        config=config.svd_ode.__dict__,
                    )
                    logger.info("SVD trainer initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize SVD trainer: {e}")
                    logger.info("Continuing with standard training")
            else:
                logger.info("SVD optimizers not available, using standard training")

        trainer.train(state, train_loader)


if __name__ == "__main__":
    levanter.config.main(main)()