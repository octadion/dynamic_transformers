import logging, os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
try:
    from huggingface_hub import login
    token = os.environ.get("HF_TOKEN")
    if token: login(token=token)
except ImportError: pass

import jax, optax, levanter
import jax.random as jrandom
import jax.numpy as jnp
import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.trainer import Trainer
from levanter.utils.jax_utils import parameter_count
from levanter.models.lm_model import LmExample
from qkvflow.data.multi_task_dataset import MultiTaskDataset, MultiTaskConfig, create_haliax_batch
from qkvflow.train_pretrained_svd_llama import create_model_from_pretrained

logger = logging.getLogger(__name__)

@dataclass
class LocalSVDConfig:
    pretrained_model_name: str = "meta-llama/Meta-Llama-3-8B"
    pretrained_cache_dir: Optional[str] = None
    type: str = "llama_svd"
    hidden_dim: Optional[int] = None
    intermediate_dim: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    num_layers: Optional[int] = None
    seq_len: int = 2048
    svd_config: Any = None
    gradient_checkpointing: bool = True
    gradient_checkpointing_block_size: int = 5

@dataclass
class MultiTaskTrainingConfig:
    model: LocalSVDConfig
    trainer: levanter.trainer.TrainerConfig
    optimizer: Any = field(default=None) 
    multi_task: MultiTaskConfig = field(default_factory=MultiTaskConfig)
    data: Dict[str, Any] = field(default_factory=dict)
    svd_config: Dict[str, Any] = field(default_factory=dict)
    freeze_backbone: bool = True
    train_policy_only: bool = True
    train_svd_from_scratch: bool = False
    pretrained_model_name: str = "meta-llama/Meta-Llama-3-8B"
    log_task_performance: bool = True
    log_task_every: int = 100
    
    model_choice: str = "llamaode-svd"
    pretrained_cache_dir: Optional[str] = None
    sinusodial_dim: int = 96
    task_batch_mixing: str = "random"
    time_embed_dim: int = 100

class MultiTaskTrainer:
    def __init__(self, config, trainer, dataset):
        self.config = config; self.trainer = trainer; self.dataset = dataset
    def create_train_loader(self):
        Batch = self.config.trainer.TrainBatch
        if not hasattr(self.config.model, 'Pos'): self.config.model.Pos = Axis("position", self.config.model.seq_len)
        Pos = self.config.model.Pos
        base_iterator = self.dataset.create_iterator(batch_size=Batch.size, shuffle=True, seed=self.config.trainer.seed)
        def data_iterator():
            for tokens, task_ids in base_iterator:
                input_ids, task_ids_named = create_haliax_batch(tokens, task_ids, Batch, Pos)
                attn_mask = hax.named(jnp.array(tokens != self.dataset.tokenizer.pad_token_id), (Batch, Pos))
                yield LmExample(tokens=input_ids, attn_mask=attn_mask, loss_mask=attn_mask), task_ids_named
        return data_iterator()
    def compute_loss_with_task_tracking(self, model, example_and_task, key=None):
        example, task_ids = example_and_task
        reg = 0.01
        if self.config.svd_config and 'policy_reg_strength' in self.config.svd_config: reg = self.config.svd_config['policy_reg_strength']
        return model.compute_loss(example, key=key, policy_reg_strength=reg).scalar()
    def train(self, initial_state):
        train_loader = self.create_train_loader()
        orig_loss = self.trainer.loss_fn
        self.trainer.loss_fn = self.compute_loss_with_task_tracking
        final_state = self.trainer.train(initial_state, train_loader)
        self.trainer.loss_fn = orig_loss
        return final_state

def build_memory_efficient_optimizer(learning_rate=1e-4):
    def param_labels(path, _):
        path_str = "/".join([str(p) for p in path])
        # Hanya latih policy/svd, backbone di-freeze total (RAM=0)
        return "train" if ("policy" in path_str or "svd" in path_str) else "freeze"
    
    transforms = {
        "train": optax.adafactor(learning_rate=learning_rate, decay_rate=0.8, clippling_threshold=1.0),
        "freeze": optax.set_to_zero(),
    }
    return optax.multi_transform(transforms, param_labels)

def main(config: MultiTaskTrainingConfig):
    logging.basicConfig(level=logging.INFO)
    from transformers import AutoTokenizer
    model_name = config.model.pretrained_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    config.trainer.initialize(config)
    seed = config.trainer.seed
    model_key, train_key = jrandom.split(jrandom.PRNGKey(seed), 2)

    optimizer = build_memory_efficient_optimizer(learning_rate=5e-4)
    
    dataset = MultiTaskDataset(tokenizer, config.multi_task)
    if not hasattr(config.model, 'Pos'): config.model.Pos = Axis("position", config.model.seq_len)
    Pos = config.model.Pos
    Vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), config.trainer.parameter_axis_mapping)

    with config.trainer.device_mesh:
        if hasattr(config.model, "gradient_checkpointing"): config.model.gradient_checkpointing = True
        model = create_model_from_pretrained(pretrained_model_name=model_name, config=config.model, Vocab=Vocab, key=model_key)
        def dummy_loss(m, e, key): return m.compute_loss(e, key=key).scalar()
        trainer = Trainer(config.trainer, optimizer, dummy_loss)
        state = trainer.initial_state(train_key, model=model)
        mt_trainer = MultiTaskTrainer(config, trainer, dataset)
        mt_trainer.train(state)

if __name__ == "__main__":
    levanter.config.main(main)()