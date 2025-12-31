import logging
import jax
import jax.numpy as jnp
import equinox as eqx
import haliax as hax
from haliax import Axis
from typing import Any
from dataclasses import dataclass
import copy

# Dynamic imports for Levanter Llama Class
import levanter.models.llama as levanter_llama
LlamaConfig = levanter_llama.LlamaConfig

# Find the correct model class name dynamically
if hasattr(levanter_llama, "LlamaLMHeadModel"):
    LlamaLmHeadModel = levanter_llama.LlamaLMHeadModel
elif hasattr(levanter_llama, "LlamaLmHeadModel"):
    LlamaLmHeadModel = levanter_llama.LlamaLmHeadModel
else:
    LlamaLmHeadModel = levanter_llama.LlamaForCausalLM

from qkvflow.utils.hf_loader import (
    load_llama_from_hf,
    extract_mlp_weight,
    convert_llama_config_from_hf
)
# from qkvflow.nn.svd_from_pretrained import initialize_svd_mlp_from_pretrained # Uncomment if ready

logger = logging.getLogger(__name__)

@dataclass
class PretrainedSVDConfig:
    pretrained_model_name_or_path: str = "meta-llama/Meta-Llama-3-8B"
    type: str = "llama_svd"

def create_model_from_pretrained(pretrained_model_name, config, Vocab, key):
    logger.info(f"Loading weights from {pretrained_model_name}...")
    
    # 1. Load HF Weights & Info
    pt_model, tokenizer, model_info = load_llama_from_hf(model_name=pretrained_model_name)
    
    # 2. Convert Config
    levanter_config = convert_llama_config_from_hf(model_info)
    
    # Override seq_len if provided in config
    if hasattr(config, "seq_len") and config.seq_len is not None:
        # Create a new config with updated seq_len safely
        conf_dict = copy.deepcopy(levanter_config.__dict__)
        conf_dict["seq_len"] = config.seq_len
        # Filter again just to be safe
        valid_keys = levanter_config.__dataclass_fields__.keys()
        clean_dict = {k: v for k, v in conf_dict.items() if k in valid_keys}
        levanter_config = LlamaConfig(**clean_dict)
    
    # 3. Initialize Levanter Model
    logger.info("Initializing Levanter model structure...")
    with jax.default_device(jax.devices("cpu")[0]):
        model = LlamaLmHeadModel.init(Vocab, levanter_config, key=key)
    
    # 4. SVD Weight Injection Logic (Disabled for OOM Debugging Base)
    # model = initialize_svd_mlp_from_pretrained(model, pt_model, ...)
    
    # Clean up PyTorch model to free RAM
    del pt_model
    import gc
    gc.collect()

    return model