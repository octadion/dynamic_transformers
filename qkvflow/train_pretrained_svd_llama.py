import logging
import jax
import jax.numpy as jnp
import equinox as eqx
import haliax as hax
from dataclasses import dataclass
import copy
import levanter.models.llama as levanter_llama

if hasattr(levanter_llama, "LlamaLMHeadModel"): LlamaLmHeadModel = levanter_llama.LlamaLMHeadModel
elif hasattr(levanter_llama, "LlamaLmHeadModel"): LlamaLmHeadModel = levanter_llama.LlamaLmHeadModel
else: LlamaLmHeadModel = levanter_llama.LlamaForCausalLM

from qkvflow.utils.hf_loader import load_llama_from_hf, convert_llama_config_from_hf
logger = logging.getLogger(__name__)

@dataclass
class PretrainedSVDConfig:
    pretrained_model_name_or_path: str = "meta-llama/Meta-Llama-3-8B"
    type: str = "llama_svd"

def create_model_from_pretrained(pretrained_model_name, config, Vocab, key):
    logger.info(f"Loading weights from {pretrained_model_name}...")
    pt_model, tokenizer, model_info = load_llama_from_hf(model_name=pretrained_model_name)
    levanter_config = convert_llama_config_from_hf(model_info)
    
    if hasattr(config, "seq_len") and config.seq_len is not None:
        conf_dict = copy.deepcopy(levanter_config.__dict__)
        conf_dict["seq_len"] = config.seq_len
        valid_keys = levanter_config.__dataclass_fields__.keys()
        clean_dict = {k: v for k, v in conf_dict.items() if k in valid_keys}
        levanter_config = levanter_llama.LlamaConfig(**clean_dict)
    
    logger.info("Initializing Levanter model structure...")
    with jax.default_device(jax.devices("cpu")[0]):
        model = LlamaLmHeadModel.init(Vocab, levanter_config, key=key)
    
    del pt_model
    import gc; gc.collect()
    return model