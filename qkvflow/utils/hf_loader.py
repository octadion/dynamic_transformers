import logging
import os
from dataclasses import fields
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import levanter.models.llama as levanter_llama

logger = logging.getLogger(__name__)

def load_llama_from_hf(model_name: str, cache_dir: Optional[str] = None, torch_dtype: str = "float16"):
    logger.info(f"Loading {model_name} (Low Memory Mode)...")
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(model_name, token=token, cache_dir=cache_dir)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    try:
        pt_model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, cache_dir=cache_dir, token=token,
            torch_dtype=getattr(torch, torch_dtype), low_cpu_mem_usage=True, device_map="cpu"
        )
    except Exception:
        pt_model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, cache_dir=cache_dir, token=token,
            torch_dtype=getattr(torch, torch_dtype), low_cpu_mem_usage=True
        )

    model_info = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "intermediate_size": getattr(config, "intermediate_size", 4 * config.hidden_size),
        "num_key_value_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-5),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "max_position_embeddings": getattr(config, "max_position_embeddings", 2048),
    }
    return pt_model, tokenizer, model_info

def convert_llama_config_from_hf(model_info: Dict[str, Any]):
    LlamaConfig = levanter_llama.LlamaConfig
    valid_fields = {f.name for f in fields(LlamaConfig)}
    args = {
        "seq_len": model_info.get("max_position_embeddings", 2048),
        "hidden_dim": model_info["hidden_size"],
        "num_layers": model_info["num_hidden_layers"],
        "num_heads": model_info["num_attention_heads"],
        "intermediate_dim": model_info["intermediate_size"],
        "use_flash_attention": True,
    }
    kv_heads = model_info["num_key_value_heads"]
    if "num_kv_heads" in valid_fields: args["num_kv_heads"] = kv_heads
    elif "kv_heads" in valid_fields: args["kv_heads"] = kv_heads
    
    final_args = {k: v for k, v in args.items() if k in valid_fields}
    return LlamaConfig(**final_args)

def extract_mlp_weight(model, layer_idx):
    layer = model.model.layers[layer_idx]
    gate_proj = layer.mlp.gate_proj.weight.detach().cpu().numpy()
    up_proj = layer.mlp.up_proj.weight.detach().cpu().numpy()
    down_proj = layer.mlp.down_proj.weight.detach().cpu().numpy()
    return gate_proj, up_proj, down_proj