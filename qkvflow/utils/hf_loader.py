# qkvflow/utils/hf_loader.py
"""
Load pretrained Llama models from HuggingFace and convert to JAX/Levanter format.
"""
import logging
from typing import Optional, Dict, Any
import jax
import jax.numpy as jnp
import equinox as eqx
import haliax as hax
from transformers import AutoModelForCausalLM, AutoTokenizer
from levanter.models.llama import LlamaConfig

logger = logging.getLogger(__name__)


def load_llama_from_hf(
    model_name: str = "meta-llama/Llama-3.1-8B",
    cache_dir: Optional[str] = None,
    torch_dtype: str = "float32",
) -> tuple[AutoModelForCausalLM, AutoTokenizer, Dict[str, Any]]:
    """
    Load Llama model from HuggingFace Hub.
    
    Args:
        model_name: HF model identifier
        cache_dir: Local cache directory
        torch_dtype: PyTorch dtype for loading
        
    Returns:
        (pt_model, tokenizer, model_info)
    """
    logger.info(f"Loading {model_name} from HuggingFace Hub...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load PyTorch model
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    
    pt_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=dtype_map[torch_dtype],
        device_map="cpu",  # Load to CPU first
    )
    
    # Extract model info
    config = pt_model.config
    model_info = {
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "rms_norm_eps": config.rms_norm_eps,
    }
    
    logger.info(f"✓ Loaded {model_name}")
    logger.info(f"  - Parameters: {sum(p.numel() for p in pt_model.parameters()) / 1e9:.2f}B")
    logger.info(f"  - Vocab size: {model_info['vocab_size']}")
    logger.info(f"  - Hidden size: {model_info['hidden_size']}")
    logger.info(f"  - Layers: {model_info['num_layers']}")
    
    return pt_model, tokenizer, model_info


def convert_llama_config_from_hf(model_info: Dict[str, Any]) -> LlamaConfig:
    """
    Convert HF model info to Levanter LlamaConfig.
    """
    return LlamaConfig(
        seq_len=model_info["max_position_embeddings"],
        hidden_dim=model_info["hidden_size"],
        intermediate_dim=model_info["intermediate_size"],
        num_layers=model_info["num_layers"],
        num_heads=model_info["num_heads"],
        num_kv_heads=model_info["num_kv_heads"],
        activation_function="silu",
        use_bias=False,
        use_flash_attention=True,
        rope_scaling=None,
    )


def extract_mlp_weight(pt_model, layer_idx: int, proj_name: str) -> jnp.ndarray:
    """
    Extract MLP weight from PyTorch model and convert to JAX.
    
    Args:
        pt_model: PyTorch Llama model
        layer_idx: Layer index (0-indexed)
        proj_name: 'gate_proj', 'up_proj', or 'down_proj'
        
    Returns:
        JAX array of shape [out_features, in_features]
    """
    import torch
    
    # Navigate to the specific layer
    layer = pt_model.model.layers[layer_idx]
    mlp_proj = getattr(layer.mlp, proj_name)
    
    # Get weight tensor [out_features, in_features]
    weight_pt = mlp_proj.weight.detach().cpu()
    
    # Convert to JAX
    weight_jax = jnp.array(weight_pt.numpy())
    
    logger.info(f"Extracted {proj_name} from layer {layer_idx}: shape {weight_jax.shape}")
    
    return weight_jax


def create_levanter_config_from_pretrained(
    model_name: str = "meta-llama/Llama-3.1-8B",
    seq_len: int = 512,  # Override for training
    cache_dir: Optional[str] = None,
) -> tuple[LlamaConfig, AutoTokenizer]:
    """
    Create Levanter config from pretrained HF model.
    
    Args:
        model_name: HF model identifier
        seq_len: Sequence length for training (can differ from pretrained)
        cache_dir: Cache directory
        
    Returns:
        (levanter_config, tokenizer)
    """
    _, tokenizer, model_info = load_llama_from_hf(model_name, cache_dir)
    
    # Create Levanter config
    config = LlamaConfig(
        seq_len=seq_len,  # Custom seq_len for training
        hidden_dim=model_info["hidden_size"],
        intermediate_dim=model_info["intermediate_size"],
        num_layers=model_info["num_layers"],
        num_heads=model_info["num_heads"],
        num_kv_heads=model_info["num_kv_heads"],
        activation_function="silu",
        use_bias=False,
        use_flash_attention=True,
    )
    
    return config, tokenizer


if __name__ == "__main__":
    # Test loading
    model_name = "meta-llama/Llama-3.1-8B"
    
    print("Testing HF loader...")
    pt_model, tokenizer, info = load_llama_from_hf(model_name)
    
    print("\nModel Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    print("\nTesting weight extraction...")
    weight = extract_mlp_weight(pt_model, layer_idx=0, proj_name="gate_proj")
    print(f"Weight shape: {weight.shape}")
    print(f"Weight dtype: {weight.dtype}")
    
    print("\n✓ All tests passed!")