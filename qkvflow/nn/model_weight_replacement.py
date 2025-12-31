# qkvflow/nn/model_weight_replacement.py
"""
Proper weight replacement for Equinox models.
Handles nested structure modification correctly.
"""
import logging
import equinox as eqx
import jax
from typing import Callable

logger = logging.getLogger(__name__)


def replace_mlp_with_svd(
    model,
    layer_idx: int,
    svd_gate,
    svd_up,
    svd_down,
):
    """
    Replace MLP projections in SVDLlamaOdeLMHeadModel with SVD versions.
    
    This uses proper Equinox tree manipulation to avoid JIT issues.
    
    Args:
        model: SVDLlamaOdeLMHeadModel instance
        layer_idx: Layer index (0-indexed) - for Stacked layers
        svd_gate: SVDLinear for gate_proj
        svd_up: SVDLinear for up_proj
        svd_down: SVDLinear for down_proj
        
    Returns:
        Modified model with replaced weights
    """
    # For Neural ODE models, all layers share the same block
    # So we only need to replace once (not per layer)
    
    # Access the adaptive MLP
    mlp_path = lambda m: m.transformer.block.mlp.adaptive_mlp
    
    # Get current adaptive MLP
    adaptive_mlp = mlp_path(model)
    
    # Create new adaptive MLP with replaced projections
    new_adaptive_mlp = eqx.tree_at(
        lambda mlp: (mlp.gate_proj, mlp.up_proj, mlp.down_proj),
        adaptive_mlp,
        (svd_gate, svd_up, svd_down),
    )
    
    # Replace in full model
    new_model = eqx.tree_at(
        mlp_path,
        model,
        new_adaptive_mlp,
    )
    
    logger.info(f"✓ Replaced MLP projections with SVD")
    
    return new_model


def replace_all_mlps_with_pretrained_svd(
    model,
    pt_model,
    rank_ratio: float = 0.5,
):
    """
    Replace all MLP layers with pretrained SVD decomposition.
    
    For Neural ODE models with shared blocks, this only replaces once.
    
    Args:
        model: SVDLlamaOdeLMHeadModel
        pt_model: PyTorch Llama model from HuggingFace
        rank_ratio: SVD rank ratio
        
    Returns:
        Model with pretrained SVD weights
    """
    from qkvflow.nn.svd_adaptive_extensions import load_llama_mlp_as_svd
    
    config = model.config
    Embed = config.Embed
    Mlp = config.Mlp
    
    logger.info("Replacing MLP weights with pretrained SVD...")
    
    # For Neural ODE models, we use layer 0 as the template
    # since all timesteps share the same block
    svd_mlp = load_llama_mlp_as_svd(
        pt_model=pt_model,
        layer_idx=0,  # Use first layer as template
        Embed=Embed,
        Mlp=Mlp,
        rank_ratio=rank_ratio,
    )
    
    # Replace in model
    new_model = replace_mlp_with_svd(
        model=model,
        layer_idx=0,
        svd_gate=svd_mlp["gate_proj"],
        svd_up=svd_mlp["up_proj"],
        svd_down=svd_mlp["down_proj"],
    )
    
    logger.info("✓ All MLP weights replaced with pretrained SVD")
    
    return new_model


if __name__ == "__main__":
    print("✅ Proper Equinox model weight replacement")
    print("Usage:")
    print("  from qkvflow.nn.model_weight_replacement import replace_all_mlps_with_pretrained_svd")
    print("  model = replace_all_mlps_with_pretrained_svd(model, pt_model)")