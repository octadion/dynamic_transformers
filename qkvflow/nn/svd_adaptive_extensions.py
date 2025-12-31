# qkvflow/nn/svd_adaptive_extensions.py
"""
Extensions to existing SVDLinear for pretrained weight loading.
Properly grounded to qkvflow/nn/svd_adaptive.py
"""
import logging
import jax.numpy as jnp
import haliax as hax

from qkvflow.nn.svd_adaptive import SVDLinear  # ✅ IMPORT EXISTING

logger = logging.getLogger(__name__)


def create_svd_from_pretrained_pytorch(
    weight_pytorch: jnp.ndarray,
    bias_pytorch: jnp.ndarray = None,
    In: hax.AxisSpec = None,
    Out: hax.AxisSpec = None,
    rank_ratio: float = 0.5,
) -> SVDLinear:
    """
    Create SVDLinear from pretrained PyTorch weight.
    
    This extends existing SVDLinear by accepting raw PyTorch tensors
    and constructing proper NamedArrays for Haliax compatibility.
    
    Args:
        weight_pytorch: [out_features, in_features] from PyTorch
        bias_pytorch: Optional [out_features] bias
        In: Input axis specification
        Out: Output axis specification
        rank_ratio: Fraction of singular values to keep
        
    Returns:
        SVDLinear with frozen U, V, S_base from pretrained weights
    """
    import haliax.nn as hnn
    
    # Convert to NamedArray for Haliax
    weight_named = hax.NamedArray(weight_pytorch, axes=(Out, In))
    bias_named = hax.NamedArray(bias_pytorch, axes=(Out,)) if bias_pytorch is not None else None
    
    # Create temporary Linear layer
    temp_linear = hnn.Linear(
        weight=weight_named,
        bias=bias_named,
        In=In,
        Out=Out,
    )
    
    # ✅ USE EXISTING from_linear() method!
    svd_linear = SVDLinear.from_linear(
        linear=temp_linear,
        rank_ratio=rank_ratio,
        key=None,  # No randomness needed for pretrained
    )
    
    logger.info(f"Created SVDLinear from pretrained weights:")
    logger.info(f"  - U shape: {svd_linear.U.shape}")
    logger.info(f"  - S shape: {svd_linear.S_base.shape}")
    logger.info(f"  - V shape: {svd_linear.V.shape}")
    logger.info(f"  - Rank: {svd_linear.Rank.size}")
    
    return svd_linear


def load_llama_mlp_as_svd(
    pt_model,
    layer_idx: int,
    Embed: hax.Axis,
    Mlp: hax.Axis,
    rank_ratio: float = 0.5,
) -> dict:
    """
    Load Llama MLP projections as SVD layers.
    
    Args:
        pt_model: PyTorch Llama model from HuggingFace
        layer_idx: Layer index (0-indexed)
        Embed: Embedding dimension axis
        Mlp: MLP hidden dimension axis
        rank_ratio: SVD rank ratio
        
    Returns:
        Dict with 'gate_proj', 'up_proj', 'down_proj' as SVDLinear
    """
    import torch
    
    # Access the specific layer
    layer = pt_model.model.layers[layer_idx]
    mlp = layer.mlp
    
    # Extract weights (PyTorch format: [out_features, in_features])
    gate_weight = mlp.gate_proj.weight.detach().cpu().numpy()  # [mlp_dim, embed_dim]
    up_weight = mlp.up_proj.weight.detach().cpu().numpy()      # [mlp_dim, embed_dim]
    down_weight = mlp.down_proj.weight.detach().cpu().numpy()  # [embed_dim, mlp_dim]
    
    logger.info(f"Loading layer {layer_idx} MLP as SVD:")
    logger.info(f"  - gate_proj: {gate_weight.shape}")
    logger.info(f"  - up_proj: {up_weight.shape}")
    logger.info(f"  - down_proj: {down_weight.shape}")
    
    # Convert to SVD (Llama doesn't use bias in MLP)
    svd_gate = create_svd_from_pretrained_pytorch(
        weight_pytorch=jnp.array(gate_weight),
        In=Embed,
        Out=Mlp,
        rank_ratio=rank_ratio,
    )
    
    svd_up = create_svd_from_pretrained_pytorch(
        weight_pytorch=jnp.array(up_weight),
        In=Embed,
        Out=Mlp,
        rank_ratio=rank_ratio,
    )
    
    svd_down = create_svd_from_pretrained_pytorch(
        weight_pytorch=jnp.array(down_weight),
        In=Mlp,
        Out=Embed,
        rank_ratio=rank_ratio,
    )
    
    return {
        "gate_proj": svd_gate,
        "up_proj": svd_up,
        "down_proj": svd_down,
    }


if __name__ == "__main__":
    print("✅ SVDLinear extensions properly grounded to existing code")
    print("Usage:")
    print("  from qkvflow.nn.svd_adaptive_extensions import load_llama_mlp_as_svd")
    print("  svd_mlp = load_llama_mlp_as_svd(pt_model, layer_idx=0, ...)")