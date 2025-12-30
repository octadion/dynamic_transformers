# qkvflow/nn/svd_from_pretrained.py
"""
Initialize SVD layers from pretrained weights.
"""
import logging
import jax
import jax.numpy as jnp
import jax.random as jrandom
import haliax as hax
import haliax.nn as hnn
from typing import Optional

from qkvflow.nn.svd_adaptive import SVDLinear
from qkvflow.utils.hf_loader import extract_mlp_weight

logger = logging.getLogger(__name__)


def create_svd_from_pretrained_weight(
    weight_jax: jnp.ndarray,
    In: hax.AxisSpec,
    Out: hax.AxisSpec,
    rank_ratio: float = 0.5,
    bias: Optional[jnp.ndarray] = None,
) -> SVDLinear:
    """
    Create SVDLinear from pretrained weight matrix.
    
    Args:
        weight_jax: Pretrained weight [out_features, in_features]
        In: Input axis spec
        Out: Output axis spec
        rank_ratio: Fraction of singular values to keep
        bias: Optional bias vector
        
    Returns:
        SVDLinear with frozen U, V, S_base
    """
    out_size, in_size = weight_jax.shape
    
    logger.info(f"Performing SVD on pretrained weight {weight_jax.shape}")
    
    # Full SVD
    U_arr, S_arr, Vh_arr = jnp.linalg.svd(weight_jax, full_matrices=False)
    
    # Truncate to rank
    full_rank = min(in_size, out_size)
    rank = max(1, int(full_rank * rank_ratio))
    Rank = hax.Axis("rank", rank)
    
    U_arr = U_arr[:, :rank]           # [out_size, rank]
    S_arr_truncated = S_arr[:rank]    # [rank]
    Vh_arr = Vh_arr[:rank, :]         # [rank, in_size]
    
    # Normalize S for stability
    s_norm = jnp.linalg.norm(S_arr_truncated)
    S_arr_normalized = S_arr_truncated / (s_norm + 1e-8)
    
    # Absorb scale into U
    U_arr_scaled = U_arr * s_norm
    
    # Create NamedArrays
    S_base = hax.NamedArray(S_arr_normalized, axes=(Rank,))
    
    if isinstance(Out, tuple):
        u_axes = Out + (Rank,)
    else:
        u_axes = (Out, Rank)
    
    if isinstance(In, tuple):
        v_axes = In + (Rank,)
    else:
        v_axes = (In, Rank)
    
    U = hax.NamedArray(U_arr_scaled, axes=u_axes)
    V = hax.NamedArray(Vh_arr.T, axes=v_axes)  # V = Vh.T
    
    # Convert bias if provided
    bias_named = None
    if bias is not None:
        bias_named = hax.NamedArray(bias, axes=(Out,) if not isinstance(Out, tuple) else Out)
    
    # Compute reconstruction error
    W_reconstructed = U_arr_scaled @ jnp.diag(S_arr_normalized) @ Vh_arr
    recon_error = jnp.linalg.norm(weight_jax - W_reconstructed, ord='fro')
    rel_error = recon_error / jnp.linalg.norm(weight_jax, ord='fro')
    
    logger.info(f"✓ SVD completed:")
    logger.info(f"  - Rank: {rank}/{full_rank} ({rank_ratio*100:.0f}%)")
    logger.info(f"  - Singular values: [{S_arr.min():.4f}, {S_arr.max():.4f}]")
    logger.info(f"  - Reconstruction error: {rel_error*100:.2f}%")
    
    # Verify orthogonality
    U_orth_err = jnp.linalg.norm(U_arr_scaled.T @ U_arr_scaled - jnp.eye(rank) * s_norm**2, ord='fro')
    V_orth_err = jnp.linalg.norm(Vh_arr @ Vh_arr.T - jnp.eye(rank), ord='fro')
    
    assert U_orth_err < 1e-3, f"U not orthogonal! Error: {U_orth_err}"
    assert V_orth_err < 1e-3, f"V not orthogonal! Error: {V_orth_err}"
    
    return SVDLinear(
        U=U,
        S_base=S_base,
        V=V,
        In=In,
        Out=Out,
        Rank=Rank,
        use_bias=bias is not None,
        bias=bias_named,
    )


def initialize_svd_mlp_from_pretrained(
    pt_model,
    layer_idx: int,
    Embed: hax.Axis,
    Mlp: hax.Axis,
    rank_ratio: float = 0.5,
) -> dict:
    """
    Initialize all 3 MLP projections (Llama) from pretrained weights.
    
    Args:
        pt_model: PyTorch Llama model
        layer_idx: Layer index
        Embed: Embedding axis
        Mlp: MLP hidden axis
        rank_ratio: SVD rank ratio
        
    Returns:
        Dict with 'gate_proj', 'up_proj', 'down_proj' SVDLinear layers
    """
    svd_projections = {}
    
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        # Extract pretrained weight
        weight_jax = extract_mlp_weight(pt_model, layer_idx, proj_name)
        
        # Determine axes
        if proj_name in ["gate_proj", "up_proj"]:
            In, Out = Embed, Mlp
        else:  # down_proj
            In, Out = Mlp, Embed
        
        # Create SVD layer
        svd_layer = create_svd_from_pretrained_weight(
            weight_jax=weight_jax,
            In=In,
            Out=Out,
            rank_ratio=rank_ratio,
            bias=None,  # Llama doesn't use bias
        )
        
        svd_projections[proj_name] = svd_layer
    
    logger.info(f"✓ Initialized SVD MLP for layer {layer_idx}")
    
    return svd_projections


if __name__ == "__main__":
    from qkvflow.utils.hf_loader import load_llama_from_hf
    
    print("Testing SVD initialization from pretrained...")
    
    # Load model
    pt_model, _, info = load_llama_from_hf("meta-llama/Llama-3.1-8B")
    
    # Define axes
    Embed = hax.Axis("embed", info["hidden_size"])
    Mlp = hax.Axis("mlp", info["intermediate_size"])
    
    # Initialize SVD for layer 0
    svd_mlp = initialize_svd_mlp_from_pretrained(
        pt_model=pt_model,
        layer_idx=0,
        Embed=Embed,
        Mlp=Mlp,
        rank_ratio=0.5,
    )
    
    print("\n✓ SVD MLP created:")
    for proj_name, svd_layer in svd_mlp.items():
        print(f"  {proj_name}:")
        print(f"    - U: {svd_layer.U.shape}")
        print(f"    - S: {svd_layer.S_base.shape}")
        print(f"    - V: {svd_layer.V.shape}")
    
    print("\n✓ All tests passed!")