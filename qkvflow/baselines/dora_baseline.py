# qkvflow/baselines/dora_baseline.py
"""
DoRA (Weight-Decomposed LoRA) baseline implementation.
DoRA decomposes weights into magnitude and direction, applying LoRA to direction.

Reference: DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)
"""
import logging
from typing import Optional
import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, NamedArray

from qkvflow.lora import LowRankLinear

logger = logging.getLogger(__name__)


class DoRALinear(eqx.Module):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation.
    
    W' = m * (W + ΔW) / ||W + ΔW||
    
    where:
    - W: pretrained weight
    - ΔW: LoRA adaptation (B @ A)
    - m: learned magnitude vector
    """
    
    base_weight: NamedArray  # Frozen pretrained weight
    lora: LowRankLinear      # LoRA adaptation
    magnitude: NamedArray    # Learned magnitude
    
    In: hax.AxisSpec = eqx.field(static=True)
    Out: hax.AxisSpec = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    bias: Optional[NamedArray]
    
    @staticmethod
    def from_linear(
        linear: hnn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        *,
        key: jax.random.PRNGKey,
    ) -> "DoRALinear":
        """Create DoRA from existing linear layer."""
        
        # Initialize LoRA
        lora = LowRankLinear.init(
            In=linear.In,
            Out=linear.Out,
            r=r,
            alpha=alpha,
            dropout_prob=dropout,
            key=key,
        )
        
        # Initialize magnitude from pretrained weight norms
        # m_i = ||W_i|| for each output dimension
        weight = linear.weight
        
        # Compute column norms (for each output neuron)
        if isinstance(linear.Out, tuple):
            # Multiple output axes - flatten first
            out_size = 1
            for ax in linear.Out:
                out_size *= weight.axis_size(ax)
            weight_flat = weight.rearrange((..., linear.In, linear.Out))
        else:
            weight_flat = weight
        
        # Compute L2 norm for each output dimension
        norms = hax.sqrt(hax.sum(weight_flat ** 2, axis=linear.In))
        
        logger.info(f"DoRA magnitude init: mean={float(hax.mean(norms).array):.4f}, "
                   f"std={float(hax.std(norms).array):.4f}")
        
        return DoRALinear(
            base_weight=linear.weight,
            lora=lora,
            magnitude=norms,
            In=linear.In,
            Out=linear.Out,
            use_bias=linear.bias is not None,
            bias=linear.bias,
        )
    
    def __call__(self, x: NamedArray, *, key: Optional[jax.random.PRNGKey] = None) -> NamedArray:
        """
        Forward pass with weight decomposition.
        
        W' = m * (W + ΔW) / ||W + ΔW||_col
        """
        # Get LoRA adaptation: ΔW = scale * (B @ A)
        delta_w = self.lora.merge()  # This returns the full ΔW matrix
        
        # Combine: W + ΔW
        combined_weight = self.base_weight + delta_w
        
        # Normalize by column norms
        if isinstance(self.Out, tuple):
            combined_flat = combined_weight.rearrange((..., self.In, self.Out))
        else:
            combined_flat = combined_weight
        
        # Compute new norms
        new_norms = hax.sqrt(hax.sum(combined_flat ** 2, axis=self.In))
        
        # Apply magnitude scaling: m / ||W + ΔW||
        scale = self.magnitude / (new_norms + 1e-8)
        
        # Scale the weight
        if isinstance(self.Out, tuple):
            scale_expanded = scale.broadcast_axis(self.In)
            weight_scaled = combined_weight * scale_expanded
        else:
            weight_scaled = combined_weight * scale.broadcast_axis(self.In)
        
        # Apply linear transformation
        y = hax.dot(self.In, x, weight_scaled)
        
        if self.use_bias and self.bias is not None:
            y = y + self.bias
        
        return y


def doraize_linear(linear: hnn.Linear, r: int, alpha: float, *, key: jax.random.PRNGKey) -> DoRALinear:
    """Convert linear layer to DoRA."""
    return DoRALinear.from_linear(linear, r=r, alpha=alpha, key=key)


def doraize_model(model, r: int = 8, alpha: float = 16.0, target_modules: str = "mlp.*_proj", *, key: jax.random.PRNGKey):
    """
    Apply DoRA to all matching linear layers in model.
    
    Similar to loraize but uses DoRA instead.
    """
    import re
    from levanter.utils.jax_utils import leaf_key_paths
    
    compiled_regex = re.compile(target_modules)
    key_iter = iter(jrandom.split(key, 1000))
    
    def _doraize_if_match(module, key_path):
        if isinstance(module, hnn.Linear) and compiled_regex.search(key_path):
            return doraize_linear(module, r=r, alpha=alpha, key=next(key_iter))
        return module
    
    return jax.tree_util.tree_map(
        _doraize_if_match,
        model,
        leaf_key_paths(model, is_leaf=lambda x: isinstance(x, hnn.Linear)),
        is_leaf=lambda x: isinstance(x, hnn.Linear),
    )


if __name__ == "__main__":
    print("DoRA (Weight-Decomposed LoRA) implementation")
    print("\nKey differences from LoRA:")
    print("  - LoRA: W' = W + ΔW")
    print("  - DoRA: W' = m * (W + ΔW) / ||W + ΔW||")
    print("\nAdvantage: Better preserves magnitude information from pretrained weights")