"""
SVD Neural ODE Transformer Block
Integrates SVD Temporal Linear layers into Neural ODE Transformer architecture
"""

import dataclasses
import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Callable, Dict, Optional
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call
from levanter.models.gpt2 import ACT2FN, dot_product_attention, Gpt2Config

from qkvflow.nn.time_embed import SinusoidalPosEmb, AlternativeTimeEmbeding
from qkvflow.nn.svd_temporal_linear import SVDTemporalLinear


class SVDAttention(eqx.Module):
    """
    SVD-based attention module for Neural ODE Transformer
    """
    
    config: Gpt2Config = eqx.field(static=True)
    inference: bool = eqx.field(static=True)
    
    c_attn: SVDTemporalLinear  # QKV projection with SVD
    c_proj: SVDTemporalLinear  # Output projection with SVD
    
    @staticmethod
    def init(config: Gpt2Config, SinusodialDim, TembedDim: hax.Axis, rank: int = 64, num_experts: int = 4, *, key):
        Qkv = hax.Axis("qkv", size=3)
        use_bias = config.use_bias
        Embed = config.Embed
        
        k_c, k_proj = jrandom.split(key, 2)
        
        # SVD-based QKV projection
        c_attn = SVDTemporalLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=Embed,
            Out=(Qkv, config.Heads, config.HeadSize),
            rank=rank,
            num_experts=num_experts,
            key=k_c,
            use_bias=use_bias,
        )
        
        # SVD-based output projection
        c_proj = SVDTemporalLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=(config.Heads, config.HeadSize),
            Out=Embed,
            rank=rank,
            num_experts=num_experts,
            key=k_proj,
            use_bias=use_bias,
        )
        
        return SVDAttention(
            config=config, 
            inference=False, 
            c_attn=c_attn, 
            c_proj=c_proj
        )
    
    @named_call
    def __call__(self, time_embed: NamedArray, x: NamedArray, mask, layer_idx, *, key):
        qkv_out = self.c_attn(time_embed, x).rearrange(
            (..., "qkv", "heads", "position", "head_size")
        )
        
        q, k, v = qkv_out.unbind("qkv")
        
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})
        
        attn_output = dot_product_attention(
            QPos="position",
            KPos="key_position", 
            Key="head_size",
            query=q,
            key=k,
            value=v,
            mask=mask,
            inference=self.inference,
            use_flash=self.config.use_flash_attention,
            flash_block_size=self.config.flash_attention_block_size,
            prng=key,
            attention_dtype=jnp.float32 if self.config.upcast_attn else None,
        )
        
        attn_output = self.c_proj(time_embed, attn_output)
        
        if self.config.upcast_attn:
            attn_output = attn_output.astype(x.dtype)
            
        return attn_output


class SVDMLP(eqx.Module):
    """
    SVD-based MLP module for Neural ODE Transformer
    """
    
    config: Gpt2Config = eqx.field(static=True)
    
    c_fc: SVDTemporalLinear    # First linear layer with SVD
    c_proj: SVDTemporalLinear  # Second linear layer with SVD 
    act: Callable = eqx.field(static=True)
    
    @staticmethod
    def init(
        config: Gpt2Config,
        SinusodialDim,
        TembedDim: hax.Axis,
        rank: int = 64,
        num_experts: int = 4,
        *,
        key,
        use_bias: bool = True,
    ):
        k_fc, k_proj = jrandom.split(key, 2)
        Embed, Mlp, activation_fn = config.Embed, config.Mlp, config.activation_function
        
        # SVD-based first layer (embed -> mlp)
        c_fc = SVDTemporalLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=Embed,
            Out=Mlp,
            rank=rank,
            num_experts=num_experts,
            key=k_fc,
            use_bias=use_bias,
        )
        
        # SVD-based second layer (mlp -> embed)
        c_proj = SVDTemporalLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=Mlp,
            Out=Embed,
            rank=rank,
            num_experts=num_experts,
            key=k_proj,
            use_bias=use_bias,
        )
        
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn
        
        return SVDMLP(config, c_fc, c_proj, act)
    
    @named_call
    def __call__(self, time_embed: NamedArray, x: NamedArray, *, key=None):
        del key
        
        x = self.c_fc(time_embed, x)
        x = self.act(x)
        x = self.c_proj(time_embed, x)
        return x


class SVDNeuralODEBlock(eqx.Module):
    """
    SVD Neural ODE Transformer Block
    Combines SVD decomposition with Neural ODE evolution
    """
    
    config: Gpt2Config = eqx.field(static=True)
    
    attn_ln: hnn.LayerNorm
    attn: SVDAttention
    mlp_ln: hnn.LayerNorm
    mlp: SVDMLP
    resid_dropout: hnn.Dropout
    
    # Policy components for adaptive mixing
    use_policy: bool = eqx.field(static=True)
    attn_policy: Optional[hnn.Linear] = None
    mlp_policy: Optional[hnn.Linear] = None
    
    @staticmethod
    def init(
        config: Gpt2Config, 
        SinusodialDim, 
        TembedDim,
        rank: int = 64,
        num_experts: int = 4,
        use_policy: bool = False,
        *, 
        key
    ):
        k_attn, k_mlp, k_policy = jrandom.split(key, 3)
        
        # Layer normalization (standard)
        attn_ln = hnn.LayerNorm.init(
            config.Embed,
            eps=config.layer_norm_epsilon,
            use_bias=config.use_bias
        )
        
        # SVD Attention
        attn = SVDAttention.init(
            config, SinusodialDim, TembedDim, 
            rank=rank, num_experts=num_experts, 
            key=k_attn
        )
        
        # Layer normalization for MLP
        mlp_ln = hnn.LayerNorm.init(
            config.Embed,
            eps=config.layer_norm_epsilon, 
            use_bias=config.use_bias
        )
        
        # SVD MLP
        mlp = SVDMLP.init(
            config, SinusodialDim, TembedDim,
            rank=rank, num_experts=num_experts,
            key=k_mlp, use_bias=config.use_bias
        )
        
        # Residual dropout
        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)
        
        # Optional policy networks for adaptive expert mixing
        attn_policy = None
        mlp_policy = None
        if use_policy:
            k_attn_policy, k_mlp_policy = jrandom.split(k_policy, 2)
            ExpertDim = hax.Axis("experts", num_experts)
            
            attn_policy = hnn.Linear.init(
                In=TembedDim, Out=ExpertDim, key=k_attn_policy
            )
            mlp_policy = hnn.Linear.init(
                In=TembedDim, Out=ExpertDim, key=k_mlp_policy  
            )
        
        return SVDNeuralODEBlock(
            config, attn_ln, attn, mlp_ln, mlp, resid_dropout,
            use_policy, attn_policy, mlp_policy
        )
    
    def __call__(self, time_embed, x: NamedArray, mask, layer_idx, *, key):
        k1, k2, k3, k4 = maybe_rng_split(key, 4)
        
        # Self-attention with SVD
        attn_output = self.attn(
            time_embed=time_embed,
            x=self.attn_ln(x),
            mask=mask,
            layer_idx=layer_idx,
            key=k1,
        )
        attn_output = self.resid_dropout(attn_output, key=k2)
        
        # MLP with SVD
        ff_output = self.mlp(
            time_embed=time_embed,
            x=self.mlp_ln(x),
            key=k3,
        )
        ff_output = self.resid_dropout(ff_output, key=k4)
        
        return attn_output + ff_output
    
    def get_expert_weights(self, time_embed: NamedArray):
        """Get expert mixing weights for analysis"""
        attn_weights = None
        mlp_weights = None
        
        if self.use_policy and self.attn_policy is not None:
            attn_logits = self.attn_policy(time_embed)
            attn_weights = hnn.softmax(attn_logits, axis="experts")
            
        if self.use_policy and self.mlp_policy is not None:
            mlp_logits = self.mlp_policy(time_embed)
            mlp_weights = hnn.softmax(mlp_logits, axis="experts")
            
        return {
            "attention": attn_weights,
            "mlp": mlp_weights
        }


class SVDNeuralODETransformer(eqx.Module):
    """
    Complete SVD Neural ODE Transformer
    """
    
    config: Gpt2Config = eqx.field(static=True)
    time_embedding: AlternativeTimeEmbeding
    block: SVDNeuralODEBlock
    ln_f: hnn.LayerNorm
    
    dt: float = eqx.field(static=True)
    
    # Adaptive parameters from Transformer²
    adaptive_mixing: bool = eqx.field(static=True)
    global_policy: Optional[hnn.Linear] = None
    
    @staticmethod
    def init(
        config: Gpt2Config,
        time_embed_dim,
        sinusodial_dim,
        rank: int = 64,
        num_experts: int = 4,
        adaptive_mixing: bool = True,
        *,
        key,
    ):
        k_tembed, k_block, k_policy = jrandom.split(key, 3)
        TembedDim = hax.Axis("TembedDim", time_embed_dim)
        SinusodialDim = hax.Axis("SinusodialDim", sinusodial_dim)
        
        # Time embedding
        time_embeding = SinusoidalPosEmb.init(SinusodialDim, key=k_tembed)
        SinusodialDim = SinusodialDim.resize(sinusodial_dim * 2 + 1)
        
        # SVD Neural ODE Block
        block = SVDNeuralODEBlock.init(
            config, SinusodialDim, TembedDim,
            rank=rank, num_experts=num_experts,
            use_policy=adaptive_mixing,
            key=k_block
        )
        
        # Final layer norm
        ln_f = hnn.LayerNorm.init(
            config.Embed, 
            eps=config.layer_norm_epsilon, 
            use_bias=config.use_bias
        )
        
        # Global policy for adaptive time evolution
        global_policy = None
        if adaptive_mixing:
            TimeStepDim = hax.Axis("timesteps", config.num_layers)
            global_policy = hnn.Linear.init(
                In=TembedDim, Out=TimeStepDim, key=k_policy
            )
        
        dt = 1.0 / config.num_layers
        
        return SVDNeuralODETransformer(
            config, time_embeding, block, ln_f, dt,
            adaptive_mixing, global_policy
        )
    
    def __call__(self, x: NamedArray, attn_mask, *, key=None) -> NamedArray:
        t = (hax.arange(self.config.Layers, dtype=x.dtype) + 1) * self.dt
        dts = hax.ones((self.config.Layers,), dtype=x.dtype) * self.dt
        
        time_embed = self.time_embedding(t)
        
        if key is not None:
            keys = maybe_rng_split(key, self.config.num_layers)
        else:
            keys = None
        
        # Adaptive time step adjustment
        if self.adaptive_mixing and self.global_policy is not None:
            # Use global policy to modulate time steps
            def modulate_dt(t_embed, base_dt):
                dt_logits = self.global_policy(t_embed)
                dt_weights = hnn.softmax(dt_logits, axis="timesteps")
                # Weighted average with learnable weights
                return base_dt * (1.0 + 0.1 * dt_weights.mean())  # Small modulation
            
            dts = hax.vmap(modulate_dt, axis="layers")(time_embed, dts)
        
        def do_block(x, time_embed, dt, key=None):
            output = self.block(time_embed, x, attn_mask, None, key=key)
            return x + output * dt
        
        # Checkpoint for memory efficiency
        do_block = jax.checkpoint(do_block, prevent_cse=False)
        
        x = hax.fold(do_block, axis=self.config.Layers)(x, time_embed, dts, key=keys)
        x = self.ln_f(x)
        
        return x
    
    def compute_trajectory(self, x, attn_mask):
        """Compute full trajectory for analysis (similar to original)"""
        t = (hax.arange(self.config.Layers, dtype=x.dtype) + 1) * self.dt
        time_embed = self.time_embedding(t)
        
        def do_block(x, time_embed):
            output = self.block(time_embed, x, attn_mask, None, key=None)
            ret = x + output * self.dt
            return ret, ret
        
        do_block = jax.checkpoint(do_block)
        
        _, trajectory = hax.scan(do_block, axis=self.config.Layers)(x, time_embed)
        return trajectory
    
    def get_expert_mixing_analysis(self, x, attn_mask):
        """Analyze expert mixing throughout the trajectory"""
        t = (hax.arange(self.config.Layers, dtype=x.dtype) + 1) * self.dt
        time_embed = self.time_embedding(t)
        
        expert_weights_history = []
        
        for i in range(self.config.num_layers):
            t_embed = time_embed.take("layers", i)
            weights = self.block.get_expert_weights(t_embed)
            expert_weights_history.append(weights)
        
        return expert_weights_history