import dataclasses
from typing import Callable, Optional, Sequence

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call
from levanter.models.gpt2 import ACT2FN, dot_product_attention, Gpt2Config

from .temporal_svd_linear import TemporalSVDLinear


class AdaptiveAttention(eqx.Module):
    """
    Adaptive attention mechanism using temporal SVD experts.
    """
    
    config: Gpt2Config = eqx.field(static=True)
    inference: bool = eqx.field(static=True)
    
    c_attn: TemporalSVDLinear  # Q, K, V projection
    c_proj: TemporalSVDLinear  # Output projection
    
    attention_type_predictor: hnn.Linear
    
    @staticmethod
    def init(
        config: Gpt2Config, 
        SinusodialDim: Axis, 
        TembedDim: Axis, 
        num_experts: int = 4,
        *,
        key
    ):
        k_attn, k_proj, k_predictor = jrandom.split(key, 3)
        
        Qkv = hax.Axis("qkv", size=3)
        use_bias = config.use_bias
        Embed = config.Embed
        
        # QKV projection
        c_attn = TemporalSVDLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=Embed,
            Out=(Qkv, config.Heads, config.HeadSize),
            num_experts=num_experts,
            key=k_attn,
            use_bias=use_bias,
        )
        
        # Output projection 
        c_proj = TemporalSVDLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=(config.Heads, config.HeadSize),
            Out=Embed,
            num_experts=num_experts,
            key=k_proj,
            use_bias=use_bias,
        )
        
        # Attention pattern type predictor
        AttentionTypes = hax.Axis("attention_types", 4)  # local, global, sparse, dense
        attention_type_predictor = hnn.Linear.init(
            In=TembedDim, Out=AttentionTypes, key=k_predictor, use_bias=True
        )
        
        return AdaptiveAttention(
            config=config,
            inference=False,
            c_attn=c_attn,
            c_proj=c_proj,
            attention_type_predictor=attention_type_predictor,
        )
    
    @named_call
    def __call__(
        self, 
        time_embed: NamedArray, 
        x: NamedArray, 
        mask, 
        layer_idx, 
        *, 
        key
    ):
        attention_type_logits = self.attention_type_predictor(time_embed)
        attention_type_weights = hax.nn.softmax(attention_type_logits, axis="attention_types")
        
        qkv_out = self.c_attn(time_embed, x).rearrange(
            ("batch", "position", "qkv", "heads", "head_size")
        )
        
        q, k, v = qkv_out.unbind("qkv")

        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        attn_outputs = []
        
        # Pattern 1: Standard attention
        standard_attn = dot_product_attention(
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
        attn_outputs.append(standard_attn)
        
        # Pattern 2: Local attention (attend to nearby positions)
        local_mask = self._create_local_mask(q.axis_size("position"), window_size=64)
        local_combined_mask = mask.combine(local_mask) if mask is not None else local_mask
        local_attn = dot_product_attention(
            QPos="position",
            KPos="key_position",
            Key="head_size", 
            query=q,
            key=k,
            value=v,
            mask=local_combined_mask,
            inference=self.inference,
            use_flash=False,
            prng=key,
            attention_dtype=jnp.float32 if self.config.upcast_attn else None,
        )
        attn_outputs.append(local_attn)
        
        # Pattern 3: Sparse attention (attend to every k-th position)
        sparse_mask = self._create_sparse_mask(q.axis_size("position"), stride=8)
        sparse_combined_mask = mask.combine(sparse_mask) if mask is not None else sparse_mask
        sparse_attn = dot_product_attention(
            QPos="position",
            KPos="key_position",
            Key="head_size",
            query=q,
            key=k, 
            value=v,
            mask=sparse_combined_mask,
            inference=self.inference,
            use_flash=False,
            prng=key,
            attention_dtype=jnp.float32 if self.config.upcast_attn else None,
        )
        attn_outputs.append(sparse_attn)
        
        # Pattern 4: Global attention (full attention with different scaling)
        scaled_q = q * 0.5
        global_attn = dot_product_attention(
            QPos="position",
            KPos="key_position",
            Key="head_size",
            query=scaled_q,
            key=k,
            value=v,
            mask=mask,
            inference=self.inference,
            use_flash=self.config.use_flash_attention,
            flash_block_size=self.config.flash_attention_block_size,
            prng=key,
            attention_dtype=jnp.float32 if self.config.upcast_attn else None,
        )
        attn_outputs.append(global_attn)
        
        # Mix attention patterns based on predicted weights
        mixed_attn = None
        for i, attn_out in enumerate(attn_outputs):
            weight = attention_type_weights.take("attention_types", i)
            weighted_attn = attn_out * weight
            
            if mixed_attn is None:
                mixed_attn = weighted_attn
            else:
                mixed_attn = mixed_attn + weighted_attn
        
        attn_output = self.c_proj(time_embed, mixed_attn)
        
        if self.config.upcast_attn:
            attn_output = attn_output.astype(x.dtype)
        
        return attn_output
    
    def _create_local_mask(self, seq_len: int, window_size: int):
        """Create a local attention mask."""
        from levanter.models.attention import AttentionMask
        
        mask_matrix = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
        
        # Set to False (mask out) positions outside the window
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask_matrix = mask_matrix.at[i, :start].set(False)
            mask_matrix = mask_matrix.at[i, end:].set(False)
        
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        final_mask = mask_matrix & causal_mask
        
        return AttentionMask.explicit(final_mask)
    
    def _create_sparse_mask(self, seq_len: int, stride: int):
        """Create a sparse attention mask."""
        from levanter.models.attention import AttentionMask
        
        # Create sparse attention pattern (attend to every stride-th position)
        mask_matrix = jnp.zeros((seq_len, seq_len), dtype=jnp.bool_)
        
        for i in range(seq_len):
            # Always attend to self
            mask_matrix = mask_matrix.at[i, i].set(True)
            
            # Attend to positions at stride intervals
            for j in range(0, i + 1, stride):
                mask_matrix = mask_matrix.at[i, j].set(True)
                
            last_positions = max(0, i - 4)
            mask_matrix = mask_matrix.at[i, last_positions:i+1].set(True)
        
        return AttentionMask.explicit(mask_matrix)
    
    def evaluate_at(self, time_embed: NamedArray):
        """Evaluate at specific time point."""
        c_attn = self.c_attn.evaluate_at(time_embed)
        c_proj = self.c_proj.evaluate_at(time_embed)

        from levanter.models.gpt2 import Gpt2Attention
        return Gpt2Attention(
            config=self.config,
            c_attn=c_attn,
            c_proj=c_proj,
            inference=self.inference,
        )


class AdaptiveMLP(eqx.Module):
    """
    Adaptive MLP with expert specialization via SVD.
    """
    
    config: Gpt2Config = eqx.field(static=True)
    
    # SVD-based projections with expert routing
    c_fc: TemporalSVDLinear
    c_proj: TemporalSVDLinear
    act: Callable = eqx.field(static=True)
    
    # Expert routing
    expert_router: hnn.Linear  # Routes to different expert combinations
    
    @staticmethod
    def init(
        config: Gpt2Config,
        SinusodialDim: Axis,
        TembedDim: Axis,
        num_experts: int = 4,
        *,
        key,
        use_bias: bool = True,
    ):
        k_fc, k_proj, k_router = jrandom.split(key, 3)
        
        Embed, Mlp, activation_fn = config.Embed, config.Mlp, config.activation_function
        
        # First projection with expert specialization
        c_fc = TemporalSVDLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=Embed,
            Out=Mlp,
            num_experts=num_experts,
            key=k_fc,
            use_bias=use_bias,
        )
        
        # Second projection with expert specialization  
        c_proj = TemporalSVDLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=Mlp,
            Out=Embed,
            num_experts=num_experts,
            key=k_proj,
            use_bias=use_bias,
        )
        
        # Expert routing mechanism
        ExpertRoutes = hax.Axis("expert_routes", num_experts)
        expert_router = hnn.Linear.init(
            In=TembedDim, Out=ExpertRoutes, key=k_router, use_bias=True
        )
        
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn
        
        return AdaptiveMLP(
            config=config,
            c_fc=c_fc,
            c_proj=c_proj,
            act=act,
            expert_router=expert_router,
        )
    
    @named_call
    def __call__(self, time_embed: NamedArray, x: NamedArray, *, key=None):
        del key
        
        # Route through experts based on time embedding
        expert_routing_logits = self.expert_router(time_embed)
        expert_routing_weights = hax.nn.softmax(expert_routing_logits, axis="expert_routes")
        
        # First transformation with expert routing
        hidden = self.c_fc(time_embed, x)
        
        # Apply activation function
        hidden = self.act(hidden)
        
        # Second transformation with expert routing
        output = self.c_proj(time_embed, hidden)
        
        return output
    
    def evaluate_at(self, time_embed: NamedArray):
        """Evaluate at specific time point."""
        c_fc = self.c_fc.evaluate_at(time_embed)
        c_proj = self.c_proj.evaluate_at(time_embed)
        
        from levanter.models.gpt2 import Gpt2Mlp
        return Gpt2Mlp(c_fc=c_fc, c_proj=c_proj, act=self.act)