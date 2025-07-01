"""
Neural ODE Transformer with SVD-based adaptive weights (Llama architecture).
Combines temporal weight evolution with dynamic SVD adaptation.
"""

import dataclasses
from typing import Optional, Dict, Callable

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmExample

from .dynamic import TemporalLinear, SinusoidalPosEmb, AlternativeTimeEmbeding
from .dynamic_llama import (
    LlamaAttention, LlamaRMSNorm, LlamaEmbedding, LlamaMlp,
    _apply_rotary_pos_emb, _rotate_half
)
from .svd_adaptive import SVDLinear, DynamicSVDPolicy


class LlamaAdaptiveMLP(eqx.Module):
    """Llama MLP with SVD decomposition and adaptive weights (3-projection SwiGLU style)."""
    
    gate_proj: SVDLinear
    up_proj: SVDLinear  
    down_proj: SVDLinear
    act: callable = eqx.field(static=True)
    
    @staticmethod
    def from_mlp(mlp: LlamaMlp, rank_ratio: float = 0.5, *, key: jax.random.PRNGKey) -> "LlamaAdaptiveMLP":
        """Create LlamaAdaptiveMLP from existing LlamaMlp."""
        k1, k2, k3 = jrandom.split(key, 3)
        
        # Extract temporal linear components at t=0 to get base weights
        t0_embed = hax.zeros((mlp.gate_proj.TembedDim,))
        gate_proj_t0 = mlp.gate_proj.evaluate_at(t0_embed)
        up_proj_t0 = mlp.up_proj.evaluate_at(t0_embed)
        down_proj_t0 = mlp.down_proj.evaluate_at(t0_embed)
        
        # Create SVD versions
        gate_proj = SVDLinear.from_linear(gate_proj_t0, rank_ratio, key=k1)
        up_proj = SVDLinear.from_linear(up_proj_t0, rank_ratio, key=k2)
        down_proj = SVDLinear.from_linear(down_proj_t0, rank_ratio, key=k3)
        
        return LlamaAdaptiveMLP(gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj, act=mlp.act)
    
    def __call__(self, x: NamedArray, multipliers: Dict[str, NamedArray], *, key: Optional[jax.random.PRNGKey] = None) -> NamedArray:
        k1, k2, k3 = maybe_rng_split(key, 3)
        
        gate = self.gate_proj(x, s_multiplier=multipliers["gate_proj"], key=k1)
        gate = self.act(gate)
        up = self.up_proj(x, s_multiplier=multipliers["up_proj"], key=k2)
        hidden = gate * up
        output = self.down_proj(hidden, s_multiplier=multipliers["down_proj"], key=k3)
        return output


class LlamaAdaptiveTemporalMLP(eqx.Module):
    """Llama MLP with temporal evolution and SVD adaptation combined additively."""
    
    config: LlamaConfig = eqx.field(static=True)
    gate_proj_temporal: TemporalLinear
    up_proj_temporal: TemporalLinear  
    down_proj_temporal: TemporalLinear
    adaptive_mlp: LlamaAdaptiveMLP
    act: Callable = eqx.field(static=True)
    
    @named_call
    def __call__(self, time_embed: NamedArray, x: NamedArray, multipliers: Dict[str, NamedArray], *, key=None):
        # Get temporal weight and bias components
        w_gate_temporal, b_gate_temporal = self.gate_proj_temporal.evaluate_at_components(time_embed)
        w_up_temporal, b_up_temporal = self.up_proj_temporal.evaluate_at_components(time_embed)
        w_down_temporal, b_down_temporal = self.down_proj_temporal.evaluate_at_components(time_embed)
        
        batch_axis = multipliers['gate_proj'].axes[0]
 
        w_gate_temporal = w_gate_temporal.broadcast_axis(batch_axis)
        w_up_temporal = w_up_temporal.broadcast_axis(batch_axis)
        w_down_temporal = w_down_temporal.broadcast_axis(batch_axis)
        
        # Get SVD weight and bias components  
        w_gate_svd = self.adaptive_mlp.gate_proj.get_effective_weight(s_multiplier=multipliers['gate_proj'])
        b_gate_svd = self.adaptive_mlp.gate_proj.bias
        w_up_svd = self.adaptive_mlp.up_proj.get_effective_weight(s_multiplier=multipliers['up_proj'])
        b_up_svd = self.adaptive_mlp.up_proj.bias
        w_down_svd = self.adaptive_mlp.down_proj.get_effective_weight(s_multiplier=multipliers['down_proj'])
        b_down_svd = self.adaptive_mlp.down_proj.bias

        w_gate_eff = w_gate_svd + w_gate_temporal
        w_up_eff = w_up_svd + w_up_temporal
        w_down_eff = w_down_svd + w_down_temporal
        
        def combine_bias(svd_bias, temp_bias):
            if temp_bias is not None:
                temp_bias = temp_bias.broadcast_axis(batch_axis)
            if svd_bias is not None and temp_bias is not None:
                return svd_bias + temp_bias
            elif svd_bias is not None:
                return svd_bias
            elif temp_bias is not None:
                return temp_bias
            else:
                return None
                
        b_gate_eff = combine_bias(b_gate_svd, b_gate_temporal)
        b_up_eff = combine_bias(b_up_svd, b_up_temporal)
        b_down_eff = combine_bias(b_down_svd, b_down_temporal)
        
        # SwiGLU forward pass with effective weights
        gate = hax.dot(self.adaptive_mlp.gate_proj.In, x, w_gate_eff)
        if b_gate_eff is not None:
            gate = gate + b_gate_eff
        gate = self.act(gate)
        
        up = hax.dot(self.adaptive_mlp.up_proj.In, x, w_up_eff) 
        if b_up_eff is not None:
            up = up + b_up_eff
            
        hidden = gate * up
        
        output = hax.dot(self.adaptive_mlp.down_proj.In, hidden, w_down_eff)
        if b_down_eff is not None:
            output = output + b_down_eff
            
        return output


class LlamaAdaptiveBlock(eqx.Module):
    """Llama transformer block with SVD-adaptive MLP and temporal evolution."""
    
    config: LlamaConfig = eqx.field(static=True)
    
    self_attn: LlamaAttention
    mlp: LlamaAdaptiveTemporalMLP
    input_layernorm: LlamaRMSNorm
    post_attention_layernorm: LlamaRMSNorm
    
    @staticmethod
    def init(
        config: LlamaConfig, 
        SinusodialDim, 
        TembedDim,
        rank_ratio: float = 0.5,
        *, 
        key
    ):
        k_attn, k_mlp, k_adapt, ln_1_key, ln_2_key = jrandom.split(key, 5)

        # Create attention with temporal linear layers
        self_attn = LlamaAttention.init(config, SinusodialDim, TembedDim, key=k_attn)
        
        # Create temporal MLP first
        regular_mlp = LlamaMlp.init(
            SinusodialDim, TembedDim, config.Embed, config.Mlp, 
            config.activation_function, key=k_mlp, use_bias=config.use_bias
        )

        # Create adaptive version from the temporal MLP
        adaptive_mlp = LlamaAdaptiveMLP.from_mlp(regular_mlp, rank_ratio, key=k_adapt)

        # Combine into adaptive temporal MLP
        adaptive_temporal_mlp = LlamaAdaptiveTemporalMLP(
            config=config,
            gate_proj_temporal=regular_mlp.gate_proj,
            up_proj_temporal=regular_mlp.up_proj,
            down_proj_temporal=regular_mlp.down_proj,
            adaptive_mlp=adaptive_mlp,
            act=regular_mlp.act,
        )
        
        # Layer norms with temporal embedding support
        input_layernorm = LlamaRMSNorm.init(
            config.Embed, SinusodialDim=SinusodialDim, TembedDim=TembedDim, key=ln_1_key
        )
        post_attention_layernorm = LlamaRMSNorm.init(
            config.Embed, SinusodialDim=SinusodialDim, TembedDim=TembedDim, key=ln_2_key
        )
        
        return LlamaAdaptiveBlock(config, self_attn, adaptive_temporal_mlp, input_layernorm, post_attention_layernorm)
    
    def __call__(self, time_embed, x: NamedArray, mask, layer_idx, multipliers: Dict[str, NamedArray], *, key):
        k1, k2 = maybe_rng_split(key, 2)

        attn_output = self.self_attn(
            time_embed=time_embed,
            x=self.input_layernorm(time_embed, x),
            mask=mask,
            key=k1,
        )
        
        intermediate_x = x + attn_output

        mlp_output = self.mlp(
            time_embed=time_embed,
            x=self.post_attention_layernorm(time_embed, intermediate_x),
            multipliers=multipliers,
            key=k2,
        )

        return attn_output + mlp_output


class SVDLlamaOdeTransformer(eqx.Module):
    """Llama Neural ODE Transformer with SVD-based weight adaptation."""
    
    config: LlamaConfig = eqx.field(static=True)
    time_embedding: AlternativeTimeEmbeding
    block: LlamaAdaptiveBlock
    norm: LlamaRMSNorm
    policy: DynamicSVDPolicy
    
    dt: float = eqx.field(static=True)
    rank_ratio: float = eqx.field(static=True)
    
    @staticmethod
    def init(
        config: LlamaConfig,
        time_embed_dim,
        sinusodial_dim,
        rank_ratio: float = 0.5,
        policy_init_scale: float = 0.1,
        *,
        key,
    ):
        k_tembed, k_block, k_policy, k_norm = jrandom.split(key, 4)
        TembedDim = hax.Axis("TembedDim", time_embed_dim)
        SinusodialDim = hax.Axis("SinusodialDim", sinusodial_dim)
        time_embeding = SinusoidalPosEmb.init(SinusodialDim, key=k_tembed)
        SinusodialDim = SinusodialDim.resize(sinusodial_dim * 2 + 1)

        block = LlamaAdaptiveBlock.init(
            config, SinusodialDim, TembedDim, rank_ratio, key=k_block
        )
        norm = LlamaRMSNorm.init(
            config.Embed, SinusodialDim=SinusodialDim, TembedDim=TembedDim, key=k_norm
        )
        dt = 1.0 / config.num_layers

        # Create policy for all MLP projections across layers
        mlp_in_size = config.Embed.size
        mlp_hidden_size = config.Mlp.size

        rank_per_layer = {}
        rank_gate = max(1, int(min(mlp_in_size, mlp_hidden_size) * rank_ratio))
        rank_up = max(1, int(min(mlp_in_size, mlp_hidden_size) * rank_ratio))  
        rank_down = max(1, int(min(mlp_hidden_size, mlp_in_size) * rank_ratio))
        
        for layer_idx in range(config.num_layers):
            rank_per_layer[f"layer_{layer_idx}_gate_proj"] = rank_gate
            rank_per_layer[f"layer_{layer_idx}_up_proj"] = rank_up
            rank_per_layer[f"layer_{layer_idx}_down_proj"] = rank_down
        
        policy = DynamicSVDPolicy.init(
            num_layers=config.num_layers,
            rank_per_layer=rank_per_layer,
            task_vector_dim=config.Embed,
            key=k_policy,
        )
        
        return SVDLlamaOdeTransformer(
            config, time_embeding, block, norm, policy, dt, rank_ratio
        )
    
    def __call__(self, x: NamedArray, attn_mask, *, key=None) -> NamedArray:
        # Calculate task vector for SVD policy  
        task_vector = x.mean(axis=self.config.Pos)
        multipliers = self.policy(task_vector)

        # Time evolution setup
        t = (hax.arange(self.config.Layers, dtype=x.dtype) + 1) * self.dt
        dts = hax.ones((self.config.Layers,), dtype=x.dtype) * self.dt
        time_embed = self.time_embedding(t)
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else [None] * self.config.num_layers

        def make_do_block(layer_idx, layer_key):
            def do_block(x_in):
                layer_multipliers = {
                    "gate_proj": multipliers.get(f"layer_{layer_idx}_gate_proj"),
                    "up_proj": multipliers.get(f"layer_{layer_idx}_up_proj"),
                    "down_proj": multipliers.get(f"layer_{layer_idx}_down_proj"),
                }
                
                output = self.block(
                    time_embed.take("layers", layer_idx), 
                    x_in, 
                    attn_mask, 
                    layer_idx, 
                    multipliers=layer_multipliers, 
                    key=layer_key
                )
                return x_in + output * dts.take("layers", layer_idx)
            return do_block
        
        for i in range(self.config.num_layers):
            do_block = make_do_block(i, keys[i])
            x = jax.checkpoint(do_block, prevent_cse=False)(x)
        
        # Final layer norm (temporal)
        final_time_embed = time_embed.take("layers", -1)  # Use last time step
        x = self.norm(final_time_embed, x)
        return x

    def get_policy_loss(self, reg_strength: float = 0.01) -> jax.Array:
        """Policy regularization loss."""
        params = eqx.filter(self.policy.policy_net, eqx.is_array)
        loss = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
        return loss * reg_strength


class SVDLlamaOdeLMHeadModel(eqx.Module):
    """Llama language model with SVD-adaptive Neural ODE transformer."""
    
    transformer: SVDLlamaOdeTransformer
    embeddings: LlamaEmbedding
    lm_head: hnn.Linear
    
    @property
    def config(self):
        return self.transformer.config
    
    @property
    def Vocab(self) -> hax.Axis:
        return self.embeddings.Vocab
    
    @property
    def Pos(self) -> hax.Axis:
        return self.config.Pos
    
    @classmethod
    def init(
        cls,
        Vocab: hax.Axis,
        config: LlamaConfig,
        time_embed_dim=100,
        sinusodial_dim=16,
        rank_ratio=0.5,
        policy_init_scale=0.1,
        *,
        key,
    ) -> "SVDLlamaOdeLMHeadModel":
        k_t, k_emb, k_head = jrandom.split(key, 3)
        transformer = SVDLlamaOdeTransformer.init(
            config,
            time_embed_dim=time_embed_dim,
            sinusodial_dim=sinusodial_dim,
            rank_ratio=rank_ratio,
            policy_init_scale=policy_init_scale,
            key=k_t,
        )
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_head, use_bias=False)
        
        return SVDLlamaOdeLMHeadModel(transformer, embeddings, lm_head)
    
    def __call__(
        self, input_ids: NamedArray, attn_mask=None, *, key=None
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask, key=key)
        lm_logits = self.lm_head(x)
        
        return lm_logits
    
    def compute_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
        policy_reg_strength: float = 0.01,
    ) -> NamedArray:
        """Compute language modeling loss with policy regularization."""
        logits = self(example.tokens, example.attn_mask, key=key)
        targets = hax.roll(example.tokens, -1, axis=self.Pos.name)
        target_y = hax.nn.one_hot(targets, self.Vocab, dtype=logits.dtype)
        
        lm_loss = hnn.cross_entropy_loss(
            logits,
            self.Vocab,
            target_y,
            reduction,
            reduction_axis=reduction_axis,
            where=example.loss_mask,
        )
        
        # Add policy regularization
        policy_loss = self.transformer.get_policy_loss(policy_reg_strength)
        
        return lm_loss + policy_loss
    
    @property
    def vocab_size(self) -> int:
        return self.Vocab.size
    
    def resize_vocab(self, new_size: int, key=None) -> "SVDLlamaOdeLMHeadModel":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        new_lm_head = hax.tree_util.resize_axis(
            self.lm_head.weight, self.Vocab, new_size, key=key
        )
        new_lm_head = dataclasses.replace(
            self.lm_head, Out=self.Vocab.resize(new_size), weight=new_lm_head
        )
        return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
    
    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": None, "embeddings": None, "lm_head": None}

    def get_policy_params(self) -> Dict[str, NamedArray]:
        """Get policy parameters for saving/loading."""
        return self.transformer.policy.get_policy_params()
    
    def set_policy_params(self, params: Dict[str, NamedArray]):
        """Set policy parameters from loaded values."""
        new_policy_net = params["policy_net"]
        new_policy = dataclasses.replace(self.transformer.policy, policy_net=new_policy_net)
        new_transformer = dataclasses.replace(
            self.transformer,
            policy=new_policy
        )
        return dataclasses.replace(self, transformer=new_transformer)