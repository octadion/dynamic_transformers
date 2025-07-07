"""
Neural ODE Transformer with SVD-based adaptive weights (Transformer-Squared integration).
"""

import dataclasses
from typing import Optional, Dict, Callable

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call
from levanter.models.gpt2 import Gpt2Config, Gpt2Embeddings
from levanter.models.lm_model import LmExample
import wandb

from .dynamic import (
    Attention, TemporalLayerNorm, MLP, NeuralOdeTransformer,
    TemporalLinear, SinusoidalPosEmb, AlternativeTimeEmbeding
)
from .svd_adaptive import AdaptiveMLP, SVDPolicy, SVDLinear, DynamicSVDPolicy


class AdaptiveBlock(eqx.Module):
    """Transformer block with SVD-adaptive MLP."""
    
    config: Gpt2Config = eqx.field(static=True)
    
    attn_ln: TemporalLayerNorm
    attn: Attention
    mlp_ln: TemporalLayerNorm
    mlp: AdaptiveMLP
    resid_dropout: hnn.Dropout
    
    @staticmethod
    def init(
        config: Gpt2Config, 
        SinusodialDim, 
        TembedDim,
        rank_ratio: float = 0.5,
        *, 
        key
    ):
        k_attn, k_mlp, k_adapt, k_gate = jrandom.split(key, 4)

        attn_ln = TemporalLayerNorm.init(
            config.Embed,
            eps=config.layer_norm_epsilon,
            use_bias=config.use_bias,
            TembedDim=TembedDim,
            SinusodialDim=SinusodialDim,
            key=key,
        )
        attn = Attention.init(config, SinusodialDim, TembedDim, key=k_attn)
        mlp_ln = TemporalLayerNorm.init(
            config.Embed,
            eps=config.layer_norm_epsilon,
            use_bias=config.use_bias,
            TembedDim=TembedDim,
            SinusodialDim=SinusodialDim,
            key=key,
        )
        regular_mlp = MLP.init(
            config, SinusodialDim, TembedDim, key=k_mlp, use_bias=config.use_bias
        )

        t0_embed = hax.zeros((TembedDim,))
        c_fc_t0 = regular_mlp.c_fc.evaluate_at(t0_embed)
        c_proj_t0 = regular_mlp.c_proj.evaluate_at(t0_embed)

        k_fc_svd, k_proj_svd = jrandom.split(k_adapt)

        c_fc_svd = SVDLinear.from_linear(c_fc_t0, rank_ratio, key=k_fc_svd)
        c_proj_svd = SVDLinear.from_linear(c_proj_t0, rank_ratio, key=k_proj_svd)

        k_gate_fc, k_gate_proj = jrandom.split(k_gate)
        
        initial_logit_value = 0.0 
        
        Scalar = hax.Axis("scalar", 1)
        gate_logit_fc = hax.named(jnp.full((), initial_logit_value, dtype=jnp.float32), ())
        gate_logit_proj = hax.named(jnp.full((), initial_logit_value, dtype=jnp.float32), ())
    
        adaptive_mlp = AdaptiveMLP(c_fc=c_fc_svd, c_proj=c_proj_svd, act=regular_mlp.act)

        adaptive_mlp = AdaptiveTemporalMLP(
            config=config,
            c_fc_temporal=regular_mlp.c_fc,
            c_proj_temporal=regular_mlp.c_proj,
            adaptive_mlp=adaptive_mlp,
            act=regular_mlp.act,
            gate_logit_fc=gate_logit_fc,
            gate_logit_proj=gate_logit_proj
        )
        
        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)
        
        return AdaptiveBlock(config, attn_ln, attn, mlp_ln, adaptive_mlp, resid_dropout)
    
    def __call__(self, time_embed, x: NamedArray, mask, layer_idx, multipliers: Dict[str, NamedArray], *, key):
        k1, k2, k3, k4 = maybe_rng_split(key, 4)
        
        attn_output = self.attn(
            time_embed=time_embed,
            x=self.attn_ln(time_embed, x),
            mask=mask,
            layer_idx=layer_idx,
            key=k1,
        )
        attn_output = self.resid_dropout(attn_output, key=k2)
        
        ff_output = self.mlp(
            time_embed=time_embed,
            x=self.mlp_ln(time_embed, x),
            multipliers=multipliers,
            key=k3,
        )
        ff_output = self.resid_dropout(ff_output, key=k4)
        
        return attn_output + ff_output


class AdaptiveTemporalMLP(eqx.Module):
    config: Gpt2Config = eqx.field(static=True)
    c_fc_temporal: TemporalLinear
    c_proj_temporal: TemporalLinear
    adaptive_mlp: AdaptiveMLP
    act: Callable = eqx.field(static=True)
    gate_logit_fc: hax.NamedArray
    gate_logit_proj: hax.NamedArray
    
    @named_call
    def __call__(self, time_embed: NamedArray, x: NamedArray, multipliers: Dict[str, NamedArray], *, key=None):
        w_fc_temporal, b_fc_temporal = self.c_fc_temporal.evaluate_at_components(time_embed)
        w_proj_temporal, b_proj_temporal = self.c_proj_temporal.evaluate_at_components(time_embed)
        
        batch_axis = multipliers['c_fc'].axes[0]

        w_fc_temporal = w_fc_temporal.broadcast_axis(batch_axis)
        w_proj_temporal = w_proj_temporal.broadcast_axis(batch_axis)

        w_fc_svd = self.adaptive_mlp.c_fc.get_effective_weight(s_multiplier=multipliers['c_fc'])
        w_proj_svd = self.adaptive_mlp.c_proj.get_effective_weight(s_multiplier=multipliers['c_proj'])

        b_fc_svd = self.adaptive_mlp.c_fc.bias
        b_proj_svd = self.adaptive_mlp.c_proj.bias
        
        gate_fc_raw = jnn.sigmoid(self.gate_logit_fc.astype(jnp.float32).array)
        gate_proj_raw = jnn.sigmoid(self.gate_logit_proj.astype(jnp.float32).array)

        w_fc_svd_raw = w_fc_svd.array
        w_fc_temporal_raw = w_fc_temporal.array
        w_proj_svd_raw = w_proj_svd.array
        w_proj_temporal_raw = w_proj_temporal.array

        w_fc_svd_raw = jnp.swapaxes(w_fc_svd_raw, -1, -2)
        w_proj_svd_raw = jnp.swapaxes(w_proj_svd_raw, -1, -2)

        w_fc_eff_raw = gate_fc_raw * w_fc_svd_raw + (1 - gate_fc_raw) * w_fc_temporal_raw
        w_proj_eff_raw = gate_proj_raw * w_proj_svd_raw + (1 - gate_proj_raw) * w_proj_temporal_raw

        w_fc_eff = hax.named(w_fc_eff_raw, w_fc_temporal.axes)
        w_proj_eff = hax.named(w_proj_eff_raw, w_proj_temporal.axes)

        b_fc_eff = b_fc_svd
        if b_fc_temporal is not None:
            b_fc_temporal = b_fc_temporal.broadcast_axis(batch_axis)
            b_fc_eff = b_fc_temporal if b_fc_eff is None else b_fc_eff + b_fc_temporal
            
        b_proj_eff = b_proj_svd
        if b_proj_temporal is not None:
            b_proj_temporal = b_proj_temporal.broadcast_axis(batch_axis)
            b_proj_eff = b_proj_temporal if b_proj_eff is None else b_proj_eff + b_proj_temporal
        
        x = hax.dot(self.adaptive_mlp.c_fc.In, x, w_fc_eff)
        if b_fc_eff is not None:
            x = x + b_fc_eff
        x = self.act(x)
        
        x = hax.dot(self.adaptive_mlp.c_proj.In, x, w_proj_eff)
        if b_proj_eff is not None:
            x = x + b_proj_eff
            
        return x


class SVDNeuralOdeTransformer(eqx.Module):
    """Neural ODE Transformer with SVD-based weight adaptation."""
    
    config: Gpt2Config = eqx.field(static=True)
    time_embedding: AlternativeTimeEmbeding
    block: AdaptiveBlock
    ln_f: hnn.LayerNorm
    policy: DynamicSVDPolicy
    
    dt: float = eqx.field(static=True)
    rank_ratio: float = eqx.field(static=True)
    
    @staticmethod
    def init(
        config: Gpt2Config,
        time_embed_dim,
        sinusodial_dim,
        rank_ratio: float = 0.5,
        policy_init_scale: float = 0.1,
        *,
        key,
    ):
        k_tembed, k_block, k_policy = jrandom.split(key, 3)
        TembedDim = hax.Axis("TembedDim", time_embed_dim)
        SinusodialDim = hax.Axis("SinusodialDim", sinusodial_dim)
        time_embeding = SinusoidalPosEmb.init(SinusodialDim, key=k_tembed)
        SinusodialDim = SinusodialDim.resize(sinusodial_dim * 2 + 1)

        block = AdaptiveBlock.init(
            config, SinusodialDim, TembedDim, rank_ratio, key=k_block
        )
        ln_f = hnn.LayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias
        )
        dt = 1.0 / config.num_layers

        mlp_in_size = config.Embed.size
        mlp_hidden_size = config.Mlp.size

        rank_per_layer = {}
        rank_fc = max(1, int(min(mlp_in_size, mlp_hidden_size) * rank_ratio))
        rank_proj = max(1, int(min(mlp_hidden_size, mlp_in_size) * rank_ratio))
        
        for layer_idx in range(config.num_layers):
            rank_per_layer[f"layer_{layer_idx}_c_fc"] = rank_fc
            rank_per_layer[f"layer_{layer_idx}_c_proj"] = rank_proj
        
        policy = DynamicSVDPolicy.init(
            num_layers=config.num_layers,
            rank_per_layer=rank_per_layer,
            task_vector_dim=config.Embed,
            key=k_policy,
        )
        
        return SVDNeuralOdeTransformer(
            config, time_embeding, block, ln_f, policy, dt, rank_ratio
        )
    
    def __call__(self, x: NamedArray, attn_mask, *, key=None) -> NamedArray:
        task_vector = hax.named(jnp.mean(x.array, axis=1), (x.axes[0], x.axes[2]))
        multipliers = self.policy(task_vector)

        t = (hax.arange(self.config.Layers, dtype=x.dtype) + 1) * self.dt
        dts = hax.ones((self.config.Layers,), dtype=x.dtype) * self.dt
        time_embed = self.time_embedding(t)
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else [None] * self.config.num_layers

        def make_do_block(layer_idx, layer_key):
            def do_block(x_in):
                layer_multipliers = {
                    "c_fc": multipliers.get(f"layer_{layer_idx}_c_fc"),
                    "c_proj": multipliers.get(f"layer_{layer_idx}_c_proj"),
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
        
        x = self.ln_f(x)
        return x

    def get_policy_loss(self, reg_strength: float = 0.01) -> jax.Array:
        params = eqx.filter(self.policy.policy_net, eqx.is_array)
        loss = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
        return loss * reg_strength


class SVDNeuralOdeLMHeadModel(eqx.Module):
    """Language model with SVD-adaptive Neural ODE transformer."""
    
    transformer: SVDNeuralOdeTransformer
    embeddings: Gpt2Embeddings
    
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
        config: Gpt2Config,
        time_embed_dim=100,
        sinusodial_dim=16,
        rank_ratio=0.5,
        policy_init_scale=0.1,
        *,
        key,
    ) -> "SVDNeuralOdeLMHeadModel":
        k_t, k_embeddings = jrandom.split(key, 2)
        transformer = SVDNeuralOdeTransformer.init(
            config,
            time_embed_dim=time_embed_dim,
            sinusodial_dim=sinusodial_dim,
            rank_ratio=rank_ratio,
            policy_init_scale=policy_init_scale,
            key=k_t,
        )
        embeddings = Gpt2Embeddings.init(Vocab, config, key=k_embeddings)
        
        return SVDNeuralOdeLMHeadModel(transformer, embeddings)
    
    def __call__(
        self, input_ids: NamedArray, attn_mask=None, *, key=None
    ) -> NamedArray:
        k_embed, k_transformer = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids, key=k_embed)
        x = self.transformer(x, attn_mask, key=k_transformer)
        lm_logits = self.embeddings.unembed(x)
        
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
    
    def resize_vocab(self, new_size: int, key=None) -> "SVDNeuralOdeLMHeadModel":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)
    
    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": None, "embeddings": None}

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