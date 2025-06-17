import dataclasses
from typing import Dict, Optional, Sequence

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call
from levanter.models.gpt2 import Gpt2Config, Gpt2Embeddings

from .temporal_svd_linear import TemporalSVDLinear
from .adaptive_attention_mlp import AdaptiveAttention, AdaptiveMLP
from qkvflow.nn.time_embed import SinusoidalPosEmb


class AdaptiveBlock(eqx.Module):
    """
    Adaptive transformer block combining temporal dynamics with expert specialization.
    """
    
    config: Gpt2Config = eqx.field(static=True)

    attn_ln: TemporalSVDLinear  # Adaptive layer norm
    attn: AdaptiveAttention      # Adaptive attention
    mlp_ln: TemporalSVDLinear   # Adaptive layer norm  
    mlp: AdaptiveMLP            # Adaptive MLP
    resid_dropout: hnn.Dropout
    
    task_predictor: hnn.Linear
    
    @staticmethod
    def init(
        config: Gpt2Config, 
        SinusodialDim: Axis, 
        TembedDim: Axis, 
        num_experts: int = 4,
        *,
        key
    ):
        k_attn_ln, k_attn, k_mlp_ln, k_mlp, k_task = jrandom.split(key, 5)
        
        attn_ln = TemporalSVDLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=config.Embed,
            Out=config.Embed.alias("embed_attn_out"),
            num_experts=num_experts,
            svd_rank=min(config.Embed.size // 4, 64),
            key=k_attn_ln,
            use_bias=config.use_bias,
        )
        
        mlp_ln = TemporalSVDLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=config.Embed,
            Out=config.Embed.alias("embed_mlp_out"),
            num_experts=num_experts,
            svd_rank=min(config.Embed.size // 4, 64),
            key=k_mlp_ln,
            use_bias=config.use_bias,
        )
        
        # Adaptive attention and MLP
        attn = AdaptiveAttention.init(
            config, SinusodialDim, TembedDim, num_experts, key=k_attn
        )
        mlp = AdaptiveMLP.init(
            config, SinusodialDim, TembedDim, num_experts, key=k_mlp, use_bias=config.use_bias
        )
        
        # Task type predictor
        TaskTypes = hax.Axis("task_types", 8)  # math, code, reasoning, text, etc.
        task_predictor = hnn.Linear.init(
            In=TembedDim, Out=TaskTypes, key=k_task, use_bias=True
        )
        
        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)
        
        return AdaptiveBlock(
            config=config,
            attn_ln=attn_ln,
            attn=attn,
            mlp_ln=mlp_ln,
            mlp=mlp,
            resid_dropout=resid_dropout,
            task_predictor=task_predictor,
        )
    
    def __call__(self, time_embed: NamedArray, x: NamedArray, mask, layer_idx, *, key):
        k1, k2, k3, k4 = maybe_rng_split(key, 4)
        
        task_logits = self.task_predictor(time_embed)
        task_weights = hax.nn.softmax(task_logits, axis="task_types")

        ln_params = self.attn_ln(time_embed, x)
        
        mean = x.mean(axis="embed")
        var = x.var(axis="embed")
        
        mean_broadcasted = mean.add_axis("embed", x.axis_size("embed"))
        var_broadcasted = var.add_axis("embed", x.axis_size("embed"))
        normalized_x = (x - mean_broadcasted) / hax.sqrt(var_broadcasted + 1e-5)
        
        normalized_x = normalized_x + ln_params
        
        attn_output = self.attn(
            time_embed=time_embed,
            x=normalized_x,
            mask=mask,
            layer_idx=layer_idx,
            key=k1,
        )
        attn_output = self.resid_dropout(attn_output, key=k2)
        
        # Apply residual connection  
        x = x + attn_output
        
        ln_params = self.mlp_ln(time_embed, x)
        
        mean = x.mean(axis="embed")
        var = x.var(axis="embed")
        
        mean_broadcasted = mean.add_axis("embed", x.axis_size("embed"))
        var_broadcasted = var.add_axis("embed", x.axis_size("embed"))
        normalized_x = (x - mean_broadcasted) / hax.sqrt(var_broadcasted + 1e-5)
        
        # Apply adaptive scaling and shifting
        normalized_x = normalized_x + ln_params
        
        ff_output = self.mlp(
            time_embed=time_embed,
            x=normalized_x,
            key=k3,
        )
        ff_output = self.resid_dropout(ff_output, key=k4)
        
        # Apply residual connection
        output = x + ff_output
        
        return output
    
    def evaluate_at(self, time_embed: NamedArray):
        """Evaluate block at specific time point."""
        attn_ln = self.attn_ln.evaluate_at(time_embed)
        attn = self.attn.evaluate_at(time_embed)
        mlp_ln = self.mlp_ln.evaluate_at(time_embed)
        mlp = self.mlp.evaluate_at(time_embed)
        
        # Create a standard block for evaluation
        from qkvflow.nn.dynamic import _Block
        return _Block(
            attn_ln=attn_ln,
            attn=attn,
            mlp_ln=mlp_ln,
            mlp=mlp,
            resid_dropout=self.resid_dropout,
        )


class AdaptiveNeuralOdeTransformer(eqx.Module):
    """
    Adaptive Neural ODE Transformer that combines continuous depth with expert specialization.
    """
    
    config: Gpt2Config = eqx.field(static=True)
    time_embedding: SinusoidalPosEmb
    block: AdaptiveBlock
    ln_f: hnn.LayerNorm
    
    # Adaptive control
    difficulty_predictor: hnn.Linear  # Predicts problem difficulty for depth control
    dt: float = eqx.field(static=True)
    
    @staticmethod
    def init(
        config: Gpt2Config,
        time_embed_dim: int,
        sinusodial_dim: int,
        num_experts: int = 4,
        *,
        key,
    ):
        k_tembed, k_block, k_diff = jrandom.split(key, 3)
        
        TembedDim = hax.Axis("TembedDim", time_embed_dim)
        SinusodialDim = hax.Axis("SinusodialDim", sinusodial_dim)
        
        # Time embedding
        time_embedding = SinusoidalPosEmb.init(SinusodialDim, key=k_tembed)
        SinusodialDim = SinusodialDim.resize(sinusodial_dim * 2 + 1)
        
        # Adaptive block
        block = AdaptiveBlock.init(
            config, SinusodialDim, TembedDim, num_experts, key=k_block
        )
        
        # Final layer norm
        ln_f = hnn.LayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias
        )
        
        # Difficulty predictor for adaptive depth
        DifficultyLevels = hax.Axis("difficulty", 5)  # very_easy to very_hard
        difficulty_predictor = hnn.Linear.init(
            In=TembedDim, Out=DifficultyLevels, key=k_diff, use_bias=True
        )
        
        dt = 1.0 / config.num_layers
        
        return AdaptiveNeuralOdeTransformer(
            config=config,
            time_embedding=time_embedding,
            block=block,
            ln_f=ln_f,
            difficulty_predictor=difficulty_predictor,
            dt=dt,
        )
    
    def __call__(self, x: NamedArray, attn_mask, *, key=None) -> NamedArray:
        # Generate time steps
        t = (hax.arange(self.config.Layers, dtype=x.dtype) + 1) * self.dt
        time_embed = self.time_embedding(t)
        
        # Predict difficulty to potentially adjust step size
        avg_time_embed = time_embed.mean(axis="layers")
        difficulty_logits = self.difficulty_predictor(avg_time_embed)
        difficulty_weights = hax.nn.softmax(difficulty_logits, axis="difficulty")
        
        # Adjust time steps based on difficulty (harder problems = smaller steps)
        difficulty_factor = jnp.sum(
            difficulty_weights.array * jnp.array([1.0, 0.8, 0.6, 0.4, 0.2])
        )
        adaptive_dt = self.dt * difficulty_factor
        dts = hax.ones((self.config.Layers,), dtype=x.dtype) * adaptive_dt
        
        if key is not None:
            keys = maybe_rng_split(key, self.config.num_layers)
        else:
            keys = None
        
        def do_block(x, time_embed, dt, key=None):
            output = self.block(time_embed, x, attn_mask, None, key=key)
            return x + output * dt
        
        # Checkpointed computation for memory efficiency
        do_block = jax.checkpoint(do_block, prevent_cse=False)
        
        x = hax.fold(do_block, axis=self.config.Layers)(x, time_embed, dts, key=keys)
        x = self.ln_f(x)
        
        return x
    
    def compute_trajectory(self, x, attn_mask):
        """Compute the full trajectory through the ODE."""
        t = (hax.arange(self.config.Layers, dtype=x.dtype) + 1) * self.dt
        time_embed = self.time_embedding(t)
        
        def do_block(x, time_embed):
            output = self.block(time_embed, x, attn_mask, None, key=None)
            ret = x + output * self.dt
            return ret, ret
        
        do_block = jax.checkpoint(do_block)
        _, trajectory = hax.scan(do_block, axis=self.config.Layers)(x, time_embed)
        
        return trajectory
    
    def evaluate_at(self, dt: float):
        """Evaluate at specific step size for discrete inference."""
        dtype = self.ln_f.weight.dtype
        new_axis = self.config.Layers.resize(int(1.0 / dt))
        t = (hax.arange(new_axis, dtype=dtype) + 1) * dt
        dts = hax.ones((new_axis,), dtype=dtype) * dt
        time_embed: NamedArray = self.time_embedding(t)
        
        blocks = []
        for i in range(time_embed.axis_size("layers")):
            tb = time_embed.take("layers", i)
            block = self.block.evaluate_at(tb)
            blocks.append(block)
        
        from qkvflow.nn.dynamic import _NeuralOdeTransformer
        return _NeuralOdeTransformer(
            config=self.config, 
            blocks=blocks, 
            ln_f=self.ln_f, 
            dts=np.array(dts.array)
        )


class AdaptiveNeuralOdeLMHeadModel(eqx.Module):
    """
    Complete language model with adaptive Neural ODE transformer.
    """
    
    transformer: AdaptiveNeuralOdeTransformer
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
        time_embed_dim: int = 100,
        sinusodial_dim: int = 16,
        num_experts: int = 4,
        *,
        key,
    ) -> "AdaptiveNeuralOdeLMHeadModel":
        k_t, k_embeddings = jrandom.split(key, 2)
        
        transformer = AdaptiveNeuralOdeTransformer.init(
            config,
            time_embed_dim=time_embed_dim,
            sinusodial_dim=sinusodial_dim,
            num_experts=num_experts,
            key=k_t,
        )
        embeddings = Gpt2Embeddings.init(Vocab, config, key=k_embeddings)
        
        return AdaptiveNeuralOdeLMHeadModel(transformer, embeddings)
    
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
        example,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> NamedArray:
        logits = self(example.tokens, example.attn_mask, key=key)
        targets = hax.roll(example.tokens, -1, axis=self.Pos.name)
        target_y = hax.nn.one_hot(targets, self.Vocab, dtype=logits.dtype)
        return hnn.cross_entropy_loss(
            logits,
            self.Vocab,
            target_y,
            reduction,
            reduction_axis=reduction_axis,
            where=example.loss_mask,
        )
    
    @property
    def vocab_size(self) -> int:
        return self.Vocab.size
    
    def resize_vocab(self, new_size: int, key=None) -> "AdaptiveNeuralOdeLMHeadModel":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)
    
    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": None, "embeddings": None}
    
    def evaluate_at(self, dt: float):
        """Evaluate at specific step size."""
        transformer = self.transformer.evaluate_at(dt)
        
        from qkvflow.nn.dynamic import NeuralOdeLMHeadModel
        return NeuralOdeLMHeadModel(
            transformer=transformer,
            embeddings=self.embeddings,
        )
    
    def get_expert_analysis(self, input_ids: NamedArray, attn_mask=None, *, key=None):
        """Analyze expert contributions and task predictions."""
        k_embed, k_transformer = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids, key=k_embed)
        
        # Get time embeddings
        t = (hax.arange(self.config.Layers, dtype=x.dtype) + 1) * self.transformer.dt
        time_embed = self.transformer.time_embedding(t)
        
        analysis = {}
        
        # Analyze expert contributions over time
        for layer_idx in range(self.config.num_layers):
            te = time_embed.take("layers", layer_idx)
            
            # Get attention expert contributions
            attn_contribs = self.transformer.block.attn.c_attn.get_expert_contributions(te)
            analysis[f"layer_{layer_idx}_attn_experts"] = attn_contribs
            
            # Get MLP expert contributions  
            mlp_contribs = self.transformer.block.mlp.c_fc.get_expert_contributions(te)
            analysis[f"layer_{layer_idx}_mlp_experts"] = mlp_contribs
            
            # Get task predictions
            task_logits = self.transformer.block.task_predictor(te)
            task_probs = hax.nn.softmax(task_logits, axis="task_types")
            analysis[f"layer_{layer_idx}_task_probs"] = task_probs
        
        return analysis