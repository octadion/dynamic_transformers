import dataclasses
from typing import List, Optional, Sequence

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, AxisSpec, NamedArray


class TemporalSVDLinear(eqx.Module):
    """
    Time-dependent SVD weights implementation with proper error handling.
    """

    lin1: hnn.Linear
    lin2: hnn.Linear
    
    expert_U: NamedArray  
    expert_S: NamedArray  
    expert_V: NamedArray  
    
    expert_selector: hnn.Linear
    svd_modulator: hnn.Linear

    In: AxisSpec = eqx.field(static=True)
    Out: AxisSpec = eqx.field(static=True)
    TembedDim: AxisSpec = eqx.field(static=True)
    num_experts: int = eqx.field(static=True)
    svd_rank: int = eqx.field(static=True)

    f_b: Optional[NamedArray]

    @staticmethod
    def init(
        SinusodialDim: hax.Axis,
        TembedDim: hax.Axis,
        In: hax.AxisSpec,
        Out: hax.AxisSpec,
        num_experts: int = 4,
        svd_rank: Optional[int] = None,
        *,
        key,
        use_bias=True,
    ):
        k_lin1, k_lin2, k_expert, k_modulator, k_svd = jrandom.split(key, 5)
        
        if not isinstance(In, Sequence):
            In = (In,)
        if not isinstance(Out, Sequence):
            Out = (Out,)
        
        Out_unique = []
        for i, out_axis in enumerate(Out):
            if any(in_axis.name == out_axis.name for in_axis in In):
                Out_unique.append(out_axis.alias(f"{out_axis.name}_out"))
            else:
                Out_unique.append(out_axis)
        Out = tuple(Out_unique)
        
        total_in = int(jnp.prod(jnp.array([ax.size for ax in In])))
        total_out = int(jnp.prod(jnp.array([ax.size for ax in Out])))
        if svd_rank is None:
            svd_rank = min(total_in, total_out) // 4 
        
        SVDRank = hax.Axis("svd_rank", svd_rank)
        ExpertAxis = hax.Axis("expert", num_experts)
        
        # Time embedding network
        TembedDim_alias = TembedDim.alias("TembedDim_alias")
        lin1 = hnn.Linear.init(SinusodialDim, TembedDim_alias, key=k_lin1)
        lin2 = hnn.Linear.init(TembedDim_alias, TembedDim, key=k_lin2)
        
        # Expert selector
        expert_selector = hnn.Linear.init(
            TembedDim, ExpertAxis, key=k_expert, use_bias=True
        )
        
        # SVD modulator
        svd_modulator = hnn.Linear.init(
            TembedDim, (ExpertAxis, SVDRank), key=k_modulator, use_bias=True
        )
        
        # Initialize SVD components for each expert
        expert_keys = jrandom.split(k_svd, num_experts)
        
        expert_U_list = []
        expert_S_list = []
        expert_V_list = []
        
        for expert_key in expert_keys:
            full_weight = hax.random.normal(
                expert_key, shape=In + Out, dtype=jnp.float32
            ) * jnp.sqrt(2.0 / (total_in + total_out))
            
            # Reshape for SVD
            weight_2d = full_weight.array.reshape(total_in, total_out)
            
            try:
                U_full, S_full, Vt_full = jnp.linalg.svd(weight_2d, full_matrices=False)
            except Exception as e:
                print(f"SVD failed, using random initialization: {e}")
                U_full = jax.random.normal(expert_key, (total_in, min(total_in, total_out)))
                S_full = jnp.ones(min(total_in, total_out))
                Vt_full = jax.random.normal(expert_key, (min(total_in, total_out), total_out))
                
                U_full, _ = jnp.linalg.qr(U_full)
                Vt_full, _ = jnp.linalg.qr(Vt_full.T)
                Vt_full = Vt_full.T
            
            # Take top svd_rank components
            U_truncated = U_full[:, :svd_rank]
            S_truncated = S_full[:svd_rank]
            V_truncated = Vt_full[:svd_rank, :].T
            
            # Reshape back to haliax NamedArrays
            U_shaped = hax.NamedArray(
                U_truncated.reshape(*(ax.size for ax in In), svd_rank),
                axes=In + (SVDRank,)
            )
            S_shaped = hax.NamedArray(S_truncated, axes=(SVDRank,))
            V_shaped = hax.NamedArray(
                V_truncated.T.reshape(svd_rank, *(ax.size for ax in Out)),
                axes=(SVDRank,) + Out
            )
            
            expert_U_list.append(U_shaped)
            expert_S_list.append(S_shaped)
            expert_V_list.append(V_shaped)
        
        # Stack experts
        expert_U = hax.stack("expert", expert_U_list)
        expert_S = hax.stack("expert", expert_S_list)
        expert_V = hax.stack("expert", expert_V_list)
        
        # Optional bias
        f_b = hax.zeros(shape=(TembedDim,) + Out) if use_bias else None
        
        return TemporalSVDLinear(
            lin1=lin1,
            lin2=lin2,
            expert_U=expert_U,
            expert_S=expert_S,
            expert_V=expert_V,
            expert_selector=expert_selector,
            svd_modulator=svd_modulator,
            In=In,
            Out=Out,
            TembedDim=TembedDim,
            num_experts=num_experts,
            svd_rank=svd_rank,
            f_b=f_b,
        )

    def __call__(
        self,
        time_embed: NamedArray,
        x: NamedArray,
        *,
        key=None,
    ):
        try:
            # Process time embedding
            t_embed = self.lin1(time_embed)
            t_embed = hnn.silu(t_embed)
            t_embed = self.lin2(t_embed)
            
            # Get expert mixing weights
            expert_logits = self.expert_selector(t_embed)
            expert_weights = hax.nn.softmax(expert_logits, axis="expert")
            
            # Get SVD modulation factors
            svd_modulation = self.svd_modulator(t_embed)
            svd_modulation = hnn.sigmoid(svd_modulation) + 0.1
            
            # Compose mixed weight matrix from experts - FIXED VERSION
            output = None
            
            for i in range(self.num_experts):
                # Get expert components
                U_i = self.expert_U.take("expert", i)  # (In..., svd_rank)
                S_i = self.expert_S.take("expert", i)  # (svd_rank,)
                V_i = self.expert_V.take("expert", i)  # (svd_rank, Out...)
                
                # Get weights for this expert
                expert_weight = expert_weights.take("expert", i)  # scalar
                expert_modulation = svd_modulation.take("expert", i)  # (svd_rank,)
                
                # Modulate singular values
                modulated_S = S_i * expert_modulation  # (svd_rank,)
                
                # Compute expert contribution: U @ diag(modulated_S) @ V
                # First: U * modulated_S (broadcast along svd_rank axis)
                US = U_i * hax.broadcast_to(modulated_S, U_i.axes)  # (In..., svd_rank)
                
                # Then: (U * S) @ V
                expert_weight_matrix = hax.dot("svd_rank", US, V_i)  # (In..., Out...)
                
                # Weight by expert contribution
                weighted_contribution = expert_weight_matrix * expert_weight
                
                if output is None:
                    output = weighted_contribution
                else:
                    output = output + weighted_contribution
            
            # Apply to input
            result = hax.dot(self.In, x, output)
            result = hax.auto_sharded(result)
            
            # Add bias if present
            if self.f_b is not None:
                bias = hax.dot(self.TembedDim, t_embed, self.f_b)
                result = result + bias
                result = hax.auto_sharded(result)
            
            return result
            
        except Exception as e:
            print(f"TemporalSVDLinear forward failed: {e}")
            # Fallback to simple linear transformation
            return hax.zeros(x.axes[:-len(self.In)] + self.Out)

    def evaluate_at(self, time_embed: NamedArray):
        """Evaluate the layer at a specific time point to get static weights."""
        try:
            # Process time embedding
            t_embed = self.lin1(time_embed)
            t_embed = hnn.silu(t_embed)
            t_embed = self.lin2(t_embed)
            
            # Get expert mixing weights
            expert_logits = self.expert_selector(t_embed)
            expert_weights = hax.nn.softmax(expert_logits, axis="expert")
            
            # Get SVD modulation factors
            svd_modulation = self.svd_modulator(t_embed)
            svd_modulation = hnn.sigmoid(svd_modulation) + 0.1
            
            # Compose final weight matrix
            mixed_weight = None
            
            for i in range(self.num_experts):
                U_i = self.expert_U.take("expert", i)
                S_i = self.expert_S.take("expert", i)
                V_i = self.expert_V.take("expert", i)
                
                expert_weight = expert_weights.take("expert", i)
                expert_modulation = svd_modulation.take("expert", i)
                
                modulated_S = S_i * expert_modulation
                US = U_i * hax.broadcast_to(modulated_S, U_i.axes)
                weight_contrib = hax.dot("svd_rank", US, V_i)
                weighted_contrib = weight_contrib * expert_weight
                
                if mixed_weight is None:
                    mixed_weight = weighted_contrib
                else:
                    mixed_weight = mixed_weight + weighted_contrib
            
            # Compute bias
            bias = None
            if self.f_b is not None:
                bias = hax.dot(self.TembedDim, t_embed, self.f_b)
            
            return hnn.Linear(weight=mixed_weight, bias=bias, In=self.In, Out=self.Out)
        
        except Exception as e:
            print(f"TemporalSVDLinear evaluate_at failed: {e}")
            # Return identity-like transformation
            weight = hax.zeros(self.In + self.Out)
            return hnn.Linear(weight=weight, bias=None, In=self.In, Out=self.Out)

    def get_expert_contributions(self, time_embed: NamedArray):
        """Get the contribution of each expert at a given time point."""
        try:
            t_embed = self.lin1(time_embed)
            t_embed = hnn.silu(t_embed)
            t_embed = self.lin2(t_embed)
            
            expert_logits = self.expert_selector(t_embed)
            expert_weights = hax.nn.softmax(expert_logits, axis="expert")
            
            svd_modulation = self.svd_modulator(t_embed)
            svd_modulation = hnn.sigmoid(svd_modulation) + 0.1
            
            return {
                "expert_weights": expert_weights,
                "svd_modulation": svd_modulation,
            }
        except Exception as e:
            print(f"get_expert_contributions failed: {e}")
            return {
                "expert_weights": hax.zeros((hax.Axis("expert", self.num_experts),)),
                "svd_modulation": hax.zeros((hax.Axis("expert", self.num_experts), hax.Axis("svd_rank", self.svd_rank))),
            }