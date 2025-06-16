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
    Time-dependent SVD weights implementation combining Neural ODE with Transformer Squared concepts.
    
    Based on the SVD decomposition approach from self-adaptive-llms repo.
    """

    lin1: hnn.Linear
    lin2: hnn.Linear
    
    expert_U: NamedArray  # Shape: (num_experts, *In, svd_rank)
    expert_S: NamedArray  # Shape: (num_experts, svd_rank)  
    expert_V: NamedArray  # Shape: (num_experts, svd_rank, *Out)
    
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
        
        # Expert selector (outputs mixing weights)
        expert_selector = hnn.Linear.init(
            TembedDim, ExpertAxis, key=k_expert, use_bias=True
        )
        
        # SVD modulator (modulates singular values)
        svd_modulator = hnn.Linear.init(
            TembedDim, (ExpertAxis, SVDRank), key=k_modulator, use_bias=True
        )
        
        full_weight_key = jrandom.split(k_svd, num_experts)
        
        expert_U_list = []
        expert_S_list = []
        expert_V_list = []
        
        for i, expert_key in enumerate(full_weight_key):
            full_weight = hax.random.normal(
                expert_key, shape=In + Out, dtype=jnp.float32
            ) * jnp.sqrt(2.0 / (total_in + total_out))  # Xavier initialization
            
            # Reshape for SVD
            weight_2d = full_weight.array.reshape(total_in, total_out)
            
            # Perform SVD
            U_full, S_full, Vt_full = jnp.linalg.svd(weight_2d, full_matrices=False)
            
            # Take top svd_rank components
            U_truncated = U_full[:, :svd_rank]  # (total_in, svd_rank)
            S_truncated = S_full[:svd_rank]     # (svd_rank,)
            V_truncated = Vt_full[:svd_rank, :].T  # (total_out, svd_rank)
            
            # Reshape back to original dimensions
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
        if use_bias:
            f_b = hax.zeros(shape=(TembedDim,) + Out)
        else:
            f_b = None
        
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
        # Process time embedding (MLP block from neural ODE)
        t_embed = self.lin1(time_embed)
        t_embed = hnn.silu(t_embed)
        t_embed = self.lin2(t_embed)
        
        # Get expert mixing weights (softmax over experts)
        expert_logits = self.expert_selector(t_embed)
        expert_weights = hax.nn.softmax(expert_logits, axis="expert")
        
        # Get SVD modulation factors (sigmoid to keep positive)
        svd_modulation = self.svd_modulator(t_embed)
        svd_modulation = hnn.sigmoid(svd_modulation) + 0.1
        
        # Compose mixed weight matrix from experts
        mixed_weight = None
        
        for i in range(self.num_experts):
            U_i = self.expert_U.take("expert", i)  # (*In, svd_rank)
            S_i = self.expert_S.take("expert", i)  # (svd_rank,)
            V_i = self.expert_V.take("expert", i)  # (svd_rank, *Out)

            expert_weight = expert_weights.take("expert", i)
            expert_modulation = svd_modulation.take("expert", i)  # (svd_rank,)
            
            modulated_S = S_i * expert_modulation
            
            # Reconstruct weight matrix: U @ diag(S) @ V
            # U: (*In, svd_rank), modulated_S: (svd_rank,), V: (svd_rank, *Out)
            US = U_i * modulated_S  # Broadcast multiply
            weight_contrib = US.dot("svd_rank", V_i)  # Result: (*In, *Out)
            
            # Weight by expert mixing coefficient
            weighted_contrib = weight_contrib * expert_weight
            
            if mixed_weight is None:
                mixed_weight = weighted_contrib
            else:
                mixed_weight = mixed_weight + weighted_contrib
        
        # Apply mixed weight to input
        output = x.dot(self.In, mixed_weight)
        output = hax.auto_sharded(output)
        
        # Add bias if present
        if self.f_b is not None:
            bias = t_embed.dot(self.TembedDim, self.f_b)
            output = output + bias
            output = hax.auto_sharded(output)
        
        return output

    def evaluate_at(self, time_embed: NamedArray):
        """Evaluate the layer at a specific time point to get static weights."""
        # Process time embedding
        t_embed = self.lin1(time_embed)
        t_embed = hnn.silu(t_embed)
        t_embed = self.lin2(time_embed)
        
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
            US = U_i * modulated_S
            weight_contrib = US.dot("svd_rank", V_i)
            weighted_contrib = weight_contrib * expert_weight
            
            if mixed_weight is None:
                mixed_weight = weighted_contrib
            else:
                mixed_weight = mixed_weight + weighted_contrib
        
        # Compute bias
        bias = None
        if self.f_b is not None:
            bias = t_embed.dot(self.TembedDim, self.f_b)
        
        return hnn.Linear(weight=mixed_weight, bias=bias, In=self.In, Out=self.Out)

    def get_expert_contributions(self, time_embed: NamedArray):
        """Get the contribution of each expert at a given time point."""
        t_embed = self.lin1(time_embed)
        t_embed = hnn.silu(t_embed)
        t_embed = self.lin2(time_embed)
        
        expert_logits = self.expert_selector(t_embed)
        expert_weights = hax.nn.softmax(expert_logits, axis="expert")
        
        svd_modulation = self.svd_modulator(t_embed)
        svd_modulation = hnn.sigmoid(svd_modulation) + 0.1
        
        return {
            "expert_weights": expert_weights,
            "svd_modulation": svd_modulation,
        }