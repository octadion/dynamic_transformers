"""
SVD-based Temporal Linear Layer for Neural ODE Transformer
Combines SVD decomposition from Transformer² with time-dependent weights
"""

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Optional, Sequence
from haliax import Axis, NamedArray


class SVDTemporalLinear(eqx.Module):
    """
    SVD-based temporal linear layer that combines:
    1. SVD decomposition for parameter efficiency
    2. Time-dependent mixing of multiple expert decompositions
    3. Learnable singular value modulation
    """
    
    # Base SVD components
    U_base: NamedArray  # Left singular vectors
    S_base: NamedArray  # Singular values  
    V_base: NamedArray  # Right singular vectors
    
    # Expert SVD components (multiple experts)
    expert_Us: list[NamedArray]
    expert_Ss: list[NamedArray] 
    expert_Vs: list[NamedArray]
    
    # Time-dependent mixing networks
    time_proj: hnn.Linear
    expert_mixer: hnn.Linear  # Projects time to expert mixing weights
    singular_modulator: hnn.Linear  # Modulates singular values
    
    # Network for time embedding processing
    time_mlp1: hnn.Linear
    time_mlp2: hnn.Linear
    
    In: hax.AxisSpec = eqx.field(static=True)
    Out: hax.AxisSpec = eqx.field(static=True)
    TembedDim: hax.AxisSpec = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    num_experts: int = eqx.field(static=True)
    
    @staticmethod
    def init(
        SinusodialDim: hax.Axis,
        TembedDim: hax.Axis,
        In: hax.AxisSpec,
        Out: hax.AxisSpec,
        rank: int = 64,
        num_experts: int = 4,
        *,
        key,
        use_bias=True,
    ):
        keys = jrandom.split(key, 10)
        
        if not isinstance(In, Sequence):
            In = (In,)
        if not isinstance(Out, Sequence):
            Out = (Out,)
            
        # Calculate input and output dimensions
        in_dim = 1
        for axis in In:
            in_dim *= axis.size
        out_dim = 1  
        for axis in Out:
            out_dim *= axis.size
            
        # Rank for SVD decomposition
        actual_rank = min(rank, min(in_dim, out_dim))
        Rank = hax.Axis("rank", actual_rank)
        
        # Initialize base SVD components with proper scaling
        U_base = hax.random.normal(keys[0], shape=In + (Rank,)) * 0.1
        S_base = hax.ones(shape=(Rank,)) * 0.1  # Small initial singular values
        V_base = hax.random.normal(keys[1], shape=(Rank,) + Out) * 0.1
        
        # Initialize expert SVD components
        expert_Us = []
        expert_Ss = []
        expert_Vs = []
        
        for i in range(num_experts):
            expert_Us.append(
                hax.random.normal(keys[2+i], shape=In + (Rank,)) * 0.05
            )
            expert_Ss.append(
                hax.random.normal(keys[2+i], shape=(Rank,)) * 0.05
            )
            expert_Vs.append(
                hax.random.normal(keys[2+i], shape=(Rank,) + Out) * 0.05
            )
        
        # Time embedding processing
        TembedDim_alias = TembedDim.alias("TembedDim_alias")
        time_mlp1 = hnn.Linear.init(SinusodialDim, TembedDim_alias, key=keys[6])
        time_mlp2 = hnn.Linear.init(TembedDim_alias, TembedDim, key=keys[7])
        
        # Expert mixing network
        ExpertDim = hax.Axis("experts", num_experts)
        expert_mixer = hnn.Linear.init(
            In=TembedDim,
            Out=ExpertDim,
            key=keys[8],
            use_bias=True
        )
        
        # Singular value modulation network
        singular_modulator = hnn.Linear.init(
            In=TembedDim,
            Out=Rank,
            key=keys[9],
            use_bias=True
        )
        
        return SVDTemporalLinear(
            U_base=U_base,
            S_base=S_base, 
            V_base=V_base,
            expert_Us=expert_Us,
            expert_Ss=expert_Ss,
            expert_Vs=expert_Vs,
            time_proj=None,
            expert_mixer=expert_mixer,
            singular_modulator=singular_modulator,
            time_mlp1=time_mlp1,
            time_mlp2=time_mlp2,
            In=In,
            Out=Out,
            TembedDim=TembedDim,
            rank=actual_rank,
            num_experts=num_experts,
        )
    
    def __call__(
        self,
        time_embed: NamedArray,
        x: NamedArray,
        *,
        key=None,
    ):
        # Process time embedding
        t_embed = self.time_mlp1(time_embed)
        t_embed = hnn.silu(t_embed)  
        t_embed = self.time_mlp2(t_embed)
        
        # Get expert mixing weights (softmax normalized)
        expert_weights = self.expert_mixer(t_embed)
        expert_weights = hnn.softmax(expert_weights, axis="experts")
        
        # Get singular value modulation
        s_modulation = self.singular_modulator(t_embed)
        s_modulation = hnn.sigmoid(s_modulation)  # Keep in [0,1] range
        
        # Combine expert components using mixing weights
        combined_U = self.U_base
        combined_S = self.S_base
        combined_V = self.V_base
        
        for i, (U_expert, S_expert, V_expert) in enumerate(
            zip(self.expert_Us, self.expert_Ss, self.expert_Vs)
        ):
            weight = expert_weights.take("experts", i)
            combined_U = combined_U + weight * U_expert
            combined_S = combined_S + weight * S_expert  
            combined_V = combined_V + weight * V_expert
        
        # Apply singular value modulation
        modulated_S = combined_S * s_modulation
        
        # Reconstruct weight matrix: W = U @ diag(S) @ V^T
        # For haliax: W = U @ diag(S) @ V (since V is already transposed)
        US = combined_U * modulated_S.broadcast_axis(*combined_U.axes[:-1])
        W = hax.dot("rank", US, combined_V)
        
        # Apply linear transformation
        output = x.dot(self.In, W)
        output = hax.auto_sharded(output)
        
        return output
    
    def get_effective_weight(self, time_embed: NamedArray) -> NamedArray:
        """Get the effective weight matrix at given time (for analysis)"""
        # Process time embedding
        t_embed = self.time_mlp1(time_embed)
        t_embed = hnn.silu(t_embed)
        t_embed = self.time_mlp2(t_embed)
        
        # Get expert mixing weights
        expert_weights = self.expert_mixer(t_embed)
        expert_weights = hnn.softmax(expert_weights, axis="experts")
        
        # Get singular value modulation
        s_modulation = self.singular_modulator(t_embed)
        s_modulation = hnn.sigmoid(s_modulation)
        
        # Combine expert components
        combined_U = self.U_base
        combined_S = self.S_base
        combined_V = self.V_base
        
        for i, (U_expert, S_expert, V_expert) in enumerate(
            zip(self.expert_Us, self.expert_Ss, self.expert_Vs)
        ):
            weight = expert_weights.take("experts", i)
            combined_U = combined_U + weight * U_expert
            combined_S = combined_S + weight * S_expert
            combined_V = combined_V + weight * V_expert
        
        # Apply modulation and reconstruct
        modulated_S = combined_S * s_modulation
        US = combined_U * modulated_S.broadcast_axis(*combined_U.axes[:-1])
        W = hax.dot("rank", US, combined_V)
        
        return W


class PolicyBasedSVDTemporalLinear(eqx.Module):
    """
    Policy-based version that uses learnable policies to control expert mixing
    Similar to the policy approach in Transformer²
    """
    
    base_svd_layer: SVDTemporalLinear
    policy_network: hnn.Linear
    value_network: hnn.Linear
    
    trainable_params: list = eqx.field(init=False)
    
    @staticmethod
    def init(
        SinusodialDim: hax.Axis,
        TembedDim: hax.Axis, 
        In: hax.AxisSpec,
        Out: hax.AxisSpec,
        rank: int = 64,
        num_experts: int = 4,
        *,
        key,
    ):
        keys = jrandom.split(key, 3)
        
        # Initialize base SVD layer
        base_layer = SVDTemporalLinear.init(
            SinusodialDim=SinusodialDim,
            TembedDim=TembedDim,
            In=In,
            Out=Out,
            rank=rank,
            num_experts=num_experts,
            key=keys[0]
        )
        
        # Policy network (outputs action probabilities for expert selection)
        ExpertDim = hax.Axis("experts", num_experts)
        policy_network = hnn.Linear.init(
            In=TembedDim,
            Out=ExpertDim,
            key=keys[1]
        )
        
        # Value network (estimates value of current state)
        value_network = hnn.Linear.init(
            In=TembedDim,
            Out=hax.Axis("value", 1),
            key=keys[2]
        )
        
        instance = PolicyBasedSVDTemporalLinear(
            base_svd_layer=base_layer,
            policy_network=policy_network,
            value_network=value_network
        )
        
        # Set trainable parameters (policy and value networks)
        instance.trainable_params = [
            instance.policy_network.weight,
            instance.policy_network.bias,
            instance.value_network.weight, 
            instance.value_network.bias
        ]
        
        return instance
    
    def __call__(self, time_embed: NamedArray, x: NamedArray, *, key=None):
        # Use policy network to control expert mixing
        policy_logits = self.policy_network(time_embed)
        expert_probs = hnn.softmax(policy_logits, axis="experts")
        
        # Replace expert mixer output in base layer
        # This is a bit hacky but allows us to use policy control
        modified_layer = eqx.tree_at(
            lambda layer: layer.expert_mixer,
            self.base_svd_layer,
            lambda _: MockExpertMixer(expert_probs)
        )
        
        return modified_layer(time_embed, x, key=key)
    
    def get_policy_action(self, time_embed: NamedArray) -> NamedArray:
        """Get policy action (expert probabilities)"""
        return hnn.softmax(self.policy_network(time_embed), axis="experts")
    
    def get_value(self, time_embed: NamedArray) -> NamedArray:
        """Get state value estimate"""
        return self.value_network(time_embed)


class MockExpertMixer(eqx.Module):
    """Helper class to mock expert mixer with predetermined probabilities"""
    probs: NamedArray
    
    def __call__(self, x):
        return self.probs