"""
SVD-based adaptive weight module for Transformer-Squared integration.
"""
import math
import dataclasses
import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split
from typing import Callable, Dict, Tuple, List, Optional


class SVDLinear(eqx.Module):
    """
    Linear layer with SVD decomposition.
    W = U @ S @ V^T, where S is modulated by multipliers passed during forward pass.
    """
    U: NamedArray
    S_base: NamedArray
    V: NamedArray
    
    In: hax.AxisSpec = eqx.field(static=True)
    Out: hax.AxisSpec = eqx.field(static=True)
    Rank: hax.Axis = eqx.field(static=True)
    
    use_bias: bool = eqx.field(static=True)
    bias: Optional[NamedArray]
    
    @staticmethod
    def from_linear(
        linear: hnn.Linear,
        rank_ratio: float = 0.5,
        *,
        key: jax.random.PRNGKey,
    ) -> "SVDLinear":
        """Create SVDLinear from existing Linear layer."""
        import math

        weight = linear.weight
        
        if isinstance(linear.Out, Axis):
            out_size = linear.weight.axis_size(linear.Out)
        elif isinstance(linear.Out, tuple):
            out_size = math.prod(linear.weight.axis_size(ax) for ax in linear.Out)
        else:
            raise TypeError(f"Unsupported AxisSpec type for linear.Out: {type(linear.Out)}")

        if isinstance(linear.In, Axis):
            in_size = linear.weight.axis_size(linear.In)
        elif isinstance(linear.In, tuple):
            in_size = math.prod(linear.weight.axis_size(ax) for ax in linear.In)
        else:
            raise TypeError(f"Unsupported AxisSpec type for linear.In: {type(linear.In)}")

        weight_matrix = weight.array.reshape(out_size, in_size)

        U_arr, S_arr, Vh_arr = jnp.linalg.svd(weight_matrix, full_matrices=False)
        
        full_rank = min(in_size, out_size)
        rank = max(1, int(full_rank * rank_ratio))
        Rank = hax.Axis("rank", rank)
        
        U_arr = U_arr[:, :rank]     # [out_size, rank]
        S_arr = S_arr[:rank]        # [rank]
        Vh_arr = Vh_arr[:rank, :]   # [rank, in_size]

        U_arr, S_arr, Vh_arr = jnp.linalg.svd(weight_matrix, full_matrices=False)

        full_rank = min(in_size, out_size)
        rank = max(1, int(full_rank * rank_ratio))
        Rank = hax.Axis("rank", rank)

        U_arr = U_arr[:, :rank]
        S_arr_truncated = S_arr[:rank]
        Vh_arr = Vh_arr[:rank, :]

        s_norm = jnp.linalg.norm(S_arr_truncated)
        S_arr_normalized = S_arr_truncated / (s_norm + 1e-8)

        S_base = hax.NamedArray(S_arr_normalized, axes=(Rank,))

        if isinstance(linear.Out, tuple):
            u_axes = linear.Out + (Rank,)
        else:
            u_axes = (linear.Out, Rank)

        if isinstance(linear.In, tuple):
            v_axes = linear.In + (Rank,)
        else:
            v_axes = (linear.In, Rank)

        U = hax.NamedArray(U_arr, axes=u_axes)
        V = hax.NamedArray(Vh_arr.T, axes=v_axes)  # V = Vh.T

        return SVDLinear(
            U=U,
            S_base=S_base,
            V=V,
            In=linear.In,
            Out=linear.Out,
            Rank=Rank,
            use_bias=linear.bias is not None,
            bias=linear.bias,
        )
    
    def get_effective_weight(self, s_multiplier: NamedArray) -> NamedArray:
        """Reconstruct weight matrix with modulated singular values."""
        assert self.Rank in s_multiplier.axes, f"Rank axis {self.Rank} tidak ditemukan di s_multiplier {s_multiplier.axes}"
        batch_axes = tuple(ax for ax in s_multiplier.axes if ax != self.Rank)

        S_effective = self.S_base * s_multiplier

        U_broadcasted = self.U.broadcast_axis(batch_axes)

        S_effective_broadcasted = S_effective.broadcast_to(U_broadcasted.axes)

        U_scaled = U_broadcasted * S_effective_broadcasted

        V_broadcasted = self.V.broadcast_axis(batch_axes)
        W = hax.dot(self.Rank, U_scaled, V_broadcasted)

        return W
    
    def __call__(self, x: NamedArray, s_multiplier: NamedArray, *, key: Optional[jax.random.PRNGKey] = None) -> NamedArray:
        W = self.get_effective_weight(s_multiplier)
        y = hax.dot(self.In, x, W)
        
        if self.use_bias and self.bias is not None:
            y = y + self.bias
            
        return y


class AdaptiveMLP(eqx.Module):
    """MLP with SVD decomposition and adaptive weights."""
    
    c_fc: SVDLinear
    c_proj: SVDLinear
    act: callable = eqx.field(static=True)
    
    @staticmethod
    def from_mlp(mlp, rank_ratio: float = 0.5, *, key: jax.random.PRNGKey) -> "AdaptiveMLP":
        """Create AdaptiveMLP from existing MLP."""
        k1, k2 = jrandom.split(key)
        
        c_fc = SVDLinear.from_linear(mlp.c_fc, rank_ratio, key=k1)
        c_proj = SVDLinear.from_linear(mlp.c_proj, rank_ratio, key=k2)
        
        return AdaptiveMLP(c_fc=c_fc, c_proj=c_proj, act=mlp.act)

    def __call__(self, x: NamedArray, multipliers: Dict[str, NamedArray], *, key: Optional[jax.random.PRNGKey] = None) -> NamedArray:
        k1, k2 = maybe_rng_split(key, 2)
        x = self.c_fc(x, s_multiplier=multipliers["c_fc"], key=k1)
        x = self.act(x)
        x = self.c_proj(x, s_multiplier=multipliers["c_proj"], key=k2)
        return x


class SVDPolicy(eqx.Module):
    """
    Simple policy for learning singular value multipliers.
    This is a basic version that learns direct multipliers.
    """
    
    num_layers: int = eqx.field(static=True)
    rank_per_layer: Dict[str, int] = eqx.field(static=True)
    
    # Learnable parameters
    multipliers: Dict[str, NamedArray]
    
    @staticmethod
    def init(
        num_layers: int,
        rank_per_layer: Dict[str, int],
        init_scale: float = 0.1,
        *,
        key: jax.random.PRNGKey,
    ) -> "SVDPolicy":
        """Initialize policy with small random values around 1."""
        multipliers = {}
        
        for layer_idx in range(num_layers):
            for proj in ["c_fc", "c_proj"]:
                param_key = f"layer_{layer_idx}_{proj}"
                if param_key in rank_per_layer:
                    rank = rank_per_layer[param_key]
                    k = jrandom.fold_in(key, hash(param_key))
                    # Initialize close to 1
                    mult = hax.ones((hax.Axis("rank", rank),)) + \
                           hax.random.normal(k, (hax.Axis("rank", rank),)) * init_scale
                    multipliers[param_key] = mult
        
        return SVDPolicy(
            num_layers=num_layers,
            rank_per_layer=rank_per_layer,
            multipliers=multipliers,
        )
    
    def get_multipliers_for_layer(self, layer_idx: int) -> Dict[str, NamedArray]:
        """Get multipliers for a specific layer."""
        return {
            "c_fc": self.multipliers.get(f"layer_{layer_idx}_c_fc"),
            "c_proj": self.multipliers.get(f"layer_{layer_idx}_c_proj"),
        }
    
    def apply_regularization(self, reg_strength: float = 0.01) -> jax.Array:
        """L2 regularization to keep multipliers close to 1."""
        reg_loss = 0.0
        for mult in self.multipliers.values():
            reg_loss = reg_loss + hax.sum((mult - 1.0) ** 2).scalar()
        return reg_loss * reg_strength


class _PolicyNet(eqx.Module):
    layer1: hnn.Linear
    layer2: hnn.Linear
    act: Callable = eqx.field(static=True)

    def __call__(self, x: NamedArray) -> NamedArray:
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x


class DynamicSVDPolicy(eqx.Module):
    """
    A dynamic policy that generates SVD multipliers from a task vector.
    """

    policy_net: _PolicyNet
    
    rank_shapes: Dict[str, Tuple[Axis]] = eqx.field(static=True)
    param_names: List[str] = eqx.field(static=True)
    
    @staticmethod
    def init(
        num_layers: int,
        rank_per_layer: dict[str, int],
        task_vector_dim: Axis,
        hidden_dim_ratio: int = 4,
        *,
        key: jax.random.PRNGKey,
    ) -> "DynamicSVDPolicy":
        """Initializes the dynamic policy network."""
        
        total_multipliers = sum(rank_per_layer.values())
        MlpOutput = hax.Axis("MlpOutput", total_multipliers)

        Hidden = hax.Axis("PolicyHidden", task_vector_dim.size * hidden_dim_ratio)

        k_l1, k_l2 = jrandom.split(key)

        l1 = hnn.Linear.init(In=task_vector_dim, Out=Hidden, key=k_l1, use_bias=True)
        l2 = hnn.Linear.init(In=Hidden, Out=MlpOutput, key=k_l2, use_bias=True)
        activation = hnn.relu

        policy_net = _PolicyNet(layer1=l1, layer2=l2, act=activation)

        param_names = sorted(rank_per_layer.keys())
        rank_shapes = {name: (hax.Axis("rank", rank_per_layer[name]),) for name in param_names}

        return DynamicSVDPolicy(policy_net, rank_shapes, param_names)
        
    def __call__(self, task_vector: NamedArray) -> Dict[str, NamedArray]:
        import math

        flat_multipliers = self.policy_net(task_vector)
        
        output_dict = {}
        current_idx = 0
        
        Batch = task_vector.axes[0]
        
        for name in self.param_names:
            shape_info = self.rank_shapes[name]
            num_elements = math.prod(ax.size for ax in shape_info)
            
            chunk = flat_multipliers.array[..., current_idx:current_idx + num_elements]

            batch_shape = chunk.shape[:-1]
            target_shape = batch_shape + tuple(ax.size for ax in shape_info)
            chunk_reshaped = chunk.reshape(target_shape)

            new_axes = (Batch,) + shape_info

            center = 1.0
            span = 0.5
            output_dict[name] = center + span * hax.tanh(hax.named(chunk_reshaped, new_axes))
            
            current_idx += num_elements
        
        return output_dict
        
    def get_policy_params(self):
        return {"policy_net": self.policy_net}