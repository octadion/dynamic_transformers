import jax
import jax.numpy as jnp
import haliax as hax
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

from qkvflow.nn.dynamic import NeuralOdeLMHeadModel
from qkvflow.nn.dynamic_svd import SVDNeuralOdeLMHeadModel


def convert_ode_to_svd(
    ode_model: NeuralOdeLMHeadModel,
    rank_ratio: float = 0.5,
    policy_init_scale: float = 0.1,
    *,
    key: jax.random.PRNGKey,
) -> SVDNeuralOdeLMHeadModel:
    """
    Convert a regular Neural ODE model to SVD-adaptive version.
    """
    # TODO: Implement actual conversion logic
    raise NotImplementedError("Model conversion not yet implemented")


def analyze_policy_multipliers(
    model: SVDNeuralOdeLMHeadModel,
    save_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Analyze the learned policy multipliers."""
    if not hasattr(model, 'get_policy_params'):
        raise ValueError("Model does not have policy parameters")
    
    policy_params = model.get_policy_params()
    analysis = {}
    
    for name, param in policy_params.items():
        param_array = np.array(param.array)
        analysis[name] = {
            'values': param_array,
            'mean': float(np.mean(param_array)),
            'std': float(np.std(param_array)),
            'min': float(np.min(param_array)),
            'max': float(np.max(param_array)),
            'num_above_one': int(np.sum(param_array > 1.0)),
            'num_below_one': int(np.sum(param_array < 1.0)),
        }
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path / 'policy_analysis.json', 'w') as f:
            json_data = {k: {kk: vv for kk, vv in v.items() if kk != 'values'} 
                        for k, v in analysis.items()}
            json.dump(json_data, f, indent=2)

        np.savez(save_path / 'policy_values.npz', 
                 **{k: v['values'] for k, v in analysis.items()})
    
    return analysis


def visualize_policy_evolution(
    checkpoints: Dict[int, SVDNeuralOdeLMHeadModel],
    save_path: Optional[Path] = None,
):
    """Visualize how policy multipliers evolve during training."""

    all_multipliers = {}
    steps = sorted(checkpoints.keys())
    
    for step in steps:
        model = checkpoints[step]
        if hasattr(model, 'get_policy_params'):
            policy_params = model.get_policy_params()
            for name, param in policy_params.items():
                if name not in all_multipliers:
                    all_multipliers[name] = []
                all_multipliers[name].append(np.array(param.array))

    fig, axes = plt.subplots(len(all_multipliers), 2, figsize=(12, 4*len(all_multipliers)))
    if len(all_multipliers) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, values_list) in enumerate(all_multipliers.items()):
        # Evolution of mean and std
        means = [np.mean(v) for v in values_list]
        stds = [np.std(v) for v in values_list]
        
        ax = axes[idx, 0]
        ax.errorbar(steps, means, yerr=stds, marker='o')
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Multiplier Value')
        ax.set_title(f'{name} - Mean ± Std')
        ax.grid(True, alpha=0.3)

        ax = axes[idx, 1]
        final_values = values_list[-1]
        ax.hist(final_values, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Multiplier Value')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} - Final Distribution')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / 'policy_evolution.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_effective_rank(model: SVDNeuralOdeLMHeadModel) -> Dict[str, float]:
    """
    Compute the effective rank of each adapted layer.
    
    Effective rank is computed as: exp(entropy of normalized squared singular values)
    This gives a smooth measure of how many singular values are "active".
    """
    if not hasattr(model.transformer, 'block'):
        raise ValueError("Model structure not as expected")
    
    effective_ranks = {}

    block = model.transformer.block
    if hasattr(block, 'mlp') and hasattr(block.mlp, 'adaptive_mlp'):
        for name in ['c_fc', 'c_proj']:
            svd_layer = getattr(block.mlp.adaptive_mlp, name)
            if hasattr(svd_layer, 'S_base') and hasattr(svd_layer, 's_multiplier'):
                s_eff = svd_layer.S_base.array * svd_layer.s_multiplier.array

                s_norm = s_eff ** 2
                s_norm = s_norm / jnp.sum(s_norm)

                entropy = -jnp.sum(s_norm * jnp.log(s_norm + 1e-10))

                eff_rank = jnp.exp(entropy)
                effective_ranks[name] = float(eff_rank)
    
    return effective_ranks


def save_policy_checkpoint(
    model: SVDNeuralOdeLMHeadModel,
    save_path: Path,
):
    """Save only the policy parameters for efficient storage."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(model, 'get_policy_params'):
        policy_params = model.get_policy_params()
        
        np_params = {k: np.array(v.array) for k, v in policy_params.items()}
        
        np.savez_compressed(save_path, **np_params)
        print(f"Saved policy parameters to {save_path}")
    else:
        raise ValueError("Model does not have policy parameters")


def load_policy_checkpoint(
    model: SVDNeuralOdeLMHeadModel,
    load_path: Path,
) -> SVDNeuralOdeLMHeadModel:
    """Load policy parameters from checkpoint."""
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {load_path}")

    loaded = np.load(load_path)
    
    policy_params = {}
    for k, v in loaded.items():
        axis = hax.Axis("rank", v.shape[0])
        policy_params[k] = hax.NamedArray(jnp.array(v), axes=(axis,))
    
    return model.set_policy_params(policy_params)


def compare_models(
    base_model: NeuralOdeLMHeadModel,
    svd_model: SVDNeuralOdeLMHeadModel,
    input_ids: hax.NamedArray,
    attn_mask=None,
) -> Dict[str, float]:
    """Compare outputs between base and SVD-adapted models."""

    base_output = base_model(input_ids, attn_mask)
    svd_output = svd_model(input_ids, attn_mask)

    abs_diff = jnp.abs(base_output.array - svd_output.array)
    rel_diff = abs_diff / (jnp.abs(base_output.array) + 1e-8)
    
    metrics = {
        'max_abs_diff': float(jnp.max(abs_diff)),
        'mean_abs_diff': float(jnp.mean(abs_diff)),
        'max_rel_diff': float(jnp.max(rel_diff)),
        'mean_rel_diff': float(jnp.mean(rel_diff)),
        'cosine_similarity': float(
            jnp.sum(base_output.array * svd_output.array) / 
            (jnp.linalg.norm(base_output.array) * jnp.linalg.norm(svd_output.array))
        ),
    }
    
    return metrics


if __name__ == "__main__":
    print("SVD Model Utilities")
    print("This module provides functions for:")
    print("- Converting Neural ODE models to SVD-adaptive versions")
    print("- Analyzing policy multipliers")
    print("- Visualizing policy evolution during training")
    print("- Computing effective rank of adapted layers")
    print("- Saving/loading policy checkpoints")
    print("- Comparing base and adapted models")