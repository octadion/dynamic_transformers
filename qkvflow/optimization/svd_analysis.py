"""
Analysis tools for SVD Neural ODE models
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import jax.numpy as jnp
import jax.random as jrandom
import haliax as hax
import logging

logger = logging.getLogger(__name__)


def analyze_expert_specialization(model, test_examples, save_path="expert_analysis.png"):
    """Analyze how different experts specialize"""
    
    expert_usage_by_example = []
    
    logger.info(f"Analyzing expert specialization for {len(test_examples)} examples")
    
    for i, example in enumerate(test_examples):
        try:
            usage = model.analyze_expert_usage(example)
            expert_usage_by_example.append(usage)
        except Exception as e:
            logger.warning(f"Failed to analyze example {i}: {e}")
            continue
    
    if not expert_usage_by_example:
        logger.error("No examples could be analyzed")
        return None
    
    # Aggregate expert usage statistics
    all_attention_usage = np.stack([u["attention_experts"] for u in expert_usage_by_example])
    all_mlp_usage = np.stack([u["mlp_experts"] for u in expert_usage_by_example])
    
    # Plot expert usage patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Attention expert usage across layers
    sns.heatmap(
        all_attention_usage.mean(axis=0),  # Average across examples
        ax=axes[0, 0],
        cmap="viridis",
        cbar_kws={"label": "Usage Probability"}
    )
    axes[0, 0].set_title("Attention Expert Usage Across Layers")
    axes[0, 0].set_xlabel("Expert ID")
    axes[0, 0].set_ylabel("Layer")
    
    # MLP expert usage across layers  
    sns.heatmap(
        all_mlp_usage.mean(axis=0),
        ax=axes[0, 1],
        cmap="viridis", 
        cbar_kws={"label": "Usage Probability"}
    )
    axes[0, 1].set_title("MLP Expert Usage Across Layers")
    axes[0, 1].set_xlabel("Expert ID")
    axes[0, 1].set_ylabel("Layer")
    
    # Expert diversity over layers
    attn_diversity = -np.sum(all_attention_usage.mean(0) * np.log(all_attention_usage.mean(0) + 1e-8), axis=1)
    mlp_diversity = -np.sum(all_mlp_usage.mean(0) * np.log(all_mlp_usage.mean(0) + 1e-8), axis=1)
    
    axes[1, 0].plot(attn_diversity, label="Attention", marker="o")
    axes[1, 0].plot(mlp_diversity, label="MLP", marker="s")
    axes[1, 0].set_title("Expert Diversity Across Layers")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Entropy (nats)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Example-wise expert preference
    example_preferences = []
    for i, usage in enumerate(expert_usage_by_example):
        attn_pref = usage["attention_experts"].mean(axis=0)  # Average across layers
        mlp_pref = usage["mlp_experts"].mean(axis=0)
        example_preferences.append(np.concatenate([attn_pref, mlp_pref]))
    
    example_preferences = np.stack(example_preferences)
    sns.heatmap(
        example_preferences.T,
        ax=axes[1, 1],
        cmap="RdBu_r",
        center=0.25,  # Expected uniform probability for 4 experts
        cbar_kws={"label": "Preference"}
    )
    axes[1, 1].set_title("Expert Preferences by Example")
    axes[1, 1].set_xlabel("Example ID")
    axes[1, 1].set_ylabel("Expert (Attn: 0-3, MLP: 4-7)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        "attention_usage": all_attention_usage,
        "mlp_usage": all_mlp_usage,
        "attention_diversity": attn_diversity,
        "mlp_diversity": mlp_diversity,
        "example_preferences": example_preferences,
    }


def analyze_parameter_evolution(model, time_steps=None, layer_idx=0):
    """Analyze how parameters evolve over time in Neural ODE"""
    
    if time_steps is None:
        time_steps = np.linspace(0, 1, 21)  # 21 time points
    
    parameter_norms = {
        "attention_qkv": [],
        "mlp_fc": [],
        "mlp_proj": [],
    }
    
    logger.info(f"Analyzing parameter evolution for layer {layer_idx}")
    
    for t in time_steps:
        try:
            # Get effective parameters at time t
            effective_params = model.get_effective_parameters(layer_idx)
            
            for param_name, param_tensor in effective_params.items():
                if hasattr(param_tensor, 'array'):
                    norm = jnp.linalg.norm(param_tensor.array)
                else:
                    norm = jnp.linalg.norm(param_tensor)
                parameter_norms[param_name].append(float(norm))
        except Exception as e:
            logger.warning(f"Failed to get parameters at time {t}: {e}")
            for param_name in parameter_norms:
                parameter_norms[param_name].append(0.0)
    
    # Plot parameter evolution
    plt.figure(figsize=(12, 6))
    
    for param_name, norms in parameter_norms.items():
        plt.plot(time_steps, norms, marker="o", label=param_name, linewidth=2)
    
    plt.xlabel("Layer Index (Time)")
    plt.ylabel("Parameter Norm")
    plt.title(f"Parameter Evolution Across Layers (Starting from Layer {layer_idx})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return parameter_norms


def compare_models(standard_model, svd_ode_model, test_examples):
    """Compare standard Neural ODE with SVD version"""
    
    results = {
        "standard": {"losses": [], "perplexities": []},
        "svd_ode": {"losses": [], "perplexities": []},
    }
    
    logger.info(f"Comparing models on {len(test_examples)} examples")
    
    for i, example in enumerate(test_examples):
        try:
            # Standard model
            std_loss = standard_model.compute_loss(example)
            if hasattr(std_loss, 'item'):
                std_loss_val = std_loss.item()
            else:
                std_loss_val = float(std_loss)
            std_ppl = np.exp(std_loss_val)
            results["standard"]["losses"].append(std_loss_val)
            results["standard"]["perplexities"].append(std_ppl)
            
            # SVD ODE model
            svd_loss = svd_ode_model.compute_loss(example)
            if hasattr(svd_loss, 'item'):
                svd_loss_val = svd_loss.item()
            else:
                svd_loss_val = float(svd_loss)
            svd_ppl = np.exp(svd_loss_val)
            results["svd_ode"]["losses"].append(svd_loss_val)
            results["svd_ode"]["perplexities"].append(svd_ppl)
            
        except Exception as e:
            logger.warning(f"Failed to compare models on example {i}: {e}")
            continue
    
    if not results["standard"]["losses"]:
        logger.error("No examples could be compared")
        return None
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss comparison
    axes[0].scatter(results["standard"]["losses"], results["svd_ode"]["losses"], alpha=0.7)
    min_loss = min(min(results["standard"]["losses"]), min(results["svd_ode"]["losses"]))
    max_loss = max(max(results["standard"]["losses"]), max(results["svd_ode"]["losses"]))
    axes[0].plot([min_loss, max_loss], [min_loss, max_loss], 'r--', alpha=0.5)
    axes[0].set_xlabel("Standard Neural ODE Loss")
    axes[0].set_ylabel("SVD Neural ODE Loss")
    axes[0].set_title("Loss Comparison")
    axes[0].grid(True, alpha=0.3)
    
    # Perplexity comparison
    axes[1].scatter(results["standard"]["perplexities"], results["svd_ode"]["perplexities"], alpha=0.7)
    min_ppl = min(min(results["standard"]["perplexities"]), min(results["svd_ode"]["perplexities"]))
    max_ppl = max(max(results["standard"]["perplexities"]), max(results["svd_ode"]["perplexities"]))
    axes[1].plot([min_ppl, max_ppl], [min_ppl, max_ppl], 'r--', alpha=0.5)
    axes[1].set_xlabel("Standard Neural ODE Perplexity")
    axes[1].set_ylabel("SVD Neural ODE Perplexity")
    axes[1].set_title("Perplexity Comparison")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def analyze_svd_components(model, layer_idx=0, save_path="svd_components.png"):
    """Analyze SVD components (U, S, V) of the model"""
    
    try:
        # Get SVD components for a specific layer
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'block'):
            block = model.transformer.block
            
            # Analyze attention SVD components
            if hasattr(block, 'attn') and hasattr(block.attn, 'c_attn'):
                attn_layer = block.attn.c_attn
                if hasattr(attn_layer, 'U_base'):
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    
                    # Attention components
                    U_attn = np.array(attn_layer.U_base.array)
                    S_attn = np.array(attn_layer.S_base.array)
                    V_attn = np.array(attn_layer.V_base.array)
                    
                    # Plot U matrix (left singular vectors)
                    im1 = axes[0, 0].imshow(U_attn, cmap='RdBu_r', aspect='auto')
                    axes[0, 0].set_title("Attention U (Left Singular Vectors)")
                    axes[0, 0].set_xlabel("Rank")
                    axes[0, 0].set_ylabel("Input Dimension")
                    plt.colorbar(im1, ax=axes[0, 0])
                    
                    # Plot S values (singular values)
                    axes[0, 1].bar(range(len(S_attn)), S_attn)
                    axes[0, 1].set_title("Attention S (Singular Values)")
                    axes[0, 1].set_xlabel("Rank")
                    axes[0, 1].set_ylabel("Singular Value")
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Plot V matrix (right singular vectors)
                    im2 = axes[0, 2].imshow(V_attn, cmap='RdBu_r', aspect='auto')
                    axes[0, 2].set_title("Attention V (Right Singular Vectors)")
                    axes[0, 2].set_xlabel("Output Dimension")
                    axes[0, 2].set_ylabel("Rank")
                    plt.colorbar(im2, ax=axes[0, 2])
                    
            # Analyze MLP SVD components  
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'c_fc'):
                mlp_layer = block.mlp.c_fc
                if hasattr(mlp_layer, 'U_base'):
                    U_mlp = np.array(mlp_layer.U_base.array)
                    S_mlp = np.array(mlp_layer.S_base.array)
                    V_mlp = np.array(mlp_layer.V_base.array)
                    
                    # Plot U matrix
                    im3 = axes[1, 0].imshow(U_mlp, cmap='RdBu_r', aspect='auto')
                    axes[1, 0].set_title("MLP U (Left Singular Vectors)")
                    axes[1, 0].set_xlabel("Rank")
                    axes[1, 0].set_ylabel("Input Dimension")
                    plt.colorbar(im3, ax=axes[1, 0])
                    
                    # Plot S values
                    axes[1, 1].bar(range(len(S_mlp)), S_mlp)
                    axes[1, 1].set_title("MLP S (Singular Values)")
                    axes[1, 1].set_xlabel("Rank")
                    axes[1, 1].set_ylabel("Singular Value")
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Plot V matrix
                    im4 = axes[1, 2].imshow(V_mlp, cmap='RdBu_r', aspect='auto')
                    axes[1, 2].set_title("MLP V (Right Singular Vectors)")
                    axes[1, 2].set_xlabel("Output Dimension")
                    axes[1, 2].set_ylabel("Rank")
                    plt.colorbar(im4, ax=axes[1, 2])
                    
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            return {
                "attention": {"U": U_attn, "S": S_attn, "V": V_attn},
                "mlp": {"U": U_mlp, "S": S_mlp, "V": V_mlp}
            }
            
    except Exception as e:
        logger.error(f"Failed to analyze SVD components: {e}")
        return None


def create_test_examples(model, tokenizer, num_examples=10, seq_len=64):
    """Create test examples for analysis"""
    
    examples = []
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    
    for i in range(num_examples):
        # Generate random tokens
        tokens = hax.random.randint(
            jrandom.PRNGKey(i),
            (seq_len,),
            minval=0,
            maxval=vocab_size
        )
        
        # Create example object
        example = type('Example', (), {
            'tokens': tokens,
            'attn_mask': None,
            'loss_mask': hax.ones_like(tokens, dtype=bool)
        })()
        
        examples.append(example)
    
    return examples


def run_full_analysis(model, tokenizer, save_dir="analysis_results"):
    """Run complete analysis suite"""
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("Starting full SVD Neural ODE analysis")
    
    # Create test examples
    test_examples = create_test_examples(model, tokenizer, num_examples=20, seq_len=64)
    
    results = {}
    
    try:
        # Expert specialization analysis
        logger.info("Analyzing expert specialization...")
        expert_results = analyze_expert_specialization(
            model, test_examples, 
            save_path=os.path.join(save_dir, "expert_analysis.png")
        )
        results["expert_analysis"] = expert_results
    except Exception as e:
        logger.error(f"Expert analysis failed: {e}")
    
    try:
        # Parameter evolution analysis
        logger.info("Analyzing parameter evolution...")
        param_results = analyze_parameter_evolution(model, layer_idx=0)
        results["parameter_evolution"] = param_results
    except Exception as e:
        logger.error(f"Parameter evolution analysis failed: {e}")
    
    try:
        # SVD components analysis
        logger.info("Analyzing SVD components...")
        svd_results = analyze_svd_components(
            model, layer_idx=0,
            save_path=os.path.join(save_dir, "svd_components.png")
        )
        results["svd_components"] = svd_results
    except Exception as e:
        logger.error(f"SVD components analysis failed: {e}")
    
    logger.info(f"Analysis complete. Results saved to {save_dir}")
    return results


if __name__ == "__main__":
    print("SVD Neural ODE Analysis Tools")
    print("Available functions:")
    print("1. analyze_expert_specialization(model, test_examples)")
    print("2. analyze_parameter_evolution(model)")
    print("3. compare_models(standard_model, svd_ode_model, test_examples)")
    print("4. analyze_svd_components(model)")
    print("5. run_full_analysis(model, tokenizer)")
    print("\nExample usage:")
    print("from qkvflow.optimization.svd_analysis import run_full_analysis")
    print("results = run_full_analysis(model, tokenizer)")