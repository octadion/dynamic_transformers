import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import haliax as hax

from qkvflow.nn.adaptive_transformer import AdaptiveNeuralOdeLMHeadModel


def analyze_expert_specialization(
    model: AdaptiveNeuralOdeLMHeadModel,
    test_examples: List[Dict],
    tokenizer,
    save_path: str = None
) -> Dict:
    """
    Analyze how experts specialize for different types of inputs.
    """
    
    results = {
        "expert_usage": [],
        "task_predictions": [],
        "attention_patterns": [],
        "difficulty_predictions": []
    }
    
    for i, example in enumerate(test_examples):
        try:
            # Tokenize input
            input_ids = tokenizer.encode(example["text"])
            if len(input_ids) > model.config.Pos.size:
                input_ids = input_ids[:model.config.Pos.size]
            
            while len(input_ids) < 4:
                input_ids.append(tokenizer.pad_token_id or 0)

            Batch = hax.Axis("batch", 1)
            Pos = model.config.Pos.resize(len(input_ids))
            input_tensor = hax.NamedArray(
                jnp.array(input_ids)[None, :], 
                axes=(Batch, Pos)
            )
  
            analysis = model.get_expert_analysis(input_tensor, key=jrandom.PRNGKey(42))
            
            expert_usage = {}
            task_predictions = {}
            
            for layer_idx in range(min(model.config.num_layers, 4)):
                # Attention expert weights
                attn_key = f"layer_{layer_idx}_attn_experts"
                if attn_key in analysis:
                    attn_experts = analysis[attn_key]["expert_weights"]
                    expert_usage[f"layer_{layer_idx}_attn"] = attn_experts.array.tolist()
                
                # MLP expert weights
                mlp_key = f"layer_{layer_idx}_mlp_experts"
                if mlp_key in analysis:
                    mlp_experts = analysis[mlp_key]["expert_weights"] 
                    expert_usage[f"layer_{layer_idx}_mlp"] = mlp_experts.array.tolist()
                
                # Task predictions
                task_key = f"layer_{layer_idx}_task_probs"
                if task_key in analysis:
                    task_probs = analysis[task_key]
                    task_predictions[f"layer_{layer_idx}"] = task_probs.array.tolist()
            
            results["expert_usage"].append({
                "example_id": i,
                "text": example["text"][:100] + "...", 
                "label": example.get("label", "unknown"),
                "expert_weights": expert_usage
            })
            
            results["task_predictions"].append({
                "example_id": i,
                "label": example.get("label", "unknown"), 
                "predictions": task_predictions
            })
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue

    if results["expert_usage"]:
        stats = compute_specialization_stats(results)
        results["statistics"] = stats
        
        if save_path:
            try:
                visualize_expert_analysis(results, save_path)
            except Exception as e:
                print(f"Visualization failed: {e}")
    
    return results


def compute_specialization_stats(results: Dict) -> Dict:
    """Compute statistics about expert specialization."""
    
    stats = {
        "expert_entropy": {}, 
        "task_consistency": {}, 
        "expert_task_correlation": {} 
    }
    
    task_groups = {}
    for item in results["expert_usage"]:
        label = item["label"]
        if label not in task_groups:
            task_groups[label] = []
        task_groups[label].append(item)
    
    for task_label, items in task_groups.items():
        expert_activations = []
        
        for item in items:
            for layer_key, weights in item["expert_weights"].items():
                if "attn" in layer_key and weights:
                    expert_activations.append(weights)
        
        if expert_activations:
            avg_activations = np.mean(expert_activations, axis=0)
            
            avg_activations = np.clip(avg_activations, 1e-8, 1.0)
            
            entropy = -np.sum(avg_activations * np.log(avg_activations))
            stats["expert_entropy"][task_label] = float(entropy)
    
    return stats


def visualize_expert_analysis(results: Dict, save_path: str):
    """Create visualizations of expert analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Expert usage heatmap
    expert_data = []
    labels = []
    
    for item in results["expert_usage"]:
        if "layer_0_attn" in item["expert_weights"] and item["expert_weights"]["layer_0_attn"]:
            first_layer_attn = item["expert_weights"]["layer_0_attn"]
            expert_data.append(first_layer_attn)
            labels.append(item["label"])
    
    if expert_data:
        expert_matrix = np.array(expert_data)
        im1 = axes[0, 0].imshow(expert_matrix, aspect='auto', cmap='viridis')
        axes[0, 0].set_title("Expert Usage by Example")
        axes[0, 0].set_xlabel("Expert Index")
        axes[0, 0].set_ylabel("Example Index")
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Expert specialization by task
        task_expert_means = {}
        unique_labels = list(set(labels))
        
        for label in unique_labels:
            task_indices = [i for i, l in enumerate(labels) if l == label]
            if task_indices:
                task_expert_means[label] = np.mean(expert_matrix[task_indices], axis=0)
        
        if task_expert_means:
            task_matrix = np.array(list(task_expert_means.values()))
            im2 = axes[0, 1].imshow(task_matrix, aspect='auto', cmap='coolwarm')
            axes[0, 1].set_title("Average Expert Usage by Task")
            axes[0, 1].set_xlabel("Expert Index")
            axes[0, 1].set_ylabel("Task Type")
            axes[0, 1].set_yticks(range(len(unique_labels)))
            axes[0, 1].set_yticklabels(unique_labels)
            plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Expert entropy distribution
    if "statistics" in results and "expert_entropy" in results["statistics"]:
        entropies = list(results["statistics"]["expert_entropy"].values())
        task_names = list(results["statistics"]["expert_entropy"].keys())
        
        if entropies and task_names:
            axes[1, 0].bar(task_names, entropies)
            axes[1, 0].set_title("Expert Specialization (Lower = More Specialized)")
            axes[1, 0].set_ylabel("Entropy")
            axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Layer-wise expert evolution
    if results["expert_usage"]:
        example_item = results["expert_usage"][0]
        layer_evolution = []
        layer_names = []
        
        for key, weights in example_item["expert_weights"].items():
            if "attn" in key and weights:
                layer_evolution.append(weights)
                layer_names.append(key.replace("_attn", ""))
        
        if layer_evolution:
            evolution_matrix = np.array(layer_evolution)
            im4 = axes[1, 1].imshow(evolution_matrix, aspect='auto', cmap='plasma')
            axes[1, 1].set_title("Expert Usage Across Layers (First Example)")
            axes[1, 1].set_xlabel("Expert Index")
            axes[1, 1].set_ylabel("Layer")
            axes[1, 1].set_yticks(range(len(layer_names)))
            axes[1, 1].set_yticklabels(layer_names)
            plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    from qkvflow.analysis.utils import get_model
    import levanter
    
    config_path = "config/owt_10k/adaptive_nano.yaml"
    checkpoint_path = "checkpoints/your_adaptive_checkpoint"
    
    try:
        trainer, model, eval_loader, tokenizer = levanter.config.main(
            get_model,
            args=[
                "--config_path", config_path,
                "--model_choice", "adaptive_neuralode", 
                "--trainer.load_checkpoint_path", checkpoint_path,
                "--trainer.wandb.mode", "disabled",
            ],
        )()
        
        test_examples = [
            {"text": "Calculate the derivative of x^2 + 3x - 2", "label": "math"},
            {"text": "def fibonacci(n): return n if n <= 1 else", "label": "code"},
            {"text": "The capital of France is", "label": "factual"},
            {"text": "Once upon a time in a distant galaxy", "label": "creative"},
        ]
        
        results = analyze_expert_specialization(
            model, test_examples, tokenizer, "expert_analysis.png"
        )
        
        print("Expert Analysis Complete!")
        print(f"Found {len(results['expert_usage'])} examples")
        print(f"Statistics: {results['statistics']}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        print("Please ensure model is trained and checkpoint exists")