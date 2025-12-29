#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from evaluate_neural_ode import (
    EvaluationConfig, ModelCheckpointConfig, EvalDatasetConfig,
    MultipleChoiceEvaluator, ModelLoader, main
)


def example_basic_evaluation():
    """Example 1: Basic zero-shot evaluation."""
    print("Example 1: Basic Zero-Shot Evaluation")
    print("-" * 50)
    
    # Create configuration
    config = EvaluationConfig(
        model_checkpoints=[
            ModelCheckpointConfig(
                name="GPT2 Baseline",
                model_type="gpt2",
                checkpoint_path="./checkpoints/gpt2_baseline/checkpoint_5000"
            ),
            ModelCheckpointConfig(
                name="GPT2 Neural ODE",
                model_type="neuralode",
                checkpoint_path="./checkpoints/gpt2_neuralode/checkpoint_5000",
                time_embed_dim=100,
                sinusodial_dim=16
            ),
        ],
        datasets=[
            EvalDatasetConfig(
                name="PIQA",
                dataset_name="piqa",
                split="validation",
                question_key="goal",
                choices_key="choices",
                answer_key="label",
                num_samples=100  # Limit for example
            ),
        ],
        batch_size=8,
        max_seq_length=512,
        few_shot_k=0,  # Zero-shot
        output_dir="./example_results/zero_shot"
    )
    
    # Run evaluation
    main(config)
    print("\nResults saved to ./example_results/zero_shot")


def example_few_shot_evaluation():
    """Example 2: Few-shot evaluation with custom settings."""
    print("\nExample 2: Few-Shot Evaluation")
    print("-" * 50)
    
    config = EvaluationConfig(
        model_checkpoints=[
            ModelCheckpointConfig(
                name="GPT2 SVD Neural ODE",
                model_type="neuralode-svd",
                checkpoint_path="./checkpoints/gpt2_svd_neuralode/checkpoint_5000",
                time_embed_dim=100,
                sinusodial_dim=16,
                rank_ratio=0.5
            ),
        ],
        datasets=[
            EvalDatasetConfig(
                name="ARC-Easy",
                dataset_name="ai2_arc",
                dataset_config="ARC-Easy",
                split="test",
                question_key="question",
                choices_key="choices.text",
                answer_key="answerKey",
                num_samples=200
            ),
        ],
        batch_size=4,  # Smaller batch for memory
        max_seq_length=256,  # Shorter sequences
        few_shot_k=5,  # 5-shot
        output_dir="./example_results/few_shot",
        save_individual_predictions=True  # Save predictions
    )
    
    main(config)
    print("\nResults saved to ./example_results/few_shot")


def example_custom_evaluation():
    """Example 3: Custom evaluation with specific model handling."""
    print("\nExample 3: Custom Evaluation")
    print("-" * 50)
    
    from transformers import AutoTokenizer
    import jax.random as jrandom
    import haliax as hax
    from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
    
    # Initialize components manually
    config = EvaluationConfig(
        batch_size=8,
        max_seq_length=512,
        output_dir="./example_results/custom"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create model manually
    model_config = Gpt2Config()
    Vocab = hax.Axis("vocab", len(tokenizer))
    model = Gpt2LMHeadModel.init(Vocab, config=model_config, key=jrandom.PRNGKey(0))
    
    # Create evaluator
    evaluator = MultipleChoiceEvaluator(config, tokenizer)
    
    # Define custom dataset
    dataset_config = EvalDatasetConfig(
        name="PIQA-Small",
        dataset_name="piqa",
        split="validation",
        question_key="goal",
        choices_key="choices", 
        answer_key="label",
        num_samples=50
    )
    
    # Evaluate
    results = evaluator.evaluate_dataset(model, dataset_config)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct']}/{results['total']}")


def example_comparative_analysis():
    """Example 4: Comparative analysis across models."""
    print("\nExample 4: Comparative Analysis")
    print("-" * 50)
    
    import pandas as pd
    
    # Configure all models for comparison
    all_models = [
        ("GPT2 Baseline", "gpt2", "./checkpoints/gpt2_baseline/checkpoint_5000"),
        ("GPT2 Neural ODE", "neuralode", "./checkpoints/gpt2_neuralode/checkpoint_5000"),
        ("GPT2 SVD Neural ODE", "neuralode-svd", "./checkpoints/gpt2_svd_neuralode/checkpoint_5000"),
    ]
    
    model_checkpoints = []
    for name, model_type, path in all_models:
        checkpoint = ModelCheckpointConfig(
            name=name,
            model_type=model_type,
            checkpoint_path=path,
            time_embed_dim=100,
            sinusodial_dim=16,
            rank_ratio=0.5
        )
        model_checkpoints.append(checkpoint)
    
    # Configure multiple datasets
    datasets = [
        EvalDatasetConfig(
            name="PIQA",
            dataset_name="piqa",
            split="validation",
            question_key="goal",
            choices_key="choices",
            answer_key="label",
            num_samples=500
        ),
        EvalDatasetConfig(
            name="SciQ",
            dataset_name="sciq",
            split="test",
            question_key="question",
            choices_key="choices",
            answer_key="correct_answer",
            context_key="support",
            num_samples=500
        ),
    ]
    
    config = EvaluationConfig(
        model_checkpoints=model_checkpoints,
        datasets=datasets,
        batch_size=8,
        output_dir="./example_results/comparative"
    )
    
    # Run evaluation
    main(config)
    
    # Load and analyze results
    import json
    with open("./example_results/comparative/evaluation_results.json", 'r') as f:
        results = json.load(f)
    
    # Create comparison dataframe
    data = []
    for model_name, model_results in results.items():
        for dataset_name, dataset_results in model_results.items():
            data.append({
                "Model": model_name,
                "Dataset": dataset_name,
                "Accuracy": dataset_results["accuracy"]
            })
    
    df = pd.DataFrame(data)
    pivot = df.pivot(index="Model", columns="Dataset", values="Accuracy")
    
    print("\nAccuracy Comparison:")
    print(pivot.round(4))
    
    # Calculate improvements
    baseline_avg = pivot.loc["GPT2 Baseline"].mean()
    for model in pivot.index:
        if model != "GPT2 Baseline":
            model_avg = pivot.loc[model].mean()
            improvement = (model_avg - baseline_avg) * 100
            print(f"\n{model} improvement over baseline: {improvement:+.2f}%")


def example_memory_efficient_evaluation():
    """Example 5: Memory-efficient evaluation for large models."""
    print("\nExample 5: Memory-Efficient Evaluation")
    print("-" * 50)
    
    import gc
    import jax
    
    # Models to evaluate one at a time
    models = [
        ModelCheckpointConfig(
            name="Llama Baseline",
            model_type="llama",
            checkpoint_path="./checkpoints/llama_baseline/checkpoint_5000"
        ),
        ModelCheckpointConfig(
            name="Llama SVD Neural ODE",
            model_type="llamaode-svd",
            checkpoint_path="./checkpoints/llama_svd_neuralode/checkpoint_5000",
            time_embed_dim=100,
            sinusodial_dim=16,
            rank_ratio=0.5
        ),
    ]
    
    # Process each model separately
    all_results = {}
    
    for model_config in models:
        print(f"\nEvaluating {model_config.name}...")
        
        # Create config with single model
        config = EvaluationConfig(
            model_checkpoints=[model_config],
            datasets=[
                EvalDatasetConfig(
                    name="PIQA",
                    dataset_name="piqa",
                    split="validation",
                    num_samples=1000
                ),
            ],
            batch_size=4,  # Small batch for large models
            clear_cache_between_models=True,
            output_dir=f"./example_results/memory_efficient/{model_config.name}"
        )
        
        # Run evaluation
        main(config)
        
        # Explicitly clear memory
        gc.collect()
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
        
        print(f"Completed {model_config.name}, memory cleared")
    
    print("\nAll evaluations completed!")


def example_custom_dataset():
    """Example 6: Adding a custom dataset."""
    print("\nExample 6: Custom Dataset Evaluation")
    print("-" * 50)
    
    # Custom dataset configuration
    custom_dataset = EvalDatasetConfig(
        name="CustomMC",
        dataset_name="your_custom_dataset",  # Must be available via HuggingFace
        split="test",
        question_key="query",  # Adjust to your dataset's schema
        choices_key="options",
        answer_key="correct_option",
        context_key="background",  # Optional
        num_samples=None  # Use all samples
    )
    
    config = EvaluationConfig(
        model_checkpoints=[
            ModelCheckpointConfig(
                name="GPT2 Baseline",
                model_type="gpt2",
                checkpoint_path="./checkpoints/gpt2_baseline/checkpoint_5000"
            ),
        ],
        datasets=[custom_dataset],
        output_dir="./example_results/custom_dataset"
    )
    
    # Note: This will fail unless 'your_custom_dataset' exists
    # main(config)
    print("Custom dataset configuration created (not executed)")


if __name__ == "__main__":
    # Choose which examples to run
    examples_to_run = [1, 3, 4]  # Modify this list to select examples
    
    print("Neural ODE Transformer Evaluation Examples")
    print("=" * 50)
    
    # Create example directories
    os.makedirs("./example_results", exist_ok=True)
    
    # Run selected examples
    if 1 in examples_to_run:
        example_basic_evaluation()
    
    if 2 in examples_to_run:
        example_few_shot_evaluation()
    
    if 3 in examples_to_run:
        example_custom_evaluation()
    
    if 4 in examples_to_run:
        example_comparative_analysis()
    
    if 5 in examples_to_run:
        example_memory_efficient_evaluation()
    
    if 6 in examples_to_run:
        example_custom_dataset()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("Check ./example_results/ for outputs")
