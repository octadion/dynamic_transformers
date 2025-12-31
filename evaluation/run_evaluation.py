#!/usr/bin/env python3
import argparse
import yaml
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import json

import asyncio
import jax
import levanter.tensorstore_serialization as ts_ser
import tensorstore as ts
from typing import Any, Dict

async def _patched_load_array_from_tensorstore(spec: Dict[str, Any]) -> jax.Array:
    context = ts.Context()
    t = await ts.open(ts.Spec(spec), context=context)
    
    return await t.read(order='C')

ts_ser.load_array_from_tensorstore = _patched_load_array_from_tensorstore

print("INFO: TensorStore monkey patch v2 applied successfully.")
sys.path.append(str(Path(__file__).parent.parent))

from evaluate_neural_ode import (
    EvaluationConfig, ModelCheckpointConfig, EvalDatasetConfig,
    main as run_evaluation
)


def load_yaml_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_config_from_yaml(yaml_config: Dict) -> "EvaluationConfig":
    from evaluate_neural_ode import EvaluationConfig, ModelCheckpointConfig, EvalDatasetConfig
    model_checkpoints = []
    for model_cfg in yaml_config.get('model_checkpoints', []):
        checkpoint = ModelCheckpointConfig(
            name=model_cfg.get('name'),
            config_path=model_cfg.get('config_path'),
            checkpoint_path=model_cfg.get('checkpoint_path'), 
            vanilla_checkpoint_path=model_cfg.get('vanilla_checkpoint_path'), 
            lora_checkpoint_path=model_cfg.get('lora_checkpoint_path')
        )
        model_checkpoints.append(checkpoint)
    datasets = []
    for dataset_cfg in yaml_config.get('datasets', []):
        dataset = EvalDatasetConfig(
            name=dataset_cfg.get('name'),
            dataset_name=dataset_cfg.get('dataset_name'),
            split=dataset_cfg.get('split', 'validation'),
            dataset_config=dataset_cfg.get('dataset_config'),
            num_samples=dataset_cfg.get('num_samples')
        )
        datasets.append(dataset)
    eval_settings = yaml_config.get('evaluation', {})
    return EvaluationConfig(
        model_checkpoints=model_checkpoints, datasets=datasets,
        batch_size=eval_settings.get('batch_size', 4),
        max_seq_length=eval_settings.get('max_seq_length', 1024),
        few_shot_k=eval_settings.get('few_shot_k', 0),
        seed=eval_settings.get('seed', 42),
        output_dir=eval_settings.get('output_dir', './evaluation_results')
    )


def validate_config(config: EvaluationConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    issues = []
    
    for model in config.model_checkpoints:
        is_lora = model.lora_checkpoint_path is not None

        if is_lora:
            if not model.vanilla_checkpoint_path or not Path(model.vanilla_checkpoint_path).exists():
                issues.append(f"WARNING: Vanilla Checkpoint not found for LoRA model '{model.name}': {model.vanilla_checkpoint_path}")
            if not Path(model.lora_checkpoint_path).exists():
                issues.append(f"WARNING: LoRA Checkpoint not found for LoRA model '{model.name}': {model.lora_checkpoint_path}")
        else:
            if not model.checkpoint_path or not Path(model.checkpoint_path).exists():
                issues.append(f"WARNING: Pretrained Checkpoint not found for model '{model.name}': {model.checkpoint_path}")
    
    try:
        os.makedirs(config.output_dir, exist_ok=True)
    except Exception as e:
        issues.append(f"ERROR: Cannot create output directory: {e}")
    
    if config.batch_size <= 0:
        issues.append("ERROR: Batch size must be positive")
    
    if config.few_shot_k < 0:
        issues.append("ERROR: few_shot_k must be non-negative")
    
    return issues


def setup_logging(verbose: bool = True):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.root.setLevel(level)
    logging.root.addHandler(console_handler)
    
    # Suppress some verbose libraries
    logging.getLogger('datasets').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate Neural ODE Transformer models on multiple choice benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run with default configuration
            python run_evaluation.py
            
            # Run with custom YAML config
            python run_evaluation.py --config my_config.yaml
            
            # Run only specific models
            python run_evaluation.py --models "GPT2 Baseline" "GPT2 SVD Neural ODE"
            
            # Run only specific datasets
            python run_evaluation.py --datasets PIQA ARC-Easy
            
            # Run few-shot evaluation
            python run_evaluation.py --few-shot 5
            
            # Quick test run with limited samples
            python run_evaluation.py --test-run --num-samples 100
        """
    )
    
    # Configuration options
    parser.add_argument('--config', type=str, default='eval_config.yaml',
                        help='Path to YAML configuration file')
    
    # Model selection
    parser.add_argument('--models', nargs='+', type=str,
                        help='Specific model names to evaluate')
    parser.add_argument('--skip-models', nargs='+', type=str,
                        help='Model names to skip')
    
    # Dataset selection  
    parser.add_argument('--datasets', nargs='+', type=str,
                        help='Specific dataset names to evaluate')
    parser.add_argument('--skip-datasets', nargs='+', type=str,
                        help='Dataset names to skip')
    
    # Evaluation settings
    parser.add_argument('--batch-size', type=int,
                        help='Batch size for evaluation')
    parser.add_argument('--max-seq-length', type=int,
                        help='Maximum sequence length')
    parser.add_argument('--few-shot', type=int, dest='few_shot_k',
                        help='Number of few-shot examples (0 for zero-shot)')
    parser.add_argument('--seed', type=int,
                        help='Random seed')
    
    # Output settings
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save individual predictions')
    
    # Memory management
    parser.add_argument('--no-clear-cache', action='store_true',
                        help='Do not clear cache between models')
    
    # Test/debug options
    parser.add_argument('--test-run', action='store_true',
                        help='Quick test run with limited data')
    parser.add_argument('--num-samples', type=int,
                        help='Limit number of samples per dataset')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configuration without running')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if Path(args.config).exists():
        logger.info(f"Loading configuration from {args.config}")
        yaml_config = load_yaml_config(args.config)
        config = create_config_from_yaml(yaml_config)
    else:
        logger.warning(f"Config file {args.config} not found, using defaults")
        from evaluate_neural_ode import create_default_eval_config
        config = create_default_eval_config()
    
    # Apply command-line overrides
    if args.models:
        config.model_checkpoints = [m for m in config.model_checkpoints 
                                  if m.name in args.models]
    
    if args.skip_models:
        config.model_checkpoints = [m for m in config.model_checkpoints 
                                  if m.name not in args.skip_models]
    
    if args.datasets:
        config.datasets = [d for d in config.datasets 
                          if d.name in args.datasets]
    
    if args.skip_datasets:
        config.datasets = [d for d in config.datasets 
                          if d.name not in args.skip_datasets]
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    if args.max_seq_length is not None:
        config.max_seq_length = args.max_seq_length
    
    if args.few_shot_k is not None:
        config.few_shot_k = args.few_shot_k
    
    if args.seed is not None:
        config.seed = args.seed
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    
    if args.save_predictions:
        config.save_individual_predictions = True
    
    if args.no_clear_cache:
        config.clear_cache_between_models = False
    
    # Apply test run settings
    if args.test_run:
        logger.info("Running in test mode with limited data")
        # config.model_checkpoints = config.model_checkpoints[:2]  # Only first 2 models
        # config.datasets = config.datasets[:2]  # Only first 2 datasets
        for dataset in config.datasets:
            dataset.num_samples = 1000 # Limit samples
    
    if args.num_samples is not None:
        for dataset in config.datasets:
            dataset.num_samples = args.num_samples
    
    # Validate configuration
    issues = validate_config(config)
    if issues:
        logger.warning("Configuration issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        
        # Exit if there are errors
        if any("ERROR" in issue for issue in issues):
            logger.error("Cannot proceed due to configuration errors")
            sys.exit(1)
    
    # Print configuration summary
    logger.info("="*80)
    logger.info("EVALUATION CONFIGURATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Models to evaluate: {len(config.model_checkpoints)}")
    for model in config.model_checkpoints:
        logger.info(f"  - {model.name}")
    logger.info(f"Datasets to evaluate: {len(config.datasets)}")
    for dataset in config.datasets:
        samples_str = f" ({dataset.num_samples} samples)" if dataset.num_samples else ""
        logger.info(f"  - {dataset.name}{samples_str}")
    logger.info(f"Evaluation mode: {'Few-shot' if config.few_shot_k > 0 else 'Zero-shot'}")
    if config.few_shot_k > 0:
        logger.info(f"Few-shot examples: {config.few_shot_k}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("="*80)
    
    # Dry run - just print config and exit
    if args.dry_run:
        logger.info("Dry run mode - configuration printed above")
        
        # Save config for reference
        config_output = Path(config.output_dir) / "evaluation_config.json"
        os.makedirs(config.output_dir, exist_ok=True)
        
        config_dict = {
            "models": [{"name": m.name, "type": m.model_type, "path": m.checkpoint_path} 
                      for m in config.model_checkpoints],
            "datasets": [{"name": d.name, "dataset": d.dataset_name, "samples": d.num_samples} 
                        for d in config.datasets],
            "settings": {
                "batch_size": config.batch_size,
                "max_seq_length": config.max_seq_length,
                "few_shot_k": config.few_shot_k,
                "seed": config.seed
            }
        }
        
        with open(config_output, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {config_output}")
        sys.exit(0)
    
    # Run evaluation
    try:
        logger.info("Starting evaluation...")
        run_evaluation(config)
        logger.info("Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
