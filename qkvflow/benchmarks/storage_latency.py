# qkvflow/benchmarks/storage_latency.py
"""
Storage and latency benchmarking for fair comparison.
"""
import logging
import time
from typing import Dict, List
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from pathlib import Path

from qkvflow.nn.dynamic_svd_llama import SVDLlamaOdeLMHeadModel

logger = logging.getLogger(__name__)


class StorageBenchmark:
    """Measure storage requirements for different methods."""
    
    @staticmethod
    def measure_checkpoint_size(checkpoint_path: str) -> float:
        """
        Measure size of checkpoint directory in MB.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Size in MB
        """
        total_size = 0
        
        if os.path.isfile(checkpoint_path):
            total_size = os.path.getsize(checkpoint_path)
        else:
            for dirpath, dirnames, filenames in os.walk(checkpoint_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        
        size_mb = total_size / (1024 * 1024)
        return size_mb
    
    @staticmethod
    def measure_model_params(model) -> Dict[str, int]:
        """
        Count parameters in model.
        
        Returns:
            Dict with 'total', 'trainable', 'frozen'
        """
        def count_arrays(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            return sum(np.prod(leaf.shape) for leaf in leaves if isinstance(leaf, (jnp.ndarray, np.ndarray)))
        
        total = count_arrays(model)
        
        # For PCSVM, count policy params
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'policy'):
            policy_params = count_arrays(model.transformer.policy)
        else:
            policy_params = 0
        
        return {
            "total": total,
            "trainable": policy_params,
            "frozen": total - policy_params,
        }
    
    @staticmethod
    def compare_storage(
        pcsvm_checkpoint: str,
        lora_checkpoints: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Compare storage requirements.
        
        Args:
            pcsvm_checkpoint: Path to PCSVM checkpoint
            lora_checkpoints: Dict mapping task name to LoRA checkpoint path
            
        Returns:
            Dict with storage metrics in MB
        """
        results = {}
        
        # PCSVM: single policy
        pcsvm_size = StorageBenchmark.measure_checkpoint_size(pcsvm_checkpoint)
        results["pcsvm_single"] = pcsvm_size
        
        # LoRA: sum of all adapters
        lora_total = 0
        for task_name, checkpoint_path in lora_checkpoints.items():
            task_size = StorageBenchmark.measure_checkpoint_size(checkpoint_path)
            results[f"lora_{task_name}"] = task_size
            lora_total += task_size
        
        results["lora_total"] = lora_total
        results["storage_ratio"] = lora_total / pcsvm_size if pcsvm_size > 0 else 0
        
        logger.info("Storage comparison:")
        logger.info(f"  PCSVM (single): {pcsvm_size:.2f} MB")
        logger.info(f"  LoRA (total): {lora_total:.2f} MB")
        logger.info(f"  Ratio (LoRA/PCSVM): {results['storage_ratio']:.2f}x")
        
        return results


class LatencyBenchmark:
    """Measure inference latency."""
    
    @staticmethod
    def benchmark_forward_pass(
        model,
        input_ids: jnp.ndarray,
        attn_mask: jnp.ndarray,
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark forward pass latency.
        
        Args:
            model: Model to benchmark
            input_ids: Input token IDs [batch, seq_len]
            attn_mask: Attention mask
            num_warmup: Warmup iterations
            num_iterations: Measurement iterations
            
        Returns:
            Dict with latency metrics in milliseconds
        """
        logger.info(f"Benchmarking forward pass ({num_iterations} iterations)...")
        
        # Warmup
        for _ in range(num_warmup):
            _ = model(input_ids, attn_mask)
            jax.block_until_ready(_)
        
        # Measure
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            output = model(input_ids, attn_mask)
            jax.block_until_ready(output)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        results = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
        
        logger.info(f"  Mean: {results['mean_ms']:.2f} ms")
        logger.info(f"  Std: {results['std_ms']:.2f} ms")
        logger.info(f"  P95: {results['p95_ms']:.2f} ms")
        
        return results
    
    @staticmethod
    def compare_latency(
        models: Dict[str, any],
        input_ids: jnp.ndarray,
        attn_mask: jnp.ndarray,
        num_iterations: int = 100,
    ) -> Dict[str, Dict]:
        """
        Compare latency across multiple models.
        
        Args:
            models: Dict mapping model name to model
            input_ids: Input for benchmarking
            attn_mask: Attention mask
            num_iterations: Number of iterations
            
        Returns:
            Dict mapping model name to latency metrics
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Benchmarking {name}...")
            results[name] = LatencyBenchmark.benchmark_forward_pass(
                model,
                input_ids,
                attn_mask,
                num_iterations=num_iterations,
            )
        
        # Compute overhead
        if "baseline" in results and "pcsvm" in results:
            baseline_latency = results["baseline"]["mean_ms"]
            pcsvm_latency = results["pcsvm"]["mean_ms"]
            overhead_pct = (pcsvm_latency - baseline_latency) / baseline_latency * 100
            
            results["pcsvm_overhead_pct"] = overhead_pct
            logger.info(f"\nPCSVM overhead: {overhead_pct:.2f}%")
        
        return results


def run_full_benchmark(
    pcsvm_model,
    lora_models: Dict[str, any],
    baseline_model,
    pcsvm_checkpoint: str,
    lora_checkpoints: Dict[str, str],
    test_input_ids: jnp.ndarray,
    test_attn_mask: jnp.ndarray,
    output_dir: str,
):
    """
    Run complete storage + latency benchmarking.
    
    Args:
        pcsvm_model: PCSVM model
        lora_models: Dict mapping task to LoRA model
        baseline_model: Zero-shot baseline
        pcsvm_checkpoint: Path to PCSVM checkpoint
        lora_checkpoints: Dict mapping task to LoRA checkpoint
        test_input_ids: Input for latency test
        test_attn_mask: Attention mask
        output_dir: Output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Running Complete Benchmark")
    logger.info("="*60)
    
    # 1. Storage benchmark
    logger.info("\n1. STORAGE BENCHMARK")
    storage_results = StorageBenchmark.compare_storage(
        pcsvm_checkpoint=pcsvm_checkpoint,
        lora_checkpoints=lora_checkpoints,
    )
    
    # 2. Latency benchmark
    logger.info("\n2. LATENCY BENCHMARK")
    
    models_for_latency = {
        "baseline": baseline_model,
        "pcsvm": pcsvm_model,
    }
    
    # Add first LoRA model for comparison
    if lora_models:
        first_task = list(lora_models.keys())[0]
        models_for_latency[f"lora_{first_task}"] = lora_models[first_task]
    
    latency_results = LatencyBenchmark.compare_latency(
        models=models_for_latency,
        input_ids=test_input_ids,
        attn_mask=test_attn_mask,
        num_iterations=100,
    )
    
    # 3. Save results
    all_results = {
        "storage": storage_results,
        "latency": latency_results,
    }
    
    import json
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_path}")
    
    return all_results


if __name__ == "__main__":
    print("Storage and Latency Benchmarking Utilities")
    print("\nUsage:")
    print("  from qkvflow.benchmarks.storage_latency import run_full_benchmark")
    print("  results = run_full_benchmark(...)")