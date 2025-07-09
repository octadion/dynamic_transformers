import os
import gc
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
import yaml
import asyncio
import jax
import jax.numpy as jnp
import equinox as eqx
import haliax as hax
import haliax.nn as hnn
from haliax.partitioning import round_axis_for_partitioning
from haliax import Axis
import levanter.config
from levanter.trainer import Trainer
from levanter.utils.tree_utils import inference_mode
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from draccus import field

from qkvflow.train_lm import TrainLmConfig
from levanter.models.gpt2 import Gpt2LMHeadModel
from levanter.models.llama import LlamaLMHeadModel as LevanterLlamaLMHeadModel
from levanter.models.lm_model import LmHeadModel
from qkvflow.nn.dynamic import NeuralOdeLMHeadModel
from qkvflow.nn.dynamic_llama import LlamaLMHeadModel as LlamaODELMHeadModel
from qkvflow.nn.dynamic_svd import SVDNeuralOdeLMHeadModel
from qkvflow.nn.dynamic_svd_llama import SVDLlamaOdeLMHeadModel

logger = logging.getLogger(__name__)

@dataclass
class SvdConfig:
    rank_ratio: float = 0.5
    policy_init_scale: float = 0.1
    policy_reg_strength: float = 0.01

@dataclass
class EvalTrainConfig(TrainLmConfig):
    time_embed_dim: Optional[int] = None
    sinusodial_dim: Optional[int] = None
    multiplier: Optional[float] = None
    num_check_points: Optional[int] = None

    svd_config: Optional[SvdConfig] = field(default_factory=SvdConfig)
    train_policy_only: bool = False
    train_svd_from_scratch: bool = True
    load_pretrained_ode: Optional[str] = None


def get_model_for_eval(config: EvalTrainConfig):
    tokenizer = config.data.the_tokenizer
    model_key = jax.random.PRNGKey(config.trainer.seed)
    Vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), config.trainer.parameter_axis_mapping)
    model_type = config.model_choice
    logger.info(f"Initializing model of type: {model_type} from training config")

    if model_type == "gpt2": model = Gpt2LMHeadModel.init(Vocab, config=config.model, key=model_key)
    elif model_type == "neuralode": model = NeuralOdeLMHeadModel.init(Vocab, config=config.model, time_embed_dim=config.time_embed_dim, sinusodial_dim=config.sinusodial_dim, key=model_key)
    elif model_type == "neuralode-svd": model = SVDNeuralOdeLMHeadModel.init(Vocab, config=config.model, time_embed_dim=config.time_embed_dim, sinusodial_dim=config.sinusodial_dim, rank_ratio=config.svd_config.rank_ratio, key=model_key)
    elif model_type == "llama": model = LevanterLlamaLMHeadModel.init(Vocab, config=config.model, key=model_key)
    elif model_type == "llamaode": model = LlamaODELMHeadModel.init(Vocab, config=config.model, time_embed_dim=config.time_embed_dim, sinusodial_dim=config.sinusodial_dim, key=model_key)
    elif model_type == "llamaode-svd": model = SVDLlamaOdeLMHeadModel.init(Vocab, config=config.model, time_embed_dim=config.time_embed_dim, sinusodial_dim=config.sinusodial_dim, rank_ratio=config.svd_config.rank_ratio, key=model_key)
    else: raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    
    optimizer = config.optimizer.build(1)
    trainer = Trainer(config.trainer, optimizer, lambda m, ex, key: 0.0)
    with trainer.device_mesh: state = trainer.initial_state(model_key, model=model)
    return trainer, state.model, None, tokenizer

def load_model_from_levanter_config(checkpoint_config: "ModelCheckpointConfig"):
    logger.info(f"Loading model '{checkpoint_config.name}' via Levanter...")
    logger.info(f"  - Training Config: {checkpoint_config.config_path}")
    logger.info(f"  - Checkpoint: {checkpoint_config.checkpoint_path}")

    loaded_components = levanter.config.main(
        get_model_for_eval,
        args=[
            "--config_path", checkpoint_config.config_path,
            "--trainer.load_checkpoint_path", checkpoint_config.checkpoint_path,
            "--trainer.wandb.mode", "disabled",
            "--trainer.id", f"eval-{checkpoint_config.name.replace(' ', '-')}",
        ],
    )()

    model, tokenizer = loaded_components[1], loaded_components[3]
    if model is None or tokenizer is None: raise RuntimeError(f"Failed to load model '{checkpoint_config.name}'.")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Model '{checkpoint_config.name}' loaded successfully.")
    return inference_mode(model, True), tokenizer


@dataclass
class ModelCheckpointConfig:
    name: str
    checkpoint_path: str
    config_path: str

@dataclass
class EvalDatasetConfig:
    name: str
    dataset_name: str
    split: str = "validation"
    dataset_config: Optional[str] = None
    num_samples: Optional[int] = None
    context_key: Optional[str] = None

@dataclass
class EvaluationConfig:
    model_checkpoints: List[ModelCheckpointConfig] = field(default_factory=list)
    datasets: List[EvalDatasetConfig] = field(default_factory=list)
    batch_size: int = 16
    max_seq_length: int = 512
    few_shot_k: int = 0
    seed: int = 42
    output_dir: str = "./evaluation_results"
    save_individual_predictions: bool = False
    clear_cache_between_models: bool = True
    device: str = "gpu"

class MultipleChoiceEvaluator:
    def __init__(self, config: EvaluationConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def prepare_prompt(self, example: Dict, dataset_config: EvalDatasetConfig, few_shot_examples: List[Dict]) -> Tuple[List[str], int]:
        def get_details(ex, ds_name):
            if ds_name == "PIQA": return ex["goal"], [ex["sol1"], ex["sol2"]], int(ex["label"])
            elif ds_name == "ARC-Easy":
                answer = ex["answerKey"]; ans_idx = ord(answer) - ord('A') if not answer.isdigit() else int(answer) - 1
                return ex["question"], ex["choices"]["text"], ans_idx
            elif ds_name == "SciQ":
                choices = [
                    ex["correct_answer"], 
                    ex["distractor1"], 
                    ex["distractor2"], 
                    ex["distractor3"]
                ]
                correct_text = ex["correct_answer"]
                import random
                rng = random.Random(42)
                rng.shuffle(choices)
                correct_answer_index = choices.index(correct_text)
                return ex["question"], choices, correct_answer_index
            elif ds_name == "Winogrande": return ex["sentence"], [ex["option1"], ex["option2"]], int(ex["answer"]) - 1
            else: raise ValueError(f"Dataset logic not implemented for: {ds_name}")
        question, choices, correct_answer_index = get_details(example, dataset_config.name)
        prompts = [f"Question: {question}\nAnswer: {choice}" for choice in choices]
        return prompts, correct_answer_index


    def compute_loglikelihood(self, model: LmHeadModel, prompts: List[str], batch_size: int) -> List[float]:
        all_log_likelihoods = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            inputs = self.tokenizer(batch_prompts, padding="longest", return_tensors="np", truncation=True, max_length=self.config.max_seq_length)
            
            labels = np.roll(inputs['input_ids'], -1, axis=-1)
            
            prompts_only = [p.split("Answer:")[0] + "Answer:" for p in batch_prompts]
            prompts_tok_ids = self.tokenizer(prompts_only, padding=False)["input_ids"]

            loss_masks = np.zeros_like(inputs['input_ids'])
            for j, p_tok in enumerate(prompts_tok_ids):
                prompt_len = len(p_tok)
                if prompt_len < loss_masks.shape[1]:
                    completion_len = inputs['attention_mask'][j].sum() - prompt_len
                    if completion_len > 0:
                        loss_masks[j, prompt_len - 1 : prompt_len - 1 + completion_len] = 1

            Pos = hax.Axis("position", inputs['input_ids'].shape[1])
            Batch = hax.Axis("batch", inputs['input_ids'].shape[0])

            tokens = hax.named(jnp.array(inputs['input_ids']), (Batch, Pos))
            attn_mask = hax.named(jnp.array(inputs['attention_mask']), (Batch, Pos))
            loss_mask_ax = hax.named(loss_masks, (Batch, Pos))
            
            logits, _ = model(tokens, attn_mask)
            Vocab = logits.axes[-1]

            target_y = hax.nn.one_hot(hax.named(labels, tokens.axes), Vocab, dtype=logits.dtype)

            per_token_loss = hnn.cross_entropy_loss(logits, Vocab, target_y, reduction=None)

            masked_loss = per_token_loss * loss_mask_ax
            
            sum_loss = hax.sum(masked_loss, axis=Pos)
            num_completion_tokens = hax.maximum(hax.sum(loss_mask_ax, axis=Pos), 1)
            
            avg_loss = sum_loss / num_completion_tokens

            log_likelihoods = -avg_loss.array
            all_log_likelihoods.extend(log_likelihoods.tolist())

        return all_log_likelihoods

    def evaluate_dataset(self, model: LmHeadModel, dataset_config: EvalDatasetConfig) -> Dict[str, float]:
        logger.info(f"Evaluating on {dataset_config.name}")
        dataset = load_dataset(
            dataset_config.dataset_name,
            name=dataset_config.dataset_config,
            split=dataset_config.split,
            verification_mode="no_checks"
        )
        if dataset_config.num_samples: dataset = dataset.select(range(min(dataset_config.num_samples, len(dataset))))

        correct = 0; total = 0
        for example in tqdm(dataset, desc=f"Evaluating {dataset_config.name}"):
            prompts, actual_idx = self.prepare_prompt(example, dataset_config, [])
            # Split prompts into choices for batching
            log_likelihoods_per_choice = self.compute_loglikelihood(model, prompts, self.config.batch_size)
            predicted_idx = np.argmax(log_likelihoods_per_choice)
            if predicted_idx == actual_idx: correct += 1
            total += 1
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy, "correct": correct, "total": total}

def save_results(results: Dict, output_dir: str):
    output_path = os.path.join(output_dir, "evaluation_results.json")
    with open(output_path, 'w') as f: json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

def generate_comparison_report(results: Dict, config: EvaluationConfig):
    if not results: logger.warning("No results to generate report."); return
    data = []
    for model_name, model_results in results.items():
        for dataset_name, dataset_results in model_results.items():
            data.append({"Model": model_name, "Dataset": dataset_name, "Accuracy": dataset_results["accuracy"]})
    df = pd.DataFrame(data)
    if df.empty: logger.warning("Result DataFrame is empty."); return
    pivot_df = df.pivot(index="Model", columns="Dataset", values="Accuracy")
    pivot_df["Average"] = pivot_df.mean(axis=1)
    csv_path = os.path.join(config.output_dir, "comparison_table.csv")
    pivot_df.to_csv(csv_path); logger.info(f"Comparison table saved to {csv_path}")
    print("\n" + "="*80 + "\nEVALUATION SUMMARY\n" + "="*80)
    print(pivot_df.round(4))

def main(config: EvaluationConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    os.makedirs(config.output_dir, exist_ok=True)
    all_results = {}
    for model_checkpoint in config.model_checkpoints:
        logger.info(f"\n{'='*50}\nEvaluating: {model_checkpoint.name}\n{'='*50}")
        start_time = time.time()
        model, tokenizer = None, None
        try:
            model, tokenizer = load_model_from_levanter_config(model_checkpoint)
            evaluator = MultipleChoiceEvaluator(config, tokenizer)
            model_results = {}
            for dataset_config in config.datasets:
                results = evaluator.evaluate_dataset(model, dataset_config)
                model_results[dataset_config.name] = results
                logger.info(f"  {dataset_config.name} Accuracy: {results['accuracy']:.4f}")
            all_results[model_checkpoint.name] = model_results
            save_results(all_results, config.output_dir)
        except Exception as e:
            logger.error(f"Error evaluating {model_checkpoint.name}: {e}")
            import traceback; traceback.print_exc()
        finally:
            del model, tokenizer; gc.collect()
            if jax.devices() and jax.devices()[0].platform == 'gpu': jax.clear_caches()
        elapsed_time = time.time() - start_time
        logger.info(f"Evaluation time for {model_checkpoint.name}: {elapsed_time:.2f} seconds")
    generate_comparison_report(all_results, config)
    logger.info("\nEvaluation complete!")