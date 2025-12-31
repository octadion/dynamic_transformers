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
from levanter.checkpoint import load_checkpoint
import pandas as pd
import numpy as np
from tqdm import tqdm
import draccus
from datasets import load_dataset
from draccus import field
from draccus.parsers import decoding
from qkvflow.lora import loraize, is_lora_param
from qkvflow.train_lm import TrainLmConfig
from levanter.models.gpt2 import Gpt2LMHeadModel
from levanter.models.llama import LlamaLMHeadModel as LevanterLlamaLMHeadModel
from levanter.models.lm_model import LmHeadModel
from qkvflow.nn.dynamic import NeuralOdeLMHeadModel
from qkvflow.nn.dynamic_llama import LlamaLMHeadModel as LlamaODELMHeadModel
from qkvflow.nn.dynamic_svd import SVDNeuralOdeLMHeadModel
from qkvflow.nn.dynamic_svd_llama import SVDLlamaOdeLMHeadModel
from qkvflow.finetune import FinetuneLmConfig
from qkvflow.lora import loraize
import random
from transformers import AutoTokenizer  # noqa
logger = logging.getLogger(__name__)

@dataclass
class SvdConfig:
    rank_ratio: float = 0.5
    policy_init_scale: float = 0.1
    policy_reg_strength: float = 0.01
    policy_hidden_dim_ratio: Optional[float] = None
    policy_activation_strength: Optional[float] = None

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

@dataclass
class ModelCheckpointConfig:
    name: str
    config_path: str
    checkpoint_path: Optional[str] = NoneA
    lora_checkpoint_path: Optional[str] = None
    vanilla_checkpoint_path: Optional[str] = None

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
    max_seq_length: int = 1024
    few_shot_k: int = 0
    seed: int = 42
    output_dir: str = "./evaluation_results"
    save_individual_predictions: bool = False
    clear_cache_between_models: bool = True
    device: str = "gpu"

def get_model_for_eval(config: EvalTrainConfig):
    tokenizer = config.data.the_tokenizer
    model_key = jax.random.PRNGKey(config.trainer.seed)
    Vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), config.trainer.parameter_axis_mapping)
    model_type = config.model_choice
    logger.info(f"Initializing model of type: {model_type} from training config")
    if model_type == "gpt2": model = Gpt2LMHeadModel.init(Vocab, config=config.model, key=model_key)
    elif model_type == "neuralode": model = NeuralOdeLMHeadModel.init(Vocab, config=config.model, time_embed_dim=config.time_embed_dim, sinusodial_dim=config.sinusodial_dim, key=model_key)
    elif model_type == "neuralode-svd": model = SVDNeuralOdeLMHeadModel.init(Vocab, config=config.model, time_embed_dim=config.time_embed_dim, sinusodial_dim=config.sinusodial_dim, rank_ratio=config.svd_config.rank_ratio, policy_hidden_dim_ratio=config.svd_config.policy_hidden_dim_ratio,
            policy_init_scale=config.svd_config.policy_init_scale,key=model_key)
    elif model_type == "llama": model = LevanterLlamaLMHeadModel.init(Vocab, config=config.model, key=model_key)
    elif model_type == "llamaode": model = LlamaODELMHeadModel.init(Vocab, config=config.model, time_embed_dim=config.time_embed_dim, sinusodial_dim=config.sinusodial_dim, key=model_key)
    elif model_type == "llamaode-svd": model = SVDLlamaOdeLMHeadModel.init(Vocab, config=config.model, time_embed_dim=config.time_embed_dim, sinusodial_dim=config.sinusodial_dim, rank_ratio=config.svd_config.rank_ratio, policy_hidden_dim_ratio=config.svd_config.policy_hidden_dim_ratio,policy_init_scale=config.svd_config.policy_init_scale, key=model_key)
    else: raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    optimizer = config.optimizer.build(1)
    trainer = Trainer(config.trainer, optimizer, lambda m, ex, key: 0.0)
    with trainer.device_mesh: state = trainer.initial_state(model_key, model=model)
    return trainer, state.model, None, tokenizer

def load_model(checkpoint_config: "ModelCheckpointConfig"):
    is_lora = checkpoint_config.lora_checkpoint_path is not None
    
    if is_lora:
        logger.info(f"Loading finetuned LoRA model '{checkpoint_config.name}'...")
        with open(checkpoint_config.config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config: FinetuneLmConfig = decoding.decode(FinetuneLmConfig, config_dict)
        
        pretrain_config = config.pretrain_config
        lora_config = config.lora
        tokenizer = pretrain_config.data.the_tokenizer
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        Vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), pretrain_config.trainer.parameter_axis_mapping)
        model_key = jax.random.PRNGKey(pretrain_config.trainer.seed)
        
        if pretrain_config.model_choice == "gpt2":
            vanilla_scaffold = Gpt2LMHeadModel.init(Vocab, config=pretrain_config.model, key=model_key)
        elif pretrain_config.model_choice == "llama":
            vanilla_scaffold = LevanterLlamaLMHeadModel.init(Vocab, config=pretrain_config.model, key=model_key)
        else:
            raise NotImplementedError(f"Model type '{pretrain_config.model_choice}' is not supported.")

        vanilla_model, _, _ = load_checkpoint(vanilla_scaffold, training_state=None, checkpoint_path=checkpoint_config.vanilla_checkpoint_path)
        if vanilla_model is None: raise RuntimeError(f"Failed to load VANILLA checkpoint from '{checkpoint_config.vanilla_checkpoint_path}'")

        lora_model_structure = loraize(vanilla_model, config=lora_config, key=model_key)
        lora_only_scaffold = eqx.filter(lora_model_structure, is_lora_param, is_leaf=is_lora_param)
        loaded_lora_params, _, _ = load_checkpoint(lora_only_scaffold, training_state=None, checkpoint_path=checkpoint_config.lora_checkpoint_path)
        if loaded_lora_params is None: raise RuntimeError(f"Failed to load LORA checkpoint from '{checkpoint_config.lora_checkpoint_path}'")

        final_model = eqx.combine(lora_model_structure, loaded_lora_params)
        logger.info(f"LoRA model '{checkpoint_config.name}' loaded successfully.")
        return inference_mode(final_model, True), tokenizer

    else:
        logger.info(f"Loading pretrained model '{checkpoint_config.name}' via Levanter...")
        loaded_components = levanter.config.main(
            get_model_for_eval,
            args=["--config_path", checkpoint_config.config_path, "--trainer.load_checkpoint_path", checkpoint_config.checkpoint_path, "--trainer.wandb.mode", "disabled", "--trainer.id", f"eval-{checkpoint_config.name.replace(' ', '-')}"],
        )()
        model, tokenizer = loaded_components[1], loaded_components[3]
        if model is None or tokenizer is None: raise RuntimeError(f"Failed to load model '{checkpoint_config.name}'.")
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Pretrained model '{checkpoint_config.name}' loaded successfully.")
        return inference_mode(model, True), tokenizer



def load_single_dataset(path_or_name: str, config: Optional[str], split: Optional[str]):
    if os.path.isfile(path_or_name):
        logger.info(f"Loading local data file: {path_or_name}")
        file_type = os.path.splitext(path_or_name)[1][1:]
        if not file_type: raise ValueError(f"Cannot determine file type from path: {path_or_name}")
        dataset = load_dataset(file_type, data_files=path_or_name)
        return dataset[next(iter(dataset.keys()))]
    else:
        logger.info(f"Loading dataset from Hub: {path_or_name}")
        return load_dataset(path_or_name, name=config, split=split, verification_mode="no_checks")


class MultipleChoiceEvaluator:
    def __init__(self, config: EvaluationConfig, tokenizer, few_shot_cache: Optional[Dict[str, List[Dict]]] = None):
        self.config = config
        self.tokenizer = tokenizer
        self._few_shot_cache = few_shot_cache if few_shot_cache is not None else {}

    def get_few_shot_examples(self, dataset_name: str) -> List[Dict]:
        return self._few_shot_cache.get(dataset_name, [])

    def get_details_from_example(self, ex: Dict, ds_name: str) -> Tuple[str, Optional[List[str]], Any, Optional[str]]:
        context = ex.get('support') or ex.get('context') or ex.get('hint')
        if ds_name.startswith("ARC"):
            answer_key = ex["answerKey"]
            ans_idx = ord(answer_key.upper()) - ord('A') if 'A' <= answer_key.upper() <= 'Z' else int(answer_key) - 1
            return ex["question"], ex["choices"]["text"], ans_idx, context
        elif ds_name == "SciQ":
            choices = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
            rng = random.Random(hash(ex["question"]))
            rng.shuffle(choices)
            ans_idx = choices.index(ex["correct_answer"])
            return ex["question"], choices, ans_idx, context
        elif ds_name == "Winogrande":
            return ex["sentence"], [ex["option1"], ex["option2"]], int(ex["answer"]) - 1, context
        else:
            raise ValueError(f"Dataset logic not implemented for: {ds_name}")

    def format_prompt_for_dataset(self, question: str, choices: Optional[List[str]], answer: Optional[Any] = None, context: Optional[str] = None) -> str:
        prompt = ""
        if context: prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n"
        if choices:
            labels = [chr(ord('A') + i) for i in range(len(choices))]
            for label, choice in zip(labels, choices):
                prompt += f"{label}. {choice}\n"
            prompt += "Answer:"
            if answer is not None:
                prompt += f" {labels[answer]}"
        return prompt

    def prepare_prompt(self, example: Dict, dataset_config: EvalDatasetConfig) -> Tuple[List[str], int]:
        few_shot_header = ""
        few_shot_examples = self.get_few_shot_examples(dataset_config.name)
        if few_shot_examples:
            for fs_ex in few_shot_examples:
                fs_question, fs_choices, fs_answer, fs_context = self.get_details_from_example(fs_ex, dataset_config.name)
                few_shot_header += self.format_prompt_for_dataset(fs_question, fs_choices, fs_answer, fs_context) + "\n\n"

        current_question, current_choices, correct_answer_index, current_context = self.get_details_from_example(example, dataset_config.name)
        if not current_choices or not all(isinstance(c, str) for c in current_choices):
            return [], -1

        base_prompt = self.format_prompt_for_dataset(current_question, current_choices, answer=None, context=current_context)
        prompts = [few_shot_header + base_prompt] * len(current_choices)
        return prompts, correct_answer_index

    def compute_loglikelihood(self, model: LmHeadModel, prompts: List[str], choices: List[str], batch_size: int) -> List[float]:
        all_log_likelihoods = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_choices = choices[i:i + batch_size]
            batch_texts = [p + f" {c}" for p, c in zip(batch_prompts, batch_choices)]

            inputs = self.tokenizer(
                batch_texts, padding="longest", return_tensors="np",
                truncation=True, max_length=self.config.max_seq_length
            )
            labels_for_loss = np.roll(inputs['input_ids'], -1, axis=-1)
            loss_masks = np.zeros_like(inputs['input_ids'])

            for j in range(len(batch_texts)):
                prompt_len = len(self.tokenizer.encode(batch_prompts[j]))
                choice_len = len(self.tokenizer.encode(f" {batch_choices[j]}", add_special_tokens=False))
                if prompt_len + choice_len <= inputs['input_ids'].shape[1]:
                    loss_masks[j, prompt_len - 1 : prompt_len - 1 + choice_len] = 1

            Pos = hax.Axis("position", inputs['input_ids'].shape[1])
            Batch = hax.Axis("batch", inputs['input_ids'].shape[0])
            tokens = hax.named(jnp.array(inputs['input_ids']), (Batch, Pos))
            attn_mask = hax.named(jnp.array(inputs['attention_mask']), (Batch, Pos))
            loss_mask_ax = hax.named(loss_masks, (Batch, Pos))
            logits = model(tokens, attn_mask=attn_mask)
            Vocab = logits.axes[-1]
            target_y = hax.nn.one_hot(hax.named(labels_for_loss, tokens.axes), Vocab, dtype=logits.dtype)
            per_token_loss = hnn.cross_entropy_loss(logits, Vocab, target_y, reduction=None)
            masked_loss = per_token_loss * loss_mask_ax
            sum_loss = hax.sum(masked_loss, axis=Pos)
            num_completion_tokens = hax.maximum(hax.sum(loss_mask_ax, axis=Pos), 1)
            avg_loss = sum_loss / num_completion_tokens
            all_log_likelihoods.extend((-avg_loss.array).tolist())
        return all_log_likelihoods

    def evaluate_dataset(self, model: LmHeadModel, dataset_config: EvalDatasetConfig) -> Dict[str, Any]:
        logger.info(f"Evaluating '{dataset_config.name}' with fixed {self.config.few_shot_k}-shot set.")
        
        dataset = load_single_dataset(dataset_config.dataset_name, dataset_config.dataset_config, dataset_config.split)
        
        if dataset_config.num_samples and dataset_config.num_samples < len(dataset):
            dataset = dataset.select(range(dataset_config.num_samples))
            
        correct, total, skipped_count = 0, 0, 0
        
        for example in tqdm(dataset, desc=f"Evaluating {dataset_config.name}"):
            prompts, actual_idx = self.prepare_prompt(example, dataset_config)
            if not prompts:
                skipped_count += 1
                continue
            
            _, choices, _, _ = self.get_details_from_example(example, dataset_config.name)
            log_likelihoods = self.compute_loglikelihood(model, prompts, choices, self.config.batch_size)
            predicted_idx = np.argmax(log_likelihoods)
            
            if predicted_idx == actual_idx:
                correct += 1
            total += 1
            
        accuracy = correct / total if total > 0 else 0.0
        final_results = {"accuracy": accuracy, "correct": correct, "total": total, "skipped_invalid": skipped_count}
        logger.info(f"Result for {dataset_config.name}: Accuracy = {accuracy:.4f} ({correct}/{total})")
        return final_results


def save_results(results: Dict, output_dir: str):
    output_path = os.path.join(output_dir, "evaluation_results.json")
    with open(output_path, 'w') as f: json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

def generate_comparison_report(results: Dict, config: EvaluationConfig):
    if not results: 
        logger.warning("No results to generate report.")
        return
    all_data = []
    for model_name, model_results in results.items():
        for dataset_name, dataset_results in model_results.items():
            row = {"Model": model_name, "Dataset": dataset_name}
            if isinstance(dataset_results, dict):
                row.update(dataset_results)
            all_data.append(row)
    df = pd.DataFrame(all_data)
    if df.empty: 
        logger.warning("Result DataFrame is empty.")
        return
    raw_csv_path = os.path.join(config.output_dir, "full_results.csv")
    df.to_csv(raw_csv_path, index=False)
    logger.info(f"Full results saved to {raw_csv_path}")
    if "accuracy" in df.columns:
        pivot_df = df.pivot(index="Model", columns="Dataset", values="accuracy")
        if not pivot_df.empty:
            pivot_df["Average"] = pivot_df.mean(axis=1)
            summary_csv_path = os.path.join(config.output_dir, "accuracy_summary.csv")
            pivot_df.to_csv(summary_csv_path)
            logger.info(f"Accuracy summary saved to {summary_csv_path}")
            print("\n" + "="*80 + "\nACCURACY SUMMARY\n" + "="*80)
            print(pivot_df.round(4))

def main(config: EvaluationConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    os.makedirs(config.output_dir, exist_ok=True)
    random.seed(config.seed)
    
    logger.info("Pre-generating a fixed, token-efficient few-shot set for all datasets...")
    few_shot_cache = {}
    if config.few_shot_k > 0:
        temp_evaluator = MultipleChoiceEvaluator(config, None, None)
        try:
            temp_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            logger.error(f"Could not load reference tokenizer 'gpt2'. Exiting. Error: {e}")
            return

        for dataset_config in config.datasets:
            if dataset_config.name in ["humaneval", "MATH"]: continue
            logger.info(f"  - Selecting shortest {config.few_shot_k} examples from '{dataset_config.name}'")
            
            temp_dataset = load_single_dataset(dataset_config.dataset_name, dataset_config.dataset_config, dataset_config.split)
            
            example_lengths = []
            for i, ex in enumerate(temp_dataset):
                try:
                    q, choices, ans, ctx = temp_evaluator.get_details_from_example(ex, dataset_config.name)
                    if not choices: continue
                    full_prompt_text = temp_evaluator.format_prompt_for_dataset(q, choices, ans, ctx)
                    token_length = len(temp_tokenizer.encode(full_prompt_text))
                    example_lengths.append((token_length, i))
                except Exception:
                    continue
            
            example_lengths.sort()
            k = config.few_shot_k
            if len(example_lengths) < k: k = len(example_lengths)
            shortest_indices = [idx for length, idx in example_lengths[:k]]
            few_shot_cache[dataset_config.name] = [temp_dataset[i] for i in shortest_indices]
            logger.info(f"  - Selected indices: {shortest_indices}")

    logger.info("Fixed few-shot set generated.")

    all_results = {}
    for model_checkpoint in config.model_checkpoints:
        logger.info(f"\n{'='*50}\nEvaluating: {model_checkpoint.name}\n{'='*50}")
        start_time = time.time()
        model, tokenizer = None, None
        try:
            model, tokenizer = load_model(model_checkpoint)
            evaluator = MultipleChoiceEvaluator(config, tokenizer, few_shot_cache)
            
            model_results = {}
            for dataset_config in config.datasets:
                if dataset_config.name in ["humaneval", "MATH"]: continue
                results = evaluator.evaluate_dataset(model, dataset_config)
                model_results[dataset_config.name] = results
            all_results[model_checkpoint.name] = model_results
            save_results(all_results, config.output_dir)
        except Exception as e:
            logger.error(f"Error evaluating {model_checkpoint.name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del model, tokenizer
            gc.collect()
            if jax.devices() and jax.devices()[0].platform == 'gpu':
                jax.clear_caches()
        elapsed_time = time.time() - start_time
        logger.info(f"Evaluation time for {model_checkpoint.name}: {elapsed_time:.2f} seconds")
        
    generate_comparison_report(all_results, config)
    logger.info("\nEvaluation complete!")