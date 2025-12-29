import levanter
import os
import jax
import logging
from dataclasses import dataclass

from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.utils.tree_utils import inference_mode

from qkvflow.finetune import FinetuneLmConfig
from qkvflow.lora import merge_lora_modules
from levanter.models.llama import LlamaLMHeadModel
from levanter.models.gpt2 import Gpt2LMHeadModel
from haliax import Axis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class MergeJob:
    job_name: str
    finetune_config_path: str
    finetune_checkpoint_base_path: str
    best_step: int
    merged_output_path: str

def process_job(job: MergeJob):
    print("\n" + "="*50)
    logging.info(f"STARTING JOB: {job.job_name}")
    print("="*50)

    config: FinetuneLmConfig = levanter.config.load_config(FinetuneLmConfig, job.finetune_config_path)

    lora_checkpoint_path = os.path.join(job.finetune_checkpoint_base_path, f"step-{job.best_step}")

    pretrain_config = config.pretrain_config
    tokenizer = config.finetune_data.the_tokenizer
    Vocab = Axis("vocab", len(tokenizer))
    model_key = jax.random.PRNGKey(0)

    if pretrain_config.model_choice == "gpt2":
        model = Gpt2LMHeadModel.init(Vocab, config=pretrain_config.model, key=model_key)
    elif pretrain_config.model_choice == "llama":
        model = LlamaLMHeadModel.init(Vocab, config=pretrain_config.model, key=model_key)
    else:
        raise ValueError(f"Model choice '{pretrain_config.model_choice}' not supported.")

    from qkvflow.lora import loraize
    model = loraize(model, config=config.lora, key=model_key)

    logging.info(f"Loading LoRA model from: {lora_checkpoint_path}")
    model = load_checkpoint(model, path=lora_checkpoint_path)[0]

    if model is None:
        logging.error(f"FAILED to load checkpoint for job '{job.job_name}'. Path not found: {lora_checkpoint_path}")
        return

    logging.info("Merging LoRA modules...")
    merged_model = merge_lora_modules(model)
    merged_model = inference_mode(merged_model, True)

    logging.info(f"Saving merged model to: {job.merged_output_path}")
    os.makedirs(job.merged_output_path, exist_ok=True)
    save_checkpoint(merged_model, path=job.merged_output_path, step=0)
    logging.info(f"SUCCESSFULLY completed job: {job.job_name}")


if __name__ == "__main__":

    jobs_to_run = [
        MergeJob(
            job_name="GPT2_Winogrande",
            finetune_config_path="/content/dynamic_transformers/config/finetune/finetune_gpt2_lora_winogrande.yaml",
            finetune_checkpoint_base_path="/content/drive/MyDrive/lora_finetuned_checkpoints_gpt2_winogrande/d0el4lyy",
            best_step=1500,
            merged_output_path="/content/drive/MyDrive/merged_checkpoints/gpt2_winogrande_merged",
        ),
        MergeJob(
            job_name="Llama_Winogrande",
            finetune_config_path="/content/dynamic_transformers/config/finetune/finetune_llama_lora_winogrande.yaml",
            finetune_checkpoint_base_path="/content/drive/MyDrive/lora_finetuned_checkpoints_winogrande/6ukoar5l",
            best_step=2500,
            merged_output_path="/content/drive/MyDrive/merged_checkpoints/llama_winogrande_merged",
        ),
        MergeJob(
            job_name="GPT2_SciQ",
            finetune_config_path="/content/dynamic_transformers/config/finetune/finetune_gpt2_lora_sciq.yaml",
            finetune_checkpoint_base_path="/content/drive/MyDrive/lora_finetuned_checkpoints_gpt2_sciq/050tikvn", 
            best_step=500,
            merged_output_path="/content/drive/MyDrive/merged_checkpoints/gpt2_sciq_merged",
        ),
        MergeJob(
            job_name="Llama_SciQ",
            finetune_config_path="/content/dynamic_transformers/config/finetune/finetune_llama_lora_sciq.yaml",
            finetune_checkpoint_base_path="/content/drive/MyDrive/lora_finetuned_checkpoints_llama_sciq/k0c37lc5",
            best_step=1000,
            merged_output_path="/content/drive/MyDrive/merged_checkpoints/llama_sciq_merged",
        ),
    ]
    for job in jobs_to_run:
        try:
            process_job(job)
        except Exception as e:
            logging.error(f"An error occurred while processing job '{job.job_name}': {e}", exc_info=True)