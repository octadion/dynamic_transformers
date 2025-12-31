import os
import logging
from dataclasses import dataclass

import draccus
import equinox as eqx
import haliax as hax
import jax

from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.utils.tree_utils import inference_mode

from qkvflow.finetune import FinetuneLmConfig
from qkvflow.lora import is_lora_param, loraize, merge_lora_modules
from qkvflow.train_lm import TrainLmConfig
from levanter.models.gpt2 import Gpt2LMHeadModel
from levanter.models.llama import LlamaLMHeadModel
from haliax import Axis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class MergeJob:
    job_name: str
    vanilla_config_path: str
    vanilla_checkpoint_path: str
    lora_finetune_config_path: str
    lora_checkpoint_path: str
    merged_output_path: str


def process_job(job: MergeJob):
    print("\n" + "="*80)
    logging.info(f"STARTING JOB: {job.job_name}")
    print("="*80)

    logging.info(f"Parsing vanilla config: {job.vanilla_config_path}")
    vanilla_config: TrainLmConfig = draccus.parse(TrainLmConfig, config_path=job.vanilla_config_path)
    
    logging.info(f"Parsing LoRA fine-tune config: {job.lora_finetune_config_path}")
    lora_finetune_config: FinetuneLmConfig = draccus.parse(FinetuneLmConfig, config_path=job.lora_finetune_config_path)

    logging.info(f"Building vanilla model architecture ('{vanilla_config.model_choice}')")
    tokenizer = vanilla_config.data.the_tokenizer
    Vocab = Axis("vocab", len(tokenizer))
    model_key = jax.random.PRNGKey(0)

    if vanilla_config.model_choice == "gpt2":
        model_scaffold = Gpt2LMHeadModel.init(Vocab, config=vanilla_config.model, key=model_key)
    elif vanilla_config.model_choice == "llama":
        model_scaffold = LlamaLMHeadModel.init(Vocab, config=vanilla_config.model, key=model_key)
    else:
        raise ValueError(f"Model choice '{vanilla_config.model_choice}' not supported.")

    logging.info(f"Loading FULL vanilla model weights from {job.vanilla_checkpoint_path}")
    vanilla_model = load_checkpoint(model_scaffold, None, job.vanilla_checkpoint_path)[0]
    if vanilla_model is None:
        logging.error(f"FAILED to load vanilla model. Path: {job.vanilla_checkpoint_path}")
        return

    logging.info("Applying LoRA structure to the loaded vanilla model")
    lora_model_scaffold = loraize(vanilla_model, config=lora_finetune_config.lora, key=model_key)

    logging.info(f"Loading SPARSE LoRA adapter weights from {job.lora_checkpoint_path}")
    lora_only_scaffold = eqx.filter(lora_model_scaffold, is_lora_param, is_leaf=is_lora_param)
    loaded_lora_params = load_checkpoint(lora_only_scaffold, None, job.lora_checkpoint_path)[0]

    if loaded_lora_params is None:
        logging.error(f"FAILED to load LoRA adapter. Path: {job.lora_checkpoint_path}")
        return

    logging.info("Combining loaded LoRA weights with the full model")
    final_lora_model = eqx.combine(lora_model_scaffold, loaded_lora_params)

    logging.info("Merging LoRA modules mathematically")
    merged_model = merge_lora_modules(final_lora_model)
    merged_model = inference_mode(merged_model, True)

    if vanilla_config.model_choice == "llama":
        logging.info("Transposing lm_head weight for Llama model to match loading convention...")
        merged_model = eqx.tree_at(
            lambda m: m.lm_head.weight,
            merged_model,
            hax.rearrange(merged_model.lm_head.weight, "embed vocab -> vocab embed")
        )

    logging.info(f"Step 9: Saving merged model to: {job.merged_output_path}")
    os.makedirs(job.merged_output_path, exist_ok=True)

    save_checkpoint(model=merged_model, training_state=None, step=0, checkpoint_path=job.merged_output_path, exist_ok=True)
    
    logging.info(f"SUCCESSFULLY completed job: {job.job_name}")


if __name__ == "__main__": 
    jobs_to_run = [
        MergeJob(
            job_name="GPT2_Winogrande",
            vanilla_config_path="/content/dynamic_transformers/config/owt_10k/gpt2_nano.yaml",
            vanilla_checkpoint_path="/content/drive/MyDrive/nano_gpt2_checkpoints/u4c2ysp0",
            lora_finetune_config_path="/content/dynamic_transformers/config/finetune/finetune_gpt2_lora_winogrande.yaml",
            lora_checkpoint_path="/content/drive/MyDrive/lora_finetuned_checkpoints_gpt2_winogrande/7o20jpiu/step-1500",
            merged_output_path="/content/drive/MyDrive/merged_checkpoints/gpt2_winogrande_merged",
        ),
        # MergeJob(
        #     job_name="Llama_Winogrande",
        #     vanilla_config_path="/content/dynamic_transformers/config/owt_10k/llama_nano.yaml",
        #     vanilla_checkpoint_path="/content/drive/MyDrive/nano_llama_checkpoints/62t179nh",
        #     lora_finetune_config_path="/content/dynamic_transformers/config/finetune/finetune_llama_lora_winogrande.yaml",
        #     lora_checkpoint_path="/content/drive/MyDrive/lora_finetuned_checkpoints_winogrande/oqtjxo7a/step-2500",
        #     merged_output_path="/content/drive/MyDrive/merged_checkpoints/llama_winogrande_merged",
        # ),
        # MergeJob(
        #     job_name="GPT2_SciQ",
        #     vanilla_config_path="/content/dynamic_transformers/config/owt_10k/gpt2_nano.yaml",
        #     vanilla_checkpoint_path="/content/drive/MyDrive/nano_gpt2_checkpoints/u4c2ysp0",
        #     lora_finetune_config_path="/content/dynamic_transformers/config/finetune/finetune_gpt2_lora_sciq.yaml",
        #     lora_checkpoint_path="/content/drive/MyDrive/lora_finetuned_checkpoints_gpt2/iygk0cyp/step-500",
        #     merged_output_path="/content/drive/MyDrive/merged_checkpoints/gpt2_sciq_merged",
        # ),
        # MergeJob(
        #     job_name="Llama_SciQ",
        #     vanilla_config_path="/content/dynamic_transformers/config/owt_10k/llama_nano.yaml",
        #     vanilla_checkpoint_path="/content/drive/MyDrive/nano_llama_checkpoints/62t179nh",
        #     lora_finetune_config_path="/content/dynamic_transformers/config/finetune/finetune_llama_lora_sciq.yaml",
        #     lora_checkpoint_path="/content/drive/MyDrive/lora_finetuned_checkpoints/i2kxioug/step-1000",
        #     merged_output_path="/content/drive/MyDrive/merged_checkpoints/llama_sciq_merged",
        # ),
    ]

    for job in jobs_to_run:
        try:
            process_job(job)
        except Exception as e:
            logging.error(f"An error occurred while processing job '{job.job_name}': {e}", exc_info=True)