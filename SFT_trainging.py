from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch
from datasets import Dataset
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0

# Step 1: Format + Tokenize
def format_and_tokenize(example):
    result = tokenizer(
        example["prompt"] + example["label"],
        truncation=True,
        max_length=550,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

if __name__ == "__main__":
    tldr_dataset = load_dataset("CarperAI/openai_summarize_tldr")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    checkpoint_dir = "/home/godwinkhalko/LLMs/qwen-tldr-sft-merged"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("./qwen-tldr-sft-merged", torch_dtype=torch.float16)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    #LoRA configuration 
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                     # Rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[               # Target more layers
        "q_proj", 
        "v_proj", 
        "k_proj",                  # Add key projection
        "o_proj",                  # Add output projection
        "gate_proj",               # Add FFN layers
        "up_proj",
        "down_proj"]
        )

    #Get the PEFT model with LoRA applied
    model = get_peft_model(model, lora_config)

    tokenized_dataset = tldr_dataset.map(
        format_and_tokenize,
        remove_columns=tldr_dataset["train"].column_names,  # remove ALL original columns
        batched=False,
    )

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Step 3: Train with standard Trainer
    training_args = TrainingArguments(
        output_dir="./qwen-tldr-sft",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,           # Save less frequently
        save_total_limit=2,        # Keep only 2 checkpoints at a time
        load_best_model_at_end=True,
        fp16=True,
        remove_unused_columns=False,
        # Add these to prevent overfitting with more epochs
        warmup_ratio=0.1,
        weight_decay=0.01
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        data_collator=data_collator,
    )

    trainer.train()