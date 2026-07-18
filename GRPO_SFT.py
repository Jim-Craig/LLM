import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import random
import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, TaskType, get_peft_model



def generate_arithmetic_samples(num_samples=1000, ops=("+",)):
    data = []
    for _ in range(num_samples):
        op = random.choice(ops)
        a = torch.randint(0, 100, (1,)).item()
        b = torch.randint(1, 100, (1,)).item()  # avoid 0 as divisor

        if op == "+":
            answer = a + b
            reasoning = f"I need to add {a} and {b}. {a} + {b} = {answer}."
        elif op == "-":
            answer = a - b
            reasoning = f"I need to subtract {b} from {a}. {a} - {b} = {answer}."
        elif op == "*":
            answer = a * b
            reasoning = f"I need to multiply {a} and {b}. {a} * {b} = {answer}."
        elif op == "/":
            answer = round(a / b, 2)
            reasoning = f"I need to divide {a} by {b}. {a} / {b} = {answer}."

        prompt = (
            "You are a reasoning assistant.\n\nSolve the following problem.\n\n"
            "Respond in exactly this format.\n\n<think>\nYour reasoning\n</think>\n\n"
            f"<answer>\nFinal answer\n</answer>\n\nQuestion: What is {a} {op} {b}?"
        )
        completion = f"<think>\n{reasoning}\n</think>\n\n<answer>\n{answer}\n</answer>"

        data.append({"prompt": prompt, "completion": completion, "answer": answer})
    return data


def format_and_tokenize(example, tokenizer, max_length=550):
    prompt = example["prompt"]
    completion = example["completion"]

    full_text = prompt + " " + completion + tokenizer.eos_token

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    # Match add_special_tokens with the full_text call so prompt_len lines up
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    prompt_len = len(prompt_tokens["input_ids"])

    labels = tokenized["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len

    # Mask padding too, so the model isn't trained to predict pad tokens
    for i, mask_val in enumerate(tokenized["attention_mask"]):
        if mask_val == 0:
            labels[i] = -100

    tokenized["labels"] = labels
    return tokenized


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    checkpoint_dir = "/home/godwinkhalko/LLMs/GRPO-sft-merged"  # used after training, to save/merge LoRA weights

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen base tokenizer may not set this by default

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)


    raw_data = generate_arithmetic_samples(num_samples=1000, ops=("+",))
    hf_dataset = HFDataset.from_list(raw_data)
    split = hf_dataset.train_test_split(test_size=0.1, seed=42)

    dataset = {"train": split["train"], "valid": split["test"]}

    tokenized_dataset = {
        "train": dataset["train"].map(
            lambda x: format_and_tokenize(x, tokenizer),
            remove_columns=dataset["train"].column_names,
            batched=False,
        ),
        "valid": dataset["valid"].map(
            lambda x: format_and_tokenize(x, tokenizer),
            remove_columns=dataset["valid"].column_names,
            batched=False,
        ),
    }

    tokenized_dataset["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_dataset["valid"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # --- Sanity check before training: confirm the label mask lands on the completion ---
    sample = tokenized_dataset["train"][0]
    visible_ids = [tid for tid, lab in zip(sample["input_ids"].tolist(), sample["labels"].tolist()) if lab != -100]
    print("Decoded label span (should start at <think> and contain only the completion):")
    print(tokenizer.decode(visible_ids))


    
    training_args = TrainingArguments(
        output_dir="./qwen-arithmetic-sft",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=100,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        data_collator=default_data_collator,  # labels already computed — don't let a collator overwrite them
    )

    trainer.train()

    # Save the LoRA adapter, then merge + save full model for use in GRPO
    trainer.model.save_pretrained(checkpoint_dir + "-adapter")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)