from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset
import torch
import copy
from torch.optim.lr_scheduler import LinearLR
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0

#Since the preference dataset has a different structure, we need to format it similarly to the TL;DR dataset for training. The following function will help us achieve that.
def format_like_tldr(example):
    prompt = "SUBREDDIT: r/" + example.get("info", {}).get("subreddit", "") + " \nTITLE: " + example.get("info", {}).get("title", "") + " \nPOST: " + example.get("info", {}).get("post", "") + " \nTL;DR: "
    text1 = {"prompt": prompt, "completion": example.get("summaries", [{}])[0].get("text", "")}
    text2 = {"prompt": prompt, "completion": example.get("summaries", [{}])[1].get("text", "")}
    choice = example.get("choice", 0)
    return {"text1": text1, "text2": text2, "choice": choice}

#Let's create a Dataset from the preference dataset that is formatted like the TL;DR dataset. This will allow us to use the same training pipeline for both datasets.
class PreferenceDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        result_dict = format_like_tldr(example)
        return result_dict
    
def get_logProbsRatio(text, dpo_model, sft_model,tokenizer, device):
        # Tokenize the prompt and label separately
        tokens_prompt = tokenizer(text["prompt"], return_tensors="pt",padding=True, truncation=True).to(device)
        tokens_label = tokenizer(text["completion"], return_tensors="pt",padding=True, truncation=True).to(device)

        # Concatenate the prompt and label tokens for the model input
        tokens = {
        "input_ids":      torch.cat([tokens_prompt["input_ids"],      tokens_label["input_ids"]],      dim=1),
        "attention_mask": torch.cat([tokens_prompt["attention_mask"],  tokens_label["attention_mask"]], dim=1),
        }

        prompt_len = tokens_prompt["input_ids"].shape[1]
        label_len = tokens_label["input_ids"].shape[1]
        # Get the model's output logits for the label tokens given the prompt tokens for both the SFT and DPO models
        
        with torch.no_grad():
                output_sft = sft_model(**tokens)
        output_dpo = dpo_model(**tokens)

        # Extract the logits corresponding to the label tokens (i.e., the tokens after the prompt tokens)
        label_logits_sft = output_sft.logits[:, prompt_len - 1:prompt_len + label_len - 1, :]
        label_logits_dpo = output_dpo.logits[:, prompt_len - 1:prompt_len + label_len - 1, :]

        # Compute the probabilities of the label tokens for both models
        label_probs_sft = torch.log_softmax(label_logits_sft, dim=-1)
        label_probs_dpo = torch.log_softmax(label_logits_dpo, dim=-1)

        label_ids = tokens_label["input_ids"].unsqueeze(-1)
        label_probs_sft = label_probs_sft.gather(2, label_ids).squeeze(-1)
        label_probs_dpo = label_probs_dpo.gather(2, label_ids).squeeze(-1)

        #Compute the sequence probabilities by taking the product of the token probabilities
        label_mask = tokens_label["attention_mask"].bool()
        log_probs_sft = (label_probs_sft * label_mask).sum(dim=1)
        log_probs_dpo = (label_probs_dpo * label_mask).sum(dim=1)

        return log_probs_dpo - log_probs_sft 

def DPO_Loss(example, beta, dpo_model, sft_model, tokenizer, device):
    choice = example["choice"]
    log_probs_diff1 = get_logProbsRatio(example["text1"], dpo_model=dpo_model, sft_model=sft_model, tokenizer=tokenizer, device=device)
    log_probs_diff2 = get_logProbsRatio(example["text2"], dpo_model=dpo_model, sft_model=sft_model, tokenizer=tokenizer, device=device)
    sign = [1 if c == 0 else -1 for c in choice]
    loss = -torch.log(torch.sigmoid(beta * torch.tensor(sign, dtype=torch.float32).to(device) * (log_probs_diff1 - log_probs_diff2)))

    return loss.mean()

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

#Let's define the training loop for DPO. We will iterate through the preference dataset, compute the DPO loss for each example, and update the model parameters accordingly. 
# We will also evaluate the model on the validation set after each epoch to monitor its performance.
def train_dpo(sft_model, train_dataloader, validation_dataloader, tokenizer, device, beta=0.5, num_epochs=3):
    sft_model = sft_model.to(dtype=torch.bfloat16, device=device)
    dpo_model = copy.deepcopy(sft_model)

    # optimizer = torch.optim.RMSprop(dpo_model.parameters(), lr=1e-6)
    optimizer = torch.optim.AdamW(dpo_model.parameters(), lr=1e-6)
    scheduler = LinearLR(
    optimizer,
    start_factor=1e-9,   # effectively starts near 0 (1e-9 * 1e-6 ≈ 0)
    end_factor=1.0,       # ends at full lr (1.0 * 1e-6 = 1e-6)
    total_iters=150
    )
    #Since we don't want to update the SFT model during DPO training, we set it to evaluation mode and disable gradient updates for its parameters. The DPO model will be updated 
    # based on the computed loss and have disabled the batch normalization and dropout layers to ensure consistent behavior with the SFT model during training. This is important
    #  because the DPO loss relies on the outputs of both models, and we want to ensure that the SFT model's behavior remains stable throughout the training process.
    sft_model.eval()
    dpo_model.eval()  
    for param in sft_model.parameters():
        param.requires_grad = False
    val_loss_checkpoint = float('inf')
    for epoch in range(num_epochs):
        
        total_loss = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            loss = DPO_Loss(batch, beta, dpo_model, sft_model, tokenizer, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dpo_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss.append(loss.item())

        avg_loss = sum(total_loss) / len(total_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        # Validation loop
        with torch.no_grad():
            val_loss = []
            pbar = tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
            for batch_idx, batch in enumerate(pbar):
                loss = DPO_Loss(batch, beta, dpo_model, sft_model, tokenizer, device)
                val_loss.append(loss.item())
                
        avg_val_loss = sum(val_loss) / len(val_loss)
        if(avg_val_loss < val_loss_checkpoint):
            val_loss_checkpoint = avg_val_loss
            save_checkpoint(dpo_model, optimizer, scheduler, epoch, avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
    # Free up memory after training
    del optimizer
    del scheduler

    return dpo_model

if __name__ == "__main__":
    # Load the preference dataset
    preferences_dataset = load_dataset("openai/summarize_from_feedback", "comparisons")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("./qwen-tldr-sft-merged", torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("./qwen-tldr-sft-merged") 
    validation_dataset = [example for example in preferences_dataset["validation"] if example["info"]["site"] is None]
    
    train_pref_dataset = PreferenceDataset(preferences_dataset["train"])
    validation_pref_dataset = PreferenceDataset(validation_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_pref_dataset, batch_size=4, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_pref_dataset, batch_size=4)

    print("Starting DPO training...")
    dpo_model = train_dpo(model, train_dataloader, validation_dataloader, tokenizer, device)
    print("DPO training completed.")


    #Save the DPO model and tokenizer to a directory for later use
    dpo_model.save_pretrained("./qwen-tldr-dpo")
    tokenizer.save_pretrained("./qwen-tldr-dpo")


    del model
    del dpo_model
    del tokenizer
    del train_dataloader
    del validation_dataloader