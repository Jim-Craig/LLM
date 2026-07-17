import random

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset
import torch
import copy
from torch.optim.lr_scheduler import LinearLR
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

#Define a custom dataset class to handle the data. The data in this case is a simple arithmatic addition. 
# We'll randomly generate 2 numbers between 0-100 and generate the answer, in this case addition of the 2 numbers. 
# The prompt will be in the following format:
# You are a reasoning assistant.

# Solve the following problem.

# Respond in exactly this format.

# <think>
# Your reasoning
# </think>

# <answer>
# Final answer
# </answer>

# Question:

class ArthimaticAddition(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.data = []
        for _ in range(num_samples):
            a = torch.randint(0, 100, (1,)).item()
            b = torch.randint(0, 100, (1,)).item()
            answer = a + b
            prompt = f"You are a reasoning assistant.\n\nSolve the following problem.\n\nRespond in exactly this format.\n\n<think>\nYour reasoning\n</think>\n\n<answer>\nFinal answer\n</answer>\n\nQuestion: What is {a} + {b}?"
            self.data.append((prompt, answer))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


#Now we define a deterministic reward template function that will give a reward of 1 if the model's answer is in the correct format else 0. The correct format is as follows:
#<think>
#Your reasoning
#</think>
#<answer>
#Final answer
#</answer>

def reward_model(model_output, correct_answer, alpha=0.5):
    template_score = 0
    answer_score = 0
    # Check if the model output contains the correct format
    if "<think>" in model_output and "</think>" in model_output and "<answer>" in model_output and "</answer>" in model_output:
        #Ensure that the <think> and <answer> tags are in the correct order
        think_start = model_output.find("<think>")
        think_end = model_output.find("</think>")
        answer_start = model_output.find("<answer>")
        answer_end = model_output.find("</answer>")
        # print(think_start, think_end, answer_start, answer_end)
        #Make sure that the <think> and <answer> tags are in the correct order and output positvely starts with <think> and ends with </answer>
        if think_start == 0 and (think_start < think_end < answer_start < answer_end) and answer_end == len(model_output) - len("</answer>"):
            template_score = 1
        # Extract the answer from the model output
        model_answer = model_output[answer_start + len("<answer>"):answer_end].strip()
        
        # Check if the model's answer matches the correct answer
        if str(model_answer) == str(correct_answer):
            answer_score = 1  # Reward of 1 for correct format and correct answer
    return alpha * template_score + (1 - alpha) * answer_score  # Reward of 0 for incorrect format or incorrect answer

def GRPO_Loss(model_theta,model_old, model_ref, inputs, outputs, advantages, device="cuda:1", epsilon=0.2, beta=0.04):
    # Concatenate the prompt and label tokens for the model input
    prompt_len = inputs["input_ids"].shape[1]
    total_output = []
    for i, output in enumerate(outputs):
        output = output.unsqueeze(0).to(device)  # Add batch dimension and move to device
        tokens = {
            "input_ids": output,
            "attention_mask": torch.ones_like(output).to(device),
        }
        answer_len = output.shape[1] - prompt_len
        
         # Forward passes run in bf16 via autocast for memory/speed.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                model_ref_output = model_ref(**tokens)
                model_old_output = model_old(**tokens)
            model_theta_output = model_theta(**tokens)

        #Extract the logits corresponding to the answer tokens for each model
        answer_logits_theta = model_theta_output.logits[:, prompt_len - 1: prompt_len - 1 + answer_len, :].float()
        answer_logits_ref = model_ref_output.logits[:, prompt_len - 1: prompt_len - 1 + answer_len, :].float()
        answer_logits_old = model_old_output.logits[:, prompt_len - 1: prompt_len - 1 + answer_len, :].float()

        #Compute the probabilities for the answer tokens using softmax
        answer_log_probs_theta = F.log_softmax(answer_logits_theta, dim=-1)
        answer_log_probs_ref = F.log_softmax(answer_logits_ref, dim=-1)
        answer_log_probs_old = F.log_softmax(answer_logits_old, dim=-1)

        #Get the output tokens for the answer from the output.
        answer_tokens = output[:, prompt_len:].unsqueeze(-1)  # Shape: (batch_size, answer_len, 1)

        #Compute the probabilities of the answer tokens for each model
        answer_token_log_probs_theta = answer_log_probs_theta.gather(2, answer_tokens).squeeze(-1)
        answer_token_log_probs_ref = answer_log_probs_ref.gather(2, answer_tokens).squeeze(-1)
        answer_token_log_probs_old = answer_log_probs_old.gather(2, answer_tokens).squeeze(-1)

        # log-ratios computed in fp32, clamped before exp() to avoid overflow
        # if theta and old/ref diverge a lot early in training.
        log_ratio_ref_theta = torch.clamp(answer_token_log_probs_ref - answer_token_log_probs_theta, min=-20, max=20)
        log_ratio_theta_old = torch.clamp(answer_token_log_probs_theta - answer_token_log_probs_old, min=-20, max=20)


        #Get the probablity ratios of the answer tokens for the theta model with respect to the reference and old models
        ratio_ref_theta = torch.exp(log_ratio_ref_theta)
        ratio_theta_old = torch.exp(log_ratio_theta_old)

        #Copmute the GRPO CLIP value for ratio_theta_old
        clip = torch.clamp(
                ratio_theta_old,
                min=1 - epsilon,
                max=1 + epsilon
            ).to(device)
        
        #compute the minimum of the ratio_theta_ref and the clipped ratio_theta_old for each token
        policy_loss = torch.min(ratio_theta_old * advantages[i], clip * advantages[i])


        #Compute the KL divergence between the theta model and the reference model for the each answer tokens
        kl_divergence = ratio_ref_theta - torch.log(ratio_ref_theta) - 1

        #Sum the policy_loss and KL-Divergence
        output_sum = policy_loss - beta * kl_divergence
        total_output.append(output_sum.mean())  # Normalize by the length of the answer
    # Keep this as a torch stack/mean rather than np.mean over tensors —
    # np.mean() on a list of tensors with grad_fn will silently break
    # the autograd graph (it forces a numpy conversion under the hood
    # via __array__, detaching gradients). Use torch.stack().mean() instead.
    grpo_loss = -torch.stack(total_output).mean()

    
    return grpo_loss

def GRPO_Loop(model_old, model_ref, model_theta, dataset, tokenizer, optimizer, alpha=0.5, device="cuda:1", num_return_sequences=5, grpo_update_iters = 10):
    #Prompst and correct_answer would be of the dimension (batch_size,) for the data loader. The prompt is a string and the correct_answer is an integer.
    prompt, correct_answer = dataset[random.randint(0, len(dataset) - 1)]  # sample directly from dataset
    # Tokenize the prompt
    #inputs would be of the dimension (batch_size, seq_len) for the data loader. The prompt is a string and the correct_answer is an integer.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate model output from the old model as per the number of return sequences specified
    #outputs 
     # Forward passes run in bf16 via autocast for memory/speed.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            outputs = model_old.generate(input_ids=input_ids, 
                                        attention_mask=attention_mask,
                                        max_length=200, 
                                        num_return_sequences=num_return_sequences, 
                                        do_sample=True, 
                                        temperature=1.0)
    #Calculate the reward for each output based on the correct answer and the alpha value
    rewards = []
    model_outputs = []
    for output in outputs:
        model_output = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
        reward = reward_model(model_output, correct_answer, alpha)
        rewards.append(reward)
        model_outputs.append(model_output)

    #Advantage calculation
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    advantages = [(r - mean_reward) / (std_reward + 1e-8) for r in rewards]
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=grpo_update_iters)
    model_old.eval()
    model_ref.eval()
    model_theta.train()
    for i in range(grpo_update_iters):
        optimizer.zero_grad()
        grpo_loss = GRPO_Loss(model_theta, model_old, model_ref, inputs, outputs, advantages, device=device)
        print(f"GRPO Loss at update Number {i+1}: {grpo_loss.item()}")
        grpo_loss.backward()
        optimizer.step()
        scheduler.step()

    
    return model_theta

def GRPO_Training(model, epochs, tokenizer, num_iterations=10, alpha=0.5, num_return_sequences=5, grpo_update_iters=5, learning_rate=1e-5, device="cuda:1", ):
    dataset = ArthimaticAddition(num_samples=10000)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(epochs):
        model_ref = copy.deepcopy(model)
        model_ref.to(dtype = torch.bfloat16, device=device)
        print(f"Epoch {i+1}/{epochs}")
        for j in range(num_iterations):
            model_old = copy.deepcopy(model)
            model_old.to(dtype = torch.bfloat16, device=device)
            print(f"Iteration {j+1}/{num_iterations}")
            model = GRPO_Loop(model_old, model_ref, model, dataset, tokenizer, optimizer, alpha=alpha, device=device, num_return_sequences=num_return_sequences, grpo_update_iters=grpo_update_iters)
    del model_ref
    del model_old
    del dataset

    return model

if __name__ == "__main__":
    #Let's start with a base Quen 2.5 0.5B instruct model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model.to(device)

    grpo_model = GRPO_Training(model, epochs=1, tokenizer=tokenizer, num_iterations=2, alpha=0.5, num_return_sequences=2, grpo_update_iters=2, learning_rate=1e-5, device=device)

    del model
    del grpo_model
    del tokenizer
