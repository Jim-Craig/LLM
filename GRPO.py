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
import re
import random
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # harmless even if val_dataset generation is on CPU

class MultiOperandArithmetic(Dataset):
    """4 random integers, 3 random operators (+, -, *; no division), with a
    randomly chosen bracket pattern. Intended for GRPO — only prompt + ground
    truth answer are produced; no reasoning trace is templated, since the
    point is to let the model discover correct precedence handling via
    reward rather than imitate a scripted solution."""

    OPS = ("+", "-", "*")

    # A handful of valid parenthesization patterns for 4 operands / 3 ops.
    # {a},{b},{c},{d} = operands, {op1},{op2},{op3} = operators, in order.
    BRACKET_PATTERNS = [
        "{a} {op1} {b} {op2} {c} {op3} {d}",              # no brackets
        "({a} {op1} {b}) {op2} {c} {op3} {d}",
        "{a} {op1} ({b} {op2} {c}) {op3} {d}",
        "{a} {op1} {b} {op2} ({c} {op3} {d})",
        "({a} {op1} {b} {op2} {c}) {op3} {d}",
        "{a} {op1} ({b} {op2} {c} {op3} {d})",
        "({a} {op1} {b}) {op2} ({c} {op3} {d})",
        "(({a} {op1} {b}) {op2} {c}) {op3} {d}",
        "{a} {op1} (({b} {op2} {c}) {op3} {d})",
    ]

    def __init__(self, num_samples=1000, low=0, high=20):
        self.num_samples = num_samples
        self.data = []
        for _ in range(num_samples):
            nums = [torch.randint(low, high, (1,)).item() for _ in range(4)]
            ops = [random.choice(self.OPS) for _ in range(3)]
            pattern = random.choice(self.BRACKET_PATTERNS)

            expr = pattern.format(
                a=nums[0], b=nums[1], c=nums[2], d=nums[3],
                op1=ops[0], op2=ops[1], op3=ops[2],
            )

            # Ground truth computed from the exact same string the model sees,
            # so precedence/bracket handling is guaranteed self-consistent —
            # never hand-derived separately from the displayed expression.
            answer = eval(expr)

            prompt = (
                "You are a reasoning assistant.\n\nSolve the following problem.\n\n"
                "Respond in exactly this format.\n\n<think>\nYour reasoning\n</think>\n\n"
                f"<answer>\nFinal answer\n</answer>\n\nQuestion: What is {expr}?"
            )

            self.data.append({"prompt": prompt, "answer": answer, "expression": expr})

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["prompt"], item["answer"]


#Now we define a deterministic reward template function that will give a reward of 1 if the model's answer is in the correct format else 0. The correct format is as follows:
#<think>
#Your reasoning
#</think>
#<answer>
#Final answer
#</answer>

def reward_model(model_output, correct_answer):
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
        stripped = model_output.lstrip()
        if stripped.startswith("<think>"):
            template_score += 0.25
        if think_start < think_end < answer_start < answer_end:
            template_score +=0.25  
        between_think_and_answer = model_output[think_end + len("</think>"):answer_start]
        if think_start == 0 and re.fullmatch(r"\s*", between_think_and_answer):
            template_score += 0.25
        if answer_end == len(model_output) - len("</answer>"):
            template_score += 0.25  
        
        model_answer_raw = model_output[answer_start + len("<answer>"):answer_end].strip()
        try:
            model_answer = int(model_answer_raw)
            answer_score = max(0, 1 - abs(model_answer - correct_answer) / (abs(correct_answer) + 1))
        except ValueError:
            answer_score = 0  # not a clean bare integer — correctly penalized, format discipline not yet learned
            print(model_answer)
    # print(f"Model Output: {model_output} | Template Score: {template_score} | Answer Score: {answer_score}")
    return template_score, answer_score # Return both scores

def reward_model_binary(model_output, correct_answer):
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
        stripped = model_output.lstrip()
        between_think_and_answer = model_output[think_end + len("</think>"):answer_start]
        if (stripped.startswith("<think>") 
        and (think_start < think_end < answer_start < answer_end) 
        and (think_start == 0 and re.fullmatch(r"\s*", between_think_and_answer))
        and (answer_end == len(model_output) - len("</answer>"))):
            template_score = 1.0
        
        model_answer_raw = model_output[answer_start + len("<answer>"):answer_end].strip()
        try:
            model_answer = int(model_answer_raw)
            answer_score = max(0, 1 - abs(model_answer - correct_answer) / (abs(correct_answer) + 1))
        except ValueError:
            answer_score = 0  # not a clean bare integer — correctly penalized, format discipline not yet learned
            print(model_answer)
    # print(f"Model Output: {model_output} | Template Score: {template_score} | Answer Score: {answer_score}")
    return template_score, answer_score # Return both scores

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

def GRPO_Loop(model_old, model_ref, model_theta, dataset, tokenizer, optimizer, alpha=0.5, device="cuda:0", num_return_sequences=5, grpo_update_iters = 10):
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
        template_score, answer_score = reward_model_binary(model_output, correct_answer, alpha)
        reward = alpha * template_score + (1 - alpha) * answer_score
        rewards.append(reward)
        model_outputs.append(model_output)

    #Advantage calculation
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    advantages = [(r - mean_reward) / (std_reward + 1e-8) for r in rewards]

    print(f"rewards: {rewards}")
    print(f"advantages: {advantages}")
    # scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=grpo_update_iters)
    model_old.eval()
    model_ref.eval()
    model_theta.train()
    for i in range(grpo_update_iters):
        optimizer.zero_grad()
        grpo_loss = GRPO_Loss(model_theta, model_old, model_ref, inputs, outputs, advantages, device=device)
        grpo_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_theta.parameters(), max_norm=1.0)
        grad_norm = sum(p.grad.norm().item() for p in model_theta.parameters() if p.grad is not None)
        # print(f"GRPO Loss at update {i+1}: {grpo_loss.item()}  |  grad_norm: {grad_norm}")
        optimizer.step()
        # scheduler.step()

    
    return model_theta

def GRPO_Training(model, epochs, tokenizer, num_iterations=10, alpha=0.5, num_return_sequences=5, grpo_update_iters=5, learning_rate=1e-5, device="cuda:0", ):
    dataset = MultiOperandArithmetic(num_samples=10000)
    set_seed(42)  # fix the RNG state right before building val_dataset
    val_dataset = MultiOperandArithmetic(num_samples=10)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    set_seed(random.randint(0, 2**31))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    validation_rewards, template_scores, answer_scores = [], [], []
    best_model = copy.deepcopy(model)
    best_validation_reward = -float("inf")
    for i in range(epochs):
        model_ref = copy.deepcopy(best_model)
        model_ref.to(dtype = torch.bfloat16, device=device)
        print(f"Epoch {i+1}/{epochs}")
        for j in range(num_iterations):
            model_old = copy.deepcopy(model)
            model_old.to(dtype = torch.bfloat16, device=device)
            print(f"Iteration {j+1}/{num_iterations}")
            model = GRPO_Loop(model_old, model_ref, model, dataset, tokenizer, optimizer, alpha=alpha, device=device, num_return_sequences=num_return_sequences, grpo_update_iters=grpo_update_iters)
            #evaluate the model on the validation dataset and print the average reward for the validation dataset
            val_rewards = []
            template_scores, answer_scores = [], []
            for val_prompt, val_correct_answer in val_dataset:
                val_inputs = tokenizer(val_prompt, return_tensors="pt").to(device)
                val_input_ids = val_inputs["input_ids"]
                val_attention_mask = val_inputs["attention_mask"]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        val_outputs = model.generate(input_ids=val_input_ids, 
                                                    attention_mask=val_attention_mask,
                                                    max_length=200, 
                                                    num_return_sequences=num_return_sequences, 
                                                    do_sample=True, 
                                                    temperature=1.0)
                for val_output in val_outputs:
                    val_model_output = tokenizer.decode(val_output[val_input_ids.shape[1]:], skip_special_tokens=True)
                    template_score, answer_score = reward_model_binary(val_model_output, val_correct_answer, alpha)
                    template_scores.append(template_score)
                    answer_scores.append(answer_score)
                    reward = alpha * template_score + (1 - alpha) * answer_score
                    val_rewards.append(reward)
            avg_template_score = np.mean(template_scores)
            avg_answer_score = np.mean(answer_scores)
            avg_val_reward = np.mean(val_rewards)
            print(f"Average Validation Template Score after iteration {j+1}: {avg_template_score}")
            print(f"Average Validation Answer Score after iteration {j+1}: {avg_answer_score}")
            print(f"Average Validation Reward after iteration {j+1}: {avg_val_reward}")
            validation_rewards.append(avg_val_reward)
            template_scores.append(avg_template_score)
            answer_scores.append(avg_answer_score)
            if np.mean(val_rewards) > best_validation_reward:
                best_validation_reward = np.mean(val_rewards)
                best_model = copy.deepcopy(model)
                checkpoint_path = "/home/godwinkhalko/LLMs/GRPO-checkpoint"
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                best_model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
        if i == 0:
            alpha = 0.3

    del model_ref
    del model_old
    del dataset
    del val_dataset
    return best_model, validation_rewards, template_scores, answer_scores

if __name__ == "__main__":
    #Let's start with a base Quen 2.5 0.5B instruct model
    checkpoint_path = "/home/godwinkhalko/LLMs/GRPO-sft-merged"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    grpo_model, validation_rewards, template_scores, answer_scores = GRPO_Training(model, epochs=1, tokenizer=tokenizer, num_iterations=30, alpha=0.5, num_return_sequences=8, grpo_update_iters=3, learning_rate=1e-5, device=device)

    #Compute the graph of the validation rewards over the iterations and save it as a png file
    plt.plot(validation_rewards)
    plt.xlabel("Iterations")
    plt.ylabel("Average Validation Reward")
    plt.title("Validation Reward over Iterations")
    plt.savefig("/home/godwinkhalko/LLMs/validation_rewards.png")

    #Compute the graph of the template scores over the iterations and save it as a png file
    plt.plot(template_scores)
    plt.xlabel("Iterations")
    plt.ylabel("Average Validation Template Score")
    plt.title("Validation Template Score over Iterations")
    plt.savefig("/home/godwinkhalko/LLMs/template_scores.png")

    #Compute the graph of the answer scores over the iterations and save it as a png file
    plt.plot(answer_scores)
    plt.xlabel("Iterations")
    plt.ylabel("Average Validation Answer Score")
    plt.title("Validation Answer Score over Iterations")
    plt.savefig("/home/godwinkhalko/LLMs/answer_scores.png")

    #Save the model after training
    save_path = "/home/godwinkhalko/LLMs/GRPO"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    grpo_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    del model
    del grpo_model
    del tokenizer
