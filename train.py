"""TinyZero's Countdown PPO recipe, rewritten as a minimal single-GPU program.

Prompt and PPO equations adapted from Jiayi-Pan/TinyZero (Apache-2.0).
Original algorithm notices: Copyright 2024 Bytedance Ltd. and/or its affiliates; Copyright 2022 The HuggingFace Team. All rights reserved.
"""

import argparse
import ast
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from datasets import load_dataset
from torch import Tensor, nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase


def prompt(numbers: list[int], target: int) -> str:
    return f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""


def evaluate(node: ast.AST) -> float:
    """Evaluate only arithmetic; generated answers are never executed as Python."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return float(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
        return float(operations[type(node.op)](evaluate(node.left), evaluate(node.right)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("Only numbers and +, -, *, / are allowed")


def countdown_reward(completion: str, numbers: list[int], target: int) -> float:
    # Upstream scores the final line's final answer, after the Assistant prefix.
    answers = re.findall(r"<answer>(.*?)</answer>", completion.split("\n")[-1])
    if not answers:
        return 0.0
    equation = answers[-1].strip()
    if sorted(map(int, re.findall(r"\d+", equation))) != sorted(numbers):
        return 0.1
    try:
        result = evaluate(ast.parse(equation, mode="eval").body)
        return 1.0 if abs(result - target) < 1e-5 else 0.1
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return 0.1


class Critic(nn.Module):
    def __init__(self, backbone: PreTrainedModel) -> None:
        super().__init__()
        self.backbone = backbone
        parameter = next(backbone.parameters())
        self.head = nn.Linear(backbone.config.hidden_size, 1, device=parameter.device, dtype=parameter.dtype)

    def forward(self, ids: Tensor, prompt_length: int) -> Tensor:
        with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16, enabled=ids.is_cuda):
            hidden = self.backbone(input_ids=ids, use_cache=False).last_hidden_state
            # The state BEFORE each sampled token predicts its future return.
            return cast(Tensor, self.head(hidden[:, prompt_length - 1:-1]).squeeze(0).squeeze(-1).float())


def policy_statistics(actor: PreTrainedModel, ids: Tensor, prompt_length: int) -> tuple[Tensor, Tensor]:
    with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16, enabled=ids.is_cuda):
        logits = actor(input_ids=ids, use_cache=False).logits[:, prompt_length - 1:-1].float().squeeze(0)
    log_distribution = logits.log_softmax(-1)
    log_probs = log_distribution.gather(-1, ids[0, prompt_length:].unsqueeze(-1)).squeeze(-1)
    entropy = -(log_distribution.exp() * log_distribution).sum(-1)
    return log_probs, entropy


def advantages_and_returns(rewards: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """GAE with gamma = lambda = 1, as in TinyZero's default PPO config."""
    advantage = torch.zeros_like(values[0])
    next_value = torch.zeros_like(values[0])
    advantages = torch.empty_like(values)
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + next_value - values[index]
        advantage = delta + advantage
        advantages[index] = advantage
        next_value = values[index]
    return advantages, advantages + values


def ppo_loss(log_probs: Tensor, old_log_probs: Tensor, advantages: Tensor) -> Tensor:
    ratio = (log_probs - old_log_probs).exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()


@dataclass
class Trajectory:
    ids: Tensor
    prompt_length: int
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    score: float


@torch.no_grad()
def rollout(actor: PreTrainedModel, reference: PreTrainedModel, critic: Critic, tokenizer: PreTrainedTokenizerBase, numbers: list[int], target: int, max_new_tokens: int) -> Trajectory:
    device = next(actor.parameters()).device
    inputs = tokenizer(prompt(numbers, target), return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(device)
    prompt_length = inputs.input_ids.shape[1]
    generation = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        ids = actor.generate(**inputs, generation_config=generation)
    completion = tokenizer.decode(ids[0, prompt_length:], skip_special_tokens=True)
    score = countdown_reward(completion, numbers, target)

    old_log_probs, _ = policy_statistics(actor, ids, prompt_length)
    ref_log_probs, _ = policy_statistics(reference, ids, prompt_length)
    old_values = critic(ids, prompt_length)
    rewards = -0.001 * (old_log_probs - ref_log_probs)
    rewards[-1] += score
    advantages, returns = advantages_and_returns(rewards, old_values)
    return Trajectory(ids, prompt_length, old_log_probs, old_values, advantages, returns, score)


def update(actor: PreTrainedModel, critic: Critic, trajectories: list[Trajectory], actor_optimizer: torch.optim.Optimizer, critic_optimizer: torch.optim.Optimizer, minibatch_size: int) -> None:
    all_advantages = torch.cat([trajectory.advantages for trajectory in trajectories])
    mean = all_advantages.mean()
    std = all_advantages.var(unbiased=all_advantages.numel() > 1).add(1e-8).sqrt()
    for trajectory in trajectories:
        trajectory.advantages = (trajectory.advantages - mean) / std

    # One PPO epoch. Accumulate one response at a time to save GPU memory.
    for start in range(0, len(trajectories), minibatch_size):
        minibatch = trajectories[start:start + minibatch_size]
        token_count = sum(trajectory.returns.numel() for trajectory in minibatch)
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        for trajectory in minibatch:
            weight = trajectory.returns.numel() / token_count
            log_probs, entropy = policy_statistics(actor, trajectory.ids, trajectory.prompt_length)
            actor_loss = ppo_loss(log_probs, trajectory.old_log_probs, trajectory.advantages) - 0.001 * entropy.mean()
            (weight * actor_loss).backward()

            values = critic(trajectory.ids, trajectory.prompt_length)
            clipped_values = trajectory.old_values + (values - trajectory.old_values).clamp(-0.5, 0.5)
            critic_loss = 0.5 * torch.maximum((values - trajectory.returns).square(), (clipped_values - trajectory.returns).square()).mean()
            (weight * critic_loss).backward()

        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        actor_optimizer.step()
        critic_optimizer.step()


def train(model_name: str, output_dir: str, epochs: int = 15, batch_size: int = 256, minibatch_size: int = 64, max_new_tokens: int = 1024, max_steps: int = 0) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # FP32 trainable weights preserve tiny Adam updates; forward passes use BF16 on CUDA.
    actor = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    reference = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    reference.requires_grad_(False)
    critic = Critic(AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device))
    actor.gradient_checkpointing_enable()
    critic.backbone.gradient_checkpointing_enable()
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6, weight_decay=0.01)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-5, weight_decay=0.01)

    # Preserve upstream's fixed train/test boundary; do not shuffle before splitting.
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    train_data = dataset.select(range(327680))
    validation_data = dataset.select(range(327680, 328704))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(epochs):
        data = train_data.shuffle(seed=42 + epoch)
        for start in range(0, len(data), batch_size):
            actor.eval()
            critic.eval()
            trajectories = [rollout(actor, reference, critic, tokenizer, row["nums"], row["target"], max_new_tokens) for row in data.select(range(start, min(start + batch_size, len(data))))]
            actor.train()
            critic.train()
            update(actor, critic, trajectories, actor_optimizer, critic_optimizer, minibatch_size)
            step += 1
            mean_reward = sum(trajectory.score for trajectory in trajectories) / len(trajectories)
            accuracy = sum(trajectory.score == 1.0 for trajectory in trajectories) / len(trajectories)
            print(f"step={step} epoch={epoch + 1} reward={mean_reward:.4f} accuracy={accuracy:.4f}", flush=True)

            if step % 100 == 0:
                actor.eval()
                scores = [rollout(actor, reference, critic, tokenizer, row["nums"], row["target"], max_new_tokens).score for row in validation_data]
                print(f"validation_accuracy={sum(score == 1.0 for score in scores) / len(scores):.4f}", flush=True)
                actor.save_pretrained(output / f"step-{step}")
                tokenizer.save_pretrained(output / f"step-{step}")
            if max_steps and step >= max_steps:
                actor.save_pretrained(output / "final")
                tokenizer.save_pretrained(output / "final")
                return

    actor.save_pretrained(output / "final")
    tokenizer.save_pretrained(output / "final")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs all epochs")
    args = parser.parse_args()
    train(args.model, args.output_dir, args.epochs, args.batch_size, args.minibatch_size, args.max_new_tokens, args.max_steps)


if __name__ == "__main__":
    main()
