"""Minimal TinyZero-style Countdown RL using TRL GRPO and a PEFT LoRA adapter."""

import argparse
from typing import cast

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed
from trl import GRPOConfig, GRPOTrainer

from countdown import countdown_reward, make_prompt


MODEL = "Qwen/Qwen2.5-3B"
TRAIN_SIZE = 327680


def create_trainer(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, train_dataset: Dataset, eval_dataset: Dataset, config: GRPOConfig) -> GRPOTrainer:
    adapter = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    return GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=countdown_reward, peft_config=adapter, args=config, train_dataset=train_dataset, eval_dataset=eval_dataset)


def train(output_dir: str = "outputs/countdown", max_steps: int = 1000, max_completion_length: int = 1024, eval_size: int = 1024, resume_from_checkpoint: str | None = None) -> None:
    set_seed(42)

    raw = cast(Dataset, load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train"))
    train_dataset = raw.select(range(TRAIN_SIZE)).map(make_prompt)
    eval_dataset = raw.select(range(TRAIN_SIZE, TRAIN_SIZE + eval_size)).map(make_prompt)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")

    # Eight completions of one prompt per update; microbatch one keeps training memory small.
    config = GRPOConfig(output_dir=output_dir, learning_rate=1e-6, lr_scheduler_type="constant", num_train_epochs=15, max_steps=max_steps, per_device_train_batch_size=1, gradient_accumulation_steps=8, per_device_eval_batch_size=8, num_generations=8, max_prompt_length=256, max_completion_length=max_completion_length, temperature=1.0, beta=0.001, loss_type="grpo", scale_rewards="group", bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, eval_strategy="steps", eval_steps=100, save_steps=100, save_total_limit=2, report_to="none", remove_unused_columns=False, seed=42)
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, config)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_model(f"{output_dir}/adapter")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/countdown")
    parser.add_argument("--max-steps", type=int, default=1000, help="Optimizer updates; -1 runs all 15 epochs.")
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    train(output_dir=args.output_dir, max_steps=args.max_steps, max_completion_length=args.max_completion_length, eval_size=args.eval_size, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
