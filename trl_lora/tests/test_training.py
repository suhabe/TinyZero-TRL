from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, set_seed
from trl import GRPOConfig

from countdown import make_prompt
from train import create_trainer


def test_grpo_updates_only_lora_and_saves_adapter(tmp_path: Path) -> None:
    set_seed(42)
    torch.set_num_threads(1)

    vocabulary = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, **{str(i): i + 3 for i in range(29)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]", padding_side="left", model_input_names=["input_ids", "attention_mask"])

    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512, pad_token_id=0, eos_token_id=1, bos_token_id=1))
    dataset = Dataset.from_dict({"nums": [[2, 3], [3, 4]], "target": [5, 7]}).map(make_prompt)
    config = GRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, per_device_train_batch_size=1, gradient_accumulation_steps=2, per_device_eval_batch_size=2, num_generations=2, max_completion_length=4, max_steps=1, learning_rate=0.01, beta=0.001, loss_type="grpo", scale_rewards="group", gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, report_to="none", save_strategy="no", disable_tqdm=True)
    trainer = create_trainer(model, tokenizer, dataset, dataset, config)

    assert isinstance(trainer.model, PeftModel)
    trainable = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
    frozen = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters() if not parameter.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    assert trainer.ref_model is None  # TRL disables the adapter to evaluate the frozen base policy.

    # Give the two sampled completions different rewards so this tests an actual optimizer update.
    def alternating_reward(completions: list[str], **kwargs: object) -> list[float]:
        return [float(index % 2) for index in range(len(completions))]

    trainer.reward_funcs = [alternating_reward]
    trainer.reward_func_names = ["alternating_reward"]
    trainer.train()

    parameters = dict(trainer.model.named_parameters())
    assert trainer.state.global_step == 1
    assert any(not torch.equal(initial, parameters[name]) for name, initial in trainable.items())
    assert all(torch.equal(initial, parameters[name]) for name, initial in frozen.items())
    assert torch.isfinite(torch.tensor(trainer.evaluate()["eval_loss"]))

    trainer.save_model(str(tmp_path / "adapter"))
    assert (tmp_path / "adapter" / "adapter_config.json").is_file()
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()
    assert (tmp_path / "adapter" / "tokenizer.json").is_file()
