from pathlib import Path

import pytest
import torch
from torch import Tensor
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model

from train import Critic, Trajectory, advantages_and_returns, countdown_reward, policy_statistics, ppo_loss, update


@pytest.mark.parametrize(('completion', 'numbers', 'target', 'expected'), [('<answer>(3 + 4) * 2</answer>', [3, 4, 2], 14, 1.0), ('<answer>3 + 4 + 2</answer>', [3, 4, 2], 14, 0.1), ('<answer>14</answer>', [3, 4, 2], 14, 0.1), ('no answer', [3, 4, 2], 14, 0.0), ('<answer>2 ** 3</answer>', [2, 3], 8, 0.1), ('<answer>3 / (2 - 2)</answer>', [3, 2, 2], 5, 0.1), ('<answer>2 + 2</answer>', [2, 2], 4, 1.0), ('<answer>2</answer>', [2, 2], 4, 0.1)])
def test_countdown_reward(completion: str, numbers: list[int], target: int, expected: float) -> None:
    assert countdown_reward(completion, numbers, target) == expected


def test_gae_uses_terminal_reward_and_token_kl() -> None:
    rewards = torch.tensor([-0.1, -0.2, 1.0])
    values = torch.tensor([0.2, 0.3, 0.4])
    advantages, returns = advantages_and_returns(rewards, values)
    torch.testing.assert_close(returns, torch.tensor([0.7, 0.8, 1.0]))
    torch.testing.assert_close(advantages, torch.tensor([0.5, 0.5, 0.6]))


def test_ppo_clips_both_signs() -> None:
    old = torch.zeros(2)
    current = torch.tensor([2.0, 0.5]).log().requires_grad_()
    loss = ppo_loss(current, old, torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(loss, torch.tensor(-0.2))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros(2))


def tiny_models() -> tuple[Qwen2ForCausalLM, Critic]:
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    return Qwen2ForCausalLM(config), Critic(Qwen2Model(config))


def test_log_probs_align_with_response_tokens() -> None:
    actor, _ = tiny_models()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    log_probs, entropy = policy_statistics(actor, ids, 3)
    logits = actor(ids).logits[:, 2:-1].float()
    expected = logits.log_softmax(-1).gather(-1, ids[:, 3:].unsqueeze(-1)).squeeze()
    torch.testing.assert_close(log_probs, expected)
    assert entropy.shape == (2,)


def test_real_ppo_update_changes_actor_and_critic(tmp_path: Path) -> None:
    torch.manual_seed(42)
    actor, critic = tiny_models()
    trajectories: list[Trajectory] = []
    for length, score in [(5, 1.0), (7, 0.1)]:
        ids = torch.randint(0, 32, (1, length))
        with torch.no_grad():
            old_log_probs, _ = policy_statistics(actor, ids, 3)
            old_values = critic(ids, 3)
            rewards = torch.zeros(length - 3)
            rewards[-1] = score
            advantages, returns = advantages_and_returns(rewards, old_values)
        trajectories.append(Trajectory(ids, 3, old_log_probs, old_values, advantages, returns, score))
    before_actor = actor.get_input_embeddings().weight.detach().clone()
    before_critic = critic.head.weight.detach().clone()
    update(actor, critic, trajectories, torch.optim.AdamW(actor.parameters(), lr=1e-3), torch.optim.AdamW(critic.parameters(), lr=1e-3), 2)
    assert not torch.equal(before_actor, actor.get_input_embeddings().weight)
    assert not torch.equal(before_critic, critic.head.weight)
    assert all(torch.isfinite(parameter).all() for parameter in actor.parameters())
    actor.save_pretrained(tmp_path)
    reloaded = Qwen2ForCausalLM.from_pretrained(tmp_path)
    torch.testing.assert_close(actor.get_input_embeddings().weight, reloaded.get_input_embeddings().weight)


def test_full_training_loop_exports_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    import train

    actor, _ = tiny_models()
    model_dir = tmp_path / "model"
    actor.save_pretrained(model_dir)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1}, unk_token="<unk>")), unk_token="<unk>", eos_token="<eos>")
    tokenizer.save_pretrained(model_dir)
    dataset = Dataset.from_dict({"nums": [[2, 3]] * 328704, "target": [5] * 328704})
    monkeypatch.setattr(train, "load_dataset", lambda *args, **kwargs: dataset)
    output_dir = tmp_path / "output"
    train.train(str(model_dir), str(output_dir), epochs=1, batch_size=2, minibatch_size=2, max_new_tokens=2, max_steps=1)
    restored = Qwen2ForCausalLM.from_pretrained(output_dir / "final")
    assert restored.config.vocab_size == 32
    assert (output_dir / "final" / "tokenizer.json").exists()
