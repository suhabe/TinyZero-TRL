import pytest

from countdown import countdown_reward, make_prompt, score


@pytest.mark.parametrize(("completion", "numbers", "target", "expected"), [ ("<answer>(2 + 3) * 4</answer>", [2, 3, 4], 20, 1.0), ("<answer>2 + 3 + 4</answer>", [2, 3, 4], 20, 0.1), ("The answer is 20.", [2, 3, 4], 20, 0.0), ("<answer>2 + 2 + 4</answer>", [2, 3, 4], 8, 0.1), ("<answer>2 * 2 + 3</answer>", [2, 2, 3], 7, 1.0), ("<answer>6 / (3 - 1)</answer>", [6, 3, 1], 3, 1.0), ("<answer>-2 + 3 + 4</answer>", [2, 3, 4], 5, 1.0), ("<answer>2 / (3 - 3)</answer>", [2, 3, 3], 0, 0.1), ("<answer>2 ** 3</answer>", [2, 3], 8, 0.1), ("<answer>3 // 2</answer>", [3, 2], 1, 0.1), ("<answer>2.0 + 3</answer>", [2, 0, 3], 5, 0.1), ("<answer>abs(2 - 3)</answer>", [2, 3], 1, 0.1), ("<answer>(2 + 3</answer>", [2, 3], 5, 0.1), ("<answer>2 + 3 = 5</answer>", [2, 3, 5], 5, 0.1), ("<answer>\n2 + 3\n</answer>", [2, 3], 5, 1.0), ("<answer>2 - 3</answer>\n<answer>2 + 3</answer>", [2, 3], 5, 1.0), ])
def test_reward(completion: str, numbers: list[int], target: int, expected: float) -> None:
    assert score(completion, numbers, target) == expected


def test_reward_callback_keeps_each_completion_with_its_problem() -> None:
    rewards = countdown_reward(completions=["<answer>2 + 3</answer>", "<answer>2 + 3</answer>", "No answer"], nums=[[2, 3], [2, 3], [1, 2]], target=[5, 6, 3], prompts=["Ignored"] * 3)
    assert rewards == [1.0, 0.1, 0.0]


def test_base_prompt_contains_problem_and_prefills_reasoning() -> None:
    prompt = make_prompt({"nums": [2, 3, 4], "target": 20})["prompt"]
    assert "Using the numbers [2, 3, 4], create an equation that equals 20." in prompt
    assert "each number can only be used once" in prompt
    assert prompt.endswith("Assistant: Let me solve this step by step.\n<think>")
    assert "<|im_start|>" not in prompt
