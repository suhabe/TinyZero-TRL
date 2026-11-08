from typing import TypedDict


class CountdownRow(TypedDict):
    nums: list[int]
    target: int


def make_prompt(row: CountdownRow) -> dict[str, str]:
    raise NotImplementedError


def score(completion: str, numbers: list[int], target: int) -> float:
    raise NotImplementedError


def countdown_reward(completions: list[str], nums: list[list[int]], target: list[int], **kwargs: object) -> list[float]:
    raise NotImplementedError
