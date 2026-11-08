"""TinyZero's base-model Countdown prompt and rule-based reward."""

import ast
import re
from typing import TypedDict


class CountdownRow(TypedDict):
    nums: list[int]
    target: int


def make_prompt(row: CountdownRow) -> dict[str, str]:
    prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {row["nums"]}, create an equation that equals {row["target"]}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    return {"prompt": prompt}


def evaluate(node: ast.AST) -> float:
    """Interpret the four arithmetic operators without executing generated code."""
    match node:
        case ast.Constant(value=value) if type(value) is int:
            return float(value)
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -evaluate(operand)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return evaluate(operand)
        case ast.BinOp(left=left, op=operation, right=right):
            a, b = evaluate(left), evaluate(right)
            match operation:
                case ast.Add():
                    return a + b
                case ast.Sub():
                    return a - b
                case ast.Mult():
                    return a * b
                case ast.Div():
                    return a / b
    raise ValueError("Use integer numbers and +, -, *, / only.")


def score(completion: str, numbers: list[int], target: int) -> float:
    """Give 0 for missing tags, 0.1 for a tagged wrong answer, and 1 for a solution."""
    answers = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
    if not answers:
        return 0.0

    equation = answers[-1].strip()
    if not re.fullmatch(r"[0-9+\-*/()\s]+", equation):
        return 0.1
    if sorted(map(int, re.findall(r"\d+", equation))) != sorted(numbers):
        return 0.1

    try:
        result = evaluate(ast.parse(equation, mode="eval").body)
    except (ArithmeticError, SyntaxError, ValueError, RecursionError):
        return 0.1
    return 1.0 if abs(result - target) < 1e-5 else 0.1


def countdown_reward(completions: list[str], nums: list[list[int]], target: list[int], **kwargs: object) -> list[float]:
    """TRL repeats each problem's metadata alongside its sampled completions."""
    return [score(completion, numbers, answer) for completion, numbers, answer in zip(completions, nums, target, strict=True)]
