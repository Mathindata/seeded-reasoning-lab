"""Benchmarks and answer checking.

MATH-500 (HuggingFaceH4/MATH-500): competition math with a final answer; checked with
math-verify, which compares answers as math (0.5 == 1/2), not as strings.
Game of 24: four numbers, use each once with + - * / to make 24. Instances are
generated and solved here, so no download is needed, and checking is exact.
"""
from __future__ import annotations

import ast
import itertools
import random
import re
from fractions import Fraction


# ---------------- MATH-500 ----------------
def load_math500(n: int, levels: list[int], seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [r for r in ds if int(r["level"]) in set(levels)]
    rng = random.Random(seed)
    # stratified by level, so every level is represented
    by_level: dict[int, list] = {}
    for r in rows:
        by_level.setdefault(int(r["level"]), []).append(r)
    out = []
    per = max(1, n // len(by_level))
    for lv in sorted(by_level):
        pool = by_level[lv]
        out += rng.sample(pool, min(per, len(pool)))
    return [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"],
             "level": int(r["level"]), "subject": r["subject"], "benchmark": "math500"}
            for r in out[:n]]


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k].strip()
        k += 1
    return None


def check_math(response: str, gold: str) -> bool:
    try:
        from math_verify import parse, verify
        return bool(verify(parse(f"\\boxed{{{gold}}}"), parse(response)))
    except Exception:
        pred = last_boxed(response)
        norm = lambda s: re.sub(r"\s|\\left|\\right|\\!|\$", "", s or "")  # noqa: E731
        return pred is not None and norm(pred) == norm(gold)


# ---------------- Game of 24 ----------------
def _solve24(nums: list[Fraction]) -> bool:
    if len(nums) == 1:
        return nums[0] == 24
    for i, j in itertools.permutations(range(len(nums)), 2):
        rest = [nums[k] for k in range(len(nums)) if k not in (i, j)]
        a, b = nums[i], nums[j]
        cands = [a + b, a - b, a * b] + ([a / b] if b != 0 else [])
        if any(_solve24(rest + [c]) for c in cands):
            return True
    return False


def make_game24(n: int, seed: int) -> list[dict]:
    combos = [c for c in itertools.combinations_with_replacement(range(1, 14), 4)
              if _solve24([Fraction(x) for x in c])]
    rng = random.Random(seed)
    pick = rng.sample(combos, min(n, len(combos)))
    return [{"id": "g24-" + "-".join(map(str, c)),
             "problem": (f"Use the numbers {', '.join(map(str, c))} exactly once each, with "
                         "+, -, *, / and parentheses, to make 24. Give the final expression "
                         "inside \\boxed{}."),
             "answer": list(c), "level": 0, "subject": "game24", "benchmark": "game24"}
            for c in pick]


_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.USub)


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return Fraction(node.value), [node.value]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v, used = _eval(node.operand)
        return -v, used
    if isinstance(node, ast.BinOp):
        a, ua = _eval(node.left)
        b, ub = _eval(node.right)
        op = type(node.op)
        if op is ast.Add:
            return a + b, ua + ub
        if op is ast.Sub:
            return a - b, ua + ub
        if op is ast.Mult:
            return a * b, ua + ub
        if op is ast.Div:
            if b == 0:
                raise ZeroDivisionError
            return a / b, ua + ub
    raise ValueError("not allowed")


def check_game24(response: str, numbers: list[int]) -> bool:
    expr = last_boxed(response)
    if not expr:
        return False
    expr = expr.replace("\\times", "*").replace("\\cdot", "*").replace("×", "*") \
               .replace("\\div", "/").replace("÷", "/").replace("−", "-")
    expr = expr.split("=")[0]
    try:
        tree = ast.parse(expr, mode="eval")
        if not all(isinstance(n, _ALLOWED) for n in ast.walk(tree)):
            return False
        val, used = _eval(tree)
    except Exception:
        return False
    return val == 24 and sorted(used) == sorted(numbers)


def check(item: dict, response: str) -> bool:
    if item["benchmark"] == "game24":
        return check_game24(response, item["answer"])
    return check_math(response, item["answer"])


def extract(item: dict, response: str) -> str:
    """A normalized final answer, used for majority voting and diversity counts."""
    b = last_boxed(response)
    return re.sub(r"\s", "", b) if b else ""
