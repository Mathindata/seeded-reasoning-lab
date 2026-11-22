"""Experimental conditions and the prompts they produce.

A condition = (seed bank, whether the seed is first translated into a strategy).

  plain            the problem alone (baseline)
  noise_translated random words -> one-sentence strategy -> solve   (control)
  polya_translated math heuristic -> strategy -> solve              (same domain)
  near_translated  physics/engineering/CS idea -> strategy -> solve (near domain)
  far_translated   philosophy/arts/nature idea -> strategy -> solve (far domain)
  far_raw          far idea pasted next to the problem, no translation

Conditions 2 to 5 differ only in the seed bank, so any difference among them is
about the source of the idea, not about the extra step or extra tokens.
"""
from __future__ import annotations

SOLVE_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
COACH_SYSTEM = "You are a creative problem-solving coach. You never solve the problem yourself."

CONDITIONS = {
    "plain": {"bank": None, "translate": False},
    "noise_translated": {"bank": "noise", "translate": True},
    "polya_translated": {"bank": "polya", "translate": True},
    "near_translated": {"bank": "near", "translate": True},
    "far_translated": {"bank": "far", "translate": True},
    "far_raw": {"bank": "far", "translate": False},
}


def translate_messages(problem: str, domain: str, seed: str) -> list[dict]:
    user = (f"Here is a problem:\n{problem}\n\n"
            f"Here is an idea from {domain}: \"{seed}\"\n\n"
            "In one or two sentences, describe a general problem-solving move that this idea "
            "suggests for this problem. Do not solve the problem and do not compute anything.")
    return [{"role": "system", "content": COACH_SYSTEM}, {"role": "user", "content": user}]


def solve_messages(problem: str, condition: str, domain: str = "", seed: str = "",
                   strategy: str = "") -> list[dict]:
    if condition == "plain":
        user = problem
    elif condition == "far_raw":
        user = f"{problem}\n\nA thought from {domain} that you may find useful: \"{seed}\""
    else:
        user = f"{problem}\n\nA strategy you may find useful: {strategy.strip()}"
    return [{"role": "system", "content": SOLVE_SYSTEM}, {"role": "user", "content": user}]


def clean_strategy(text: str, max_chars: int = 400) -> str:
    """Keep the coach's first two sentences; drop anything that looks like a solution."""
    t = " ".join(text.strip().split())
    cut = t.find("\\boxed")
    if cut >= 0:
        t = t[:cut]
    parts = t.replace("? ", "?|").replace(". ", ".|").split("|")
    return " ".join(parts[:2])[:max_chars].strip()
