"""Phase 2: confidence-triggered strategy injection during reasoning.

The answer is written in segments of `seg_tokens` tokens. After each segment the
model's confidence on it is measured as the mean log-probability of its tokens:
  conf = (1/n) * sum_i log p(token_i | everything before it)
A value near 0 means the model was sure of each token; very negative means unsure.

Trigger rule with hysteresis (like a Schmitt trigger, so it does not chatter):
inject only after `m` consecutive segments with conf < tau, at most `max_injections`
times per answer, and not again within `cooldown` segments of the last injection.

Pairing (common random numbers): segment s of sample j of problem p uses the same
random seed in every condition. Until the first injection, every condition therefore
writes the same text, so the difference in correctness after it is caused by the
injection itself. In signal-processing terms: the same noise realization is fed to
every system, and the outputs are compared sample by sample.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from . import data, prompts, seeds

INTRO = "\n\nWait, this might not be the right path. Let me rethink the problem"
CONDITIONS2 = {
    # name: (when to inject, what to inject)
    "seg_plain": ("never", None),
    "ctrl_generic": ("confidence", None),
    "ctrl_noise": ("confidence", "noise"),
    "ctrl_near": ("confidence", "near"),
    "ctrl_far": ("confidence", "far"),
    "rand_near": ("random", "near"),
}


@dataclass
class Rule:
    tau: float
    m: int = 2
    max_injections: int = 2
    cooldown: int = 2


@dataclass
class TriggerState:
    low_streak: int = 0
    injections: int = 0
    since_last: int = 10 ** 6

    def eligible(self, rule: Rule) -> bool:
        return self.injections < rule.max_injections and self.since_last >= rule.cooldown

    def step(self, conf: float, rule: Rule) -> bool:
        """Feed one segment's confidence; return True if the rule fires now."""
        self.since_last += 1
        self.low_streak = self.low_streak + 1 if conf < rule.tau else 0
        if self.low_streak >= rule.m and self.eligible(rule):
            self.fire()
            return True
        return False

    def fire(self) -> None:
        self.injections += 1
        self.since_last = 0
        self.low_streak = 0


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def segment_seed(base: int, pid: str, sample: int, segment: int) -> int:
    """Same value in every condition. Layout: (pair id) * 1000 + segment."""
    h = int(hashlib.sha256(f"{base}|{pid}|{sample}".encode()).hexdigest()[:8], 16) % 10 ** 6
    return h * 1000 + segment


def injection_text(kind: str | None, strategy: str = "", mock: bool = False) -> str:
    tag = f" [RETHINK:{kind or 'generic'}]" if mock else ""
    if kind is None:
        return INTRO + "." + tag + "\n\n"
    return INTRO + f" with a different strategy: {strategy}" + tag + "\n\n"


def trim_to_line(text: str, max_cut: int = 300) -> str:
    """Cut a half-written line before injecting, so the injection starts on a fresh line."""
    i = text.rfind("\n")
    return text[:i] if i >= 0 and len(text) - i <= max_cut else text


@dataclass
class Job:
    pid: str
    condition: str
    sample: int
    text: str = ""
    gen_tokens: int = 0
    done: bool = False
    segments: list = field(default_factory=list)      # dicts: seg, n, conf, finished
    injections: list = field(default_factory=list)    # dicts: seg, kind, domain, seed, strategy
    would_trigger: list = field(default_factory=list)  # seg indices where the rule would fire
    prefix_at_first_injection: str = ""
    seg_hashes: list = field(default_factory=list)    # fingerprint of the text after each segment


def run_jobs(jobs: list[Job], items: dict, backend, rule: Rule, p_random: float, cfg,
             mock: bool = False) -> list[Job]:
    """Advance all jobs segment by segment until each finishes or hits max_tokens."""
    base = {id(j): backend.chat_prompt(prompts.solve_messages(items[j.pid]["problem"], "plain"))
            for j in jobs}
    states = {id(j): TriggerState() for j in jobs}
    shadow = {id(j): TriggerState() for j in jobs}   # what the rule would do, for every job
    rngs = {id(j): random.Random(f"{cfg.seed}|timing|{j.pid}|{j.sample}") for j in jobs}
    seg = 0
    while True:
        active = [j for j in jobs if not j.done]
        if not active:
            break
        outs = backend.continue_raw(
            [base[id(j)] + j.text for j in active], cfg.seg_tokens, cfg.temperature, cfg.top_p,
            [segment_seed(cfg.seed, j.pid, j.sample, seg) for j in active])
        to_inject = []
        for j, o in zip(active, outs):
            j.text += o["text"]
            j.seg_hashes.append(_h(j.text))
            j.gen_tokens += o["n_tokens"]
            conf = o["sum_logprob"] / max(o["n_tokens"], 1)
            j.segments.append({"seg": seg, "n": o["n_tokens"], "conf": conf,
                               "finished": o["finished"]})
            if o["finished"] or o["n_tokens"] == 0 or j.gen_tokens >= cfg.max_tokens:
                j.done = True
                continue
            if shadow[id(j)].step(conf, rule):
                j.would_trigger.append(seg)
            when, bank = CONDITIONS2[j.condition]
            st = states[id(j)]
            if when == "confidence":
                fire = st.step(conf, rule)
            elif when == "random":
                st.since_last += 1
                fire = st.eligible(rule) and rngs[id(j)].random() < p_random
                if fire:
                    st.fire()
            else:
                fire = False
            if fire:
                to_inject.append((j, bank))
        # translate seeds into strategies (only for the jobs injecting now)
        seeded = [(j, b) for j, b in to_inject if b]
        draws = {}
        for j, b in seeded:
            rng = random.Random(f"{cfg.seed}|{j.pid}|{b}|{j.sample}|{len(j.injections)}")
            draws[id(j)] = seeds.draw_seed(b, rng)
        if seeded:
            outs = backend.generate(
                [prompts.translate_messages(items[j.pid]["problem"], *draws[id(j)])
                 for j, _ in seeded], cfg.translate_max_tokens, cfg.temperature, cfg.top_p)
            for (j, _), (text, _n) in zip(seeded, outs):
                draws[id(j)] = draws[id(j)] + (prompts.clean_strategy(text),)
        for j, b in to_inject:
            if not j.injections:
                j.prefix_at_first_injection = j.text
            domain, seed_text, strategy = draws.get(id(j), ("", "", ""))
            j.text = trim_to_line(j.text) + injection_text(b, strategy, mock)
            j.injections.append({"seg": seg, "kind": b or "generic", "domain": domain,
                                 "seed": seed_text, "strategy": strategy})
        seg += 1
    return jobs


def to_record(j: Job, item: dict, max_tokens: int) -> dict:
    return {"id": j.pid, "condition": j.condition, "sample": j.sample,
            "benchmark": item["benchmark"], "level": item["level"], "subject": item["subject"],
            "response": j.text, "gen_tokens": j.gen_tokens, "truncated": j.gen_tokens >= max_tokens,
            "segments": j.segments, "injections": j.injections, "n_injections": len(j.injections),
            "would_trigger": j.would_trigger,
            "prefix_hash": _h(j.prefix_at_first_injection) if j.injections else "",
            "seg_hashes": j.seg_hashes,
            "pred": data.extract(item, j.text), "correct": data.check(item, j.text)}
