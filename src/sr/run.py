"""Run the experiment: problems x conditions x samples, two stages, resumable.

Every generation is logged to results.jsonl with its seed, strategy, full response,
token counts and correctness. Later phases (the controller, the "dream" analysis)
read this same log, so nothing is thrown away.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import data, prompts, seeds


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    backend: str = "vllm"                 # vllm | hf | mock
    benchmark: str = "math500"            # math500 | game24
    n_problems: int = 200
    levels: list[int] = field(default_factory=lambda: [3, 4, 5])
    conditions: list[str] = field(default_factory=lambda: list(prompts.CONDITIONS))
    k_samples: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    translate_max_tokens: int = 96
    seed: int = 20261122
    out_dir: str = "runs/main"
    chunk: int = 512                      # prompts per backend call (saves progress often)
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    hf_batch_size: int = 16
    shard: int = 0                        # run problems shard, shard + n_shards, ... (one per GPU)
    n_shards: int = 1

    @classmethod
    def load(cls, path: str) -> "Config":
        return cls(**json.load(open(path)))


def load_problems(cfg: Config) -> list[dict]:
    if cfg.benchmark == "math500":
        return data.load_math500(cfg.n_problems, cfg.levels, cfg.seed)
    if cfg.benchmark == "game24":
        return data.make_game24(cfg.n_problems, cfg.seed)
    raise ValueError(cfg.benchmark)


def plan_jobs(cfg: Config, items: list[dict]) -> list[dict]:
    """One job per (problem, condition, sample). Seeds are drawn from an RNG keyed on the
    job, so a rerun draws the same seed for the same job."""
    jobs = []
    for it in items:
        for cond in cfg.conditions:
            spec = prompts.CONDITIONS[cond]
            for s in range(cfg.k_samples):
                rng = random.Random(f"{cfg.seed}|{it['id']}|{cond}|{s}")
                domain, seed_text = (seeds.draw_seed(spec["bank"], rng) if spec["bank"]
                                     else ("", ""))
                jobs.append({"id": it["id"], "condition": cond, "sample": s,
                             "domain": domain, "seed": seed_text})
    return jobs


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {(r["id"], r["condition"], r["sample"]) for r in map(json.loads, open(path))}


def run(cfg: Config, backend, items: list[dict] | None = None, log=print) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(asdict(cfg), open(out / "config.json", "w"), indent=2)
    items = items or load_problems(cfg)
    items = items[cfg.shard::cfg.n_shards]
    json.dump(items, open(out / "problems.json", "w"), indent=1)
    by_id = {it["id"]: it for it in items}

    res_path = out / "results.jsonl"
    done = _done_keys(res_path)
    jobs = [j for j in plan_jobs(cfg, items) if (j["id"], j["condition"], j["sample"]) not in done]
    log(f"{len(items)} problems, {len(cfg.conditions)} conditions, k={cfg.k_samples}: "
        f"{len(jobs)} generations to do ({len(done)} already done)")

    t0 = time.time()
    for start in range(0, len(jobs), cfg.chunk):
        chunk = jobs[start:start + cfg.chunk]
        # stage 1: translate seeds into strategies (only for translated conditions)
        tr = [j for j in chunk if prompts.CONDITIONS[j["condition"]]["translate"]]
        if tr:
            outs = backend.generate(
                [prompts.translate_messages(by_id[j["id"]]["problem"], j["domain"], j["seed"])
                 for j in tr], cfg.translate_max_tokens, cfg.temperature, cfg.top_p)
            for j, (text, n) in zip(tr, outs):
                j["strategy_raw"], j["strategy"], j["tokens_stage1"] = text, \
                    prompts.clean_strategy(text), n
        # stage 2: solve
        outs = backend.generate(
            [prompts.solve_messages(by_id[j["id"]]["problem"], j["condition"], j["domain"],
                                    j["seed"], j.get("strategy", "")) for j in chunk],
            cfg.max_tokens, cfg.temperature, cfg.top_p)
        with open(res_path, "a") as f:
            for j, (text, n) in zip(chunk, outs):
                it = by_id[j["id"]]
                rec = {**j, "benchmark": it["benchmark"], "level": it["level"],
                       "subject": it["subject"], "response": text,
                       "tokens_stage1": j.get("tokens_stage1", 0), "tokens_stage2": n,
                       "truncated": n >= cfg.max_tokens,
                       "pred": data.extract(it, text), "correct": data.check(it, text)}
                f.write(json.dumps(rec) + "\n")
        done_n = start + len(chunk)
        rate = done_n / max(time.time() - t0, 1e-9)
        log(f"  {done_n}/{len(jobs)} done, {rate:.1f} gen/s, "
            f"~{(len(jobs) - done_n) / max(rate, 1e-9) / 60:.0f} min left")
    return res_path


def merge(run_dirs: list[str], out_dir: str) -> Path:
    """Combine shard runs (e.g. one per GPU) into one run directory for analysis."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, items, seen = [], [], set()
    for d in run_dirs:
        items += json.load(open(Path(d) / "problems.json"))
        for line in open(Path(d) / "results.jsonl"):
            r = json.loads(line)
            key = (r["id"], r["condition"], r["sample"])
            if key not in seen:
                seen.add(key)
                rows.append(line if line.endswith("\n") else line + "\n")
    json.dump(items, open(out / "problems.json", "w"), indent=1)
    cfg = json.load(open(Path(run_dirs[0]) / "config.json"))
    cfg.update({"shard": 0, "n_shards": 1, "out_dir": str(out), "merged_from": run_dirs})
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    with open(out / "results.jsonl", "w") as f:
        f.writelines(rows)
    return out
