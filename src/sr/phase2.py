"""Phase 2 driver: calibrate on held-out problems, run, analyze.

Calibration uses MATH-500 problems (same levels) that are NOT in the Phase 1/2 test set.
It runs `seg_plain` only and fixes, before any test run:
  tau      = the `tau_quantile` quantile of segment confidences (non-final segments)
  p_random = per-segment injection probability for the random-timing control, chosen so
             that random timing injects as often as the confidence rule does
It also reports how well low confidence predicts a wrong final answer (AUROC).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .controller import CONDITIONS2, Job, Rule, TriggerState, run_jobs, to_record


@dataclass
class Phase2Config:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    backend: str = "vllm"
    levels: list[int] = field(default_factory=lambda: [3, 4, 5])
    n_test: int = 200                  # same sampler and seed as Phase 1, so the same problems
    k_samples: int = 8
    calib_k: int = 4
    n_calib: int = 0                   # 0 = every held-out problem
    conditions: list[str] = field(default_factory=lambda: list(CONDITIONS2))
    seg_tokens: int = 128
    max_tokens: int = 2048
    translate_max_tokens: int = 96
    temperature: float = 0.7
    top_p: float = 0.95
    tau_quantile: float = 0.2
    m: int = 2
    max_injections: int = 2
    cooldown: int = 2
    seed: int = 20261122
    out_dir: str = "runs/phase2"
    calibration_file: str = "runs/phase2_calib/calibration.json"
    chunk: int = 512
    shard: int = 0
    n_shards: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    hf_batch_size: int = 16
    prefix_caching: bool = True

    @classmethod
    def load(cls, path: str) -> "Phase2Config":
        return cls(**json.load(open(path)))


def split_problems(cfg: Phase2Config) -> tuple[list[dict], list[dict]]:
    """Test = Phase 1's problems (same sampler, same seed). Calibration = the rest."""
    test = data.load_math500(cfg.n_test, cfg.levels, cfg.seed)
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    test_ids = {t["id"] for t in test}
    calib = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"],
              "level": int(r["level"]), "subject": r["subject"], "benchmark": "math500"}
             for r in ds if int(r["level"]) in set(cfg.levels) and r["unique_id"] not in test_ids]
    if cfg.n_calib:
        calib = calib[:cfg.n_calib]
    return test, calib


def _run(cfg, backend, items, conditions, k, out: Path, rule, p_random, log, mock):
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    done = set()
    if res.exists():
        done = {(r["id"], r["condition"], r["sample"]) for r in map(json.loads, open(res))}
    by_id = {it["id"]: it for it in items}
    todo = [Job(it["id"], c, s) for it in items for c in conditions for s in range(k)
            if (it["id"], c, s) not in done]
    log(f"{len(items)} problems x {len(conditions)} conditions x {k}: {len(todo)} to do")
    t0 = time.time()
    for i in range(0, len(todo), cfg.chunk):
        chunk = run_jobs(todo[i:i + cfg.chunk], by_id, backend, rule, p_random, cfg, mock)
        with open(res, "a") as f:
            for j in chunk:
                f.write(json.dumps(to_record(j, by_id[j.pid], cfg.max_tokens)) + "\n")
        n = i + len(chunk)
        rate = n / max(time.time() - t0, 1e-9)
        log(f"  {n}/{len(todo)}, {rate:.2f} answers/s, ~{(len(todo) - n) / rate / 60:.0f} min left")
    return res


# ---------------- calibration ----------------
def simulate_rule(seg_confs: list[list[float]], rule: Rule) -> tuple[int, int, list[bool]]:
    """Return (injections, eligible segments, per-answer 'fired at least once') if the rule
    were applied to these confidence sequences (final segments excluded)."""
    inj = elig = 0
    hits = []
    for confs in seg_confs:
        st, any_ = TriggerState(), False
        for c in confs:
            st_since = st.since_last + 1
            if st.injections < rule.max_injections and st_since >= rule.cooldown:
                elig += 1
            if st.step(c, rule):
                inj += 1
                any_ = True
        hits.append(any_)
    return inj, elig, hits


def auroc(score_wrong: np.ndarray, wrong: np.ndarray) -> float:
    """P(score of a wrong answer > score of a right one), ties count half."""
    pos, neg = score_wrong[wrong], score_wrong[~wrong]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def confidence_features(df: pd.DataFrame) -> pd.DataFrame:
    def feats(segs):
        c = [s["conf"] for s in segs if not s["finished"]] or [s["conf"] for s in segs]
        return pd.Series({"min_conf": min(c), "mean_conf": float(np.mean(c))})
    return df["segments"].apply(feats)


def calibrate(cfg: Phase2Config, backend, calib: list[dict], log=print, mock=False) -> dict:
    out = Path(cfg.calibration_file).parent
    res = _run(cfg, backend, calib, ["seg_plain"], cfg.calib_k, out,
               Rule(tau=-1e9), 0.0, log, mock)
    df = pd.read_json(res, lines=True)
    confs = [[s["conf"] for s in segs if not s["finished"]] for segs in df["segments"]]
    flat = np.array([c for cs in confs for c in cs])
    tau = float(np.quantile(flat, cfg.tau_quantile))
    rule = Rule(tau, cfg.m, cfg.max_injections, cfg.cooldown)
    inj, elig, hits = simulate_rule(confs, rule)
    f = confidence_features(df)
    wrong = ~df["correct"].to_numpy(bool)
    cal = {"tau": tau, "m": cfg.m, "max_injections": cfg.max_injections, "cooldown": cfg.cooldown,
           "p_random": inj / max(elig, 1), "answers": len(df),
           "share_answers_triggered": float(np.mean(hits)), "injections_per_answer": inj / len(df),
           "calib_accuracy": float(df["correct"].mean()),
           "auroc_min_conf_predicts_wrong": auroc(-f["min_conf"].to_numpy(), wrong),
           "auroc_mean_conf_predicts_wrong": auroc(-f["mean_conf"].to_numpy(), wrong),
           "accuracy_if_rule_would_fire": float(df.loc[hits, "correct"].mean())
           if any(hits) else float("nan"),
           "accuracy_if_rule_silent": float(df.loc[[not h for h in hits], "correct"].mean())}
    json.dump(cal, open(cfg.calibration_file, "w"), indent=2)
    log(f"calibration: {cal}")
    return cal


def run_main(cfg: Phase2Config, backend, test: list[dict], log=print, mock=False) -> Path:
    cal = json.load(open(cfg.calibration_file))
    rule = Rule(cal["tau"], cal["m"], cal["max_injections"], cal["cooldown"])
    items = test[cfg.shard::cfg.n_shards]
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**asdict(cfg), "calibration": cal}, open(out / "config.json", "w"), indent=2)
    json.dump(items, open(out / "problems.json", "w"), indent=1)
    return _run(cfg, backend, items, cfg.conditions, cfg.k_samples, out, rule, cal["p_random"],
                log, mock)


# ---------------- analysis ----------------
def _boot_mean(d: np.ndarray, groups: np.ndarray, n_boot: int, seed: int):
    """Mean with a cluster bootstrap over problems (samples of one problem move together)."""
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        boots.append(d[rows].mean())
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def analyze2(run_dir: str, n_boot: int = 2000, seed: int = 0) -> dict:
    out = Path(run_dir) / "analysis"
    out.mkdir(exist_ok=True)
    df = pd.read_json(Path(run_dir) / "results.jsonl", lines=True)
    df["correct"] = df["correct"].astype(float)
    conds = [c for c in CONDITIONS2 if c in set(df["condition"])]

    summ = df.groupby("condition").agg(
        pass1=("correct", "mean"), share_injected=("n_injections", lambda x: (x > 0).mean()),
        injections=("n_injections", "mean"), tokens=("gen_tokens", "mean"),
        truncated=("truncated", "mean")).reindex(conds)

    base = df[df["condition"] == "seg_plain"].set_index(["id", "sample"])
    rows, pairing = [], []
    for c in conds:
        if c == "seg_plain":
            continue
        x = df[df["condition"] == c].set_index(["id", "sample"])
        common = x.index.intersection(base.index)
        x, b = x.loc[common], base.loc[common]
        diff_all = (x["correct"] - b["correct"]).to_numpy()
        g = np.array([i[0] for i in common])
        est = _boot_mean(diff_all, g, n_boot, seed)
        rows.append({"condition": c, "subset": "all answers", "n": len(diff_all),
                     "diff_vs_seg_plain": est[0], "ci_lo": est[1], "ci_hi": est[2]})
        inj = (x["n_injections"] > 0).to_numpy()
        if inj.any():
            est = _boot_mean(diff_all[inj], g[inj], n_boot, seed)
            rows.append({"condition": c, "subset": "injected answers", "n": int(inj.sum()),
                         "diff_vs_seg_plain": est[0], "ci_lo": est[1], "ci_hi": est[2],
                         "seg_plain_acc_same_answers": float(b["correct"].to_numpy()[inj].mean()),
                         "cond_acc_same_answers": float(x["correct"].to_numpy()[inj].mean())})
            # pairing check: was the text identical to seg_plain up to the first injection?
            same = []
            for (pid, s), r in x[inj].iterrows():
                k = r["injections"][0]["seg"]
                hs = base.loc[(pid, s), "seg_hashes"]
                same.append(k < len(hs) and hs[k] == r["prefix_hash"])
            pairing.append({"condition": c, "injected": int(inj.sum()),
                            "identical_prefix_share": float(np.mean(same))})
    eff = pd.DataFrame(rows)
    pair = pd.DataFrame(pairing)

    # confidence as an error detector on the test set (seg_plain only)
    sp = df[df["condition"] == "seg_plain"].reset_index(drop=True)
    f = confidence_features(sp)
    wrong = sp["correct"].to_numpy() < 0.5
    detect = {"auroc_min_conf": auroc(-f["min_conf"].to_numpy(), wrong),
              "auroc_mean_conf": auroc(-f["mean_conf"].to_numpy(), wrong),
              "accuracy_if_rule_fires": float(sp.loc[[bool(w) for w in sp["would_trigger"]],
                                                     "correct"].mean())
              if any(sp["would_trigger"]) else float("nan"),
              "accuracy_if_rule_silent": float(sp.loc[[not w for w in sp["would_trigger"]],
                                                     "correct"].mean())}

    summ.to_csv(out / "summary.csv")
    eff.to_csv(out / "effects.csv", index=False)
    pair.to_csv(out / "pairing_check.csv", index=False)
    json.dump(detect, open(out / "detection.json", "w"), indent=2)
    _plot(eff, out)
    return {"summary": summ, "effects": eff, "pairing": pair, "detection": detect}


def _plot(eff: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = eff[eff["subset"] == "injected answers"]
    if e.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    y = np.arange(len(e))
    ax.errorbar(e["diff_vs_seg_plain"], y,
                xerr=[e["diff_vs_seg_plain"] - e["ci_lo"], e["ci_hi"] - e["diff_vs_seg_plain"]],
                fmt="o", capsize=3)
    ax.axvline(0, color="0.5", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(e["condition"], e["n"])])
    ax.set_xlabel("Accuracy change on the answers that got an injection\n"
                  "(same answer without injection = seg_plain)")
    ax.set_title("Effect of injecting at that moment", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "injection_effects.png", dpi=150)
    plt.close(fig)


# ---------------- follow-up checks (added after the Phase 2 run, see README) ----------------
def _pairs(df: pd.DataFrame, cond: str):
    base = df[df["condition"] == "seg_plain"].set_index(["id", "sample"])
    x = df[df["condition"] == cond].set_index(["id", "sample"])
    x = x.loc[x.index.intersection(base.index)]
    b = base.loc[x.index]
    inj = (x["n_injections"] > 0).to_numpy()
    same = np.array([bool(r["n_injections"]) and r["injections"][0]["seg"] < len(b.loc[i, "seg_hashes"])
                     and b.loc[i, "seg_hashes"][r["injections"][0]["seg"]] == r["prefix_hash"]
                     for i, r in x.iterrows()])
    return x, b, inj, same


def extra_checks(run_dir: str, n_boot: int = 3000, seed: int = 1) -> dict:
    out = Path(run_dir) / "analysis"
    df = pd.read_json(Path(run_dir) / "results.jsonl", lines=True)
    df["correct"] = df["correct"].astype(float)
    df["no_answer"] = df["pred"].eq("")
    conds = [c for c in CONDITIONS2 if c != "seg_plain" and c in set(df["condition"])]
    rows = []
    for c in conds:
        x, b, inj, same = _pairs(df, c)
        xi, bi = x[inj], b[inj]
        g = np.array([i[0] for i in xi.index])
        d = (xi["correct"] - bi["correct"]).to_numpy()
        keep = (~xi["truncated"].to_numpy(bool)) & (~bi["truncated"].to_numpy(bool))
        e_same = _boot_mean(d[same[inj]], g[same[inj]], n_boot, seed)
        e_nt = _boot_mean(d[keep], g[keep], n_boot, seed)
        rows.append({
            "condition": c, "injected": int(inj.sum()),
            "rescues": int(((bi["correct"] == 0) & (xi["correct"] == 1)).sum()),
            "breaks": int(((bi["correct"] == 1) & (xi["correct"] == 0)).sum()),
            "truncated_without": bi["truncated"].mean(), "truncated_with": xi["truncated"].mean(),
            "no_answer_without": bi["no_answer"].mean(), "no_answer_with": xi["no_answer"].mean(),
            "tokens_without": bi["gen_tokens"].mean(), "tokens_with": xi["gen_tokens"].mean(),
            "effect_identical_prefix": e_same[0], "ip_lo": e_same[1], "ip_hi": e_same[2],
            "n_identical_prefix": int(same[inj].sum()),
            "effect_not_truncated": e_nt[0], "nt_lo": e_nt[1], "nt_hi": e_nt[2],
            "n_not_truncated": int(keep.sum())})
    checks = pd.DataFrame(rows)

    # content: each seeded condition vs generic, on answers injected in both, identical prefixes
    gx, _, ginj, gsame = _pairs(df, "ctrl_generic")
    content = []
    for c in ["ctrl_noise", "ctrl_near", "ctrl_far"]:
        x, _, inj, same = _pairs(df, c)
        idx = x.index.intersection(gx.index)
        ok = (pd.Series(ginj & gsame, index=gx.index).loc[idx]
              & pd.Series(inj & same, index=x.index).loc[idx])
        both = idx[ok.to_numpy()]
        d = (x.loc[both, "correct"] - gx.loc[both, "correct"]).to_numpy()
        e = _boot_mean(d, np.array([i[0] for i in both]), n_boot, seed)
        content.append({"condition": c, "vs": "ctrl_generic", "n": len(both), "diff": e[0],
                        "ci_lo": e[1], "ci_hi": e[2]})
    content = pd.DataFrame(content)

    # timing: per-injection effect of ctrl_near minus rand_near (cluster bootstrap over problems)
    def per_inj(c):
        x, b, inj, _ = _pairs(df, c)
        d = (x["correct"] - b["correct"])[inj]
        return {p: grp.to_numpy() for p, grp in d.groupby(level=0)}
    a, r = per_inj("ctrl_near"), per_inj("rand_near")
    ids = np.union1d(list(a), list(r))
    rng = np.random.default_rng(seed)
    est = np.concatenate(list(a.values())).mean() - np.concatenate(list(r.values())).mean()
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids))
        aa = [a[p] for p in pick if p in a]
        rr = [r[p] for p in pick if p in r]
        bs.append(np.concatenate(aa).mean() - np.concatenate(rr).mean())
    timing = {"ctrl_near_minus_rand_near": float(est), "ci_lo": float(np.percentile(bs, 2.5)),
              "ci_hi": float(np.percentile(bs, 97.5))}

    checks.to_csv(out / "checks_by_condition.csv", index=False)
    content.to_csv(out / "content_vs_generic.csv", index=False)
    json.dump(timing, open(out / "timing.json", "w"), indent=2)
    return {"checks": checks, "content": content, "timing": timing}
