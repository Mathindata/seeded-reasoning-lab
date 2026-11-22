"""Turn results.jsonl into tables and plots.

pass@k (unbiased, Chen et al. 2021): with n samples of which c are correct,
  pass@k = 1 - C(n - c, k) / C(n, k)
the chance that at least one of k samples drawn from the n is correct.
pass@1 is the plain accuracy; pass@k for larger k measures coverage, which is
where more diverse starting points should help if they help at all.

All comparisons are paired by problem: the same problems under two conditions.
Confidence intervals come from a bootstrap over problems.
"""
from __future__ import annotations

import json
from collections import Counter
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["plain", "noise_translated", "polya_translated", "near_translated",
         "far_translated", "far_raw"]


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load(run_dir: str) -> pd.DataFrame:
    df = pd.read_json(Path(run_dir) / "results.jsonl", lines=True)
    df["tokens"] = df["tokens_stage1"] + df["tokens_stage2"]
    return df


def per_problem(df: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    rows = []
    for (pid, cond), g in df.groupby(["id", "condition"]):
        n, c = len(g), int(g["correct"].sum())
        preds = [p for p in g["pred"] if p]
        maj = Counter(preds).most_common(1)[0][0] if preds else ""
        maj_ok = bool(g.loc[g["pred"] == maj, "correct"].any()) if maj else False
        row = {"id": pid, "condition": cond, "level": g["level"].iloc[0], "n": n, "c": c,
               "majority_correct": maj_ok, "distinct_answers": len(set(preds)),
               "tokens": g["tokens"].mean(), "truncated": g["truncated"].mean()}
        for k in ks:
            row[f"pass@{k}"] = pass_at_k(n, c, k) if k <= n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summary(pp: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    cols = [f"pass@{k}" for k in ks] + ["majority_correct", "distinct_answers", "tokens",
                                         "truncated"]
    t = pp.groupby("condition")[cols].mean()
    return t.reindex([c for c in ORDER if c in t.index])


def paired_diff(pp: pd.DataFrame, a: str, b: str, metric: str, n_boot: int, seed: int):
    """Mean of (a - b) over problems, with a 95% bootstrap interval."""
    w = pp.pivot(index="id", columns="condition", values=metric)[[a, b]].dropna()
    d = (w[a] - w[b]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def comparisons(pp: pd.DataFrame, metrics: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    conds = [c for c in ORDER if c in set(pp["condition"])]
    rows = []
    for ref in ["plain", "noise_translated"]:
        if ref not in conds:
            continue
        for c in conds:
            if c == ref:
                continue
            for m in metrics:
                est, lo, hi = paired_diff(pp, c, ref, m, n_boot, seed)
                rows.append({"condition": c, "vs": ref, "metric": m, "diff": est,
                             "ci_lo": lo, "ci_hi": hi, "clear": (lo > 0) or (hi < 0)})
    return pd.DataFrame(rows)


def mixture_coverage(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Equal budget of k samples: all from plain, versus half plain and half from one seeded
    condition. If seeds add *different* successes, the mix covers more problems."""
    out = []
    conds = [c for c in ORDER if c in set(df["condition"]) and c != "plain"]
    for c in ["plain"] + conds:
        hits = []
        for pid, g in df.groupby("id"):
            p = g[g["condition"] == "plain"].sort_values("sample")
            if c == "plain":
                pick = p.head(k)
            else:
                s = g[g["condition"] == c].sort_values("sample")
                pick = pd.concat([p.head(k // 2), s.head(k - k // 2)])
            hits.append(bool(pick["correct"].any()))
        out.append({"pool": "plain" if c == "plain" else f"plain + {c}",
                    f"coverage@{k}": float(np.mean(hits))})
    return pd.DataFrame(out)


def seed_table(df: pd.DataFrame) -> pd.DataFrame:
    s = df[df["seed"] != ""]
    t = s.groupby(["condition", "domain", "seed"]).agg(
        n=("correct", "size"), accuracy=("correct", "mean")).reset_index()
    return t.sort_values(["condition", "accuracy"], ascending=[True, False])


def plot(summ: pd.DataFrame, comp: pd.DataFrame, ks: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for cond, row in summ.iterrows():
        ys = [row[f"pass@{k}"] for k in ks]
        ax.plot(ks, ys, marker="o", label=cond, linestyle="--" if cond == "plain" else "-")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("k (samples per problem)")
    ax.set_ylabel("pass@k")
    ax.set_title("Coverage by condition")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pass_at_k.png", dpi=150)
    plt.close(fig)

    c = comp[(comp["vs"] == "plain")]
    if c.empty:
        return
    metrics = list(dict.fromkeys(c["metric"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        cm = c[c["metric"] == m]
        y = np.arange(len(cm))
        ax.errorbar(cm["diff"], y, xerr=[cm["diff"] - cm["ci_lo"], cm["ci_hi"] - cm["diff"]],
                    fmt="o", capsize=3)
        ax.axvline(0, color="0.5", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(cm["condition"])
        ax.set_title(f"{m}: difference vs plain", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "diff_vs_plain.png", dpi=150)
    plt.close(fig)


def analyze(run_dir: str, n_boot: int = 2000, seed: int = 0) -> dict:
    out = Path(run_dir) / "analysis"
    out.mkdir(exist_ok=True)
    df = load(run_dir)
    n = int(df.groupby(["id", "condition"]).size().min())
    ks = [k for k in (1, 2, 4, 8, 16) if k <= n]
    pp = per_problem(df, ks)
    summ = summary(pp, ks)
    metrics = ["pass@1", f"pass@{ks[-1]}", "majority_correct"]
    pp["majority_correct"] = pp["majority_correct"].astype(float)
    comp = comparisons(pp, metrics, n_boot, seed)
    mix = mixture_coverage(df, ks[-1]) if "plain" in set(df["condition"]) else pd.DataFrame()
    by_level = pp.groupby(["level", "condition"])[["pass@1", f"pass@{ks[-1]}"]].mean().unstack()

    summ.to_csv(out / "summary.csv")
    comp.to_csv(out / "comparisons.csv", index=False)
    mix.to_csv(out / "mixture_coverage.csv", index=False)
    by_level.to_csv(out / "by_level.csv")
    seed_table(df).to_csv(out / "seeds.csv", index=False)
    plot(summ, comp, ks, out)
    info = {"problems": int(df["id"].nunique()), "samples_per_condition": n, "ks": ks,
            "generations": len(df), "tokens_total": int(df["tokens"].sum())}
    json.dump(info, open(out / "info.json", "w"), indent=2)
    return {"summary": summ, "comparisons": comp, "mixture": mix, "by_level": by_level,
            "info": info}
