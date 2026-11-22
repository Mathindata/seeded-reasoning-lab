"""Phase 3b (offline): use the detector to decide how many whole answers a problem gets.

Each problem has k logged `seg_plain` answers (independent samples). A policy draws answers one
after another, up to 4. Its expected accuracy and token cost for a problem are computed exactly
by averaging over every ordered draw of 4 distinct answers out of k (k = 8: 1,680 draws).
Every policy is then compared with the whole-answer frontier at its own mean token cost
(same frontier and bootstrap as Phase 3).
"""
from __future__ import annotations

import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .phase3 import _majority, budget_table, min_conf, whole_answer_curves

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_plain(path: str) -> pd.DataFrame:
    """seg_plain answers from a Phase 2 or Phase 3 log (.jsonl or .jsonl.gz)."""
    op = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with op(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] != "seg_plain":
                continue
            segs = [{"step": s.get("step", s.get("seg")), "n": s["n"], "conf": s["conf"],
                     "finished": s["finished"]} for s in r["segments"]]
            fire = r["would_trigger"][0] if r["would_trigger"] else None
            rows.append({"id": r["id"], "sample": r["sample"], "pred": str(r["pred"] or ""),
                         "correct": float(r["correct"]), "gen_tokens": r["gen_tokens"],
                         "min_conf": min_conf(segs), "fired": fire is not None,
                         "tokens_to_fire": sum(s["n"] for s in segs if s["step"] <= fire)
                         if fire is not None else 0})
    return pd.DataFrame(rows)


def policy_table(plain: pd.DataFrame, quantiles=QUANTILES) -> pd.DataFrame:
    """Exact expected accuracy and tokens per problem for every policy."""
    thr = {q: (np.inf if q >= 1 else float(np.quantile(plain["min_conf"], q))) for q in quantiles}
    out = []
    for pid, g in plain.groupby("id"):
        g = g.reset_index(drop=True)
        k = len(g)
        pred, cor = g["pred"].tolist(), g["correct"].to_numpy()
        tok, mc = g["gen_tokens"].to_numpy(float), g["min_conf"].to_numpy()
        fired, ttf = g["fired"].to_numpy(), g["tokens_to_fire"].to_numpy(float)
        vote = {}
        for n in (3, 4):
            for s in itertools.combinations(range(k), n):
                vote[s] = _majority([pred[i] for i in s], [cor[i] for i in s], [mc[i] for i in s])
        V = lambda *ix: vote[tuple(sorted(ix))]
        acc = {}

        def add(name, c, t):
            a = acc.setdefault(name, [0.0, 0.0, 0])
            a[0] += c
            a[1] += t
            a[2] += 1
        for a, b, c, d in itertools.permutations(range(k), 4):
            if fired[a]:
                add("rule_vote3", V(a, b, c), tok[a] + tok[b] + tok[c])
                add("rule_stop_vote3", V(b, c, d), ttf[a] + tok[b] + tok[c] + tok[d])
            else:
                add("rule_vote3", cor[a], tok[a])
                add("rule_stop_vote3", cor[a], tok[a])
            if pred[a] and pred[a] == pred[b]:
                add("agree_2then4", cor[a], tok[a] + tok[b])
            else:
                add("agree_2then4", V(a, b, c, d), tok[a] + tok[b] + tok[c] + tok[d])
            for q in quantiles:
                if mc[a] < thr[q]:
                    add(f"conf_q{q:.1f}_vote3", V(a, b, c), tok[a] + tok[b] + tok[c])
                else:
                    add(f"conf_q{q:.1f}_vote3", cor[a], tok[a])
        for name, (c, t, n) in acc.items():
            out.append({"policy": name, "id": pid, "correct": c / n, "gen_tokens": t / n})
    return pd.DataFrame(out)


def allocate(plain_log: str, out_dir: str, n_boot: int = 2000, seed: int = 0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plain = load_plain(plain_log)
    pol = policy_table(plain)
    curves = whole_answer_curves(plain)
    base = plain.assign(condition="seg_plain")[["condition", "id", "correct", "gen_tokens"]]
    extra = {p: g[["id", "correct", "gen_tokens"]] for p, g in pol.groupby("policy")}
    budget = budget_table(base, curves, extra, n_boot, seed)
    flagged = plain["fired"]
    info = {"answers": len(plain), "problems": int(plain["id"].nunique()),
            "share_rule_fires": float(flagged.mean()),
            "accuracy_rule_fires": float(plain.loc[flagged, "correct"].mean()),
            "accuracy_rule_silent": float(plain.loc[~flagged, "correct"].mean()),
            "share_two_answers_agree": None}
    agree = []
    for _, g in plain.groupby("id"):
        p = g["pred"].tolist()
        pairs = [(i, j) for i in range(len(p)) for j in range(len(p)) if i != j]
        agree.append(np.mean([bool(p[i]) and p[i] == p[j] for i, j in pairs]))
    info["share_two_answers_agree"] = float(np.mean(agree))
    curves_mean = curves.groupby("N")[["tokens", "best_conf", "majority"]].mean()
    budget.to_csv(out / "budget.csv", index=False)
    pol.to_csv(out / "policies_by_problem.csv", index=False)
    curves_mean.to_csv(out / "whole_answer_curves.csv")
    json.dump(info, open(out / "info.json", "w"), indent=2)
    _plot(budget, curves_mean, out)
    return {"budget": budget, "curves": curves_mean, "info": info}


def _plot(budget: pd.DataFrame, curves: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax.plot(curves["tokens"], curves["majority"], "o-", color="0.3", label="majority of N, every problem")
    ax.plot(curves["tokens"], curves["best_conf"], "s--", color="0.6", label="most confident of N")
    sw = budget[budget["condition"].str.startswith("conf_q")].sort_values("tokens")
    ax.plot(sw["tokens"], sw["pass1"], ".-", color="tab:blue", label="flag lowest-confidence q, vote of 3")
    for _, r in sw.iterrows():
        ax.annotate(r["condition"][6:9], (r["tokens"], r["pass1"]), textcoords="offset points",
                    xytext=(3, -10), fontsize=7, color="tab:blue")
    named = budget[budget["condition"].isin(["rule_vote3", "rule_stop_vote3", "agree_2then4"])]
    for (_, r), m in zip(named.iterrows(), "D^X"):
        ax.plot(r["tokens"], r["pass1"], m, markersize=7, label=r["condition"])
    ax.set_xlabel("Mean tokens per problem")
    ax.set_ylabel("Accuracy of the final answer")
    ax.set_title("Where to spend extra whole answers", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    b = pd.concat([named, sw])
    y = np.arange(len(b))[::-1]
    bx.errorbar(b["gain_over_frontier"], y, xerr=[b["gain_over_frontier"] - b["ci_lo"],
                                                   b["ci_hi"] - b["gain_over_frontier"]],
                fmt="o", capsize=3, color="0.2")
    bx.axvline(0, color="0.5", linewidth=1)
    bx.set_yticks(y)
    bx.set_yticklabels([f"{c} ({t:.2f}x)" for c, t in zip(b["condition"], b["tokens_vs_plain"])],
                       fontsize=8)
    bx.set_xlabel("Accuracy minus the frontier at equal tokens (95% CI)")
    bx.set_title("Gain over the same tokens spread evenly", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "allocation.png", dpi=150)
    plt.close(fig)
