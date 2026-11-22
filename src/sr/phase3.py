"""Phase 3: same detector as Phase 2, different remedy. No text is ever added.

When the rule fires (2 consecutive segments with mean token log-probability below tau, the
threshold fixed on held-out problems in Phase 2), the controller changes the path instead of
talking to the model:

  backtrack  drop the 2 low segments and sample them again (new random seed)
  branch4    drop them, sample 4 candidate continuations of 2 segments from the same point,
             keep the one with the highest mean log-probability
  restart    drop the whole answer and start again
  bt_random  backtrack at random moments, at the rate the rule fires (timing control)

Pairing: every condition uses the same per-step random seeds as `seg_plain`, so all write the
same text until the first event. Branch 0 of `branch4` uses the seeds `backtrack` uses after its
event, so branch4 = backtrack + 3 extra candidates + a choice (nested by construction).

Budget: every generated token counts, including discarded segments and unchosen branches.
The comparison that decides whether a remedy is worth it is against spending the same tokens on
whole answers instead (best of N by confidence, or majority vote of N), computed from the
`seg_plain` samples of the same run.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, prompts
from .controller import Rule, TriggerState, segment_seed
from .phase2 import Phase2Config, _boot_mean, auroc

CONDITIONS3 = {
    # name: (when, action)
    "seg_plain": ("never", None),
    "backtrack": ("confidence", "backtrack"),
    "branch4": ("confidence", "branch"),
    "restart": ("confidence", "restart"),
    "bt_random": ("random", "backtrack"),
}
BRANCH_SEED_STEP = 7919          # branch b > 0 uses base seed cfg.seed + b * 7919


@dataclass
class Phase3Config(Phase2Config):
    conditions: list[str] = field(default_factory=lambda: list(CONDITIONS3))
    depth: int = 2                 # segments dropped by backtrack / branch (= the rule's m)
    n_branches: int = 4
    max_total_tokens: int = 4096   # compute cap per answer, all generated tokens included
    calibration_file: str = "results/phase2/calibration.json"   # fixed in Phase 2
    out_dir: str = "runs/phase3"


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class Job3:
    pid: str
    condition: str
    sample: int
    step: int = 0                                      # next step index (sets the seed)
    path: list = field(default_factory=list)           # kept segments: step, text, n, conf, finished
    gen_tokens: int = 0                                # every token generated
    done: bool = False
    events: list = field(default_factory=list)
    would_trigger: list = field(default_factory=list)
    seg_hashes: list = field(default_factory=list)     # hash of the kept text after each step
    prefix_hash: str = ""                              # kept text just before the first event

    @property
    def text(self) -> str:
        return "".join(s["text"] for s in self.path)

    @property
    def kept_tokens(self) -> int:
        return sum(s["n"] for s in self.path)


def _seg(step, o):
    return {"step": step, "text": o["text"], "n": o["n_tokens"],
            "conf": o["sum_logprob"] / max(o["n_tokens"], 1), "finished": o["finished"]}


def _is_done(j: Job3, last: dict, cfg) -> bool:
    return (last["finished"] or last["n"] == 0 or j.kept_tokens >= cfg.max_tokens
            or j.gen_tokens >= cfg.max_total_tokens)


def _branch(jobs: list[Job3], base: dict, backend, cfg, states, rule) -> None:
    """For each job: sample n_branches continuations of `depth` segments from its current
    path, keep the most confident (mean log-prob over the branch's tokens), log the rest."""
    B = cfg.n_branches
    cands = {(id(j), b): [] for j in jobs for b in range(B)}
    for d in range(cfg.depth):
        live = [(j, b) for j in jobs for b in range(B)
                if not (cands[(id(j), b)] and cands[(id(j), b)][-1]["finished"])]
        if not live:
            break
        outs = backend.continue_raw(
            [base[id(j)] + j.text + "".join(s["text"] for s in cands[(id(j), b)]) for j, b in live],
            cfg.seg_tokens, cfg.temperature, cfg.top_p,
            [segment_seed(cfg.seed + b * BRANCH_SEED_STEP, j.pid, j.sample, j.step + d)
             for j, b in live])
        for (j, b), o in zip(live, outs):
            j.gen_tokens += o["n_tokens"]
            cands[(id(j), b)].append(_seg(j.step + d, o))
    for j in jobs:
        scores = []
        for b in range(B):
            c = cands[(id(j), b)]
            scores.append(sum(s["conf"] * s["n"] for s in c) / max(sum(s["n"] for s in c), 1))
        best = int(np.argmax(scores))
        ev = j.events[-1]
        ev.update({"branch_confs": scores, "chosen": best,
                   "branch_texts": ["".join(s["text"] for s in cands[(id(j), b)]) for b in range(B)]})
        st = states[id(j)]
        for s in cands[(id(j), best)]:
            j.path.append(s)
            st.step(s["conf"], rule)
            j.seg_hashes.append(_h(j.text))
        j.step += cfg.depth
        if _is_done(j, j.path[-1], cfg):
            j.done = True


def run_jobs3(jobs: list[Job3], items: dict, backend, rule: Rule, p_random: float, cfg) -> list[Job3]:
    base = {id(j): backend.chat_prompt(prompts.solve_messages(items[j.pid]["problem"], "plain"))
            for j in jobs}
    states = {id(j): TriggerState() for j in jobs}
    shadow = {id(j): TriggerState() for j in jobs}
    rngs = {id(j): random.Random(f"{cfg.seed}|bt_timing|{j.pid}|{j.sample}") for j in jobs}
    while True:
        active = [j for j in jobs if not j.done]
        if not active:
            break
        outs = backend.continue_raw(
            [base[id(j)] + j.text for j in active], cfg.seg_tokens, cfg.temperature, cfg.top_p,
            [segment_seed(cfg.seed, j.pid, j.sample, j.step) for j in active])
        branching = []
        for j, o in zip(active, outs):
            s = _seg(j.step, o)
            j.path.append(s)
            j.gen_tokens += s["n"]
            j.seg_hashes.append(_h(j.text))
            j.step += 1
            if _is_done(j, s, cfg):
                j.done = True
                continue
            if shadow[id(j)].step(s["conf"], rule):
                j.would_trigger.append(s["step"])
            when, action = CONDITIONS3[j.condition]
            st = states[id(j)]
            if when == "confidence":
                fire = st.step(s["conf"], rule)
            elif when == "random":
                st.since_last += 1
                fire = (st.eligible(rule) and len(j.path) >= cfg.depth
                        and rngs[id(j)].random() < p_random)
                if fire:
                    st.fire()
            else:
                fire = False
            if not fire:
                continue
            if not j.events:
                j.prefix_hash = _h(j.text)
            keep = 0 if action == "restart" else max(len(j.path) - cfg.depth, 0)
            dropped = j.path[keep:]
            j.events.append({"step": s["step"], "action": action, "checkpoint_segments": keep,
                             "dropped_segments": len(dropped),
                             "dropped_tokens": sum(x["n"] for x in dropped),
                             "dropped_confs": [x["conf"] for x in dropped]})
            j.path = j.path[:keep]
            if action == "branch":
                branching.append(j)
        if branching:
            _branch(branching, base, backend, cfg, states, rule)
    return jobs


def to_record3(j: Job3, item: dict, cfg) -> dict:
    text = j.text
    finished = bool(j.path) and j.path[-1]["finished"]
    return {"id": j.pid, "condition": j.condition, "sample": j.sample,
            "benchmark": item["benchmark"], "level": item["level"], "subject": item["subject"],
            "response": text, "gen_tokens": j.gen_tokens, "kept_tokens": j.kept_tokens,
            "truncated": not finished,
            "segments": [{k: s[k] for k in ("step", "n", "conf", "finished")} for s in j.path],
            "events": j.events, "n_events": len(j.events), "would_trigger": j.would_trigger,
            "prefix_hash": j.prefix_hash, "seg_hashes": j.seg_hashes,
            "pred": data.extract(item, text), "correct": data.check(item, text)}


def run_main3(cfg: Phase3Config, backend, test: list[dict], log=print) -> Path:
    cal = json.load(open(cfg.calibration_file))
    rule = Rule(cal["tau"], cal["m"], cal["max_injections"], cal["cooldown"])
    assert cfg.depth == rule.m, "backtrack depth should equal the rule's m"
    items = test[cfg.shard::cfg.n_shards]
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**asdict(cfg), "calibration": cal}, open(out / "config.json", "w"), indent=2)
    json.dump(items, open(out / "problems.json", "w"), indent=1)
    res = out / "results.jsonl"
    done = set()
    if res.exists():
        done = {(r["id"], r["condition"], r["sample"]) for r in map(json.loads, open(res))}
    by_id = {it["id"]: it for it in items}
    todo = [Job3(it["id"], c, s) for it in items for c in cfg.conditions
            for s in range(cfg.k_samples) if (it["id"], c, s) not in done]
    log(f"{len(items)} problems x {len(cfg.conditions)} conditions x {cfg.k_samples}: "
        f"{len(todo)} to do")
    t0 = time.time()
    for i in range(0, len(todo), cfg.chunk):
        chunk = run_jobs3(todo[i:i + cfg.chunk], by_id, backend, rule, cal["p_random"], cfg)
        with open(res, "a") as f:
            for j in chunk:
                f.write(json.dumps(to_record3(j, by_id[j.pid], cfg)) + "\n")
        n = i + len(chunk)
        rate = n / max(time.time() - t0, 1e-9)
        log(f"  {n}/{len(todo)}, {rate:.2f} answers/s, ~{(len(todo) - n) / rate / 60:.0f} min left")
    return res


# ---------------- analysis ----------------
def min_conf(segs: list) -> float:
    c = [s["conf"] for s in segs if not s["finished"]] or [s["conf"] for s in segs] or [0.0]
    return float(min(c))


def within_problem_auroc(df: pd.DataFrame, score: str) -> float:
    """P(score of a wrong answer > score of a right answer), pairs taken within the same problem
    only. Removes the part of the signal that just says 'this problem is hard'."""
    gt = eq = n = 0
    for _, g in df.groupby("id"):
        w = g.loc[g["correct"] < 0.5, score].to_numpy()
        r = g.loc[g["correct"] >= 0.5, score].to_numpy()
        if len(w) and len(r):
            gt += (w[:, None] > r[None, :]).sum()
            eq += (w[:, None] == r[None, :]).sum()
            n += len(w) * len(r)
    return float((gt + 0.5 * eq) / n) if n else float("nan")


def _majority(preds, corrects, confs) -> float:
    votes = Counter(p for p in preds if p)
    if not votes:
        return float(corrects[0])
    top = max(votes.values())
    tied = [p for p, v in votes.items() if v == top]
    best = max(tied, key=lambda p: max(c for q, c in zip(preds, confs) if q == p))
    return float(next(c for q, c in zip(preds, corrects) if q == best))


def whole_answer_curves(plain: pd.DataFrame, n_max: int = 4) -> pd.DataFrame:
    """Per problem, exact expected accuracy of 'draw N whole answers, keep one' over all subsets
    of size N of its k samples: best by lowest-segment confidence, and majority vote (ties to the
    most confident). Cost of N answers = N x the problem's mean tokens."""
    rows = []
    for pid, g in plain.groupby("id"):
        preds, cor = g["pred"].tolist(), g["correct"].to_numpy(float)
        mc, tok = g["min_conf"].to_numpy(), float(g["gen_tokens"].mean())
        for N in range(1, min(n_max, len(g)) + 1):
            subs = list(itertools.combinations(range(len(g)), N))
            conf = np.mean([cor[max(s, key=lambda i: mc[i])] for s in subs])
            maj = np.mean([_majority([preds[i] for i in s], [cor[i] for i in s], [mc[i] for i in s])
                           for s in subs])
            rows.append({"id": pid, "N": N, "tokens": N * tok, "best_conf": conf, "majority": maj})
    return pd.DataFrame(rows)


def simulate_restart(plain: pd.DataFrame, max_restarts: int = 2) -> pd.DataFrame:
    """Offline restart from the seg_plain log: if the rule would fire on a sample, pay its tokens
    up to that point and take the next sample of the same problem (at most max_restarts times)."""
    out = []
    for pid, g in plain.sort_values("sample").groupby("id"):
        G = g.to_dict("records")
        n = len(G)
        for j in range(n):
            cost = 0
            for r in range(max_restarts + 1):
                x = G[(j + r) % n]
                if x["would_trigger"] and r < max_restarts:
                    f = x["would_trigger"][0]
                    cost += sum(s["n"] for s in x["segments"] if s["step"] <= f)
                    continue
                out.append({"id": pid, "sample": j, "correct": float(x["correct"]),
                            "gen_tokens": cost + x["gen_tokens"]})
                break
    return pd.DataFrame(out)


def _frontier(cost: np.ndarray, acc: np.ndarray, at: float) -> float:
    return float(np.interp(at, cost, np.maximum.accumulate(acc)))


def budget_table(df: pd.DataFrame, curves: pd.DataFrame, extra: dict, n_boot: int, seed: int):
    """Accuracy of each condition minus the whole-answer frontier at the same mean tokens.
    Frontier = for each N, the better of best-by-confidence and majority vote (chosen after the
    fact, so it favors the baseline), linear between N (mixing N and N+1 is a valid policy).
    Cluster bootstrap over problems."""
    pids = np.array(sorted(df["id"].unique()))
    cur = {N: g.set_index("id").loc[pids] for N, g in curves.groupby("N")}
    Ns = sorted(cur)
    groups = {c: g.groupby("id").agg(acc=("correct", "mean"), tok=("gen_tokens", "mean")).loc[pids]
              for c, g in list(df.groupby("condition")) + list(extra.items())}
    rng = np.random.default_rng(seed)
    picks = [np.arange(len(pids))] + [rng.integers(0, len(pids), len(pids)) for _ in range(n_boot)]
    rows = []
    for c, g in groups.items():
        acc, tok = g["acc"].to_numpy(), g["tok"].to_numpy()
        vals = []
        for p in picks:
            cost = np.array([cur[N]["tokens"].to_numpy()[p].mean() for N in Ns])
            env = np.array([max(cur[N]["best_conf"].to_numpy()[p].mean(),
                                cur[N]["majority"].to_numpy()[p].mean()) for N in Ns])
            vals.append(acc[p].mean() - _frontier(cost, env, tok[p].mean()))
        v = np.array(vals[1:])
        rows.append({"condition": c, "pass1": acc.mean(), "tokens": tok.mean(),
                     "tokens_vs_plain": tok.mean() / groups["seg_plain"]["tok"].mean(),
                     "gain_over_frontier": vals[0], "ci_lo": np.percentile(v, 2.5),
                     "ci_hi": np.percentile(v, 97.5)})
    return pd.DataFrame(rows)


def _paired(df, cond, ref, n_boot, seed, subset=None):
    a = df[df["condition"] == cond].set_index(["id", "sample"])
    b = df[df["condition"] == ref].set_index(["id", "sample"])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    mask = np.ones(len(idx), bool) if subset is None else subset(a, b)
    d = (a["correct"] - b["correct"]).to_numpy()[mask]
    g = np.array([i[0] for i in idx])[mask]
    if len(d) == 0:
        return {"n": 0}
    m, lo, hi = _boot_mean(d, g, n_boot, seed)
    return {"n": int(len(d)), "diff": m, "ci_lo": lo, "ci_hi": hi,
            "acc_ref": float(b["correct"].to_numpy()[mask].mean()),
            "acc_cond": float(a["correct"].to_numpy()[mask].mean()),
            "rescues": int(((b["correct"] == 0) & (a["correct"] == 1)).to_numpy()[mask].sum()),
            "breaks": int(((b["correct"] == 1) & (a["correct"] == 0)).to_numpy()[mask].sum())}


def analyze3(run_dir: str, n_boot: int = 2000, seed: int = 0) -> dict:
    out = Path(run_dir) / "analysis"
    out.mkdir(exist_ok=True)
    df = pd.read_json(Path(run_dir) / "results.jsonl", lines=True, dtype={"pred": str})
    df["pred"] = df["pred"].fillna("").astype(str)
    df["correct"] = df["correct"].astype(float)
    df["min_conf"] = df["segments"].apply(min_conf)
    df["no_answer"] = df["pred"].eq("")
    conds = [c for c in CONDITIONS3 if c in set(df["condition"])]

    summ = df.groupby("condition").agg(
        pass1=("correct", "mean"), share_with_event=("n_events", lambda x: (x > 0).mean()),
        events=("n_events", "mean"), tokens=("gen_tokens", "mean"),
        kept_tokens=("kept_tokens", "mean"), truncated=("truncated", "mean"),
        no_answer=("no_answer", "mean")).reindex(conds)

    has_event = lambda a, b: (a["n_events"] > 0).to_numpy()
    rows, pairing = [], []
    base = df[df["condition"] == "seg_plain"].set_index(["id", "sample"])
    for c in conds:
        if c == "seg_plain":
            continue
        rows.append({"condition": c, "vs": "seg_plain", "subset": "all answers",
                     **_paired(df, c, "seg_plain", n_boot, seed)})
        rows.append({"condition": c, "vs": "seg_plain", "subset": "answers with an event",
                     **_paired(df, c, "seg_plain", n_boot, seed, has_event)})
        x = df[(df["condition"] == c) & (df["n_events"] > 0)].set_index(["id", "sample"])
        same = []
        for key, r in x.iterrows():
            k = r["events"][0]["step"]
            hs = base.loc[key, "seg_hashes"] if key in base.index else []
            same.append(k < len(hs) and hs[k] == r["prefix_hash"])
        pairing.append({"condition": c, "with_event": len(x),
                        "identical_prefix_share": float(np.mean(same)) if same else float("nan")})

    # nested comparisons
    def both_event_same_step(a, b):
        return np.array([bool(x) and bool(y) and x[0]["step"] == y[0]["step"]
                         for x, y in zip(a["events"], b["events"])])

    def late(a, b):
        return both_event_same_step(a, b) & np.array(
            [bool(x) and x[0]["checkpoint_segments"] > 0 for x in b["events"]])

    def early(a, b):
        return both_event_same_step(a, b) & np.array(
            [bool(x) and x[0]["checkpoint_segments"] == 0 for x in b["events"]])
    if {"branch4", "backtrack"} <= set(conds):
        rows.append({"condition": "branch4", "vs": "backtrack", "subset": "same first event",
                     **_paired(df, "branch4", "backtrack", n_boot, seed, both_event_same_step)})
    if {"restart", "backtrack"} <= set(conds):
        rows.append({"condition": "restart", "vs": "backtrack",
                     "subset": "same first event, backtrack keeps a prefix",
                     **_paired(df, "restart", "backtrack", n_boot, seed, late)})
        rows.append({"condition": "restart", "vs": "backtrack",
                     "subset": "same first event, backtrack goes to the start",
                     **_paired(df, "restart", "backtrack", n_boot, seed, early)})
    eff = pd.DataFrame(rows)

    # branch choice: did the chosen branch differ from branch 0, and was its confidence higher?
    br = [e for evs in df.loc[df["condition"] == "branch4", "events"] for e in evs if "chosen" in e]
    branch_info = {"branch_events": len(br),
                   "share_chosen_not_branch0": float(np.mean([e["chosen"] != 0 for e in br]))
                   if br else float("nan"),
                   "mean_conf_gain_chosen_vs_branch0": float(np.mean(
                       [e["branch_confs"][e["chosen"]] - e["branch_confs"][0] for e in br]))
                   if br else float("nan")}

    # detector, and the budget comparison
    plain = df[df["condition"] == "seg_plain"].copy()
    wrong = plain["correct"].to_numpy() < 0.5
    plain["neg_min_conf"] = -plain["min_conf"]
    fired = plain["would_trigger"].apply(bool)
    detect = {"auroc_min_conf": auroc(plain["neg_min_conf"].to_numpy(), wrong),
              "auroc_min_conf_within_problem": within_problem_auroc(plain, "neg_min_conf"),
              "accuracy_if_rule_fires": float(plain.loc[fired, "correct"].mean()),
              "accuracy_if_rule_silent": float(plain.loc[~fired, "correct"].mean()),
              "first_fire_step_counts": plain.loc[fired, "would_trigger"].apply(
                  lambda w: w[0]).value_counts().sort_index().to_dict(),
              **branch_info}
    curves = whole_answer_curves(plain)
    sim = simulate_restart(plain, max_restarts=2)
    budget = budget_table(df, curves, {"restart_simulated": sim}, n_boot, seed)
    curve_mean = curves.groupby("N")[["tokens", "best_conf", "majority"]].mean()

    summ.to_csv(out / "summary.csv")
    eff.to_csv(out / "effects.csv", index=False)
    pd.DataFrame(pairing).to_csv(out / "pairing_check.csv", index=False)
    budget.to_csv(out / "budget.csv", index=False)
    curve_mean.to_csv(out / "whole_answer_curves.csv")
    json.dump(detect, open(out / "detection.json", "w"), indent=2, default=str)
    _plot_budget(budget, curve_mean, out)
    return {"summary": summ, "effects": eff, "pairing": pd.DataFrame(pairing), "budget": budget,
            "curves": curve_mean, "detection": detect}


def _plot_budget(budget: pd.DataFrame, curves: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax.plot(curves["tokens"], curves["majority"], "o-", color="0.3", label="majority of N whole answers")
    ax.plot(curves["tokens"], curves["best_conf"], "s--", color="0.6",
            label="most confident of N whole answers")
    for N, r in curves.iterrows():
        ax.annotate(f"N={N}", (r["tokens"], r["majority"]), textcoords="offset points",
                    xytext=(-8, 8), fontsize=8, color="0.3")
    b = budget[budget["condition"] != "seg_plain"].reset_index(drop=True)
    markers = "DvP^Xh"
    for i, r in b.iterrows():
        ax.plot(r["tokens"], r["pass1"], markers[i % len(markers)], markersize=7, label=r["condition"])
    ax.set_xlabel("Mean tokens generated per final answer (discarded tokens included)")
    ax.set_ylabel("Accuracy of the final answer")
    ax.set_title("Tokens spent on the path vs on more whole answers", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    y = np.arange(len(b))[::-1]
    bx.errorbar(b["gain_over_frontier"], y, xerr=[b["gain_over_frontier"] - b["ci_lo"],
                                                   b["ci_hi"] - b["gain_over_frontier"]],
                fmt="o", capsize=3, color="0.2")
    bx.axvline(0, color="0.5", linewidth=1)
    bx.set_yticks(y)
    bx.set_yticklabels([f"{c} ({t:.2f}x tokens)" for c, t in zip(b["condition"], b["tokens_vs_plain"])])
    bx.set_xlabel("Accuracy minus the frontier at equal tokens (95% CI)")
    bx.set_title("Gain over spending the tokens on whole answers", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "budget.png", dpi=150)
    plt.close(fig)

def phase2_baseline(generations_gz: str = "results/phase2/generations.jsonl.gz",
                    out_dir: str = "runs/phase2_as_phase3") -> dict:
    """Run the Phase 3 analysis on the Phase 2 `seg_plain` answers alone: detector strength
    within problems, the whole-answer frontier, and the simulated restart. These are the
    numbers quoted in the README before the Phase 3 run."""
    import gzip
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(generations_gz, "rt") as f, open(out / "results.jsonl", "w") as g:
        for line in f:
            r = json.loads(line)
            if r["condition"] != "seg_plain":
                continue
            r["segments"] = [{"step": s["seg"], "n": s["n"], "conf": s["conf"],
                              "finished": s["finished"]} for s in r["segments"]]
            r.update(kept_tokens=r["gen_tokens"], n_events=0, events=[], prefix_hash="")
            g.write(json.dumps(r) + "\n")
    return analyze3(str(out))
