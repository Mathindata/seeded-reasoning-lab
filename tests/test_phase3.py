"""Phase 3 with a mock model whose planted truth lives in the path.

Mock (MockPathBackend): each segment is 'bad' with probability 0.25 (confidence about -2.5 per
token) or 'good' (about -0.3); an answer is right only if no bad segment is left in it. With the
rule set to fire on one low segment (m = 1), backtracking drops the bad segment and resamples it,
so it rescues answers; choosing the most confident of 4 branches rescues more; random timing
drops segments that are usually good, so it rescues little.
"""
import json
import random

import numpy as np
import pandas as pd
import pytest

from sr.backends import MockPathBackend
from sr.phase3 import (Phase3Config, _frontier, analyze3, run_main3, simulate_restart,
                       whole_answer_curves)


def items(n):
    rng = random.Random(3)
    return [{"id": f"q{i}", "problem": f"Question {i}: find the value.",
             "answer": str(rng.randint(100, 999)), "level": 3 + i % 3, "subject": "Algebra",
             "benchmark": "math500"} for i in range(n)]


@pytest.fixture(scope="module")
def p3(tmp_path_factory):
    d = tmp_path_factory.mktemp("p3")
    cal = {"tau": -1.0, "m": 1, "max_injections": 3, "cooldown": 1, "p_random": 0.25}
    json.dump(cal, open(d / "cal.json", "w"))
    test = items(40)
    be = MockPathBackend(0, {it["problem"]: it["answer"] for it in test})
    cfg = Phase3Config(backend="mock", k_samples=6, depth=1, n_branches=4, chunk=500,
                       out_dir=str(d / "run"), calibration_file=str(d / "cal.json"))
    run_main3(cfg, be, test, log=lambda *a: None)
    rows = [json.loads(x) for x in open(d / "run" / "results.jsonl")]
    return cfg, rows, analyze3(cfg.out_dir, n_boot=200)


def test_all_answers_written(p3):
    _, rows, _ = p3
    assert len(rows) == 40 * 5 * 6
    assert all(r["n_events"] == 0 for r in rows if r["condition"] == "seg_plain")


def test_pairing_and_nesting(p3):
    _, rows, res = p3
    assert (res["pairing"]["identical_prefix_share"] == 1.0).all()
    by = {(r["id"], r["sample"], r["condition"]): r for r in rows}
    # branch 0 of branch4 is exactly what backtrack sampled after the same first event
    checked = 0
    for (pid, s, c), r in by.items():
        if c != "branch4" or not r["events"]:
            continue
        bt = by[(pid, s, "backtrack")]
        assert bt["events"] and bt["events"][0]["step"] == r["events"][0]["step"]
        if bt["n_events"] == 1:       # nothing dropped later, so the resample is still in the text
            assert r["events"][0]["branch_texts"][0] in bt["response"]
            checked += 1
    assert checked > 10


def test_token_accounting(p3):
    _, rows, _ = p3
    for r in rows:
        dropped = sum(e["dropped_tokens"] for e in r["events"])
        unchosen = sum(8 * t.count("\nline") for e in r["events"] if "branch_texts" in e
                       for b, t in enumerate(e["branch_texts"]) if b != e["chosen"])
        assert r["gen_tokens"] == r["kept_tokens"] + dropped + unchosen


def test_remedies_recover_planted_truth(p3):
    _, _, res = p3
    s = res["summary"]["pass1"]
    assert s["branch4"] > s["backtrack"] > s["seg_plain"] + 0.2
    assert s["restart"] > s["seg_plain"]
    assert s["bt_random"] < s["backtrack"]
    e = res["effects"].set_index(["condition", "vs", "subset"])
    assert e.loc[("branch4", "backtrack", "same first event"), "diff"] > 0
    assert res["detection"]["auroc_min_conf"] > 0.9   # a bad final segment is not scored
    assert res["detection"]["share_chosen_not_branch0"] > 0.1


def test_budget_frontier_and_outputs(p3):
    cfg, _, res = p3
    b = res["budget"].set_index("condition")
    assert abs(b.loc["seg_plain", "gain_over_frontier"]) < 1e-9   # plain sits on the frontier
    assert b.loc["branch4", "tokens_vs_plain"] > 1
    for f in ("summary.csv", "effects.csv", "budget.csv", "whole_answer_curves.csv",
              "pairing_check.csv", "detection.json", "budget.png"):
        assert (__import__("pathlib").Path(cfg.out_dir) / "analysis" / f).exists()


def test_whole_answer_curves_exact_small_case():
    # one problem, 3 samples: right (conf -1), wrong (conf -0.5), wrong (conf -2)
    plain = pd.DataFrame({"id": ["p"] * 3, "pred": ["7", "8", "9"], "correct": [1.0, 0.0, 0.0],
                          "min_conf": [-1.0, -0.5, -2.0], "gen_tokens": [100, 100, 100]})
    c = whole_answer_curves(plain, n_max=3).set_index("N")
    assert c.loc[1, "best_conf"] == pytest.approx(1 / 3)
    # pairs: (r,w1)->w1, (r,w2)->r, (w1,w2)->w1  => 1/3
    assert c.loc[2, "best_conf"] == pytest.approx(1 / 3)
    assert c.loc[2, "tokens"] == 200
    assert _frontier(np.array([100, 200]), np.array([0.3, 0.5]), 150) == pytest.approx(0.4)


def test_restart_simulation_counts_tokens():
    seg = lambda n, st: {"step": st, "n": n, "conf": -1, "finished": False}
    plain = pd.DataFrame({"id": ["p", "p"], "sample": [0, 1], "correct": [0.0, 1.0],
                          "gen_tokens": [300, 200], "would_trigger": [[1], []],
                          "segments": [[seg(100, 0), seg(100, 1), seg(100, 2)], [seg(200, 0)]]})
    sim = simulate_restart(plain).set_index("sample")
    assert sim.loc[0, "gen_tokens"] == 200 + 200 and sim.loc[0, "correct"] == 1
    assert sim.loc[1, "gen_tokens"] == 200
