"""Phase 2 with a mock model that has a planted truth.

Mock: half of all trajectories are 'lost' (confidence about -2.4 per token, wrong final
answer); the rest are 'on track' (about -0.45, right answer). An injection rescues a lost
trajectory with probability 0.6 for near-domain strategies and 0.2 for the others.
Expected: the confidence rule fires on lost trajectories only, so its per-injection effect is
about +0.6 for near and +0.2 otherwise; random timing also hits on-track trajectories, where
nothing can be gained, so its per-injection effect is diluted (about +0.3).
"""
import json
import random

import numpy as np
import pytest

from sr.backends import MockBackend
from sr.controller import Rule, TriggerState
from sr.phase2 import Phase2Config, analyze2, calibrate, run_main


def items(n, start=0):
    rng = random.Random(start)
    return [{"id": f"q{start + i}", "problem": f"Question {start + i}: find the value.",
             "answer": str(rng.randint(100, 999)), "level": 3 + i % 3, "subject": "Algebra",
             "benchmark": "math500"} for i in range(n)]


def test_hysteresis_rule():
    r = Rule(tau=-1.0, m=2, max_injections=2, cooldown=2)
    st = TriggerState()
    fired = [st.step(c, r) for c in [-2, -0.5, -2, -2, -2, -2, -2, -2, -2, -2]]
    # needs 2 lows in a row; after firing, the streak restarts and the cooldown applies
    assert fired == [False, False, False, True, False, True, False, False, False, False]
    assert st.injections == 2   # capped


@pytest.fixture(scope="module")
def p2(tmp_path_factory):
    d = tmp_path_factory.mktemp("p2")
    test, calib = items(40), items(40, start=1000)
    answers = {it["problem"]: it["answer"] for it in test + calib}
    be = MockBackend(0, answers, {"generic": 0.2, "noise": 0.2, "near": 0.6, "far": 0.2})
    cfg = Phase2Config(backend="mock", k_samples=6, calib_k=4, out_dir=str(d / "main"),
                       calibration_file=str(d / "calib" / "calibration.json"), chunk=200,
                       tau_quantile=0.55)
    cal = calibrate(cfg, be, calib, log=lambda *a: None, mock=True)
    run_main(cfg, be, test, log=lambda *a: None, mock=True)
    return cfg, cal, analyze2(cfg.out_dir, n_boot=300)


def test_calibration_sets_threshold_between_the_two_regimes(p2):
    _, cal, _ = p2
    assert -2.4 < cal["tau"] < -0.3          # below every on-track segment but one band
    assert 0.3 < cal["share_answers_triggered"] < 0.7
    assert cal["auroc_min_conf_predicts_wrong"] > 0.95
    assert 0 < cal["p_random"] < 1
    assert cal["accuracy_if_rule_would_fire"] < 0.1 < 0.9 < cal["accuracy_if_rule_silent"]


def test_pairing_is_exact_before_injection(p2):
    _, _, res = p2
    pair = res["pairing"].set_index("condition")
    assert (pair["identical_prefix_share"] == 1.0).all()


def test_effects_recover_planted_truth(p2):
    _, _, res = p2
    e = res["effects"]
    inj = e[e["subset"] == "injected answers"].set_index("condition")
    assert abs(inj.loc["ctrl_near", "diff_vs_seg_plain"] - 0.6) < 0.2
    assert abs(inj.loc["ctrl_generic", "diff_vs_seg_plain"] - 0.2) < 0.2
    assert inj.loc["ctrl_near", "diff_vs_seg_plain"] > inj.loc["ctrl_far", "diff_vs_seg_plain"]
    # random timing: diluted per-injection effect
    assert inj.loc["rand_near", "diff_vs_seg_plain"] < inj.loc["ctrl_near", "diff_vs_seg_plain"]
    # answers the rule never touched are unchanged: overall effect = share injected x effect
    allr = e[e["subset"] == "all answers"].set_index("condition")
    s = res["summary"]
    approx = s.loc["ctrl_near", "share_injected"] * inj.loc["ctrl_near", "diff_vs_seg_plain"]
    assert abs(allr.loc["ctrl_near", "diff_vs_seg_plain"] - approx) < 1e-9


def test_detection_and_outputs(p2):
    cfg, _, res = p2
    assert res["detection"]["auroc_min_conf"] > 0.95
    assert res["detection"]["accuracy_if_rule_fires"] < res["detection"]["accuracy_if_rule_silent"]
    rows = [json.loads(x) for x in open(f"{cfg.out_dir}/results.jsonl")]
    assert len(rows) == 40 * 6 * 6
    plain = [r for r in rows if r["condition"] == "seg_plain"]
    assert all(r["n_injections"] == 0 for r in plain)
    for f in ("summary.csv", "effects.csv", "pairing_check.csv", "detection.json",
              "injection_effects.png"):
        assert (__import__("pathlib").Path(cfg.out_dir) / "analysis" / f).exists()
