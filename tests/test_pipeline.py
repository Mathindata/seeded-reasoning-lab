"""End-to-end with the mock model, plus unit tests of the parts that are easy to get wrong."""
import json
import random

import numpy as np
import pytest

from sr import data, prompts, seeds
from sr.analyze import analyze, pass_at_k
from sr.backends import MockBackend
from sr.run import Config, plan_jobs, run


def fake_items(n=40):
    rng = random.Random(0)
    return [{"id": f"p{i}", "problem": f"Problem number {i}: compute something.",
             "answer": str(rng.randint(100, 999)), "level": 3 + i % 3, "subject": "Algebra",
             "benchmark": "math500"} for i in range(n)]


def test_pass_at_k_matches_definition():
    assert pass_at_k(8, 0, 4) == 0.0
    assert pass_at_k(8, 8, 1) == 1.0
    assert np.isclose(pass_at_k(8, 2, 1), 2 / 8)
    # brute force: probability that a random 3-subset of 8 contains a correct one
    from itertools import combinations
    pool = [1, 1, 0, 0, 0, 0, 0, 0]
    subs = list(combinations(range(8), 3))
    brute = np.mean([any(pool[i] for i in s) for s in subs])
    assert np.isclose(pass_at_k(8, 2, 3), brute)


def test_seeds_are_reproducible_per_job():
    cfg = Config(backend="mock", k_samples=3, conditions=["far_translated", "noise_translated"])
    a = plan_jobs(cfg, fake_items(3))
    b = plan_jobs(cfg, fake_items(3))
    assert [j["seed"] for j in a] == [j["seed"] for j in b]
    assert len({j["seed"] for j in a if j["condition"] == "far_translated"}) > 1


def test_prompts_keep_conditions_separate():
    m = prompts.solve_messages("P?", "plain")
    assert m[-1]["content"] == "P?"
    m = prompts.solve_messages("P?", "far_raw", "Daoism", "water flows")
    assert "Daoism" in m[-1]["content"] and "strategy" not in m[-1]["content"]
    m = prompts.solve_messages("P?", "near_translated", strategy="Use symmetry.")
    assert "Use symmetry." in m[-1]["content"] and "physics" not in m[-1]["content"]
    assert prompts.clean_strategy("Work backwards. Then x. Then y. \\boxed{3}") == \
        "Work backwards. Then x."


def test_banks_have_no_duplicates():
    for bank in (seeds.POLYA, seeds.NEAR, seeds.FAR):
        texts = [t for _, t in bank]
        assert len(texts) == len(set(texts))
    assert len(seeds.WORDS) == len(set(seeds.WORDS))


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    items = fake_items(60)
    cfg = Config(backend="mock", k_samples=8, out_dir=str(d), chunk=300)
    answers = {it["problem"]: it["answer"] for it in items}
    # planted effect: translated conditions are better than plain; far_raw is worse
    be = MockBackend(1, answers, {"plain": 0.35, "translated": 0.55, "far_raw": 0.2})
    run(cfg, be, items, log=lambda *a: None)
    return cfg, items, be


def test_run_is_complete_and_resumable(mock_run):
    cfg, items, be = mock_run
    rows = [json.loads(x) for x in open(f"{cfg.out_dir}/results.jsonl")]
    assert len(rows) == len(items) * len(cfg.conditions) * cfg.k_samples
    run(cfg, be, items, log=lambda *a: None)    # rerun: nothing new is added
    assert len(open(f"{cfg.out_dir}/results.jsonl").readlines()) == len(rows)
    tr = [r for r in rows if r["condition"] == "far_translated"]
    assert all(r["strategy"] and r["tokens_stage1"] > 0 for r in tr)
    assert all(r["tokens_stage1"] == 0 for r in rows if r["condition"] == "plain")


def test_analysis_detects_planted_effect(mock_run):
    cfg, *_ = mock_run
    res = analyze(cfg.out_dir, n_boot=500)
    s = res["summary"]
    assert s.loc["far_translated", "pass@1"] > s.loc["plain", "pass@1"] + 0.1
    assert s.loc["far_raw", "pass@1"] < s.loc["plain", "pass@1"]
    c = res["comparisons"]
    row = c[(c.condition == "far_translated") & (c.vs == "plain") & (c.metric == "pass@1")]
    assert bool(row["clear"].iloc[0]) and row["diff"].iloc[0] > 0
    # no planted difference among translated banks: their gap should be small
    assert abs(s.loc["far_translated", "pass@1"] - s.loc["near_translated", "pass@1"]) < 0.08
    for f in ("summary.csv", "comparisons.csv", "pass_at_k.png", "diff_vs_plain.png",
              "seeds.csv", "mixture_coverage.csv"):
        assert (cfg.out_dir and __import__("pathlib").Path(cfg.out_dir, "analysis", f).exists())


def test_game24_checker():
    assert data.check_game24("\\boxed{(13-9)*(3+3)}", [3, 3, 9, 13])
    assert not data.check_game24("\\boxed{(13-9)*(3+2)}", [3, 3, 9, 13])     # wrong numbers
    assert not data.check_game24("\\boxed{4*6}", [4, 6, 1, 1])               # unused numbers
    assert not data.check_game24("\\boxed{__import__('os')}", [1, 2, 3, 4])  # not arithmetic
    assert data.check_game24("\\boxed{8 \\div (3 - 8 \\div 3)}", [3, 3, 8, 8])


def test_math_checker_equivalence():
    assert data.check_math("so \\boxed{\\dfrac{1}{2}}", "\\frac12")
    assert data.check_math("\\boxed{0.5}", "\\frac{1}{2}")
    assert not data.check_math("\\boxed{3}", "4")
    assert data.last_boxed("a \\boxed{\\frac{1}{2}} b") == "\\frac{1}{2}"


def test_shards_cover_all_problems_once_and_merge(tmp_path):
    from sr.analyze import analyze
    from sr.run import merge
    items = fake_items(10)
    answers = {it["problem"]: it["answer"] for it in items}
    dirs = []
    for s in (0, 1):
        cfg = Config(backend="mock", k_samples=2, out_dir=str(tmp_path / f"s{s}"), shard=s,
                     n_shards=2)
        run(cfg, MockBackend(s, answers), items, log=lambda *a: None)
        dirs.append(cfg.out_dir)
    ids = [json.load(open(f"{d}/problems.json")) for d in dirs]
    a, b = {i["id"] for i in ids[0]}, {i["id"] for i in ids[1]}
    assert not a & b and a | b == {it["id"] for it in items}
    out = merge(dirs, str(tmp_path / "main"))
    rows = open(out / "results.jsonl").readlines()
    assert len(rows) == 10 * 6 * 2
    assert analyze(str(out), n_boot=100)["info"]["problems"] == 10
