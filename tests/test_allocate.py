"""Exact policy expectations on a hand-checkable problem with 4 logged answers."""
import pandas as pd
import pytest

from sr.allocate import policy_table


def plain(fired):
    return pd.DataFrame({"id": ["p"] * 4, "sample": range(4), "pred": ["7", "7", "8", "9"],
                         "correct": [1.0, 1.0, 0.0, 0.0], "gen_tokens": [100] * 4,
                         "min_conf": [-0.3, -0.1, -0.5, -0.9], "fired": fired,
                         "tokens_to_fire": [50, 0, 0, 0]})


def test_agreement_policy_exact():
    t = policy_table(plain([False] * 4), quantiles=[1.0]).set_index("policy")
    # first two agree only for the ordered pairs (0,1),(1,0): 2 of 12; otherwise the vote of all
    # 4 answers picks 7, which is right
    assert t.loc["agree_2then4", "correct"] == pytest.approx(1.0)
    assert t.loc["agree_2then4", "gen_tokens"] == pytest.approx(2 / 12 * 200 + 10 / 12 * 400)
    assert t.loc["rule_vote3", "correct"] == pytest.approx(0.5)
    assert t.loc["rule_vote3", "gen_tokens"] == pytest.approx(100)


def test_rule_stop_policy_exact():
    t = policy_table(plain([True, False, False, False]), quantiles=[1.0]).set_index("policy")
    # answer 0 first (1/4): stopped at 50 tokens, vote of 7, 8, 9 ties -> most confident (7)
    assert t.loc["rule_stop_vote3", "correct"] == pytest.approx(0.25 * 1 + 0.75 * (1 / 3))
    assert t.loc["rule_stop_vote3", "gen_tokens"] == pytest.approx(0.25 * 350 + 0.75 * 100)
    # q = 1.0 flags every answer: always the vote of 3
    assert t.loc["conf_q1.0_vote3", "gen_tokens"] == pytest.approx(300)
