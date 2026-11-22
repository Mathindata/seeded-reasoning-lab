# Seeded reasoning: do ideas from distant fields help a small LLM solve math?

**Question.** People often solve a problem with an idea borrowed from a far-away field.
If a small LLM (Qwen2.5-1.5B-Instruct) gets a one-sentence strategy derived from an idea
in Daoism, cooking or martial arts, does it solve more math problems than with a math
heuristic, a physics principle, or random words? And does the *distance* of the source
field matter, once the extra step and extra tokens are controlled for?

> Status: design, code and tests are done; predictions are written below **before** any real
> run. Results will be added after the Kaggle run, whatever they show.

## Design

Each problem is solved k = 8 times under each of six conditions (temperature 0.7):

| Condition | What the model sees | Role |
|---|---|---|
| `plain` | the problem | baseline |
| `noise_translated` | random words, turned into a one-sentence strategy, then the problem | control: same extra step, no meaningful idea |
| `polya_translated` | a math heuristic (after Polya), turned into a strategy | same domain |
| `near_translated` | a physics, engineering or CS principle, turned into a strategy | near domain |
| `far_translated` | an idea from philosophy, arts, nature, crafts or strategy, turned into a strategy | far domain |
| `far_raw` | the far idea pasted next to the problem, no translation | tests distraction |

"Translated" means two calls to the same model: first a coach prompt turns the idea into a
general problem-solving move for this problem (without solving it), then the solver gets the
problem plus that move. Conditions 2 to 5 differ only in where the seed comes from, so
differences among them are about the source of the idea. All seeds are short paraphrases written
for this project (`src/sr/seeds.py`), not quotations.

**Data.** MATH-500, levels 3 to 5, 200 problems stratified by level, checked with
[math-verify](https://github.com/huggingface/Math-Verify) (compares answers as math, so 0.5 = 1/2).
Optional: Game of 24, generated and checked exactly here.

**Metrics.** Accuracy (pass@1); coverage pass@k for k = 1, 2, 4, 8 (unbiased estimator);
majority-vote accuracy; number of distinct answers; tokens per sample (both stages counted);
and *mixture coverage*: with a fixed budget of 8 samples, does half plain plus half seeded cover
more problems than 8 plain? All comparisons are paired by problem with 95% bootstrap intervals.

## Predictions (written before running)

1. `far_raw` will be **worse** than `plain` on pass@1: a pasted quote is irrelevant context, and
   models are easily distracted by it (Shi et al., 2023).
2. On pass@1, the translated conditions will be close to **each other**, including the noise
   control. Earlier work found that self-generated random examples matched or beat relevant ones
   on GSM8K and MATH, and that the accuracy of the examples mattered more than their relevance
   (Qin et al., 2025). So any gain may come from the extra step, not from the idea.
3. If distance matters at all, it will show in **coverage** (pass@8 and mixture coverage), not in
   pass@1: far seeds should produce more *different* attempts, so some new problems get solved.
   The test is `far_translated` against `noise_translated`, not against `plain`.

A null result on prediction 3 is a valid result and will be reported as such.

## Run it

Kaggle (T4 x2, Internet on): `notebooks/run_kaggle.ipynb`. Local:

```bash
pip install -e ".[dev,vllm]"          # or [hf] if vLLM does not run on your GPU
pytest -q                             # 8 tests, uses a mock model
python -m sr run --config configs/pilot.json      # 30 problems, k = 4
python -m sr run --config configs/main.json       # 200 problems, k = 8
python -m sr analyze --run runs/main
```

Compute: the main run is 9,600 solve generations (up to 1,024 tokens) plus 6,400 short
strategy generations. Speed is measured in the pilot; the run resumes where it stopped.
Every generation (seed, strategy, full response, tokens, correctness) is logged to
`runs/*/results.jsonl`.

## Roadmap

1. **Static seeds** (this repo): does the source of the idea matter?
2. **Controller**: every k reasoning steps, check the model's confidence (mean token
   log-probability); if it stays low for m checks, switch to a new seed (hysteresis, so it does
   not switch back and forth). Sweep k: switching too often is known to hurt (Wang et al., 2025).
3. **Learned seed choice**: a bandit that picks the seed bank from problem features, using the
   per-seed table in `analysis/seeds.csv`.
4. **Abandoned branches**: log the paths the controller leaves and study what carries over into
   the final answer.

## Related work

- Yasunaga et al., *Large Language Models as Analogical Reasoners*, ICLR 2024: the model
  self-generates relevant examples before solving.
- Qin et al., *Relevant or Random: Can LLMs Truly Perform Analogical Reasoning?*, Findings of
  ACL 2025: random self-generated examples matched or beat relevant ones on math.
- Zhou et al., *Self-Discover: Large Language Models Self-Compose Reasoning Structures*,
  NeurIPS 2024: the model selects and composes general reasoning modules.
- Shi et al., *Large Language Models Can Be Easily Distracted by Irrelevant Context*, ICML 2023.
- Wang et al., *Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs*, 2025:
  frequent thought switching correlates with wrong answers.
- rSIM, *Incentivizing Reasoning Capabilities of LLMs via Reinforced Strategy Injection*
  (arXiv 2512.08300): a planner injects a strategy from a fixed set at each reasoning step.
  This project asks what happens when the strategy set is open and comes from other fields.

## Limitations

One small model, one benchmark (plus an optional second), seeds written by one person, and a
fixed translation prompt. Results say what happens for this model and setup, not for LLMs in
general.
