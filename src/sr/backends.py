"""Text generation backends with one interface.

generate(batch, max_tokens, temperature, top_p) -> list of (text, n_tokens)
where batch is a list of chat message lists. Each item gets its own sample.

  vllm  fast batched inference (try first on Kaggle T4s; fp16)
  hf    plain transformers generate (slower fallback if vLLM does not install)
  mock  no model: deterministic fake answers, for tests and dry runs
"""
from __future__ import annotations

import random
import re


class VLLMBackend:
    def __init__(self, model: str, seed: int, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85, tensor_parallel_size: int = 1,
                 prefix_caching: bool = False):
        from vllm import LLM
        self.llm = LLM(model=model, dtype="half", seed=seed, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       tensor_parallel_size=tensor_parallel_size,
                       enable_prefix_caching=prefix_caching)
        self.tok = self.llm.get_tokenizer()

    def chat_prompt(self, messages) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def continue_raw(self, prompts, max_tokens, temperature, top_p, seeds):
        """Continue raw prompt strings. One seed per request, so the same (prompt, seed) gives
        the same text in every condition. Returns dicts with text, n_tokens, sum_logprob, finished."""
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                              seed=int(s), logprobs=1) for s in seeds]
        outs = self.llm.generate(prompts, sps, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            lp = c.cumulative_logprob
            if lp is None and c.logprobs:
                lp = sum(d[t].logprob for t, d in zip(c.token_ids, c.logprobs) if t in d)
            res.append({"text": c.text, "n_tokens": len(c.token_ids), "sum_logprob": lp or 0.0,
                        "finished": c.finish_reason == "stop"})
        return res

    def generate(self, batch, max_tokens, temperature, top_p):
        from vllm import SamplingParams
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in batch]
        sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        outs = self.llm.generate(prompts, sp)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


class HFContinueMixin:
    """Phase 2 support for the transformers backend. Note: one seed per batch, not per
    request, so conditions are NOT paired sample by sample with this backend."""

    def chat_prompt(self, messages) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def continue_raw(self, prompts, max_tokens, temperature, top_p, seeds):
        torch = self.torch
        res = []
        for i in range(0, len(prompts), self.bs):
            chunk = prompts[i:i + self.bs]
            torch.manual_seed(int(seeds[i]))
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                                          temperature=temperature, top_p=top_p,
                                          pad_token_id=self.tok.pad_token_id,
                                          return_dict_in_generate=True, output_scores=True)
            lps = self.model.compute_transition_scores(gen.sequences, gen.scores,
                                                       normalize_logits=True)
            new = gen.sequences[:, enc["input_ids"].shape[1]:]
            for row, lp in zip(new, lps):
                keep = [k for k, t in enumerate(row.tolist()) if t != self.tok.pad_token_id]
                ids = [row[k].item() for k in keep]
                finished = self.tok.eos_token_id in ids or len(ids) < max_tokens
                s = float(sum(lp[k].item() for k in keep if torch.isfinite(lp[k])))
                res.append({"text": self.tok.decode(ids, skip_special_tokens=True),
                            "n_tokens": len(ids), "sum_logprob": s, "finished": finished})
        return res


class HFBackend(HFContinueMixin):
    def __init__(self, model: str, seed: int, batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, device_map="auto")
        self.bs = batch_size

    def generate(self, batch, max_tokens, temperature, top_p):
        out = []
        for i in range(0, len(batch), self.bs):
            chunk = batch[i:i + self.bs]
            prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                                          temperature=temperature, top_p=top_p,
                                          pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for row in new:
                ids = [t for t in row.tolist() if t != self.tok.pad_token_id]
                out.append((self.tok.decode(ids, skip_special_tokens=True), len(ids)))
        return out




class MockBackend:
    """Fake model. Solve prompts return a boxed answer that is right with a probability
    set per condition (read from a marker in the prompt), so analysis code can be tested."""

    def __init__(self, seed: int, answers: dict[str, str] | None = None,
                 p_correct: dict[str, float] | None = None):
        self.rng = random.Random(seed)
        self.answers = answers or {}
        self.p = p_correct or {}

    def generate(self, batch, max_tokens, temperature, top_p):
        out = []
        for msgs in batch:
            user = msgs[-1]["content"]
            if "problem-solving coach" in msgs[0]["content"]:
                out.append(("Try working backwards from the goal. Then check small cases.", 14))
                continue
            key = next((k for k in self.answers if k in user), None)
            gold = self.answers.get(key, "0")
            kind = ("far_raw" if "A thought from" in user else
                    "translated" if "A strategy you may find useful" in user else "plain")
            p = self.p.get(kind, 0.5)
            ans = gold if self.rng.random() < p else str(self.rng.randint(1, 99))
            text = f"Step 1. Think.\nStep 2. Compute.\nSo the answer is \\boxed{{{ans}}}."
            out.append((text, len(re.findall(r"\S+", text))))
        return out

    # ---- Phase 2 (segmented generation) ----
    def chat_prompt(self, messages) -> str:
        return "<SYS>" + messages[0]["content"] + "<USER>" + messages[-1]["content"] + "<ASSIST>"

    def continue_raw(self, prompts, max_tokens, temperature, top_p, seeds):
        """Deterministic per seed. Each call writes one 'line'. A trajectory is 'lost' (low
        confidence, wrong answer) or 'on track'. An injected strategy can rescue a lost one
        with probability p_rescue[kind]; generic rethink with p_rescue['generic']."""
        res = []
        for prompt, s in zip(prompts, seeds):
            rng = random.Random(int(s))
            body = prompt.split("<ASSIST>", 1)[1]
            key = next((k for k in self.answers if k in prompt), None)
            gold = self.answers.get(key, "0")
            # trajectory type fixed by the part of the seed shared across segments
            traj = random.Random(int(s) // 1000)
            lost = traj.random() < 0.5
            n_lines = body.count("\nline")
            if "[RETHINK" in body:
                kind = body.rsplit("[RETHINK:", 1)[1].split("]", 1)[0]
                rescued = random.Random(int(s) // 1000 + 7).random() < self.p.get(kind, 0.0)
                lost = lost and not rescued
            conf = -2.5 + rng.random() * 0.3 if lost else -0.3 - rng.random() * 0.3
            if n_lines >= 4:
                ans = str(int(gold) + 1) if lost else gold
                text = f"\nline final \\boxed{{{ans}}}"
                res.append({"text": text, "n_tokens": 8, "sum_logprob": conf * 8,
                            "finished": True})
            else:
                res.append({"text": f"\nline {n_lines}", "n_tokens": 8, "sum_logprob": conf * 8,
                            "finished": False})
        return res


class MockPathBackend(MockBackend):
    """Phase 3 mock with a planted truth that lives in the path, not in the problem.
    Every segment is drawn from its own seed: 'bad' with probability p_bad (confidence about
    -2.5 per token), else 'good' (about -0.3). After `n_lines` segments the answer is written;
    it is right only if no bad segment is left in the text. So discarding a bad segment and
    resampling it can rescue an answer, and choosing the most confident of several branches
    rescues it more often."""

    def __init__(self, seed: int, answers: dict[str, str] | None = None, p_bad: float = 0.25,
                 n_lines: int = 5):
        super().__init__(seed, answers)
        self.p_bad, self.n_lines = p_bad, n_lines

    def continue_raw(self, prompts, max_tokens, temperature, top_p, seeds):
        res = []
        for prompt, s in zip(prompts, seeds):
            rng = random.Random(int(s))
            body = prompt.split("<ASSIST>", 1)[1]
            key = next((k for k in self.answers if k in prompt), None)
            gold = self.answers.get(key, "0")
            bad = rng.random() < self.p_bad
            conf = -2.5 + rng.random() * 0.2 if bad else -0.3 - rng.random() * 0.2
            n = body.count("\nline")
            tag = "B" if bad else "G"
            if n >= self.n_lines:
                ok = " B" not in body and not bad
                ans = gold if ok else str(int(gold) + 1)
                res.append({"text": f"\nline end {tag} \\boxed{{{ans}}}", "n_tokens": 8,
                            "sum_logprob": conf * 8, "finished": True})
            else:
                res.append({"text": f"\nline {n} {tag}", "n_tokens": 8, "sum_logprob": conf * 8,
                            "finished": False})
        return res


def make_backend(cfg, answers=None, p_correct=None):
    if cfg.backend == "vllm":
        return VLLMBackend(cfg.model, cfg.seed, cfg.max_model_len, cfg.gpu_memory_utilization,
                           cfg.tensor_parallel_size,
                           prefix_caching=getattr(cfg, "prefix_caching", False))
    if cfg.backend == "hf":
        return HFBackend(cfg.model, cfg.seed, cfg.hf_batch_size)
    if cfg.backend == "mock":
        return MockBackend(cfg.seed, answers, p_correct)
    raise ValueError(cfg.backend)
