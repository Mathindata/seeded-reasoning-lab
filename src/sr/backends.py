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
                 gpu_memory_utilization: float = 0.85, tensor_parallel_size: int = 1):
        from vllm import LLM
        self.llm = LLM(model=model, dtype="half", seed=seed, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       tensor_parallel_size=tensor_parallel_size)
        self.tok = self.llm.get_tokenizer()

    def generate(self, batch, max_tokens, temperature, top_p):
        from vllm import SamplingParams
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in batch]
        sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        outs = self.llm.generate(prompts, sp)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


class HFBackend:
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


def make_backend(cfg, answers=None, p_correct=None):
    if cfg.backend == "vllm":
        return VLLMBackend(cfg.model, cfg.seed, cfg.max_model_len, cfg.gpu_memory_utilization,
                           cfg.tensor_parallel_size)
    if cfg.backend == "hf":
        return HFBackend(cfg.model, cfg.seed, cfg.hf_batch_size)
    if cfg.backend == "mock":
        return MockBackend(cfg.seed, answers, p_correct)
    raise ValueError(cfg.backend)
