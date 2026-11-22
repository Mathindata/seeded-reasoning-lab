"""python -m sr run --config configs/pilot.json
   python -m sr analyze --run runs/pilot"""
import argparse

import pandas as pd

from .analyze import analyze
from .backends import make_backend
from .run import Config, run


def main():
    ap = argparse.ArgumentParser(prog="sr")
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--config", default="configs/pilot.json")
    ap.add_argument("--run", default=None, help="run directory for analyze")
    ap.add_argument("--backend", default=None, help="override: vllm | hf | mock")
    ap.add_argument("--model", default=None, help="override the model name")
    a = ap.parse_args()
    if a.cmd == "run":
        cfg = Config.load(a.config)
        cfg.backend = a.backend or cfg.backend
        cfg.model = a.model or cfg.model
        run(cfg, make_backend(cfg))
        a.run = cfg.out_dir
    res = analyze(a.run or Config.load(a.config).out_dir)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(res["info"])
        print(res["summary"])
        c = res["comparisons"]
        print(c[c["vs"] == "plain"].to_string(index=False))
        print(res["mixture"].to_string(index=False))


if __name__ == "__main__":
    main()
