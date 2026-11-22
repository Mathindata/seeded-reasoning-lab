"""python -m sr run --config configs/pilot.json
   python -m sr analyze --run runs/pilot"""
import argparse

import pandas as pd

from .analyze import analyze
from .backends import make_backend
from .run import Config, merge, run


def main():
    ap = argparse.ArgumentParser(prog="sr")
    ap.add_argument("cmd", choices=["run", "analyze", "merge", "p2-calibrate", "p2-run", "p2-analyze"])
    ap.add_argument("--runs", nargs="*", default=[], help="shard run directories to merge")
    ap.add_argument("--out", default="runs/main", help="merged run directory")
    ap.add_argument("--config", default="configs/pilot.json")
    ap.add_argument("--run", default=None, help="run directory for analyze")
    ap.add_argument("--backend", default=None, help="override: vllm | hf | mock")
    ap.add_argument("--model", default=None, help="override the model name")
    a = ap.parse_args()
    if a.cmd.startswith("p2-"):
        return phase2_main(a)
    if a.cmd == "merge":
        print("merged into", merge(a.runs, a.out))
        return
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


def phase2_main(a):
    from . import phase2
    if a.cmd == "p2-analyze":
        res = phase2.analyze2(a.run)
        with pd.option_context("display.width", 200, "display.precision", 3):
            print(res["summary"]); print(res["effects"].to_string(index=False))
            print(res["pairing"].to_string(index=False)); print(res["detection"])
        return
    cfg = phase2.Phase2Config.load(a.config)
    cfg.backend = a.backend or cfg.backend
    cfg.model = a.model or cfg.model
    backend = make_backend(cfg)
    test, calib = phase2.split_problems(cfg)
    if a.cmd == "p2-calibrate":
        phase2.calibrate(cfg, backend, calib)
    else:
        phase2.run_main(cfg, backend, test)


if __name__ == "__main__":
    main()
