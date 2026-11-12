"""Command line: marsseg download | train | evaluate | predict | perseverance | figures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DATA = Path(os.environ.get("MARSSEG_DATA", Path.home() / ".marsseg"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="marsseg", description="Mars rover terrain segmentation on AI4Mars")
    ap.add_argument("--data", type=Path, default=DATA, help="data root (default $MARSSEG_DATA or ~/.marsseg)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch MSL NAVCAM (+ optional Perseverance) parts of AI4Mars v0.6")
    d.add_argument("--m2020", action="store_true", help="also fetch Perseverance NAVCAM images and labels")
    d.add_argument("--workers", type=int, default=8)

    t = sub.add_parser("train")
    t.add_argument("--arch", default="segformer-b2", choices=["segformer-b2", "deeplabv3p-r50"])
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--size", type=int, default=512)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--budget-min", type=float, default=None, help="fit the LR schedule into this many minutes")
    t.add_argument("--out", type=Path, default=Path("runs"))

    e = sub.add_parser("evaluate", help="expert test sets; fits temperature on the validation split")
    e.add_argument("checkpoint", type=Path)
    e.add_argument("--mc", type=int, default=10, help="Monte-Carlo dropout passes for the uncertainty run")
    e.add_argument("--out", type=Path, default=Path("results"))

    p = sub.add_parser("predict", help="segment arbitrary images and write overlays")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("predictions"))
    p.add_argument("--mc", type=int, default=10)

    sub.add_parser("perseverance", help="fetch a few recent Perseverance NAVCAM frames (NASA raw-image feed)")

    b = sub.add_parser("rare-bias", help="fit a big-rock logit bias on held-out frames and re-score the test set")
    b.add_argument("checkpoint", type=Path)
    b.add_argument("--results", type=Path, default=Path("results"))

    f = sub.add_parser("figures")
    f.add_argument("checkpoint", type=Path)
    f.add_argument("--results", type=Path, default=Path("results"))
    f.add_argument("--out", type=Path, default=Path("figures"))

    a = ap.parse_args(argv)
    root = a.data / "ai4mars"

    if a.cmd == "download":
        from . import download
        download.fetch(root, download.MSL_PREFIXES, a.workers)
        if a.m2020:
            download.fetch(root, download.M2020_PREFIXES + ["m2020/labels/NAV/"], a.workers)
    elif a.cmd == "train":
        from . import train
        print(train.run(root, a.out, a.arch, a.epochs, a.batch, a.size, a.lr, a.workers, budget_min=a.budget_min))
    elif a.cmd == "evaluate":
        from . import data as D, evaluate as E, model as M
        import torch
        m = M.load(a.checkpoint)
        from .train import splits
        _, _, calib = splits(root)
        T = E.fit_temperature(m, calib)
        print(f"temperature {T:.3f}")
        ck = torch.load(a.checkpoint, weights_only=False)
        ck["temperature"] = T
        torch.save(ck, a.checkpoint)
        a.out.mkdir(parents=True, exist_ok=True)
        tag = a.checkpoint.stem
        res = {"temperature": T,
               "temperature_scaled": E.evaluate(m, root, temperature=T),
               f"mc{a.mc}": E.evaluate(m, root, temperature=T, mc=a.mc)}
        pairs = D.m2020_pairs(root)
        if pairs:
            res["m2020_crowd"] = E.evaluate_pairs(m, pairs[::4], temperature=T)  # every 4th frame
            print("Perseverance (crowd labels):", {k: res["m2020_crowd"][k] for k in ("n_images", "pixel_acc", "miou")})
        E.save(res, a.out / f"{tag}_test.json")
    elif a.cmd == "predict":
        from .predict import run
        run(a.checkpoint, a.images, a.out, a.mc)
    elif a.cmd == "rare-bias":
        from . import evaluate as E, model as M
        from .train import splits
        m = M.load(a.checkpoint)
        _, _, calib = splits(root)
        fit = E.fit_class_bias(m, calib, temperature=m.temperature)
        path = a.results / f"{a.checkpoint.stem}_test.json"
        res = json.loads(path.read_text())
        res["bigrock_bias"] = {"fit_on": "val-calib (held-out crowd-labelled frames)", **fit,
                               "test": E.evaluate_with_bias(m, root, fit["chosen_bias"], temperature=m.temperature)}
        path.write_text(json.dumps(res, indent=1))
    elif a.cmd == "perseverance":
        from . import perseverance
        print(perseverance.fetch(a.data / "perseverance"))
    elif a.cmd == "figures":
        from . import figures
        figures.make_all(a.checkpoint, root, a.data, a.results, a.out)


if __name__ == "__main__":
    main()
