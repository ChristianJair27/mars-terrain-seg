"""README figures (dark theme)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from . import data as D  # noqa: E402
from . import hazard  # noqa: E402
from .evaluate import predict  # noqa: E402
from .metrics import entropy  # noqa: E402
from .model import load  # noqa: E402

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
CMAP = ListedColormap(D.COLORS / 255.0)
RISK = ListedColormap(["#2ea043", "#d29922", "#da3633"])
# Swan et al. (2021), Table 4: DeepLabv3+ ResNet-101 trained on merged MSL NAVCAM labels,
# confusion matrix (% of each true class) against the 3-label-agreement expert test set; accuracy 96.67 %.
PAPER_MIN3 = np.array([[99.10, 0.32, 0.57, 0.01],
                       [3.64, 94.90, 0.37, 1.09],
                       [0.88, 5.62, 93.45, 0.05],
                       [6.76, 0.00, 0.00, 93.24]])
PAPER_MIN3_ACC = 96.67


def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
        "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": GRID,
        "font.size": 11, "axes.titlesize": 12, "font.family": "DejaVu Sans",
    })


def _legend(ax, loc="lower center"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=D.COLORS[i] / 255, label=c) for i, c in enumerate(D.CLASSES)],
              loc=loc, ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))


def masked(cls: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(cls == D.IGNORE, cls)


@torch.no_grad()
def infer(model, sample: D.Sample, mc=10):
    img = D.load_image(sample.image)
    lab = D.load_label(sample)
    inv = D.invalid_mask(sample, img.shape)
    lab[inv] = D.IGNORE
    small = np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(img).resize((512, 512)))
    x = D.to_tensor(small).unsqueeze(0).cuda()
    p, mi = predict(model, x, mc=mc, temperature=getattr(model, "temperature", 1.0), out_size=img.shape)
    p0 = p[0]
    return img, lab, inv, p0.cpu().numpy(), entropy(p0).cpu().numpy(), mi[0].cpu().numpy()


def examples(model, samples, out: Path, n=4, seed=3):
    rng = np.random.default_rng(seed)
    pick = [samples[i] for i in rng.choice(len(samples), n, replace=False)]
    fig, axes = plt.subplots(n, 5, figsize=(17, 3.6 * n))
    titles = ["Curiosity NAVCAM", "expert label (min1)", "prediction", "uncertainty (entropy)", "drive risk"]
    for r, s in enumerate(pick):
        img, lab, inv, p, ent, mi = infer(model, s)
        pred = p.argmax(0)
        pred = np.where(inv, D.IGNORE, pred)
        risk = hazard.risk_map(p, ent, invalid=inv)
        panels = [(img, "gray", None), (masked(lab), CMAP, (0, 3)), (masked(pred), CMAP, (0, 3)),
                  (np.ma.masked_where(inv, ent), "magma", (0, 1)), (np.ma.masked_invalid(risk), "RdYlGn_r", (0, 1))]
        for c, (a, cm, lim) in enumerate(panels):
            ax = axes[r, c]
            if c in (1, 2, 4):
                ax.imshow(img, cmap="gray")
            ax.imshow(a, cmap=cm, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1],
                      alpha=1 if c in (0, 3) else 0.6, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c])
        axes[r, 0].set_ylabel(s.name[:16], fontsize=8)
    fig.subplots_adjust(wspace=0.02, hspace=0.04, bottom=0.04)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=D.COLORS[i] / 255, label=c) for i, c in enumerate(D.CLASSES)],
               loc="lower center", ncol=4, frameon=False)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def iou_bars(res: dict, out: Path, key="temperature_scaled"):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    splits = ["test-min1", "test-min2", "test-min3"]
    w = 0.26
    x = np.arange(5)
    for k, sp in enumerate(splits):
        r = res[key][sp]
        vals = [r["iou"][c] for c in D.CLASSES] + [r["miou"]]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=f"{sp.replace('test-', '')}  (acc {100 * r['pixel_acc']:.1f} %)",
                      color=["#58a6ff", "#a371f7", "#3fb950"][k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, D.CLASSES + ["mean (mIoU)"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("IoU on the expert test set")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Per-class IoU by expert agreement level (minK = K of 3 experts labelled the pixel, 100 % agreement)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def confusion_vs_paper(res: dict, out: Path, key="temperature_scaled"):
    ours = np.array(res[key]["test-min3"]["confusion_row_pct"])
    acc = 100 * res[key]["test-min3"]["pixel_acc"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, m, t in [(axes[0], PAPER_MIN3, f"Swan et al. 2021, DeepLabv3+ R101\naccuracy {PAPER_MIN3_ACC:.2f} %"),
                     (axes[1], ours, f"this repo, SegFormer-B2\naccuracy {acc:.2f} %")]:
        ax.imshow(m, cmap="Blues", vmin=0, vmax=100)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", color="black" if m[i, j] > 50 else FG,
                        fontsize=10)
        ax.set_xticks(range(4), D.CLASSES)
        ax.set_yticks(range(4), D.CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(t, fontsize=10)
    fig.suptitle("Confusion matrices (% of each true class), MSL NAVCAM expert test set, 3-label agreement",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def reliability(res: dict, out: Path, mc_key: str):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    for key, lab, col in [("raw", "softmax", "#f85149"), ("temperature_scaled", "temperature-scaled", "#58a6ff"),
                          (mc_key, "temp.-scaled + MC dropout", "#3fb950")]:
        r = res[key]["test-min1"]
        rel = r["reliability"]
        n = np.array(rel["bin_n"])
        ok = n > 200
        ax.plot(np.array(rel["bin_conf"])[ok], np.array(rel["bin_acc"])[ok], "o-", color=col, ms=4,
                label=f"{lab}: ECE {100 * r['ece']:.2f} %")
    ax.set_xlabel("confidence of the predicted class")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.25, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Reliability, expert test set (min1)", fontsize=11)
    ax.grid(color=GRID, lw=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def hazard_demo(model, sample, out: Path):
    img, lab, inv, p, ent, mi = infer(model, sample)
    risk = hazard.risk_map(p, ent, invalid=inv)
    z = hazard.zones(risk)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img, cmap="gray")
    axes[0].imshow(masked(np.where(inv, D.IGNORE, p.argmax(0))), cmap=CMAP, vmin=0, vmax=3, alpha=0.45)
    axes[0].set_title("terrain classes")
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(np.ma.masked_where(z == 255, z), cmap=RISK, vmin=0, vmax=2, alpha=0.45)
    axes[1].set_title("drive risk: green go, amber caution, red no-go (illustrative costs)")
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=D.COLORS[i] / 255, label=c) for i, c in enumerate(D.CLASSES)],
               loc="lower left", ncol=4, frameon=False)
    fig.suptitle(f"Curiosity NAVCAM {sample.name}  (NASA/JPL-Caltech); rover and >30 m masked", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def training_curves(log_path: Path, out: Path):
    log = json.loads(Path(log_path).read_text())
    ep = [r["epoch"] + 1 for r in log]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(ep, [r["loss"] for r in log], "o-", color="#f85149", label="train loss")
    ax2 = ax.twinx()
    ax2.plot(ep, [r["val_miou"] for r in log], "s-", color="#3fb950", label="val mIoU (crowd labels)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel("mIoU")
    for a in (ax, ax2):
        a.tick_params(colors=FG)
    fig.legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def perseverance_panel(model, paths: list[Path], out: Path):
    from .predict import segment
    n = len(paths)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3.4 * n))
    axes = np.atleast_2d(axes)
    for r, path in enumerate(paths):
        img, p, ent, mi = segment(model, path)
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 1].imshow(img, cmap="gray")
        axes[r, 1].imshow(p.argmax(0), cmap=CMAP, vmin=0, vmax=3, alpha=0.5, interpolation="nearest")
        axes[r, 2].imshow(ent, cmap="magma", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(path.stem[:22], fontsize=7)
        for c, t in enumerate(["Perseverance NAVCAM (raw)", "prediction", "uncertainty (entropy)"]):
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(t)
    fig.suptitle("Out of domain, qualitative only: trained on Curiosity, applied to Perseverance (NASA/JPL-Caltech)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def make_all(checkpoint: Path, root: Path, data_root: Path, results: Path, out: Path):
    style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = load(checkpoint)
    res = json.loads((Path(results) / f"{Path(checkpoint).stem}_test.json").read_text())
    mc_key = next(k for k in res if k.startswith("mc"))
    test = D.msl_samples(root, "test-min1")
    examples(model, test, out / "examples.png")
    iou_bars(res, out / "iou_by_agreement.png")
    confusion_vs_paper(res, out / "confusion_vs_paper.png")
    reliability(res, out / "reliability.png", mc_key)
    hazard_demo(model, test[len(test) // 3], out / "hazard_map.png")
    log = Path(checkpoint).with_name(Path(checkpoint).stem + "_log.json")
    if log.exists():
        training_curves(log, out / "training.png")
    pers = sorted((Path(data_root) / "perseverance").glob("*.png")) + sorted(
        (Path(data_root) / "perseverance").glob("*.jpg"))
    if pers:
        perseverance_panel(model, pers[:4], out / "perseverance_ood.png")
