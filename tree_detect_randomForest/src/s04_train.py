"""
Step 04 — Fine-tune DeepForest on the train AOI tiles. Full training run.

- Device: MPS (Apple GPU) when available, else CPU; 32-bit precision (no AMP).
- Validation: the held-out TEST AOI tiles, so early stopping tracks generalization
  (the spatial split and labels are identical to the smoke test; only the validation
  source moved from train -> test, per the README's "point it at the test tiles once
  you trust the loop"). NOTE: with only two AOIs this means the final test metrics are
  selection-biased (checkpoint chosen on the same set used to report).
- Early stopping on validation box_recall (IoU 0.4), patience from config.
- Checkpoints: best-by-val-recall AND last (saved every epoch), to outputs/model/.
- Prints per-epoch wall-clock; writes a per-epoch metrics CSV + a training-curve PNG.

Epoch count comes from config.yaml (train.epochs); set TD_EPOCHS to override.
"""
import os
# Must precede any torch import so unsupported MPS ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import csv
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

MONITOR = "box_recall"   # DeepForest logs this per epoch at IoU 0.4 (validation.iou_threshold)


def _f1(p, r):
    if p is None or r is None or (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


class EpochRecorder(Callback):
    """Per-epoch wall-clock + metric capture. Read at on_validation_end so every
    DeepForest-logged metric is already in trainer.callback_metrics."""
    def __init__(self):
        super().__init__()
        self.rows = []
        self._t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.time()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics

        def g(k):
            v = m.get(k)
            return float(v) if v is not None else None

        secs = (time.time() - self._t0) if self._t0 else float("nan")
        epoch = int(trainer.current_epoch)
        tl, vl = g("train_loss"), g("val_loss")
        prec, rec = g("box_precision"), g(MONITOR)
        f1 = _f1(prec, rec)
        self.rows.append({"epoch": epoch, "train_loss": tl, "val_loss": vl,
                          "val_precision": prec, "val_recall": rec, "val_f1": f1,
                          "seconds": round(secs, 1)})

        def fmt(x, n):
            return "n/a" if x is None else round(x, n)
        print(f"[epoch {epoch:02d}] {secs:6.1f}s  train_loss={fmt(tl,4)}  "
              f"val_precision={fmt(prec,3)}  val_recall={fmt(rec,3)}  val_f1={round(f1,3)}",
              flush=True)


def main():
    cfg = C.load_config()
    from deepforest import main as df_main

    od = C.p(cfg, cfg["outputs_dir"])
    train_dir = od / "tiles" / "train"
    test_dir = od / "tiles" / "test"
    model_dir = od / "model"; model_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    epochs = int(os.environ.get("TD_EPOCHS", cfg["train"]["epochs"]))
    batch = int(cfg["train"]["batch_size"])
    patience = int(cfg["train"].get("early_stop_patience", 8))
    iou_eval = cfg["train"]["iou_eval"]

    m = df_main.deepforest()              # deepforest 2.x auto-loads released weights

    # Train on the train AOI; validate on the held-out test AOI (identical labels/split).
    m.config["train"]["csv_file"] = str(train_dir / "train_tiles.csv")
    m.config["train"]["root_dir"] = str(train_dir)
    m.config["train"]["epochs"] = epochs
    m.config["train"]["batch_size"] = batch
    m.config["batch_size"] = batch
    m.config["score_thresh"] = cfg["train"]["score_thresh"]
    m.config["validation"]["csv_file"] = str(test_dir / "test_tiles.csv")
    m.config["validation"]["root_dir"] = str(test_dir)
    m.config["validation"]["iou_threshold"] = iou_eval                       # IoU 0.4
    m.config["validation"]["val_accuracy_interval"] = int(cfg["train"].get("val_accuracy_interval", 1))
    m.config["accelerator"] = device
    m.config["devices"] = 1

    early = EarlyStopping(monitor=MONITOR, mode="max", patience=patience, verbose=True)
    ckpt_best = ModelCheckpoint(
        dirpath=str(model_dir), filename="treedetect_best_recall",
        monitor=MONITOR, mode="max", save_top_k=1, save_last=True,
        enable_version_counter=False)
    rec = EpochRecorder()

    # kwargs override DeepForest's config-derived trainer args.
    m.create_trainer(callbacks=[early, ckpt_best, rec],
                     accelerator=device, devices=1, max_epochs=epochs,
                     precision="32-true")
    print(f"device={device}  max_epochs={epochs}  batch_size={batch}  "
          f"early_stop_patience={patience}  monitor={MONITOR}(max)  "
          f"validation=TEST AOI @ IoU {iou_eval}", flush=True)

    t_start = time.time()
    m.trainer.fit(m)
    total = time.time() - t_start

    # Legacy-named final checkpoint so s05/s06 always resolve a path.
    m.trainer.save_checkpoint(str(model_dir / "treedetect_finetuned.ckpt"))

    best_path = ckpt_best.best_model_path or "(none)"
    best_score = (float(ckpt_best.best_model_score)
                  if ckpt_best.best_model_score is not None else float("nan"))
    print(f"\nbest {MONITOR}={best_score:.3f}  ->  {best_path}", flush=True)
    print(f"total wall-clock: {total/60:.1f} min over {len(rec.rows)} epochs "
          f"(stopped_early={len(rec.rows) < epochs})", flush=True)

    _write_metrics(cfg, rec.rows)
    _curve_figure(cfg, rec.rows, best_score)


def _write_metrics(cfg, rows):
    out = C.p(cfg, cfg["outputs_dir"]) / "train_metrics.csv"
    fields = ["epoch", "train_loss", "val_precision", "val_recall", "val_f1", "val_loss", "seconds"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {out}", flush=True)


def _curve_figure(cfg, rows, best_score):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    if not rows:
        return
    ep = [r["epoch"] for r in rows]

    def series(key):
        return [(r["epoch"], r[key]) for r in rows if r.get(key) is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for key, color, lbl in [("train_loss", "#1b7837", "train loss"),
                            ("val_loss", "#d73027", "val loss")]:
        s = series(key)
        if s:
            ax1.plot(*zip(*s), "-o", color=color, label=lbl, markersize=3)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("s04 — loss"); ax1.legend()

    for key, color, lbl in [("val_precision", "#4575b4", "val precision"),
                            ("val_recall", "#d73027", "val recall"),
                            ("val_f1", "#7b3294", "val F1")]:
        s = series(key)
        if s:
            ax2.plot(*zip(*s), "-o", color=color, label=lbl, markersize=3)
    # mark best-recall epoch
    rrows = series("val_recall")
    if rrows:
        be, bv = max(rrows, key=lambda t: t[1])
        ax2.axvline(be, color="#999999", ls="--", lw=1)
        ax2.annotate(f"best recall {bv:.3f}\n(epoch {be})", xy=(be, bv),
                     xytext=(6, -4), textcoords="offset points", fontsize=8)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("score (IoU 0.4)")
    ax2.set_title("s04 — validation precision / recall / F1"); ax2.legend()
    fig.suptitle("treedetect — full training run (test-AOI validation)", fontsize=12)
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "train_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
