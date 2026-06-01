"""
Step 04 — Fine-tune DeepForest on the train AOI tiles.  [needs tiles + ideally a GPU]

Starts from DeepForest's released tree-crown model and fine-tunes on your campus
tiles. Saves the trained checkpoint to outputs/model/.

Epoch count comes from config.yaml (train.epochs); set TD_EPOCHS to override for a
quick pass without editing the committed config.
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import common as C
from pytorch_lightning import Callback


class _LossRecorder(Callback):
    """Lightning callback: capture per-epoch train/val loss for a figure."""
    def __init__(self):
        super().__init__()
        self.epochs, self.train_loss, self.val_loss = [], [], []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        def g(k):
            v = m.get(k)
            return float(v) if v is not None else None
        self.epochs.append(trainer.current_epoch)
        self.train_loss.append(g("train_loss_epoch") or g("train_loss"))
        self.val_loss.append(g("val_loss"))


def main():
    cfg = C.load_config()
    from deepforest import main as df_main

    od = C.p(cfg, cfg["outputs_dir"])
    train_dir = od / "tiles" / "train"
    train_csv = train_dir / "train_tiles.csv"

    epochs = int(os.environ.get("TD_EPOCHS", cfg["train"]["epochs"]))

    m = df_main.deepforest()              # deepforest 2.x auto-loads the released weights

    m.config["train"]["csv_file"] = str(train_csv)
    m.config["train"]["root_dir"] = str(train_dir)
    m.config["train"]["epochs"] = epochs
    m.config["train"]["batch_size"] = cfg["train"]["batch_size"]
    m.config["batch_size"] = cfg["train"]["batch_size"]
    m.config["score_thresh"] = cfg["train"]["score_thresh"]
    # light validation on the same set is fine for a first pass; swap to test tiles if desired
    m.config["validation"]["csv_file"] = str(train_csv)
    m.config["validation"]["root_dir"] = str(train_dir)

    rec = _LossRecorder()
    m.create_trainer(callbacks=[rec])
    print(f"training {epochs} epochs on accelerator={m.trainer.accelerator.__class__.__name__}")
    m.trainer.fit(m)

    model_dir = od / "model"; model_dir.mkdir(parents=True, exist_ok=True)
    ckpt = model_dir / "treedetect_finetuned.ckpt"
    m.trainer.save_checkpoint(str(ckpt))
    print(f"saved checkpoint -> {ckpt}")

    _loss_figure(cfg, rec)


def _loss_figure(cfg, rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    fig, ax = plt.subplots(figsize=(7, 5))
    tl = [(e, v) for e, v in zip(rec.epochs, rec.train_loss) if v is not None]
    vl = [(e, v) for e, v in zip(rec.epochs, rec.val_loss) if v is not None]
    if tl:
        ax.plot(*zip(*tl), "-o", color="#1b7837", label="train loss")
    if vl:
        ax.plot(*zip(*vl), "-o", color="#d73027", label="val loss")
    if not tl and not vl:
        ax.text(0.5, 0.5, "no per-epoch loss captured", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("s04 — DeepForest fine-tuning loss")
    ax.legend()
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "train_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
