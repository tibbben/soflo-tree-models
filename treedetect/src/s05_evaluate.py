"""
Step 05 — Evaluate the fine-tuned model on the held-out test AOI.  [local]

Reports precision / recall / box-recall at the configured IoU threshold and saves
a side-by-side figure (ground truth vs prediction) for one sample tile.
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
import common as C


def main():
    cfg = C.load_config()
    from deepforest import main as df_main

    od = C.p(cfg, cfg["outputs_dir"])
    test_dir = od / "tiles" / "test"
    test_csv = test_dir / "test_tiles.csv"

    m = df_main.deepforest.load_from_checkpoint(str(od / "model" / "treedetect_finetuned.ckpt"))
    m.config["score_thresh"] = cfg["train"]["score_thresh"]

    # deepforest 2.x: evaluate(csv_file, iou_threshold=None, root_dir=None, ...)
    results = m.evaluate(str(test_csv), iou_threshold=cfg["train"]["iou_eval"],
                         root_dir=str(test_dir))
    prec = float(results.get("box_precision", float("nan")))
    rec = float(results.get("box_recall", float("nan")))
    print("box precision:", round(prec, 3))
    print("box recall   :", round(rec, 3))
    summary = od / "eval_summary.csv"
    pd.DataFrame([{"box_precision": prec, "box_recall": rec,
                   "iou_eval": cfg["train"]["iou_eval"],
                   "score_thresh": cfg["train"]["score_thresh"]}]).to_csv(summary, index=False)
    print(f"wrote {summary}")

    # qualitative check on one tile: ground truth vs prediction
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image
        C.style()

        tiles = pd.read_csv(test_csv)
        sample = tiles.image_path.iloc[0]
        gt = tiles[tiles.image_path == sample]
        img = np.array(Image.open(test_dir / sample).convert("RGB"))
        pred = m.predict_image(path=str(test_dir / sample))
        if pred is None:
            pred = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax"])

        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        for ax, (title, df, color) in zip(
                axes, [("ground truth", gt, "#1b7837"), (f"prediction ({len(pred)})", pred, "#d73027")]):
            ax.imshow(img)
            for _, r in df.iterrows():
                ax.add_patch(Rectangle((r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin,
                                       fill=False, edgecolor=color, linewidth=1.0))
            ax.set_title(title); ax.axis("off")
        fig.suptitle(f"s05 — test tile {sample}  (precision {prec:.2f} / recall {rec:.2f})")
        fig.tight_layout()
        out = C.p(cfg, cfg["reports_dir"]) / "eval_sample.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")
    except Exception as e:
        print("sample figure skipped:", e)


if __name__ == "__main__":
    main()
