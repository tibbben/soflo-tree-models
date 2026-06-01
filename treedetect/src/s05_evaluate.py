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

    results = m.evaluate(str(test_csv), str(test_dir), iou_threshold=cfg["train"]["iou_eval"])
    print("box precision:", round(float(results.get("box_precision", float('nan'))), 3))
    print("box recall   :", round(float(results.get("box_recall", float('nan'))), 3))
    summary = od / "eval_summary.csv"
    pd.DataFrame([{k: results[k] for k in results if not hasattr(results[k], "__len__")}]).to_csv(summary, index=False)
    print(f"wrote {summary}")

    # qualitative check on one tile
    try:
        import matplotlib.pyplot as plt
        tiles = pd.read_csv(test_csv)
        sample = tiles.image_path.iloc[0]
        pred = m.predict_image(path=str(test_dir / sample), return_plot=True)
        C.style()
        fig, ax = plt.subplots(figsize=(6, 6)); ax.imshow(pred[..., ::-1]); ax.axis("off")
        ax.set_title("Predicted tree boxes (sample test tile)")
        out = C.p(cfg, cfg["reports_dir"]) / "eval_sample.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"wrote {out}")
    except Exception as e:
        print("sample figure skipped:", e)


if __name__ == "__main__":
    main()
