"""
Step 04 — Fine-tune DeepForest on the train AOI tiles.  [needs tiles + ideally a GPU]

Starts from DeepForest's released tree-crown model and fine-tunes on your campus
tiles. Saves the trained checkpoint to outputs/model/.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import common as C


def main():
    cfg = C.load_config()
    from deepforest import main as df_main

    od = C.p(cfg, cfg["outputs_dir"])
    train_dir = od / "tiles" / "train"
    train_csv = train_dir / "train_tiles.csv"

    m = df_main.deepforest()
    m.use_release()                      # pretrained NEON tree-crown weights

    m.config["train"]["csv_file"] = str(train_csv)
    m.config["train"]["root_dir"] = str(train_dir)
    m.config["train"]["epochs"] = cfg["train"]["epochs"]
    m.config["batch_size"] = cfg["train"]["batch_size"]
    m.config["score_thresh"] = cfg["train"]["score_thresh"]
    # light validation on the same set is fine for a first pass; swap to test tiles if desired
    m.config["validation"]["csv_file"] = str(train_csv)
    m.config["validation"]["root_dir"] = str(train_dir)

    m.create_trainer()
    m.trainer.fit(m)

    model_dir = od / "model"; model_dir.mkdir(parents=True, exist_ok=True)
    m.trainer.save_checkpoint(str(model_dir / "treedetect_finetuned.ckpt"))
    print(f"saved checkpoint -> {model_dir / 'treedetect_finetuned.ckpt'}")


if __name__ == "__main__":
    main()
