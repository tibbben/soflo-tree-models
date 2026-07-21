"""
train_yolo_cfg.py — config-driven training. Every hyperparameter comes from the
yaml's `train:` block, passed straight to model.train(**train). Nothing is added
implicitly, so anything not listed keeps the ultralytics default.

The config file is copied into the run folder as the run's provenance record.

Run from the project root ON A GPU NODE (via bsub):
    python scripts/train_yolo_cfg.py ./configs/champion_5cm_5m.yaml
"""

import sys
import shutil
from ultralytics import YOLO

sys.path.insert(0, "./scripts")
from config import load, summary

# the __main__ guard is required on Windows (multiprocessing dataloader),
# harmless on Linux/HPC — keep for portability
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/train_yolo_cfg.py <config.yaml>")
    cfg_path = sys.argv[1]
    cfg = load(cfg_path)
    summary(cfg, "train")

    model = YOLO(cfg["model"])
    model.train(**cfg["train"])

    # record exactly which config produced this run
    save_dir = model.trainer.save_dir
    shutil.copy(cfg_path, f"{save_dir}/run_config.yaml")
    print(f"Run folder: {save_dir} (config copied to run_config.yaml)")
