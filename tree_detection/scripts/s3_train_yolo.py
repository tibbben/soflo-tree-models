"""
train_yolo.py — Fine-tune YOLOv8 on the labeled tree chips.

Run 16 config: yolov8s with augmentation (rotation + flips + light mixup), imgsz
1280, batch 2. Augmentation was a clean win over the un-augmented baseline; the
larger yolov8m backbone was tested separately and did not help.

Run from the project root:
    python scripts/train_yolo.py
"""

from ultralytics import YOLO

# NOTE: the __main__ guard is REQUIRED on Windows (multiprocessing dataloader).
if __name__ == "__main__":
    # yolov8s — the small backbone (yolov8m did not improve results).
    model = YOLO("yolov8s.pt")

    model.train(
        data="./yolo_dataset/dataset.yaml",   # dataset config
        epochs=200,
        patience=50,                           # early stop after 50 epochs w/o val gain
        imgsz=1280,                            # inference/training resolution
        batch=2,                               # safe for an 8GB GPU with AMP
        device=0,                              # GPU (CUDA device 0)
        amp=True,                              # mixed precision to save VRAM

        # --- Augmentation ---
        degrees=180.0,    # full rotation — aerial imagery has no canonical orientation
        flipud=0.5,       # vertical flip
        fliplr=0.5,       # horizontal flip
        scale=0.6,        # scale jitter
        mixup=0.1,        # light mixup

        project="./runs",
        name="train1",    # subfolder; ultralytics auto-increments
    )