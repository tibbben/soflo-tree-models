"""
train_yolo.py — Fine-tune YOLOv8 on the labeled tree chips.

Run from the gables_campus/ folder:
    python scripts/train_yolo.py
"""

from ultralytics import YOLO

if __name__ == "__main__":
    # Load the pretrained YOLOv8 small model as the starting point.
    model = YOLO("yolov8s.pt")

    # Fine-tune on our dataset.
    model.train(
        data="./yolo_dataset/dataset.yaml",   # dataset config
        epochs=100,                            # max epochs (early stopping will kick in before if needed)
        patience=30,                           # stop early if no improvement for 30 epochs
        imgsz=1280,                            # chip size
        batch=2,                               # safe for RTX 4060 8GB with AMP
        device=0,                              # GPU (CUDA device 0)
        amp=True,                              # mixed precision to save VRAM
        project="./runs",                      # save results under ./runs/
        name="train1",                         # subfolder name for this run
    )