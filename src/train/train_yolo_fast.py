# train_yolo_fast.py
import os
import yaml
from ultralytics import YOLO
import torch

print("🔍 Проверка датасета...")

train_images = "./data/WeedCrop.v1i.yolov5pytorch/train/images"
train_labels = "./data/WeedCrop.v1i.yolov5pytorch/train/labels"
valid_images = "./data/WeedCrop.v1i.yolov5pytorch/valid/images"

assert os.path.exists(train_images), f"❌ Папка {train_images} не найдена!"
assert os.path.exists(train_labels), f"❌ Папка {train_labels} не найдена!"
assert os.path.exists(valid_images), f"❌ Папка {valid_images} не найдена!"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Устройство: {device}")

model = YOLO("../models/yolov8n.pt")
print("✅ Модель загружена!")


results = model.train(
    data="data/WeedCrop.v1i.yolov5pytorch/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    optimizer="SGD",

    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,

    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.2,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,

    device=device,
    workers=4,
    patience=20,
    save=True,
    project="yolo_weed_article",
    name="yolov8n_weed_100epochs",
    exist_ok=True,
    pretrained=True,
    amp=True,
    seed=42,
    verbose=True,
)

print("Обучение завершено!")

results_dir = "yolo_weed_article/yolov8n_100epochs"
os.makedirs(results_dir, exist_ok=True)

metrics = model.val(data="data/WeedCrop.v1i.yolov5pytorch/data.yaml")
print(f"mAP@0.5: {metrics.box.map50} precision: {metrics.box.mp} recall: {metrics.box.mr}")

metrics_data = {
    "mAP@0.5": metrics.box.map50,
    "precision": metrics.box.mp,
    "recall": metrics.box.mr,
}

import json

with open(os.path.join(results_dir, "metrics.json"), "w") as f:
    json.dump(metrics_data, f, indent=4)

print("Метрики сохранены")