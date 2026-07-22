# test_yolo_image.py
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


import os

label_dir = '../data/WeedCrop.v1i.yolov5pytorch/train/labels'
crop = 0
weed = 0

for f in os.listdir(label_dir):
    with open(os.path.join(label_dir, f), 'r') as file:
        for line in file:
            if line.strip():
                cls = int(line.split()[0])
                if cls == 0:
                    crop += 1
                else:
                    weed += 1

print(f'Crop (класс 0): {crop} объектов')
print(f'Weed (класс 1): {weed} объектов')
print(f'Соотношение: 1:{weed/crop:.2f}' if crop > 0 else '⚠️ Crop вообще нет!')


model = YOLO("../../runs/detect/yolo_weed_article/yolov8n_weed_100epochs/weights/best.pt")
image_path = "../data/WeedCrop.v1i.yolov5pytorch/valid/images/32683_jpg.rf.37ac73af15762f4d9ea413d9ec534810.jpg"

results = model(image_path, conf=0.2)

results[0].save("prediction_result.jpg")
results[0].show()

print("\n📊 Найденные объекты:")
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = box.conf[0].item()
    cls = int(box.cls[0].item())
    label = '🌿 Сорняк' if cls == 1 else '🌾 Культура'
    print(f"  {label}: {conf*100:.1f}% уверенности")
    print(f"    Координаты: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")