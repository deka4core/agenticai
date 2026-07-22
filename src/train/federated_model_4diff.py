# federated_distillation_bbox_fixed_v2.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from datetime import datetime

print("=" * 70)
print("🚀 ФЕДЕРАТИВНОЕ ОБУЧЕНИЕ С BOUNDING BOXES (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Устройство: {device}")

NUM_ROUNDS = 5
LOCAL_EPOCHS = 5
NUM_DISEASE_CLASSES = 11
NUM_WEED_CLASSES = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
IMAGE_SIZE_DET = 640
MAX_BOXES = 10  # максимальное количество bbox на изображение

TOMATO_TRAIN = "./data/tomato/train"
WEED_TRAIN_IMAGES = "./data/WeedCrop.v1i.yolov5pytorch/train/images"
WEED_TRAIN_LABELS = "./data/WeedCrop.v1i.yolov5pytorch/train/labels"


class GlobalModel(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.backbone = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classification_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_DISEASE_CLASSES)
        )

        self.detection_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, MAX_BOXES * (4 + NUM_WEED_CLASSES))
        )

    def forward(self, x, task='classification'):
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)

        if task == 'classification':
            return self.classification_head(features)
        elif task == 'detection':
            output = self.detection_head(features)
            output = output.view(output.size(0), MAX_BOXES, 4 + NUM_WEED_CLASSES)
            boxes = output[:, :, :4]  # [batch, MAX_BOXES, 4]
            classes = output[:, :, 4:]  # [batch, MAX_BOXES, 2]
            return boxes, classes
        return None


class WeedDetectionDataset(Dataset):
    def __init__(self, images_path, labels_path, img_size=IMAGE_SIZE_DET):
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_path, img_name)
        label_path = os.path.join(self.labels_path,
                                  img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img)

        boxes = []
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(float(parts[0]))
                        x = float(parts[1])
                        y = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        boxes.append([x, y, w, h])
                        classes.append(cls)

        num_boxes = len(boxes)
        if num_boxes > MAX_BOXES:
            boxes = boxes[:MAX_BOXES]
            classes = classes[:MAX_BOXES]
            num_boxes = MAX_BOXES

        boxes_tensor = torch.zeros(MAX_BOXES, 4, dtype=torch.float32)
        classes_tensor = torch.zeros(MAX_BOXES, dtype=torch.long)

        for i in range(num_boxes):
            boxes_tensor[i] = torch.tensor(boxes[i])
            classes_tensor[i] = classes[i]

        valid_mask = torch.zeros(MAX_BOXES, dtype=torch.bool)
        valid_mask[:num_boxes] = True

        return img, boxes_tensor, classes_tensor, valid_mask


def ciou_loss(pred_boxes, target_boxes, valid_mask):
    """Complete IoU Loss с маской"""
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    pred_boxes = pred_boxes[valid_mask]
    target_boxes = target_boxes[valid_mask]

    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    target_area = target_boxes[:, 2] * target_boxes[:, 3]
    union_area = pred_area + target_area - inter_area

    iou = inter_area / (union_area + 1e-6)

    pred_center = pred_boxes[:, :2]
    target_center = target_boxes[:, :2]
    center_dist = torch.sum((pred_center - target_center) ** 2, dim=1)

    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    v = (4 / (np.pi ** 2)) * (torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + 1e-6)) -
                              torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + 1e-6))) ** 2
    alpha = v / ((1 - iou) + v + 1e-6)

    ciou = iou - center_dist / (enclose_diag + 1e-6) - alpha * v
    return 1 - ciou.mean()


def detection_loss(pred_boxes, pred_classes, target_boxes, target_classes, valid_mask):
    """CIoU + BCE для фиксированного количества объектов"""
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    box_loss = ciou_loss(pred_boxes, target_boxes, valid_mask)

    # Классовая потеря
    pred_classes = pred_classes[valid_mask]
    target_classes = target_classes[valid_mask]
    cls_loss = nn.CrossEntropyLoss()(pred_classes, target_classes)

    return box_loss + cls_loss


class Client1(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.pool(self.backbone(dummy))
            feat_dim = features.view(1, -1).size(1)

        self.fc = nn.Linear(feat_dim, NUM_DISEASE_CLASSES)
        print(f"   Клиент 1: MobileNetV2 → классификация")

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class Client2(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.backbone = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.pool(self.backbone(dummy))
            feat_dim = features.view(1, -1).size(1)

        self.fc = nn.Linear(feat_dim, NUM_DISEASE_CLASSES)
        print(f"   Клиент 2: DenseNet121 → классификация")

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class Client3(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from ultralytics import YOLO
            yolo = YOLO("../models/yolov8n.pt")
            self.backbone = yolo.model.model[0:10]
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

            with torch.no_grad():
                dummy = torch.randn(1, 3, 640, 640)
                features = self.pool(self.backbone(dummy))
                feat_dim = features.view(1, -1).size(1)

            self.fc = nn.Linear(feat_dim, MAX_BOXES * (4 + NUM_WEED_CLASSES))
            print(f"   Клиент 3: YOLOv8 → детекция ({MAX_BOXES} bbox)")
        except:
            print("   ⚠️ YOLOv8 не найден")
            self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1024, MAX_BOXES * (4 + NUM_WEED_CLASSES))
            print(f"   Клиент 3: DenseNet121 → детекция ({MAX_BOXES} bbox)")

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        output = output.view(output.size(0), MAX_BOXES, 4 + NUM_WEED_CLASSES)
        return output[:, :, :4], output[:, :, 4:]


class Client4(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from effdet import create_model
            effdet = create_model('efficientdet_d0', bench_task='', num_classes=2, pretrained=True)
            self.backbone = effdet.backbone
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

            with torch.no_grad():
                dummy = torch.randn(1, 3, 640, 640)
                features = self.pool(self.backbone(dummy))
                feat_dim = features.view(1, -1).size(1)

            self.fc = nn.Linear(feat_dim, MAX_BOXES * (4 + NUM_WEED_CLASSES))
            print(f"   Клиент 4: EfficientDet → детекция ({MAX_BOXES} bbox)")
        except:
            print("   ⚠️ EfficientDet не найден")
            self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1024, MAX_BOXES * (4 + NUM_WEED_CLASSES))
            print(f"   Клиент 4: DenseNet121 → детекция ({MAX_BOXES} bbox)")

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        output = output.view(output.size(0), MAX_BOXES, 4 + NUM_WEED_CLASSES)
        return output[:, :, :4], output[:, :, 4:]


def collate_fn_bbox(batch):
    images = []
    all_boxes = []
    all_classes = []
    all_masks = []

    for img, boxes, classes, mask in batch:
        images.append(img)
        all_boxes.append(boxes)
        all_classes.append(classes)
        all_masks.append(mask)

    return torch.stack(images), torch.stack(all_boxes), torch.stack(all_classes), torch.stack(all_masks)


def train_client_detection(client, loader, device, client_name, epochs=LOCAL_EPOCHS):
    client.train()
    optimizer = torch.optim.Adam(client.parameters(), lr=LEARNING_RATE)

    print(f"\n  📊 Обучение {client_name}:")
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for images, boxes, classes, masks in loader:
            images = images.to(device)
            boxes = boxes.to(device)
            classes = classes.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            pred_boxes, pred_classes = client(images)

            batch_loss = 0
            for i in range(images.size(0)):
                loss = detection_loss(
                    pred_boxes[i], pred_classes[i],
                    boxes[i], classes[i],
                    masks[i]
                )
                batch_loss += loss

            batch_loss = batch_loss / images.size(0)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"    Эпоха {epoch + 1}/{epochs}: loss = {avg_loss:.6f}")

    return client


def train_client_classification(client, loader, device, client_name, epochs=LOCAL_EPOCHS):
    client.train()
    optimizer = torch.optim.Adam(client.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\n  📊 Обучение {client_name}:")
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = client(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"    Эпоха {epoch + 1}/{epochs}: loss = {avg_loss:.6f}")

    return client

transform_cls = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

tomato_train = datasets.ImageFolder(TOMATO_TRAIN, transform=transform_cls)
tomato_loader = DataLoader(tomato_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

weed_train = WeedDetectionDataset(WEED_TRAIN_IMAGES, WEED_TRAIN_LABELS, img_size=IMAGE_SIZE_DET)
weed_loader = DataLoader(
    weed_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn_bbox
)

print(f"   Томаты: {len(tomato_train)} изображений")
print(f"   Сорняки: {len(weed_train)} изображений")

print("\n🏗️ ИНИЦИАЛИЗАЦИЯ КЛИЕНТОВ:")
clients = [
    Client1().to(device),
    Client2().to(device),
    Client3().to(device),
    Client4().to(device),
]
client_tasks = ['classification', 'classification', 'detection', 'detection']
client_loaders = [tomato_loader, tomato_loader, weed_loader, weed_loader]
client_names = ['MobileNetV2', 'DenseNet121', 'YOLOv8', 'EfficientDet']

global_model = GlobalModel().to(device)
print("\n🌐 Глобальная модель: DenseNet121 + 2 головы")


start_time = datetime.now()

for round_num in range(NUM_ROUNDS):
    print("\n" + "=" * 70)
    print(f"🔄 РАУНД {round_num + 1}/{NUM_ROUNDS}")
    print("=" * 70)

    for i, (client, loader, task) in enumerate(zip(clients, client_loaders, client_tasks)):
        if task == 'classification':
            client = train_client_classification(client, loader, device, f"Клиент {i + 1} ({client_names[i]})")
        else:
            client = train_client_detection(client, loader, device, f"Клиент {i + 1} ({client_names[i]})")
        print(f"    ✅ Клиент {i + 1} завершил обучение")

print("\n💾 СОХРАНЕНИЕ...")
os.makedirs("../models", exist_ok=True)
torch.save(global_model.state_dict(), "src/models/global_model_bbox_fixed.pth")
print("✅ Модель сохранена: models/global_model_bbox_fixed.pth")

end_time = datetime.now()
print(f"⏱️ Время обучения: {end_time - start_time}")

print("\n" + "=" * 70)
print("📊 ИТОГОВЫЙ ОТЧЕТ")
print("=" * 70)
print(f"   Клиенты: 4 (MobileNetV2, DenseNet121, YOLOv8, EfficientDet)")
print(f"   Глобальная модель: DenseNet121 + 2 головы")
print(f"   Максимум bbox на изображение: {MAX_BOXES}")
print(f"   Модель сохранена: models/global_model_bbox_fixed.pth")
print("\n🏁 ОБУЧЕНИЕ ЗАВЕРШЕНО!")