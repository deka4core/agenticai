# final_model_working.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import copy
from datetime import datetime

print("=" * 70)
print("✅ РАБОЧАЯ ВЕРСИЯ: ВСЕ КЛИЕНТЫ НА DENSENET121")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Устройство: {device}")

NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
NUM_DISEASE_CLASSES = 11
NUM_WEED_CLASSES = 2
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

TOMATO_TRAIN = "./data/tomato/train"
WEED_TRAIN_IMAGES = "./data/WeedCrop.v1i.yolov5pytorch/train/images"
WEED_TRAIN_LABELS = "./data/WeedCrop.v1i.yolov5pytorch/train/labels"

print(f"\n📊 ПАРАМЕТРЫ:")
print(f"   Раундов: {NUM_ROUNDS}")
print(f"   Локальных эпох: {LOCAL_EPOCHS}")

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
            nn.Linear(512, NUM_WEED_CLASSES)
        )

    def forward(self, x, task='classification'):
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)

        if task == 'classification':
            return self.classification_head(features)
        elif task == 'detection':
            return self.detection_head(features)
        return None


class Client(GlobalModel):
    def __init__(self, client_id, task_type):
        super().__init__()
        self.client_id = client_id
        self.task_type = task_type
        task_name = "классификация" if task_type == 'classification' else "детекция"
        print(f"   Клиент {client_id}: DenseNet121 + {task_name}")


clients = [
    Client(1, 'classification').to(device),
    Client(2, 'classification').to(device),
    Client(3, 'detection').to(device),
    Client(4, 'detection').to(device),
]

client_tasks = ['classification', 'classification', 'detection', 'detection']
client_names = ['Клиент 1 (томаты)', 'Клиент 2 (томаты)', 'Клиент 3 (сорняки)', 'Клиент 4 (сорняки)']


class WeedDataset(Dataset):
    def __init__(self, images_path, labels_path, img_size=224):
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

        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                line = f.readline()
                if line:
                    label = int(float(line.split()[0]))

        return img, label


print("\n📁 ЗАГРУЗКА ДАННЫХ...")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

tomato_train = datasets.ImageFolder(TOMATO_TRAIN, transform=transform)
weed_train = WeedDataset(WEED_TRAIN_IMAGES, WEED_TRAIN_LABELS)

tomato_loader = DataLoader(tomato_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
weed_loader = DataLoader(weed_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

print(f"   Томаты: {len(tomato_train)} изображений")
print(f"   Сорняки: {len(weed_train)} изображений")

global_model = GlobalModel().to(device)
print("\n🌐 Глобальная модель: DenseNet121 + 2 головы")


def train_client(model, loader, task, device, epochs=LOCAL_EPOCHS):
    """Обучение клиента"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images, task=task)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model.state_dict()


def fed_avg_4_clients(w1, w2, w3, w4):
    global_weights = {}
    for key in w1.keys():
        global_weights[key] = (w1[key] + w2[key] + w3[key] + w4[key]) / 4
    return global_weights

start_time = datetime.now()

for round_num in range(NUM_ROUNDS):
    print(f"\n🔄 РАУНД {round_num + 1}/{NUM_ROUNDS}")

    client_weights = []

    for i, (client, task, name) in enumerate(zip(clients, client_tasks, client_names)):
        print(f"  {name}...")

        client.load_state_dict(copy.deepcopy(global_model.state_dict()))

        loader = tomato_loader if task == 'classification' else weed_loader
        w = train_client(client, loader, task, device)
        client_weights.append(w)

        print(f"    ✅ {name} завершил обучение")

    global_weights = fed_avg_4_clients(
        client_weights[0],
        client_weights[1],
        client_weights[2],
        client_weights[3]
    )
    global_model.load_state_dict(global_weights)

    if device.type == "cuda":
        torch.cuda.empty_cache()

os.makedirs("../models", exist_ok=True)
model_path = "../models/full_global_model.pth"
torch.save(global_model.state_dict(), model_path)
print(f"✅ Модель сохранена: {model_path}")

end_time = datetime.now()
print(f"⏱️ Время обучения: {end_time - start_time}")
