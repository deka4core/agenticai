import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import copy

TOMATO_TRAIN = r"./data/tomato/train"
TOMATO_VALID = r"./data/tomato/valid"
WEED_TRAIN_IMAGES = r"./data/WeedCrop.v1i.yolov5pytorch/train/images"
WEED_TRAIN_LABELS = r"./data/WeedCrop.v1i.yolov5pytorch/train/labels"


class GlobalModel(nn.Module):
    def __init__(self, num_disease_classes=11, num_weed_classes=2):
        super().__init__()

        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.backbone = densenet.features  # выход: [batch, 1024, 7, 7]

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # голова 1
        self.classification_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_disease_classes),
        )

        # голова 2
        self.detection_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_weed_classes)  # классификация crop/weed
        )

    def forward(self, x, task='classification'):
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)  # flatten

        if task == 'classification':
            return self.classification_head(features)
        elif task == 'detection':
            return self.detection_head(features)
        return None


# ДАТАСЕТ СОРНЯКОВ
class WeedDataset(Dataset):
    def __init__(self, images_path, labels_path, img_size=224):
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.images = os.listdir(images_path)
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


def fed_avg(w1, w2):
    global_weights = {}
    for key in w1.keys():
        global_weights[key] = (w1[key] + w2[key]) / 2
    return global_weights


def train_client(model, loader, task, device, epochs=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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
        print(f"  Эпоха {epoch + 1} | Loss: {total_loss / len(loader):.4f}")

    return model.state_dict()


def main():
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
    ])

    # томаты
    tomato_train = datasets.ImageFolder(TOMATO_TRAIN, transform=transform)
    tomato_loader = DataLoader(
        tomato_train,
        batch_size=64,
        shuffle=True,
        num_workers=4, # если на rtx cuda стратовать
        pin_memory=True,
        persistent_workers=True,
    )

    # сорняки
    weed_train = WeedDataset(WEED_TRAIN_IMAGES, WEED_TRAIN_LABELS)
    weed_loader = DataLoader(
        weed_train,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    NUM_ROUNDS = 20
    global_model = GlobalModel().to(device)

    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Раунд {round_num + 1}/{NUM_ROUNDS} ---")

        # на томатах
        print("CLIENT 1:")
        client1_model = GlobalModel().to(device)
        client1_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
        w1 = train_client(client1_model, tomato_loader, task='classification', device=device, epochs=5)

        # на сорняках
        print("CLIENT 2:")
        client2_model = GlobalModel().to(device)
        client2_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
        w2 = train_client(client2_model, weed_loader, task='detection', device=device, epochs=5)

        global_weights = fed_avg(w1, w2)
        global_model.load_state_dict(global_weights)

    torch.save(global_model.state_dict(), "global_model.pth")
    print("\nГлобальная модель сохранена — global_model.pth")


if __name__ == "__main__":
    main()