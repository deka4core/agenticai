import torch
from torch.utils.data import Dataset, DataLoader
from effdet import create_model, DetBenchTrain
import cv2
import os

# Пути
TRAIN_IMAGES = r"C:\Users\mywin\PycharmProjects\agenticai\data\WeedCrop.v1i.yolov5pytorch\train\images"
TRAIN_LABELS = r"C:\Users\mywin\PycharmProjects\agenticai\data\WeedCrop.v1i.yolov5pytorch\train\labels"
VALID_IMAGES = r"C:\Users\mywin\PycharmProjects\agenticai\data\WeedCrop.v1i.yolov5pytorch\valid\images"
VALID_LABELS = r"C:\Users\mywin\PycharmProjects\agenticai\data\WeedCrop.v1i.yolov5pytorch\valid\labels"


# Датасет
class WeedDataset(Dataset):
    def __init__(self, images_path, labels_path, img_size=512):
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.images = os.listdir(images_path)

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
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.split())
                    x1 = (x - w / 2) * self.img_size
                    y1 = (y - h / 2) * self.img_size
                    x2 = (x + w / 2) * self.img_size
                    y2 = (y + h / 2) * self.img_size
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls) + 1)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)

        target = {
            'bbox': boxes,  # было 'boxes'
            'cls': labels.float(),  # было 'labels', и нужен float
            'img_size': torch.tensor([self.img_size, self.img_size], dtype=torch.float32),
            'img_scale': torch.tensor(1.0),
        }

        return img, target


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)

    max_boxes = max(len(t['bbox']) for t in targets)
    if max_boxes == 0:
        max_boxes = 1

    batch_boxes = []
    batch_labels = []

    for t in targets:
        n = len(t['bbox'])
        pad = max_boxes - n
        if n > 0:
            padded_boxes = torch.cat([t['bbox'], torch.zeros(pad, 4)], dim=0)
            padded_labels = torch.cat([t['cls'], torch.zeros(pad)], dim=0)
        else:
            padded_boxes = torch.zeros(max_boxes, 4)
            padded_labels = torch.zeros(max_boxes)

        batch_boxes.append(padded_boxes)
        batch_labels.append(padded_labels)

    combined_target = {
        'bbox': torch.stack(batch_boxes),
        'cls': torch.stack(batch_labels),
        'img_size': torch.tensor([[512, 512]] * len(targets), dtype=torch.float32),
        'img_scale': torch.ones(len(targets)),
    }

    return images, combined_target


# Загрузка данных
train_dataset = WeedDataset(TRAIN_IMAGES, TRAIN_LABELS)
valid_dataset = WeedDataset(VALID_IMAGES, VALID_LABELS)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

print("Тренировочных изображений:", len(train_dataset))
print("Валидационных изображений:", len(valid_dataset))

# Модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

# base_model = create_model('efficientdet_d0', bench_task='', num_classes=2, pretrained=True)
#
# # Оборачиваем вручную с labeler
# model = DetBenchTrain(base_model, create_labeler=True)
# model = model.to(device)

# # Обучение
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#
# for epoch in range(10):
#     model.train()
#     total_loss = 0
#
#     for images, targets in train_loader:
#         images = images.to(device)
#         targets = {k: v.to(device) for k, v in targets.items()}
#
#         optimizer.zero_grad()
#         loss = model(images, targets)
#         loss['loss'].backward()
#         optimizer.step()
#         total_loss += loss['loss'].item()
#
#     print(f"Эпоха {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")
#
# torch.save(model.state_dict(), "efficientdet_weed.pth")
# print("Веса сохранены")


from effdet import create_model
from effdet.bench import DetBenchPredict

# Загружаем модель для инференса
base_model_eval = create_model('efficientdet_d0', bench_task='', num_classes=2, pretrained=False)
model_eval = DetBenchPredict(base_model_eval)
model_eval.load_state_dict(torch.load("efficientdet_weed.pth", map_location=device))
model_eval = model_eval.to(device)
model_eval.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, targets in valid_loader:
        images = images.to(device)
        output = model_eval(images)
        # output: [batch, 100, 6] — x1,y1,x2,y2,score,class
        for i in range(len(output)):
            pred_classes = output[i, :, 5].long()
            true_classes = targets['cls'][i]
            true_classes = true_classes[true_classes > 0].long()
            if len(true_classes) > 0:
                matches = (pred_classes[:len(true_classes)] == true_classes).sum().item()
                correct += matches
                total += len(true_classes)

print(f"Точность EfficientDet: {100 * correct / total:.2f}%")

