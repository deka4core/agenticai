import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Путь к валидационному датасету
TOMATO_VALID = r"C:\Users\mywin\PycharmProjects\agenticai\data\tomato\valid"

# Трансформации
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Загрузка данных
valid_dataset = datasets.ImageFolder(TOMATO_VALID, transform=valid_transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Загрузка модели с весами подруги
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=False)
model.classifier = nn.Linear(1024, len(valid_dataset.classes))
model.load_state_dict(torch.load("densenet_tomato.pth", map_location=device))
model = model.to(device)
model.eval()

# Проверка точности
correct = 0
total = 0

with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Точность DenseNet121 на валидации: {accuracy:.2f}%")