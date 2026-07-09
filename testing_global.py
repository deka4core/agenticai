# test_model.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from backbone import GlobalModel, TOMATO_TRAIN, TOMATO_VALID, WEED_TRAIN_IMAGES, WEED_TRAIN_LABELS


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # Загружаем модель
    model = GlobalModel().to(device)
    model.load_state_dict(torch.load("global_model.pth", map_location=device))
    model.eval()

    # Трансформации для теста
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Загружаем тестовые датасеты
    tomato_valid = datasets.ImageFolder(TOMATO_VALID, transform=transform)

    # Создаем тестовый датасет для сорняков из валидационной папки
    WEED_VALID_IMAGES = r".\data\WeedCrop.v1i.yolov5pytorch\valid\images"
    WEED_VALID_LABELS = r".\data\WeedCrop.v1i.yolov5pytorch\valid\labels"

    # Проверяем существует ли валидационная папка для сорняков
    import os
    if os.path.exists(WEED_VALID_IMAGES):
        from backbone import WeedDataset
        weed_valid = WeedDataset(WEED_VALID_IMAGES, WEED_VALID_LABELS)
        weed_loader = DataLoader(weed_valid, batch_size=32, shuffle=False, num_workers=2)
        test_weed = True
    else:
        print("Внимание: валидационная папка для сорняков не найдена!")
        test_weed = False

    # Загрузчики данных
    tomato_loader = DataLoader(tomato_valid, batch_size=32, shuffle=False, num_workers=2)

    # Тестирование на томатах (классификация болезней)
    print("=" * 50)
    print("Тестирование на датасете томатов (Классификация болезней)")
    print("=" * 50)

    tomato_preds = []
    tomato_true = []
    tomato_probs = []

    with torch.no_grad():
        for images, labels in tomato_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, task='classification')
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            tomato_preds.extend(predicted.cpu().numpy())
            tomato_true.extend(labels.cpu().numpy())
            tomato_probs.extend(probs.cpu().numpy())

    tomato_preds = np.array(tomato_preds)
    tomato_true = np.array(tomato_true)
    tomato_probs = np.array(tomato_probs)

    tomato_accuracy = np.mean(tomato_preds == tomato_true)
    print(f"Точность на томатах: {tomato_accuracy:.4f} ({tomato_accuracy * 100:.2f}%)")
    print(f"Количество образцов: {len(tomato_true)}")

    # Получаем названия классов томатов
    tomato_classes = tomato_valid.classes
    print("\nОтчет по классам:")
    print(classification_report(tomato_true, tomato_preds, target_names=tomato_classes))

    # Тестирование на сорняках (бинарная классификация)
    if test_weed:
        print("\n" + "=" * 50)
        print("Тестирование на датасете сорняков (Бинарная классификация)")
        print("=" * 50)

        weed_preds = []
        weed_true = []
        weed_probs = []

        with torch.no_grad():
            for images, labels in weed_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images, task='detection')
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                weed_preds.extend(predicted.cpu().numpy())
                weed_true.extend(labels.cpu().numpy())
                weed_probs.extend(probs.cpu().numpy())

        weed_preds = np.array(weed_preds)
        weed_true = np.array(weed_true)
        weed_probs = np.array(weed_probs)

        weed_accuracy = np.mean(weed_preds == weed_true)
        print(f"Точность на сорняках: {weed_accuracy:.4f} ({weed_accuracy * 100:.2f}%)")
        print(f"Количество образцов: {len(weed_true)}")
        print("\nОтчет по классам (0 - сорняк, 1 - культура):")
        print(classification_report(weed_true, weed_preds, target_names=['сорняк', 'культура']))

    # Визуализация
    visualize_results(model, tomato_loader, weed_loader if test_weed else None, tomato_classes, device)


def visualize_results(model, tomato_loader, weed_loader, tomato_classes, device):
    """Визуализация результатов тестирования"""
    model.eval()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Примеры предсказаний модели', fontsize=16)

    # Отображаем примеры для томатов
    with torch.no_grad():
        images, labels = next(iter(tomato_loader))
        images_subset = images[:3].to(device)
        labels_subset = labels[:3]

        outputs = model(images_subset, task='classification')
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        # Денормализация изображений для отображения
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images_display = images_subset.cpu() * std + mean
        images_display = torch.clamp(images_display, 0, 1)

        for i in range(3):
            ax = axes[0, i]
            img = images_display[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            true_label = tomato_classes[labels_subset[i].item()]
            pred_label = tomato_classes[preds[i].item()]
            confidence = probs[i, preds[i].item()].item()

            color = 'green' if preds[i].item() == labels_subset[i].item() else 'red'
            ax.set_title(f'Истина: {true_label}\nПредсказано: {pred_label}\nУверенность: {confidence:.2f}',
                         color=color)
            ax.axis('off')

    # Отображаем примеры для сорняков если есть
    if weed_loader is not None:
        with torch.no_grad():
            images, labels = next(iter(weed_loader))
            images_subset = images[:3].to(device)
            labels_subset = labels[:3]

            outputs = model(images_subset, task='detection')
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images_display = images_subset.cpu() * std + mean
            images_display = torch.clamp(images_display, 0, 1)

            weed_classes = ['сорняк', 'культура']

            for i in range(3):
                ax = axes[1, i]
                img = images_display[i].permute(1, 2, 0).numpy()
                ax.imshow(img)
                true_label = weed_classes[labels_subset[i].item()]
                pred_label = weed_classes[preds[i].item()]
                confidence = probs[i, preds[i].item()].item()

                color = 'green' if preds[i].item() == labels_subset[i].item() else 'red'
                ax.set_title(f'Истина: {true_label}\nПредсказано: {pred_label}\nУверенность: {confidence:.2f}',
                             color=color)
                ax.axis('off')

    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_confusion_matrix():
    """Визуализация матрицы ошибок для томатов"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем модель и данные
    model = GlobalModel().to(device)
    model.load_state_dict(torch.load("global_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tomato_valid = datasets.ImageFolder(TOMATO_VALID, transform=transform)
    tomato_loader = DataLoader(tomato_valid, batch_size=32, shuffle=False, num_workers=2)

    # Получаем предсказания
    preds = []
    true = []

    with torch.no_grad():
        for images, labels in tomato_loader:
            images = images.to(device)
            outputs = model(images, task='classification')
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            true.extend(labels.cpu().numpy())

    # Строим матрицу ошибок
    cm = confusion_matrix(true, preds)
    classes = tomato_valid.classes

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Матрица ошибок - Классификация болезней томатов')
    plt.xlabel('Предсказано')
    plt.ylabel('Истина')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Вычисляем точность по классам
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nТочность по классам:")
    for i, cls in enumerate(classes):
        print(f"{cls}: {class_accuracy[i]:.4f} ({class_accuracy[i] * 100:.2f}%)")


if __name__ == "__main__":
    # Основное тестирование
    test_model()

    # Визуализация матрицы ошибок
    print("\n" + "=" * 50)
    print("Визуализация матрицы ошибок для томатов")
    print("=" * 50)
    visualize_confusion_matrix()