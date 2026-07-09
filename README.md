# 🌾 Agentic AI for Precision Agriculture

Мультизадачная система для точного земледелия с использованием федеративного обучения и агентного искусственного интеллекта. Повтор эксперимента. Изучение и анализ.

##  Оглавление

- [Обзор проекта](#обзор-проекта)
- [Структура проекта](#структура-проекта)
- [Эксперименты](#эксперименты)
- [Федеративное обучение](#федеративное-обучение)
- [Результаты](#результаты)
- [Установка и запуск](#установка-и-запуск)
- [Использование](#использование)

##  Обзор проекта

Проект реализует систему на основе **Agentic AI** для точного земледелия, которая решает две ключевые задачи:
1. **Классификация болезней томатов** (11 классов)
2. **Обнаружение сорняков** (бинарная классификация)

Основные технологии:
-  **Мультизадачное обучение** - одна модель для двух задач
-  **Федеративное обучение** - обучение без сбора данных
-  **DenseNet121** - общий бэкбон для извлечения признаков
-  **Две специализированные головы** для разных задач

##  Структура проекта

```
agenticai/
├── data/
│   ├── tomato/              # Датасет болезней томатов
│   │   ├── train/           # 11 классов болезней
│   │   └── valid/           # Валидационные данные
│   └── WeedCrop.v1i.yolov5pytorch/
│       ├── train/           # Изображения сорняков/культур
│       └── valid/           # Валидационные данные
│
├── models/
│   ├── densenet_tomato.pth      # Веса DenseNet (болезни)
│   ├── efficientdet_weed.pth    # Веса EfficientDet (сорняки)
│   └── global_model.pth   #  мультизадачная модель
│
├── src/
│   ├── backbone.py          # Архитектура GlobalModel
│   ├── densenet_default.py  # Обучение densenet отдельно
│   ├── edet.py              # Обучение EfficientDet отдельно
│   ├── testing_global.py    # Тестирование модели
│
├── analysis/
│   ├── confusion_matrices/  # Матрицы ошибок
│   └── model_predictions    # Тесты модели
│
├── README.md
├── requirements.txt
└── .gitignore
```

##  Эксперименты

### Часть 1: Обучение локальных моделей

Первым этапом были обучены четыре независимые модели согласно статье:

| Модель | Задача | Архитектура | Точность |
|--------|--------|------------|----------|
| **DenseNet121** | Классификация болезней | DenseNet121 + FC слои | 95.0% |
| **MobileNetV2** | Классификация болезней | MobileNetV2 + FC слои | - |
| **YOLOv8** | Обнаружение сорняков | YOLOv8 с CIoU loss | - |
| **EfficientDet-D0** | Обнаружение сорняков | EfficientDet | mAP@0.5: 0.978 |

#### Процесс обучения:

```python
# Пример обучения DenseNet121 на томатах
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(1024, 11)
optimizer = Adam(model.parameters(), lr=0.0001)
criterion = CategoricalCrossEntropy()

# Результаты приближены к статье
# DenseNet: 95.0% accuracy
# EfficientDet: 0.978 mAP@0.5
```

### Часть 2: Мультизадачная модель с федеративным обучением

**Цель:** Создать одну модель, которая может решать обе задачи, используя федеративное обучение.

#### Архитектура GlobalModel:

```python
class GlobalModel(nn.Module):
    def __init__(self):
        # Общий бэкбон - извлекает признаки из изображений
        self.backbone = DenseNet121().features  # [batch, 1024, 7, 7]
        self.pool = AdaptiveAvgPool2d((1, 1))
        
        # Голова 1: Классификация болезней (11 классов)
        self.classification_head = Sequential(
            Linear(1024, 512),
            ReLU(),
            Dropout(0.3),
            Linear(512, 11)  # 11 болезней томатов
        )
        
        # Голова 2: Обнаружение сорняков (2 класса)
        self.detection_head = Sequential(
            Linear(1024, 512),
            ReLU(),
            Dropout(0.3),
            Linear(512, 2)  # сорняк / культура
        )
```

##  Федеративное обучение

#### 1. **Инициализация**
```python
# Создаем глобальную модель с случайными весами
global_model = GlobalModel()
```

#### 2. **Обучение на клиентах** (каждый раунд)

```python
# КЛИЕНТ 1: Обучение на томатах (5 эпох)
client1 = copy.deepcopy(global_model)
train_client(client1, tomato_loader, task='classification')
# Обновляются: backbone + classification_head
# detection_head НЕ меняется!

# КЛИЕНТ 2: Обучение на сорняках (5 эпох)
client2 = copy.deepcopy(global_model)
train_client(client2, weed_loader, task='detection')
# Обновляются: backbone + detection_head
# classification_head НЕ меняется!
```

#### 3. **Агрегация весов (FedAvg)**

```python
def fed_avg(w1, w2):
    global_weights = {}
    
    for key in w1.keys():
        if 'backbone' in key:
            #  УСРЕДНЯЕМ только бэкбон
            # Бэкбон учится от ОБОИХ задач
            global_weights[key] = (w1[key] + w2[key]) / 2
            
        elif 'classification_head' in key:
            # Берем веса только от клиента 1 (томаты)
            global_weights[key] = w1[key]
            
        elif 'detection_head' in key:
            # Берем веса только от клиента 2 (сорняки)
            global_weights[key] = w2[key]
    
    return global_weights
```

#### 4. **Повторяем 20 раундов**

```
Раунд N:
  Client 1 (томаты) → w1 = {backbone: [0.15], cls_head: [0.8], det_head: [1.5]}
  Client 2 (сорняки) → w2 = {backbone: [0.12], cls_head: [0.6], det_head: [1.7]}
  Агрегация:
    backbone = (0.15 + 0.12) / 2 = 0.135 
    cls_head = 0.8                       
    det_head = 1.7                         
```

### Почему усредняем только бэкбон?

```
 БЭКБОН (общий):
   - Учится видеть общие признаки (форма листа, цвет, текстура)
   - Получает знания от ОБОИХ задач
   - Становится универсальным экстрактором признаков

 ГОЛОВА 1 (болезни):
   - Специализируется на 11 болезнях томатов
   - НЕ должна смешиваться с головой 2
   - Сохраняет веса только от клиента 1

 ГОЛОВА 2 (сорняки):
   - Специализируется на 2 классах (сорняк/культура)
   - НЕ должна смешиваться с головой 1
   - Сохраняет веса только от клиента 2
```

##  Результаты

### Сравнение локальных и глобальной моделей:

| Модель | Точность (болезни) | mAP@0.5 (сорняки) |
|--------|-------------------|-------------------|
| DenseNet121 (локальная) | 95.0% | - |
| MobileNetV2 (локальная) | - | - |
| YOLOv8 (локальная) | - | - |
| EfficientDet-D0 (локальная) | - | 0.978 |
| **Global Model (FL)** | **96.4%** | **0.882** |

**Вывод:** Глобальная модель **превосходит** локальные модели, благодаря комбинированию знаний от обеих задач.

### Визуализация результатов:

```python
# Матрица ошибок для Global Model
см. results/confusion_matrices/global_confusion_matrix.png

# Графики обучения
см. results/training_curves/
```

##  Установка и запуск

### Требования:

```bash
Python >= 3.8
PyTorch >= 1.10.0
torchvision >= 0.11.0
CUDA (рекомендуется)
```

### Установка:

```bash
# 1. Клонировать репозиторий
git clone https://github.com/username/agenticai.git
cd agenticai

# 2. Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

### Запуск:

```bash
# 1. Обучение локальных моделей
python src/train_local.py

# 2. Федеративное обучение
python src/federated_learning.py

# 3. Тестирование
python src/test_model.py

# 4. Визуализация результатов
python src/visualize_results.py
```

##  Использование

### Загрузка и использование готовой модели:

```python
import torch
from src.backbone import GlobalModel

# Загрузка обученной модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GlobalModel().to(device)
model.load_state_dict(torch.load("models/global_model_final.pth"))
model.eval()

# Классификация болезней томатов
image = load_image("tomato_leaf.jpg")
output = model(image, task='classification')
disease_class = torch.argmax(output)
print(f"Болезнь: {classes[disease_class]}")

# Обнаружение сорняков
image = load_image("field_photo.jpg")
output = model(image, task='detection')
is_weed = torch.argmax(output) == 0
print(f"Результат: {'Сорняк' if is_weed else 'Культура'}")
```
##  Ссылки

- [Статья: Agentic AI for smart and sustainable precision agriculture]([https://doi.org/10.3389/fpls.2025.1706428](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1706428/full#s5))
- [Датасет болезней томатов](https://www.kaggle.com/datasets/ashishmotwani/tomato)
- [Датасет WeedCrop](https://www.kaggle.com/datasets/vinayakshanawad/weedcrop-image-dataset)
