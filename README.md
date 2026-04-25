# 🚗 Street Objects Classification

Сравнительный анализ моделей классификации изображений уличных объектов, развёртывание лучшей модели в виде API и создание пользовательского интерфейса.

---

## 📦 Датасет

**[Street Objects](https://www.kaggle.com/datasets/owm4096/street-objects)** — датасет изображений уличных объектов.

- Всего изображений: **9 879**
- Размер входа: 64×64 (для CNN) / 224×224 (для Transfer Learning)
- Классов: **7**

| Класс | Описание |
|-------|----------|
| `bicycle` | Велосипед |
| `car` | Автомобиль |
| `limit30` | Знак ограничения скорости 30 |
| `person` | Человек |
| `stop` | Знак стоп |
| `trafficlight` | Светофор |
| `truck` | Грузовик |

---

## 🤖 Сравниваемые модели

| № | Модель | Описание |
|---|--------|----------|
| 1 | MLP | Многослойный перцептрон (1024→512→256) |
| 2 | Fashion CNN | Простая свёрточная сеть (ДЗ_3) |
| 3 | VGG-like | VGG-подобная архитектура |
| 4 | ResNet-like | ResNet с остаточными блоками |
| 5 | CNN Plain | CNN без регуляризации |
| 6 | CNN + BN | CNN с BatchNormalization |
| 7 | CNN + Dropout | CNN с Dropout |
| 8 | CNN + BN + Dropout | CNN с BatchNormalization и Dropout |
| 9 | ResNet50 TL | Transfer Learning на ResNet50 |
| 10 | MobileNetV2 TL | Transfer Learning на MobileNetV2 |
| 11 | EfficientNetB0 TL | Transfer Learning на EfficientNetB0 ✅ |

---

## 📊 Результаты сравнения моделей

| Модель | Accuracy | Recall | Precision | F1 | Inference (мс) |
|--------|----------|--------|-----------|----|----------------|
| **11_efficientnetb0** | **0.9514** | **0.9507** | **0.9533** | **0.9507** | 6.564 |
| 10_mobilenetv2_tl | 0.9484 | 0.9447 | 0.9471 | 0.9447 | 6.678 |
| 9_resnet50_tl | 0.9312 | 0.9240 | 0.9261 | 0.9240 | 4.819 |
| 6_cnn_bn | 0.9155 | 0.9096 | 0.9119 | 0.9096 | 1.354 |
| 7_cnn_dropout | 0.9104 | 0.9000 | 0.9024 | 0.9000 | 1.109 |
| 8_cnn_bn_do | 0.9059 | 0.9008 | 0.9031 | 0.9008 | 0.530 |
| 5_cnn_plain | 0.8963 | 0.8912 | 0.8935 | 0.8912 | 0.686 |
| 3_vgg_like | 0.8826 | 0.8817 | 0.8840 | 0.8817 | 1.435 |
| 2_fashion_cnn | 0.8801 | 0.8698 | 0.8721 | 0.8698 | 0.458 |
| 4_resnet_like | 0.8801 | 0.8536 | 0.8559 | 0.8536 | 1.676 |
| 1_mlp | 0.7849 | 0.7635 | 0.7658 | 0.7635 | 0.533 |

**Лучшая модель по F1: EfficientNetB0 (F1 = 0.9507)**

---

## 🔗 Ссылки

- **GitHub:** https://github.com/end1ess1/deep_learning_deployment
- **API (Render):** https://deep-learning-deployment-b78b.onrender.com
- **Streamlit:** https://deeplearningdeploymentfrontend-ukd7otysgzzfwoarkbncp9.streamlit.app/

---

## 🗂 Структура репозитория

```
├── api/
│   ├── main.py               # FastAPI приложение
│   └── requirements.txt      # Зависимости бэкенда
├── frontend/
│   ├── app.py                # Streamlit приложение
│   └── requirements.txt      # Зависимости фронтенда
├── models/
│   └── best_model_11_efficientnetb0.keras
├── label_encoder.pkl
├── ДЗ_10.ipynb
└── README.md
```

---

## 🚀 Локальное развёртывание

### Бэкенд (FastAPI)

```bash
cd api
pip install -r requirements.txt
# Положи модель и label_encoder.pkl в корень проекта
uvicorn main:app --host 0.0.0.0 --port 8000
```

API будет доступен на `http://localhost:8000`

### Фронтенд (Streamlit)

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

В боковой панели укажи URL бэкенда: `http://localhost:8000`

---

## 📡 Примеры использования API

### GET /
```bash
curl https://deep-learning-deployment-b78b.onrender.com/
```
```json
{
  "status": "ok",
  "model": "EfficientNetB0",
  "classes": ["bicycle", "car", "limit30", "person", "stop", "trafficlight", "truck"]
}
```

### POST /predict
```bash
curl -X POST https://deep-learning-deployment-b78b.onrender.com/predict \
  -F "file=@image.jpg"
```
```json
{
  "predicted_class": "car",
  "confidence": 0.9823,
  "probabilities": {
    "bicycle": 0.0012,
    "car": 0.9823,
    "limit30": 0.0021,
    "person": 0.0054,
    "stop": 0.0031,
    "trafficlight": 0.0043,
    "truck": 0.0016
  }
}
```

### Python
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "https://deep-learning-deployment-b78b.onrender.com/predict",
        files={"file": f}
    )

print(response.json())
```
