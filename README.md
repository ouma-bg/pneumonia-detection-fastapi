# 🫁 Pneumonia Detection — FastAPI & Streamlit

A deep learning web application that detects pneumonia from chest X-ray images using MobileNetV2, served via a FastAPI backend and a Streamlit frontend.

---

## 🚀 Demo

| Streamlit Interface | API Endpoint |
|---------------------|--------------|
| Upload X-ray → Get prediction | REST API for integration |

---

## 🧠 Model

- **Architecture:** MobileNetV2 (Transfer Learning)
- **Task:** Binary Classification — NORMAL vs PNEUMONIA
- **Dataset:** [Chest X-Ray Images (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Training Phases:** 2-phase fine-tuning (frozen base → unfrozen)
- **Formats:** `.h5` and `.keras`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras |
| Model | MobileNetV2 |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Language | Python 3.13 |

---

## 📁 Project Structure

```
lab_pneumonia/
├── fastapi_app/
│   ├── main.py          # API endpoints
│   ├── models.py        # Pydantic schemas
│   ├── utils.py         # Image preprocessing
│   ├── config.py        # Configuration
│   └── test_api.py      # API tests
├── streamlit/
│   ├── main.py          # Streamlit UI
│   ├── model_loader.py  # Model loading logic
│   └── util.py          # Utility functions
├── train_pneumonia.py   # Training script
├── create_val_split.py  # Dataset preparation
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/ouma-bg/pneumonia-detection-fastapi.git
cd pneumonia-detection-fastapi

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset

Download the dataset from Kaggle and place it in:

```
archive/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    └── test/
```

> Dataset not included in this repo due to size. Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

---

## 🏃 Run the App

### FastAPI Backend

```bash
cd fastapi_app
uvicorn main:app --reload
```

API docs available at: `http://localhost:8000/docs`

### Streamlit Frontend

```bash
cd streamlit
streamlit run main.py
```

---

## 🔁 Retrain the Model

```bash
python train_pneumonia.py
```

> Trained model will be saved in `model/` directory.

---

## 📬 Contact

**Oumaima**
- GitHub: [@ouma-bg](https://github.com/ouma-bg)
