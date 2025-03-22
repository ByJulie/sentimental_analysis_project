# Sentiment Analysis Pipeline

## 📌 Overview
This project is a sentiment analysis pipeline using a BERT model. It is divided into three main parts:

- 🔍 **Data Extraction**: Loading and preparation of raw text data.
- 🛠 **Data Processing**: Cleaning and tokenization of texts to make them compatible with BERT.
- 🤖 **Model Training & Inference**: Fine-tuning of a pre-trained BERT model for sentiment classification and implementation of an inference script.

📂 The project uses Kaggle's **Google Play Store Reviews** dataset (📜CC0 License: Public Domain).

## ⚙️ Installation and configuration

### 1️⃣ clone the project
```sh
git clone <repository_url>
cd sentiment-analysis-pipeline
```

### 2️⃣ Create a virtual environment
```sh
python -m venv mlops_env
source mlops_env/bin/activate # For Linux/macOS
mlops_env\Scripts\activate # For Windows
```

### 3️⃣ install dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Add dataset file
Replace `file_path` in `data_extraction.py` and `data_processing.py` with the actual path to the data file downloaded from Kaggle.

## Usage

### ▶️ Execute complete pipeline
👨‍💻 Technical use case
A data scientist wants to train a sentiment classification model on another dataset (e.g. restaurant reviews). He can easily:
- Replace the original dataset.
- Run the pipeline to clean, tokenize and train a new model.
- Use inference.py to test the fine-tuned model on new data.
```sh
python -m src.data_extraction # Data extraction
python -m src.data_processing # Text pre-processing
python -m src.model # Model training
python -m src.inference # Inference on new data
```

### ▶️ Example of use (inference)
🏢 Business use case
An e-commerce company wants to analyze customer reviews of its products. Thanks to this Sentiment Analysis model, it can automatically classify reviews as Positive, Neutral or Negative, enabling it to quickly identify potential problems and improve customer satisfaction.
```python
from src.inference import predict_sentiment

text = “This application is great!”
print(predict_sentiment(text)) # Output: Positive, Negative or Neutral
```

## 📁 File description

- 📂 **src/**
  - 🗂 `data_extraction.py` : Data loading and verification.
  - ✨ `data_processing.py` : Data cleaning, tokenization and preparation.
  - 🎯 `model.py` : BERT model loading and fine-tuning.
  - 🔮 `inference.py` : Script for running predictions on new texts.

- 📂 **tests/unit/**
  - ✅ `test_data_extraction.py` : Checks data loading.
  - ✅ `test_data_processing.py` : Tests text cleaning and tokenization.
  - ✅ `test_model.py` : Checks model instantiation and behavior.
  - ✅ `test_inference.py` : Tests inference on controlled inputs.

## 🧪 Testing and validation
To run unit tests :
```sh
pytest tests/unit/
```

## ✨ Contributors
Project carried out in an academic context by Julie and Nafi.