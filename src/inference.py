import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define labels
LABELS = {0: "Négatif", 1: "Neutre", 2: "Positif"}

# Load tokenizer and trained model
model_dir = os.path.join(os.path.dirname(__file__), "..", "best_model")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
model.eval()

# Prediction function
def predict_sentiment(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=160,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return LABELS[prediction]

#if __name__ == "__main__":
    #text = input("Enter a text for sentimental analysis: ")
    #sentiment = predict_sentiment(text)
    #print(f"Predicted sentiment: {sentiment}")


if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")  
    df["predicted_sentiment"] = df["content"].apply(predict_sentiment)
    print("\n Predicted Sentiments:")
    print(df[["content", "predicted_sentiment"]].to_string(index=False))
