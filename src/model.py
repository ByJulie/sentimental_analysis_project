import argparse
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_scheduler
from tqdm import tqdm
from collections import defaultdict
import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.data_processing import train_loader, val_loader

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
MODEL_NAME = "bert-base-cased"
NUM_CLASSES = 3  # 0: Negative, 1: Neutral, 2: Positive

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_CLASSES)
model.to(device)


# Parameters for accepting command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true',
                        help='Indicate that you only need to evaluate the model, without training it.')
    return parser.parse_args()

# Function to load model and data (if required)


def load_model_and_data():
    # If training is complete, load the best model
    tokenizer = AutoTokenizer.from_pretrained("best_model")
    model = AutoModelForSequenceClassification.from_pretrained(
        "best_model", num_labels=NUM_CLASSES)
    model.to(device)
    return tokenizer, model

# Training function


def train_model(model, train_loader, val_loader, epochs, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
    total_steps = len(train_loader) * epochs
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0
    best_model_state = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}\n" + "-" * 10)
        model.train()

        total_loss, correct = 0, 0
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            inputs = {key: value.to(device) for key, value in batch.items()}
            targets = inputs.pop("targets")

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct += (outputs.logits.argmax(1) == targets).sum().item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=total_loss/len(train_loader),
                             acc=correct/len(train_loader.dataset))

        train_acc = correct / len(train_loader.dataset)
        train_loss = total_loss / len(train_loader)

        # Validation
        val_acc, val_loss = eval_model(model, val_loader, loss_fn)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}\n")

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        # Store the best model in memory
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model_state = model.state_dict().copy()

    # Save the best model only after the end of training
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.save_pretrained("best_model")
        tokenizer.save_pretrained("best_model")
        print(f"Best model saved with accuracy: {best_accuracy:.4f}")

    return history


# Evaluation function
def eval_model(model, data_loader, loss_fn):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            targets = inputs.pop("targets")

            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, targets)

            total_loss += loss.item()
            correct += (outputs.logits.argmax(1) == targets).sum().item()

    return correct / len(data_loader.dataset), total_loss / len(data_loader)

# Main function


def main():
    loss_fn = nn.CrossEntropyLoss().to(device)
    args = parse_args()

    # Si on passe --eval-only, on charge seulement le modèle et on l'évalue
    if args.eval_only:
        print("Only evaluating the model...")
        tokenizer, model = load_model_and_data()
        # Effectuer l'évaluation seulement
        eval_model(model, val_loader, loss_fn)
    else:
        print("Training the model...")
        tokenizer, model = load_model_and_data()
        history = train_model(model, train_loader, val_loader, epochs=3)
        # Après l'entraînement, tu peux aussi évaluer le modèle
        eval_model(model, val_loader, loss_fn)


if __name__ == "__main__":
    main()
