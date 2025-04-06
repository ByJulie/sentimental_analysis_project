import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.data_extraction import load_data

import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import re
import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set the model name for tokenization
MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Maximum token length
MAX_LEN = 160

def clean_text(text):
    """
    Clean raw text by removing special characters, converting to lowercase,
    and normalizing whitespace.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and trim whitespace
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text

def to_sentiment(rating):
    """
    Convert numerical rating into sentiment class.
    0 - negative, 1 - neutral, 2 - positive
    """
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

def preprocess_data(df):
    """
    Apply text cleaning and sentiment mapping to the dataset.
    """
    df['cleaned_content'] = df['content'].apply(clean_text)
    df['sentiment'] = df['score'].apply(to_sentiment)
    return df

class SentimentDataset(Dataset):
    """
    Custom dataset for tokenized text.
    """
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        target = self.targets[index]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    """
    Create a PyTorch DataLoader from a Pandas DataFrame.
    """
    dataset = SentimentDataset(
        texts=df.cleaned_content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)

df = load_data("data/dataset.csv") 
df = preprocess_data(df)

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

BATCH_SIZE = 16
train_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

if __name__ == "__main__":
    
    sample_batch = next(iter(train_loader))
    print(sample_batch.keys())
    print(sample_batch['input_ids'].shape)
    print(sample_batch['attention_mask'].shape)
    print(sample_batch['targets'].shape)

