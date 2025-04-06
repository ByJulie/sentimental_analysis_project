import pytest
from transformers import BertTokenizer
import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.data_processing import clean_text, MAX_LEN

# tokinizer initialization
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def test_clean_text_basic():
    raw = "This is a TEST! With punctuation... and  spaces.  "
    expected = "this is a test with punctuation and spaces"
    assert clean_text(raw) == expected


def test_clean_text_numbers_and_symbols():
    raw = "Text 123 !! ##@@ cleaned?"
    expected = "text 123 cleaned"
    assert clean_text(raw) == expected


def test_tokenizer_contains_cls_and_sep():
    text = "Hello world"
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = tokens["input_ids"].squeeze().tolist()
    assert input_ids[0] == tokenizer.cls_token_id
    assert tokenizer.sep_token_id in input_ids


def test_tokenizer_output_length():
    text = "Short input"
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = tokens["input_ids"].squeeze()
    attention_mask = tokens["attention_mask"].squeeze()
    assert len(input_ids) == MAX_LEN
    assert len(attention_mask) == MAX_LEN
