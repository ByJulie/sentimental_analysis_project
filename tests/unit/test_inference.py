import torch
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, AutoTokenizer
import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.inference import predict_sentiment, model_dir, LABELS

def test_model_loading():
    """Checks that the model and tokenizer load correctly"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    assert tokenizer is not None, "Tokenizer not loaded correctly"
    assert model is not None, "The model has not been loaded correctly"
    assert isinstance(
        model, BertForSequenceClassification),  f"The loaded model is the wrong type: {type(model)}"


def test_text_preprocessing():
    """Check the tensor transformation of the text"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sample_text = "This is a test sentence."

    encoding = tokenizer.encode_plus(
        sample_text,
        add_special_tokens=True,
        max_length=160,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    assert "input_ids" in encoding, "input_ids absent from encoding"
    assert "attention_mask" in encoding, "attention_mask absent from encoding"
    assert encoding["input_ids"].shape == (
        1, 160), "input_ids has an incorrect form"
    assert encoding["attention_mask"].shape == (
        1, 160), "attention_mask has an incorrect form"


def test_model_inference():
    """Checks that the model generates logits with the correct form"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    sample_text = "This is a test sentence."
    encoding = tokenizer(sample_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)

    assert hasattr(outputs, "logits"), "The model does not return logits"
    assert outputs.logits.shape == (
        1, len(LABELS)), f"Incorrect logit form: {outputs.logits.shape}"


def test_predict_sentiment():
    """Testing the entire inference process"""
    sample_text = "I love this product!"
    sentiment = predict_sentiment(sample_text)

    assert sentiment in LABELS.values(), f"Unknown label returned: {sentiment}"
