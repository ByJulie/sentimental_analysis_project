from src.model import model, tokenizer, NUM_CLASSES
import torch
import pytest
import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Load tokenizer and trained model


@pytest.fixture
def dummy_inputs():
    """Fixture for generating dummy inputs"""
    texts = ["This is a positive sentence.", "This is a negative sentence."]
    inputs = tokenizer(texts, padding=True, truncation=True,
                       return_tensors="pt").to(model.device)
    return inputs, len(texts)


def test_model_output_shape(dummy_inputs):
    """Checks that output has the expected form (batch_size, num_classes)"""
    print(f"Model is running on: {model.device}")
    inputs, batch_size = dummy_inputs
    with torch.no_grad():
        outputs = model(**inputs)
    assert outputs.logits.shape == (batch_size, NUM_CLASSES), \
        f"Expected shape ({batch_size}, {NUM_CLASSES}), but got {outputs.logits.shape}"


def test_model_runs_without_crash(dummy_inputs):
    """Checks that the model does not crash during inference"""
    inputs, _ = dummy_inputs
    try:
        with torch.no_grad():
            _ = model(**inputs)
    except Exception as e:
        pytest.fail(f"The model crashed during inference: {e}")
