from src.data_extraction import load_data
import pandas as pd
from io import StringIO
import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


def test_load_valid_csv(tmp_path):
    # Create a temporary CSV file
    file_path = tmp_path / "test_data.csv"
    data = "col1,col2,col3\n1,2,3\n4,5,6"
    file_path.write_text(data)

    df = load_data(str(file_path))

    assert df is not None, "DataFrame should not be None for a valid CSV"
    assert isinstance(df, pd.DataFrame), "Loaded data should be a DataFrame"
    assert list(df.columns) == ["col1", "col2",
                                "col3"], "Columns should match expected values"
    assert df.shape == (2, 3), "DataFrame shape should be (2,3)"


def test_load_missing_file():
    df = load_data("non_existent_file.csv")
    assert df is None, "DataFrame should be None if file is missing"


def test_load_empty_csv(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("")  # Write empty content

    df = load_data(str(file_path))
    assert df is None, "DataFrame should be None for an empty file"
