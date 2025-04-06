# function to load the file

import pandas as pd


def load_data(file_path):
    """
    Load raw data from a CSV file.
    Handles missing files and incorrect formats.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame if successful, None otherwise.
    """
    try:
        df = pd.read_csv(file_path)
        return df

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' cannot be found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: The file '{file_path}' has an incorrect format.")
    except Exception as e:
        print(f"Unknown error when loading file: {e}")

    return None  # Return None if we have failure


# usage
if __name__ == "__main__":
    data = load_data("data/dataset.csv")
    if data is not None:
        print(data.head())
