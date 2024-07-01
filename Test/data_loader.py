import pandas as pd
from sklearn.preprocessing import LabelEncoder
  


"""
    Loads the training and test data from CSV files.
    
    Args:
        train_path (str): Path to the training data CSV file.
        test_path (str, optional): Path to the test data CSV file.
        
    Returns:
        pd.DataFrame: Training data.
        pd.DataFrame: Test data, if provided, else None.
"""                        


def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data, columns_pairs):
    """
    Preprocesses the data by concatenating specified columns into a single text column.
    
    Args:
        data (pd.DataFrame): Input data to preprocess.
        columns_pairs (list of tuple): List of (column_name, new_column_name) pairs to concatenate.
        
    Returns:
        pd.DataFrame: Preprocessed data with a new 'text_data' column.
    """
    data['text_data'] = data.apply(lambda row: ' '.join(
        [f"{row[trait_score]} {row[question_text]}" for trait_score, question_text in columns_pairs]
    ), axis=1)
    return data

def encode_labels(data, column):
    """
    Encodes the labels in the specified column using LabelEncoder.
    
    Args:
        data (pd.DataFrame): Data containing the labels to encode.
        column (str): Name of the column containing the labels.
        
    Returns:
        np.ndarray: Encoded labels.
        LabelEncoder: Fitted LabelEncoder.
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data[column])
    return encoded_labels, label_encoder
