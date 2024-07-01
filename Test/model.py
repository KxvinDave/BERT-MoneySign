from transformers import BertTokenizer, BertForSequenceClassification
import torch
import random
import numpy as np  
"""
Loads the pre-trained BERT model and tokenizer from the specified path.

Args:
    model_path (str): Path to the pre-trained model and tokenizer.
    
Returns:
    model (BertForSequenceClassification): Pre-trained BERT model for sequence classification.
    tokenizer (BertTokenizer): Tokenizer corresponding to the pre-trained BERT model.
"""



def load_model(model_path):
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)  # Sets the seed for Python's built-in random module
    np.random.seed(seed_val) # Sets the seed for NumPy's random number generator
    torch.manual_seed(seed_val)  # Sets the seed for PyTorch on the CPU
    torch.cuda.manual_seed_all(seed_val)  # Sets the seed for all GPUs (if using CUDA)



def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer
