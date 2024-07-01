import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

def fit_label_encoder(labels):
    labels = [label.strip().replace('Far-sighted Eagle', 'Far-Sighted Eagle') for label in labels if pd.notna(label)]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    return label_encoder

def tokenize_data(tokenizer, texts, max_length=64):
    """
    Tokenizes a list of texts using the provided tokenizer.
    
    Args:
        tokenizer (BertTokenizer): Tokenizer to use for tokenizing the texts.
        texts (list of str): List of texts to tokenize.
        max_length (int): Maximum length of the tokenized sequences.
        
    Returns:
        torch.Tensor: Tensor of tokenized input IDs.
        torch.Tensor: Tensor of attention masks.
    """
    
    
    input_ids = []
    attention_masks = []

    for sent in texts:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens=True,
                            max_length=max_length,
                            pad_to_max_length=True,
                            return_attention_mask=True,
                            return_tensors='pt',
                            truncation=True,
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def create_dataloader(input_ids, attention_masks, batch_size=16):
    """
    Creates a DataLoader from the tokenized inputs and attention masks.
    
    Args:
        input_ids (torch.Tensor): Tensor of input IDs.
        attention_masks (torch.Tensor): Tensor of attention masks.
        batch_size (int): Batch size for the DataLoader.
        
    Returns:
        DataLoader: DataLoader for the tokenized inputs and attention masks.
    """
    data = TensorDataset(input_ids, attention_masks)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

def predict(model, dataloader, device):
    
    """
    Makes predictions using the provided model and DataLoader.
    
    Args:
        model (BertForSequenceClassification): Trained BERT model for sequence classification.
        dataloader (DataLoader): DataLoader for the tokenized inputs and attention masks.
        device (torch.device): Device to run the model on (CPU or GPU).
        
    Returns:
        np.ndarray: Array of predictions.
    """

    




    model.eval()
    model.to(device)
    predictions = []

    for batch in dataloader:
        batch = tuple(b.to(device) for b in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        predictions.extend(preds)

    return np.array(predictions)

def predictMoneySign(model, tokenizer, label_encoder, text: str, device):
    """
    Predicts the MoneySign for a given text using the provided model, tokenizer, and LabelEncoder.
    
    Args:
        model (BertForSequenceClassification): Trained BERT model for sequence classification.
        tokenizer (BertTokenizer): Tokenizer corresponding to the pre-trained BERT model.
        label_encoder (LabelEncoder): Fitted LabelEncoder for converting numerical predictions to class names.
        text (str): Input text to classify.
        device (torch.device): Device to run the model on (CPU or GPU).
        
    Returns:
        str: Predicted MoneySign.
    """

    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    pred_class = torch.argmax(logits, dim=1).item()
    try:
        answer = label_encoder.inverse_transform([pred_class])[0]
    except IndexError:
        answer = "Unknown Label"
    return answer

