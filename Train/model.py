from transformers import BertForSequenceClassification

def createModel(numLabels: int):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=numLabels, output_attentions=False, output_hidden_states=False)
    return model