import pandas as pd
from sklearn.preprocessing import LabelEncoder

TRAIN_DATA_PATH = "C:\\Users\\Chirag Sharma\\Desktop\\BERT MODEL MONEY_SIGN\\Train & Test Data for BERT Model\\FINAL MONEYSIGN TRAINING DATA (4) - Sheet1.csv"  # Update this path to your training data file

# Load training data
train_data = pd.read_csv(TRAIN_DATA_PATH)

# Clean labels
train_data['MoneySign'] = train_data['MoneySign'].str.strip().str.replace('Far-sighted Eagle', 'Far-Sighted Eagle')
train_data = train_data.dropna(subset=['MoneySign'])

# Extract labels
labels = train_data['MoneySign'].unique()

# Fit Label Encoder
label_encoder = LabelEncoder()
label_encoder.fit(train_data['MoneySign'])

print("Labels in training data:", labels)
print("Encoded labels:", label_encoder.classes_)
