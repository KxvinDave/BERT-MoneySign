import torch
from config import MODEL_PATH, DEVICE, GROQ_API_KEY, LABELS, SEED_VAL
from model import load_model
from predict import predictMoneySign, fit_label_encoder
from explain import explainMS
from database import get_user_data
import random
import numpy as np

# Set the seed value all over the place to make this reproducible.
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

# Load Model and Tokenizer
model, tokenizer = load_model(MODEL_PATH)

# Fit Label Encoder with predefined labels
label_encoder = fit_label_encoder(LABELS)

def main():
    phone_number = input("Enter the user's phone number: ")
    
    # Fetch user data based on phone number
    singular_input = get_user_data(phone_number)
    
    if not singular_input:
        print("No data found for the given phone number.")
        return

    # Predict MoneySign
    predicted_ms = predictMoneySign(model, tokenizer, label_encoder, singular_input, DEVICE)
    print(f"Predicted MoneySign for the given phone number: {predicted_ms}")

    # Explain the singular prediction
    if predicted_ms != "Unknown Label":
        explanation = explainMS(GROQ_API_KEY, singular_input, predicted_ms)
        print(f"Explanation for the prediction:\n{explanation}")
    else:
        print("The predicted MoneySign label is unknown.")

if __name__ == "__main__":
    main()



