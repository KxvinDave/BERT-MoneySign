from flask import Flask, request, jsonify
import torch
import random
import numpy as np
import logging
from config import MODEL_PATH, DEVICE, GROQ_API_KEY, SEED_VAL
from model import load_model
from predict import predictMoneySign, fit_label_encoder
#from explain import explainMS
from database import get_user_data

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Set the seed value all over the place to make this reproducible.
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

# Load Model and Tokenizer
model, tokenizer = load_model(MODEL_PATH)

# Fit Label Encoder with cleaned labels
LABELS = ["Persistent Horse", "Far-Sighted Eagle", "Tactical Tiger", "Enlightened Whale", 
          "Opportunistic Lion", "Vigilant Turtle", "Virtuous Elephant", "Stealthy Shark"]
label_encoder = fit_label_encoder(LABELS)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    phone_number = data.get('phone_number')

    if not phone_number:
        return jsonify({"error": "Phone number is required"}), 400

    # Fetch user data based on phone number
    singular_input = get_user_data(phone_number)
    
    if not singular_input:
        return jsonify({"error": "No data found for the given phone number"}), 404

    # Predict MoneySign
    predicted_ms = predictMoneySign(model, tokenizer, label_encoder, singular_input, DEVICE)
    if predicted_ms not in LABELS:
        logging.error(f"The predicted MoneySign label '{predicted_ms}' is unknown")
        return jsonify({"error": "The predicted MoneySign label is unknown"}), 400

    # Explain the singular prediction
    #explanation = explainMS(GROQ_API_KEY, singular_input, predicted_ms)

    return jsonify({
        "predicted_money_sign": predicted_ms,
        #"explanation": explanation
    })

if __name__ == '__main__':
    app.run(debug=True)
