import torch
# Path to the pre-trained BERT model configurations
MODEL_PATH = "C:\\Users\\Chirag Sharma\\Desktop\\BERT MODEL MONEY_SIGN\\Bert Configurations\\"
# Device configuration: Use GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# API key for the Groq service
GROQ_API_KEY = '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'
# List of possible MoneySign labels
LABELS = ["Persistent Horse", "Far-Sighted Eagle", "Tactical Tiger", "Enlightened Whale", "Opportunistic Lion", "Vigilant Turtle", "Virtuous Elephant", "Stealthy Shark"]
SEED_VAL = 42