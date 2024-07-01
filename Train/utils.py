import random
import torch
import numpy as np
import time
import datetime

def setSeed(seedValue=42):
    random.seed(seedValue)
    np.random.seed(seedValue)
    torch.manual_seed(seedValue)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seedValue)

def formatTime(elapsed):
    elapsedRounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsedRounded))