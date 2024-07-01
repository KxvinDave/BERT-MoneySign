import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from typing import List, Tuple
import logging
import time

from dataPreprocessing import preprocessData, tokenizeData
from model import createModel
from utils import setSeed, formatTime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, device: torch.device):
        self.device = device
    def prepareData(self, dataPath: str) -> Tuple[DataLoader, DataLoader, int]:
        data = preprocessData(dataPath)
        inputIDs, attentionMasks, labels = tokenizeData(data)
        trainInputs, valInputs, trainLabels, valLabels = train_test_split(inputIDs, labels, test_size=0.2, random_state=42)
        trainMasks, valMasks, _, _ = train_test_split(attentionMasks, labels, test_size=0.2, random_state=42)

        trainData = TensorDataset(trainInputs, trainMasks, trainLabels)
        trainSampler = RandomSampler(trainData)
        trainDataLoader = DataLoader(trainData, sampler=trainSampler, batch_size=32)

        valData = TensorDataset(valInputs, valMasks, valLabels)
        valSampler = SequentialSampler(valData)
        valDataLoader = DataLoader(valData, sampler=valSampler, batch_size=32)
        return trainDataLoader, valDataLoader, len(set(labels.numpy()))
    def trainModel(self, trainDataLoader: DataLoader, valDataLoader: DataLoader, numLabels: int, epochs: int = 47):
        model = createModel(numLabels)
        model.to(self.device)
        optimiser = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        totalSteps = len(trainDataLoader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimiser, num_warmup_steps=0, num_training_steps=totalSteps)

        setSeed(42)
        for epoch in range(epochs):
            logger.info(f'======Epoch {epoch + 1} / {epochs}======')
            logger.info('Training...')

            t0 = time.time()
            totalTrainingLoss = 0
            model.train()

            for batch in trainDataLoader:
                bInputIDs = batch[0].to(self.device)
                bInputMask = batch[1].to(self.device)
                bLabels = batch[2].to(self.device)
                model.zero_grad()
                outputs = model(bInputIDs, token_type_ids=None, attention_mask=bInputMask, labels=bLabels)
                loss = outputs.loss
                totalTrainingLoss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
            avgTrainingLoss = totalTrainingLoss / len(trainDataLoader)
            trainingTime = formatTime(time.time() - t0)

            logger.info(f'Average training loss: {avgTrainingLoss:.2f}')
            logger.info(f'Training epoch took: {trainingTime}')

            logger.info('Running Validation')
            t0 = time.time()
            model.eval()
            totalEvalLoss = 0
            nbEvalSteps = 0
            for batch in valDataLoader:
                bInputIDs = batch[0].to(self.device)
                bInputMask = batch[1].to(self.device)
                bLabels = batch[2].to(self.device)
                with torch.no_grad():
                    outputs = model(bInputIDs, token_type_ids=None, attention_mask=bInputMask, labels=bLabels)
                loss = outputs.loss
                totalEvalLoss += loss.item()
            averageValLoss = totalEvalLoss / len(valDataLoader)
            validationTime = formatTime(time.time() - t0)
            logger.info(f'Average validation loss: {averageValLoss:.2f}')
            logger.info(f'Validation took: {validationTime}')
        logger.info('Training complete')
        return model
    
def Train(dataPath: str, device: torch.device):
    trainer = Trainer(device)
    trainDataLoader, valDataLoader, numLabels = trainer.prepareData(dataPath)
    model = trainer.trainModel(trainDataLoader, valDataLoader, numLabels)
    return model