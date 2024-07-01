import torch
from train import Train
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    dataPath = '/Data/train.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device {device}')

    try:
        model = Train(dataPath, device)
        torch.save(model.state_dict(), 'MoneySign Model.pth')
        logger.info('Model saved Successfully')
    except Exception as e:
        logger.error(f'An error occurred: {str(e)}')


if __name__ == "__main__":
    main()