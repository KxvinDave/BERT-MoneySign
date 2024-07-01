import pandas as pd
import re
import torch
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple

def cleanText(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocessData(path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found on {path}")
    columnPairs = [
        ('Q1_Curiosity', 'Q1_How often I search for new investment options:'),
        ('Q2_Curiosity', 'Q2_Investment options that I\'m aware of:'),
        ('Q3_Curiosity', 'Q3_Investment options that I\'ve invested in at least once:'),
        ('Q4_Curiosity', 'Q4_I monitor news about my investments on a daily basis:'),
        ('Q5_Creativity', 'Q5_My income sources.'),
        ('Q6_Creativity', 'Q6_The percentage of my total income that my largest income source contributes to:'),
        ('Q7_Creativity', 'Q7_Statements that best describe my financial behaviour:'),
        ('Q8_Patience', 'Q8_If my long-term (>5 years) investment suddenly moved up by 50% I would:'),
        ('Q9_Patience', 'Q9_If my long-term (>5 years) investment suddenly fell by 50% (and Nifty fell by 40%) I would:'),
        ('Q10_Patience', 'Q10_When shopping I prefer items on sale over new arrivals.'),
        ('Q11_Organization', 'Q11_My financial plan covers a period of:'),
        ('Q12_Organization', 'Q12_I generally make investment decisions by:'),
        ('Q13_Discipline', 'Q13_Financial habits that I follow:'),
        ('Q14_Discipline', 'Q14_How often do I spend 10% or more of my monthly income on unplanned expenses?'),
        ('Q15_Hypercompetition', 'Q15_I frequently compare my portfolio performance to market benchmarks and/or other investors\' portfolios.'),
        ('Q16_Hypercompetition', 'Q16_How do I approach networking?'),
        ('Q17_Hypercompetition', 'Q17_How do I react when my peers acquire something I can\'t currently afford?'),
        ('Q18_Aggressiveness', 'Q18_I\'d find it hard to accept the advice of a financial planner if I thought I was right.'),
        ('Q19_Aggressiveness', 'Q19_Once I form an opinion I don\'t entertain opposing views.'),
        ('Q20_Satisfaction', 'Q20_How often I trade, bet or gamble:'),
        ('Q21_Satisfaction', 'Q21_The percentage of returns I would generally want to achieve within one year:'),
        ('Q22_Satisfaction', 'Q22_How often I buy a new phone:'),
        ('Q23_Anxiety', 'Q23_The percentage of loss on my investment portfolio that I can tolerate:'),
        ('Q24_Anxiety', 'Q24_The amount of personal time I spend on social media everyday:'),
        ('Q25_Anxiety', 'Q25_I\'m worried about my financial situation.')
        ]
    data['textData'] = data.apply(lambda row: ' '.join(
        [cleanText(f"{row[traitScore]} {questionText}") for traitScore, questionText in columnPairs]
    ), axis=1)
    return data

def tokenizeData(data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    encodedData = tokeniser.batch_encode_plus(data['textData'].to_list(),
                                              add_special_tokens=True,
                                              return_attention_mask=True,
                                              pad_to_max_length=True,
                                              max_length=64,
                                              truncation=True,
                                              return_tensors='pt')
    inputIDs = encodedData['input_ids']
    attentionMasks = encodedData['attention_mask']
    labelEncoder = LabelEncoder()
    labels = torch.tensor(labelEncoder.fit_transform(data['MoneySign']))
    return inputIDs, attentionMasks, labels