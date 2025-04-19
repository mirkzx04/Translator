import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import tqdm

from datasets import load_dataset
from pandas import DataFrame
from torch import LongTensor
from torch.utils.data import Dataset, DataLoader

from DataProcessor import DataProcessor
from CustomTokenizer import CustomTokenizer as Tokenizer
from TranslationDataset import TranslationDataset

def take_dataset() -> DataFrame:
    """
    Raccogliamo il dataset opus_books italiano - olandete
    Il dataset è fatto così : 

    Trains:{
        Translation = [{it : Traduzione italian, nl : Traduzione olandese}, ...]
    }
    """
    dataset = load_dataset("opus_books", "it-nl")
    dataset = dataset['train']['translation']
    len_dataset = len(dataset)

    trans_ita = []
    trans_nl = []

    for n in range(len_dataset):
        ita_pha = dataset[n]['it']
        nl_pha = dataset[n]['nl']

        trans_ita.append(ita_pha)
        trans_nl.append(nl_pha)

    df = DataFrame({
        'ita' : trans_ita,
        'nl' : trans_nl
    })

    return df

def prepare_data(dataset, vocab_ita, vocab_nl):
    input_token = []
    target_token = []

    dataset_ita = dataset['ita']
    dataset_nl = dataset['nl']

    token_dataset = {}

    for i in range(len(dataset)):
        input_sentence = dataset_ita[i]
        target_sentence = dataset_nl[i]

        input_token.append(vocab_ita.tokenize_sentence(input_sentence))
        target_token.append(vocab_nl.tokenize_sentence(target_sentence))

    token_dataset['ita'] = LongTensor(input_token)
    token_dataset['nl'] = LongTensor(target_token)

    return token_dataset

dataset = take_dataset()
data_processor = DataProcessor(
    dataset=dataset,
    train_ration=0.8,
)

vocab_ita = Tokenizer()
vocab_nl = Tokenizer()

vocab_ita.build_vocab('ita', dataset)
vocab_nl.build_vocab('nl', dataset)

train_dataset, val_dataset = data_processor.split_dataset()
len_train_dataset = len(train_dataset)

train_dataset = prepare_data(
    dataset=train_dataset, 
    vocab_ita=vocab_ita, 
    vocab_nl=vocab_nl
    )
val_dataset = prepare_data(
    dataset=val_dataset, 
    vocab_ita=vocab_ita, 
    vocab_nl=vocab_nl
    )

train_dataset = TranslationDataset(train_dataset['ita'], train_dataset['nl'])
val_dataset = TranslationDataset(val_dataset['ita'], val_dataset['nl'])

batch_size = 32
shuffle = False
epochs = 200

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=shuffle
)

for epoch in tqdm.tqdm(range(epochs)):
    for batch_idx, (ita_data, nl_data) in enumerate(train_loader):
        pass

