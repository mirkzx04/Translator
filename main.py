import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import tqdm
import torch

from datasets import load_dataset
from pandas import DataFrame
from torch import LongTensor
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_

from DataProcessor import DataProcessor
from CustomTokenizer import CustomTokenizer as Tokenizer
from TranslationDataset import TranslationDataset
from Transformers.Transformer import Transformer

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 16
shuffle = False
epochs = 200
embedding_dim = 64
heads_num = 4
layers_num = 4
ff_dim = 512
max_token_len = max(vocab_ita.max_lenght, vocab_nl.max_lenght)
dropout = 0.1
lr = 0.0001

inpt_vocab_size = len(vocab_ita.word2index)
tgt_vocab_size = len(vocab_nl.word2index)

transformer_model = Transformer(
    inpt_vocab_size=inpt_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    embedding_dim=embedding_dim,
    layers_num=layers_num,
    ff_dim=ff_dim,
    max_token_len=max_token_len,
    dropout=dropout,
    device=device,
    heads_num=heads_num
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=shuffle
)

optimizer = optim.Adam(transformer_model.parameters(), lr = lr)
criterion = CrossEntropyLoss(ignore_index=0)

for epoch in tqdm.tqdm(range(epochs)):
    transformer_model.train()
    total_loss = 0
    for batch_idx, (input, target) in enumerate(train_loader):
        # print(f'Shape input pre truc : {input.shape}')
        # print(f'Shape di tgt pre trun : {target.shape}')

        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()

        input_truncated = input[:, :128]
        # print(f'shape input trunc : {input_truncated.shape}')
        tgt_truncated = target[: , :128]
        # print(f'shape tgt trunc : {tgt_truncated.shape}')


        tgt_shifted = tgt_truncated[:, 1:]

        output = transformer_model(input_truncated, tgt_truncated[:, :-1]) 
        # print(f'output shape : {output.shape}')

        output_flat = output.reshape(-1, output.size(-1))  # [16*127, vocab_size]
        tgt_flat = tgt_shifted.reshape(-1)  # [16*127]

        # print(f"output_flat shape: {output_flat.shape}")
        # print(f"tgt_flat shape: {tgt_flat.shape}")
                
        loss = criterion(output_flat, tgt_flat)
        loss.backward()
        clip_grad_norm_(transformer_model.parameters(), max_norm=1)
        print(loss)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataset)

    print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
