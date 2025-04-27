import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import torch
import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_dataset
from pandas import DataFrame
from torch import LongTensor
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from transformers import MarianTokenizer
from tqdm.auto import tqdm

from DataProcessor import DataProcessor
from CustomTokenizer import CustomTokenizer as Tokenizer
from TranslationDataset import TranslationDataset
from Transformers.Transformer import Transformer

MODEL_SAVE_PATH = './save_model'

def take_dataset() -> DataFrame:
    """
    Raccogliamo il dataset opus_books italiano - olandete
    Il dataset è fatto così : 

    Trains:{
        Translation = [{it : Traduzione italian, nl : Traduzione olandese}, ...]
    }
    """
    dataset = load_dataset("iwslt2017", "iwslt2017-it-en")

    train_df = dataset['train'][:]['translation'] 
    val_df = dataset['test'][:]['translation']  
    
    return train_df, val_df

def lr_lambda(step):

    warm_up_steps = 4000
    if step < warm_up_steps:
        return float(step) / float(max(1, warm_up_steps))
    
    return (warm_up_steps ** 0.5) / (step ** 0.5)

def tokenize_sentence(sentence, vocab):
    tokens = sentence.split()
    tokens = tokens[:50]

    tokens_ids = []

    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
        
        tokens_ids.append(vocab[token])

    return tokens_ids

train_df, val_df = take_dataset()

train_fraction = 0.2
val_fraction = 0.2

train_size = int(len(train_df) * train_fraction)
val_size = int(len(val_df) * val_fraction)

train_df = train_df[:train_size]
val_df = val_df[:val_size]

train_it_tokens = []
train_en_tokens = []

val_it_token = []
val_en_tokens = []

it_vocab = {'<pad>' : 0}
en_vocab = {'<pad>' : 0}

# Tokenizzing training data
for x in range(len(train_df)):

    it_sent = train_df[x]['it']
    en_sent = train_df[x]['en']

    train_en_tokens.append(tokenize_sentence(en_sent, en_vocab))
    train_it_tokens.append(tokenize_sentence(it_sent, it_vocab))

# Tokenizzing val data
for x in range(len(val_df)):

    it_sent = val_df[x]['it']
    en_sent = val_df[x]['en']

    val_en_tokens.append(tokenize_sentence(en_sent, en_vocab))
    val_it_token.append(tokenize_sentence(it_sent, it_vocab))

# Set vocab size
src_vocab_size = len(it_vocab)
tgt_vocab_size = len(en_vocab)

# Create train dataset and train loader
train_dataset = TranslationDataset(
    it_tokens=train_it_tokens,
    en_tokens=train_en_tokens
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create val dataset and val loader
val_dataset = TranslationDataset(
    it_tokens=val_it_token,
    en_tokens=val_en_tokens
)
val_loader = DataLoader(val_dataset, batch_size=64)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_dim = 256
num_heads = 8
num_layers = 6
d_ff = 1024
max_seq_len = max(train_dataset.max_len, val_dataset.max_len)
dropout = 0.1
epochs = 500
lr = 2e-4

model = Transformer(
    inpt_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    embedding_dim=embedding_dim,
    heads_num = num_heads,
    layers_num = num_layers,
    ff_dim= d_ff,
    max_token_len=max_seq_len,
    dropout=dropout,
    device=device,
)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
criterion = CrossEntropyLoss(ignore_index=0, label_smoothing=0.15)
scheduler = LambdaLR(optimizer, lr_lambda)

best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (src, tgt) in enumerate(tqdm(train_loader)):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        logits = model(src, tgt[:,:-1])
        logits_prob_score = torch.softmax(logits, dim = 2)
        predicted_token = torch.argmax(logits_prob_score, dim = 2)

        logits_flat = logits.view(-1, logits.size(-1))
        tgt_flat = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(logits_flat, tgt_flat)
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    if epoch % 5:
        print(f'Input : {src} \n output : {tgt} \n output : {predicted_token}')

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(val_loader):
            src, tgt = src.to(device), tgt.to(device)

            output = model(src, tgt[:, :-1])
            
            output_flat = output.view(-1, output.size(-1))
            tgt_flat = tgt[:, 1:].contiguous().view(-1)  

            loss = criterion(output_flat, tgt_flat)
            
            val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = loss.item()
            torch.save(model.state_dict(), f'./save_model/model_loss_{best_val_loss}')

        print(f'Avg loss validation : {avg_val_loss}')

