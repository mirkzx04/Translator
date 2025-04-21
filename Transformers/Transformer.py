import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from torch.nn import Embedding
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import Module
from torch.nn import ModuleList

from Transformers.components.PositionEncoding import PositionalEncoding
from Transformers.Encoder import Encoder
from Transformers.Decoder import Decoder

class Transformer(Module):
    def __init__(self,
                 inpt_vocab_size,
                 tgt_vocab_size,
                 embedding_dim,
                 heads_num,
                 layers_num,
                 ff_dim,
                 max_token_len,
                 dropout,
                 pad_token_inpt = 0,
                 pad_token_tgt = 0,
                 device = 'cpu'
                ):
        """
        Inizializza il transformers

        Args:
            inpt_vocab_size -> Numero totale di token nel vocabolario di input
            tgt_vocab_size -> Numero totale di token nel vocabolario target (La traduzione)
            embedding_dim -> Dimensione del vettore di embedding per ogni token
            heads_num -> Numero di teste per MHA nll'Encoder e nel Decoder
            layers_num -> Numero di layers di encoder e decoder
            ff_dim -> Dimensione del PositionFeedForward
            max_token_len -> sequenza di token più lunga
            dropout -> prob che un neuroni si attivi da solo
            pad_token_inpt -> Token del padding nel vocabolario di input
            pad_tokent_tgt -> Token del padding nel vocabolario di target (La traduzione)
            device -> Device di esecuzione dei calcoli di PyTorch
        """
        super().__init__()

        self.encoder_embedding = Embedding(inpt_vocab_size, embedding_dim)
        self.decoder_embedding = Embedding(7840, embedding_dim)

        self.positional_enc = PositionalEncoding(embedding_dim, max_token_len)

        self.enc_layers = self.initialize_enc_layers(layers_num, embedding_dim, heads_num, ff_dim, dropout)
        self.dec_layers = self.initialize_dec_layers(layers_num, embedding_dim, heads_num, ff_dim, dropout)

        # print(f'tgt_size in class Transformers : {tgt_vocab_size}')
        self.fc = Linear(embedding_dim, tgt_vocab_size)

        self.dropout = Dropout(dropout)

        self.pad_token_inpt = pad_token_inpt
        self.pad_token_tgt = pad_token_tgt

        self.device = device
        self = self.to(self.device)
    
    def initialize_enc_layers(self, layers_num, embedding_dim, heads_num, ff_dim, dropout):
        """
        Inizializza i Layer di Encoder
        """

        encoder_layers = ModuleList([Encoder(embedding_dim, ff_dim, dropout, heads_num) for _ in range(layers_num)])

        return encoder_layers
    def initialize_dec_layers(self, layers_num, embedding_dim, heads_num, ff_dim, dropout):
        """
        Inizializza i Layer di Encoder
        """

        decoder_layers = ModuleList([Decoder(embedding_dim, ff_dim, dropout, heads_num) for _ in range(layers_num)])

        return decoder_layers
    def get_mask(self, mask, input, target):
        
        # print(f'input shape nel mask : {input.shape}')
        # print(f'tgt shape nel mask : {target.shape}')

        if mask:
            inpt_mask, tgt_mask = self.generate_mask(mask['input'], mask['target'])
        else: 
            inpt_mask, tgt_mask = self.generate_mask(input!=self.pad_token_inpt, target!=self.pad_token_tgt)
        return inpt_mask, tgt_mask
    
    def generate_mask(self, inpt_mask, tgt_mask):
        """
        Inizializza le maschere
        """
        inpt_mask = inpt_mask.unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(3)

        tgt_seq_len = tgt_mask.size(2)
        nopeak_mask = (1 - torch.triu(torch.ones(1, tgt_seq_len, tgt_seq_len), diagonal = 1)).bool()

        tgt_mask = tgt_mask & nopeak_mask.to(self.device)

        return inpt_mask, tgt_mask
    
    def execute_enc_layers(self, input, inpt_mask):
        inpt_embedding = self.encoder_embedding(input)
        # print(f'inpt embedding shape : {inpt_embedding.shape}')
        inpt_embedding = self.positional_enc(inpt_embedding)
        # print(f'inpt embedding shape : {inpt_embedding.shape}')
        inpt_embedding = self.dropout(inpt_embedding)
        # print(f'inpt embedding shape : {inpt_embedding.shape}')

        enc_output = inpt_embedding
        for enc_layer in self.enc_layers:
            enc_output = enc_layer(enc_output, inpt_mask)

        return enc_output

    def execute_dec_layers(self, target, enc_output, inpt_mask, tgt_mask):
        tgt_embedding = self.decoder_embedding(target)
        tgt_embedding = self.positional_enc(tgt_embedding)
        tgt_embedding = self.dropout(tgt_embedding)

        dec_output = tgt_embedding
        for dec_layer in self.dec_layers:
            dec_output = dec_layer(dec_output,enc_output, inpt_mask, tgt_mask)

        return dec_output
    def forward(self, input, target, mask = None):
        inpt_mask, tgt_mask = self.get_mask(mask, input, target)
        # print(f'Shape delle maschere : {inpt_mask.shape}, {tgt_mask.shape}') 

        enc_output = self.execute_enc_layers(input, inpt_mask)
        dec_output = self.execute_dec_layers(target, enc_output, inpt_mask, tgt_mask)
        output = self.fc(dec_output)

        return output