import sys
import os
import torch
import numpy as np
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
                 device = 'cpu',
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
        super(Transformer, self).__init__()
        self.enc_emb_norm = torch.nn.LayerNorm(embedding_dim)
        self.dec_emb_norm = torch.nn.LayerNorm(embedding_dim)

        self.encoder_embedding = Embedding(inpt_vocab_size, embedding_dim)
        self.decoder_embedding = Embedding(tgt_vocab_size, embedding_dim)

        self.positional_enc = PositionalEncoding(embedding_dim, max_token_len)

        self.enc_layers = self.initialize_enc_layers(layers_num, embedding_dim, heads_num, ff_dim, dropout)
        self.dec_layers = self.initialize_dec_layers(layers_num, embedding_dim, heads_num, ff_dim, dropout)

        # print(f'tgt_size in class Transformers : {tgt_vocab_size}')
        self.fc = Linear(embedding_dim, tgt_vocab_size, bias=False)

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
        """
        Restituisce le maschere per il decoder e per l'encoder
        input -> input del modello
        mask -> Maschera 
        target -> Obbiettivo del modello
        """

        if mask:
            inpt_mask, tgt_mask = self.generate_mask(mask['input'], mask['target'])
        else: 
            inpt_mask, tgt_mask = self.generate_mask(input!=self.pad_token_inpt, target!=self.pad_token_tgt)
        return inpt_mask, tgt_mask
    
    def generate_mask(self, inpt, tgt):
        """
        Inizializza le maschere
        """
        inpt_mask = inpt.unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt.unsqueeze(1).unsqueeze(3)

        seq_len = tgt_mask.size(2)


        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(self.device)
        return inpt_mask, tgt_mask
    
    def execute_enc_layers(self, src, src_mask):
        """
        Esegue i layers dell'encoder
        Args:
            src -> Input del modello
            src_mask -> Maschera di input, nasconde i token di padding
        """

        src_embedded = self.dropout(
            self.positional_enc(
                self.encoder_embedding(src)
            ))

        enc_output = src_embedded
        for enc_layer in self.enc_layers:
            enc_output = enc_layer(enc_output, src_mask)

        return enc_output

    def execute_dec_layers(self, tgt, enc_output, inpt_mask, tgt_mask):
        """
        Esegue i layers del decoder
        Args : 
            tgt -> Obbiettivo del modello
            enc_output -> Output dell'encoder
            inpt_mask -> maschera di input, nasconde i token di padding
            tgt_mask -> Maschera triang. inferiore per il sistema autoregressivo
        """
        tgt_embedded = self.dropout(
            self.positional_enc(
                self.decoder_embedding(tgt)
            )
        )    

        dec_output = tgt_embedded
        for dec_layer in self.dec_layers:
            dec_output = dec_layer(dec_output,enc_output, inpt_mask, tgt_mask)

        return dec_output
    
    def forward(self, input, target, mask = None):
        """
        Viene eseguito per il training
        Args:
            input -> Input dato al modello
            target -> L'obbiettivo del modello
            mask -> Maschera
        """
        inpt_mask, tgt_mask = self.get_mask(mask, input, target)

        enc_output = self.execute_enc_layers(input, inpt_mask)
        dec_output = self.execute_dec_layers(target, enc_output, inpt_mask, tgt_mask)
        output = self.fc(dec_output)

        return output
    
    def decode(self, input, sos_token_id, eos_token_id, mask = None, max_token_len = 25):
        """
        Metodo per l'inferenza
        Args:
            input -> input al modello
            bos_token -> token che segnano l'inizio della sequenza
            eos_token -> Token che segnano la fine della sequenza
            mask -> Maschera
            max_token_len -> Lunghezza massima di token
        """
        print(f'Input shape : {input.shape}')
        input_shape = input.shape[0]
        tgt = torch.tensor([[sos_token_id]] * input_shape).to(self.device)

        inpt_mask, tgt_mask = self.get_mask(mask, input, tgt)
        enc_output = self.execute_enc_layers(input, inpt_mask)

        out_token = tgt
        unfinished_token = np.array([1] * input_shape)
        i = 0
        while (sum(unfinished_token)>0 and i <max_token_len):
            _, tgt_mask = self.generate_mask(tgt != self.pad_token_tgt, tgt != self.pad_token_tgt)
            dec_output = self.execute_dec_layers(tgt, enc_output, inpt_mask, tgt_mask)

            output = self.fc(dec_output)

            out_token = torch.cat((out_token, output[:,-1:, :].argmax(-1)), dim = 1)

            unfinished_token[(out_token[:, -1] == eos_token_id).cpu().numpy()] = 0

            i += 1

        return out_token