import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from torch.nn import Module
from torch.nn import Dropout, LayerNorm

from components.MultiHeadAttention import MultiHeadAttention as MHA
from components.PositionFeedForward import PositionFeedForward as FF

class Decoder(Module):
    def __init__(self, embedding_dim, ff_dim, dropout, heads_num, mask = None):
        super().__init__()
        """
        Inizializza il Decoder
        embedding_dim -> Dimensione degli embedding
        ff_dim -> Dimensione del PositionFeedForward
        dropout -> Prob. che dei neuroni si attivino da soli
        heads_num -> Numero di teste per MHA
        mask -> Maschera per distogliere l'attenzione di padding
        """
        self.embedding_dim = embedding_dim
        self.ff_dim = ff_dim

        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()
        self.norm3 = LayerNorm()

        self.dropout = Dropout(dropout)

        self.attn = MHA(heads_num, embedding_dim)
        self.ff = FF(embedding_dim, ff_dim)

        self.mask = mask
    
    def sub_layer1(self, X_train):
        """
        Calcola l'attenzione del primi sub layers
        """
        attn = self.attn(X_train, X_train, X_train, self.mask)
        attn_norm = self.norm1(X_train + self.dropout(attn))

        return attn_norm
    
    def sub_layer2(self, sub_layer1, enc_output):
        """
        Calcola l'attenzione del secondo sub layers
        """
        attn = self.attn(sub_layer1, enc_output, enc_output, self.mask)
        attn_norm = self.norm2(sub_layer1 + self.dropout(attn))

        return attn_norm

    def forward(self, X_train, enc_output):
        sub_layer1 = self.sub_layer1(X_train)

        sub_layer2 = self.sub_layer2(sub_layer1, enc_output)

        ff_out = self.ff(sub_layer2)

        dec_output = self.norm3(sub_layer2 + self.dropout(ff_out))

        return dec_output