import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from torch.nn import Module
from torch.nn import Dropout, LayerNorm

from components.MultiHeadAttention import MultiHeadAttention as MHA
from components.PositionFeedForward import PositionFeedForward as FF

class Encoder(Module):
    def __init__(self, embedding_dim, ff_dim, dropout, heads_num, mask = None):
        super().__init__()
        """
        Inizializza l'Encoder
        embedding_dim -> Dimensione degli embedding
        ff_dim -> Dimensione del PositionFeedForward
        dropout -> Prob. che dei neuroni si attivino da soli
        heads_num -> Numero di teste per MHA
        mask -> Maschera per distogliere l'attenzione di padding
        """

        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()

        self.dropout = Dropout(dropout)

        self.attn = MHA(heads_num, embedding_dim)
        self.ff = FF(embedding_dim, ff_dim)

        self.mask = mask

    def forward(self, X_train):
        """
        Forward dell'Encoder
        Effettua i calcoli / trasformazioni dei due sotto layer
        """
        attn = self.attn(X_train, X_train, X_train, self.mask)
        attn_norm = self.norm1(X_train + self.dropout(attn))

        ff_out = self.ff(attn_norm)

        enc_output = self.norm2(attn_norm + self.dropout(ff_out))

        return enc_output


