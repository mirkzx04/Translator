import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from torch.nn import Module
from torch.nn import Dropout, LayerNorm

from Transformers.components.MultiHeadAttention import MultiHeadAttention as MHA
from Transformers.components.PositionFeedForward import PositionFeedForward as FF

class Decoder(Module):
    def __init__(self, embedding_dim, ff_dim, dropout, heads_num):
        super().__init__()
        """
        Inizializza il Decoder
        embedding_dim -> Dimensione degli embedding
        ff_dim -> Dimensione del PositionFeedForward
        dropout -> Prob. che dei neuroni si attivino da soli
        heads_num -> Numero di teste per MHA
        """
        self.embedding_dim = embedding_dim
        self.ff_dim = ff_dim

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(embedding_dim)

        self.dropout = Dropout(dropout)

        self.cross_attn = MHA(heads_num, embedding_dim)
        self.attn = MHA(heads_num, embedding_dim)
        self.ff = FF(embedding_dim, ff_dim)
    
    def sub_layer1(self, X_train, mask):
        """
        Calcola l'attenzione del primi sub layers
        """
        attn = self.attn(X_train, X_train, X_train, mask)
        attn_norm = self.norm1(X_train + self.dropout(attn))

        return attn_norm
    
    def sub_layer2(self, sub_layer1, enc_output, mask):
        """
        Calcola l'attenzione del secondo sub layers
        """
        attn = self.cross_attn(sub_layer1, enc_output, enc_output, mask)
        attn_norm = self.norm2(sub_layer1 + self.dropout(attn))

        return attn_norm

    def forward(self, X_train, enc_output, inpt_mask, tgt_mask):
        sub_layer1 = self.sub_layer1(X_train, tgt_mask)

        sub_layer2 = self.sub_layer2(sub_layer1=sub_layer1, enc_output=enc_output, mask=inpt_mask)

        ff_out = self.ff(sub_layer2)

        dec_output = self.norm3(sub_layer2 + self.dropout(ff_out))

        return dec_output