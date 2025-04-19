import torch
import math

from torch.nn import Module
from torch.nn import Linear

class MultiHeadAttention(Module):
    def __init__(self, heads, embedding_dim):
        super().__init__()
        """
        Inizialitta i parametri della MultiHeadAttention

        heads -> Numero di teste
        embedding_dim -> dimensione di ogni embedding
        heads_dim -> Dimensioni di ogni testa 

        W_k -> trasformazioe lineare della matrice key
        W_q -> trnasformazione lineare della matrice query
        W_v -> trasformazione lineare della matrice value
        """
        self.heads = heads
        self.embedding_dim = embedding_dim

        self.heads_dim = embedding_dim // heads

        self.W_k = Linear(embedding_dim, embedding_dim)
        self.W_q = Linear(embedding_dim, embedding_dim)
        self.W_v = Linear(embedding_dim, embedding_dim)

        self.W_o = Linear(embedding_dim, embedding_dim)


    def dot_product(self, Q,K,V, mask = None):
        """
        Calcola l'attenzione dei token sugli altri token e poi ne calcola il peso ponderato,
        restituisce la matrice di attenzione
        """
        att_scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.heads_dim)

        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, float('-inf'))

        attn_proba = torch.softmax(att_scores, dim=-1)

        attn_output = torch.matmul(attn_proba, V)

        return attn_output
    
    def split_head(self, X_train):
        """
        Riorganizza il tensore per separare la dimensione dell'embedding in più teste
        Permette a ciascuna testa di lavorare su una sottospazio diverso della rappresentazione
        """
        batch_size, token_len, _ = X_train.size()

        X_train = X_train.view(batch_size, token_len, self.heads, self.heads_dim).transpose(1,2)

        return X_train
    
    def combine_heads(self, X_train):
        """
        Riconverte la matrice di attenzione nelle dimensioni originali
        """
        batch_size, _, token_len, embeddin_dim = X_train.size()

        X_train = X_train.transpose(1,2).contiguous().view(batch_size, token_len, embeddin_dim)

        return X_train
    
    def forward(self, Q, K, V, mask = None):
        Q = self.split_head(self.W_q(Q))
        K = self.split_head(self.W_k(K))
        V = self.split_head(self.W_v(V))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

        attn_out = self.dot_product(Q,K,V, mask)

        return self.W_o(self.combine_heads(attn_out))



