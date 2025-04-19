from torch.nn import Linear
from torch import relu
from torch.nn import Module

class PositionFeedForward(Module):
    def __init__(self, embedding_dim, ff_dim):
        super().__init__()
        """
        Inizializza il PositionFeedForward

        embedding_dim -> Dimensione degli embedding
        ff_dim -> Dimensione del FeedForward
        """

        self.lin1 = Linear(embedding_dim, ff_dim)
        self.lin2 = Linear(ff_dim, embedding_dim)

    def forward(self):
        return self.lin2(relu(self.lin1))