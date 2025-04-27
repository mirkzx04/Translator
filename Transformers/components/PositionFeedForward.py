from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Module

class PositionFeedForward(Module):
    def __init__(self, embedding_dim, ff_dim):
        super(PositionFeedForward, self).__init__()
        """
        Inizializza il PositionFeedForward

        embedding_dim -> Dimensione degli embedding
        ff_dim -> Dimensione del FeedForward
        """

        self.lin1 = Linear(embedding_dim, ff_dim)
        self.lin2 = Linear(ff_dim, embedding_dim)
        self.relu = ReLU()

    def forward(self, X):
        return self.lin2(self.relu(self.lin1(X)))