from torch.nn import Module, Parameter
from torch.nn.functional import embedding
from torch import log, sigmoid

eps = 1e-15

class node2vec_layer(Module):
    def __init__(self, graph, node_type:str):
        super(node2vec_layer, self).__init__()
        self.w = graph.node_data[node_type]
    
    def forward(self, batch):
        """Returns the embeddings for the nodes in :obj:`batch`."""
        return embedding(batch, self.w, sparse = True)
    
    def loss(self, pos_rw, neg_rw):
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = embedding(start, self.w, sparse = True).view(pos_rw.size(0), 1,
                                                          self.w.shape[1])
        h_rest = embedding(rest.view(-1), self.w, sparse = True).view(pos_rw.size(0), -1,
                                                                 self.w.shape[1])

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -log(sigmoid(out) + eps).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = embedding(start, self.w, sparse = True).view(neg_rw.size(0), 1,
                                                          self.w.shape[1])
        h_rest = embedding(rest.view(-1), self.w, sparse = True).view(neg_rw.size(0), -1,
                                                                 self.w.shape[1])

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -log(1 - sigmoid(out) + eps).mean()

        return pos_loss + neg_loss
    