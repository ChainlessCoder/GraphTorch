from torch import tensor, Tensor, cat, randint, arange
from torch_cluster import random_walk
from torch.utils.data import DataLoader

class UnweightedNode2vecSampler():
    
    def __init__(self, graph, edge_type, walk_length, context_size, walks_per_node=1, 
                 p=1, q=1, num_negative_samples=1, sparse=False):
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.graph = graph
        self.edge_type = edge_type
        
    
    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)
        row = self.graph.edge_data[self.edge_type].storage.row()
        col = self.graph.edge_data[self.edge_type].storage.col()
        rw = random_walk(row, col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return cat(walks, dim=0)


    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = randint(self.graph.edge_data[self.edge_type].sparse_size(0),
                           (batch.size(0), self.walk_length))
        rw = cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return cat(walks, dim=0)
    
    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)
    
    def generator(self, batch = None, **kwargs):
        if batch == None:
            return DataLoader(arange(self.graph.edge_data[self.edge_type].sparse_size(0)),
                              collate_fn=self.sample, **kwargs)
        elif not isinstance(batch, Tensor):
            batch = tensor(batch)
        return DataLoader(batch, collate_fn=self.sample, **kwargs)
    
    