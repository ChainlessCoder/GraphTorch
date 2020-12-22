from torch import Tensor, tensor, empty, cat, arange, int64
from torch_sparse import SparseTensor
from typing import Optional


class edges:
    def __init__(self):
        self.U = empty(0)
        self.V = empty(0)
        self.values = None
        

class graphTorch():
    
    def __init__(self):
        self.node_identifiers = empty(0, dtype = int64)
        self.node_data = {}
        self.edge_data = {}
        self._counter = 0
        
    def get_node_IDs(self):
        return self.node_identifiers
    
    def get_relative_node_indices(self, node_ids: Tensor):
        identifiers = self.node_identifiers.unsqueeze(dim = 1)
        mask = identifiers == node_ids.unsqueeze(dim = 0)
        relative_indices = torch.arange(self.node_identifiers.shape[0]).unsqueeze(dim = 1) * mask
        return relative_indices.sum(dim = 0)    

    def get_node_data(self, node_type: str):
        assert (node_type in self.node_data), "the given key does not exist"
        return self.node_data[node_type]
    
    def get_edges(self, edge_type: str):
        assert (edge_type in self.edge_data), "the given key does not exist"
        return self.edge_data[edge_type]
                
    def add_nodes(self, nodes_data: dict):
        new_node_identifiers = arange(self._counter, self._counter + len(nodes_data[next(iter(nodes_data))]))
        self.node_identifiers = cat((self.node_identifiers, new_node_identifiers), dim = 0)
        self._counter += len(new_node_identifiers)
        for node_data_type, data in nodes_data.items():
            if node_data_type in self.node_data:
                self.node_data[node_data_type] = cat((self.node_data[node_data_type], data), dim=0)
            else:
                if self.node_identifiers.shape[0] != 0:
                    assert (data.shape[0] == self.node_identifiers.shape[0]), "number of nodes must be equal"
                self.node_data[node_data_type] = data  
    
    def add_edges(self, edge_type: str, U: Tensor, V: Tensor, directed: bool, values: Optional[Tensor] = None):
        if directed == False:
            a = cat((U,V), dim = 0)
            b = cat((V,U), dim = 0)
            U,V = a, b
        N = self.node_identifiers.shape[0]
        if edge_type not in self.edge_data:
            self.edge_data[edge_type] = edges()
            self._assign_processed_edges(edge_type = edge_type, row = U, col = V, N = N, v = values)
        else:
            if (self.edge_data[edge_type].values == None) and (self.edge_data[edge_type].U != 0):
                raise "the given edge_type is weighted"
            row = cat((self.edge_data[edge_type].U, U), dim = 0)
            col = cat((self.edge_data[edge_type].V, V), dim = 0)
            if values is not None:
                values = cat((self.edge_data[edge_type].values, values), dim = 0)
            self._assign_processed_edges(edge_type, row = U, col = V, N = N, v = values)
            
    def delete_edges(self, edge_type: str, U: Tensor, V: Tensor):
        E = cat((self.edge_data[edge_type].U.unsqueeze(0), self.edge_data[edge_type].V.unsqueeze(0)),dim=0).T
        Erem = cat((U.unsqueeze(0), V.unsqueeze(0)), dim=0).T
        mask = E.unsqueeze(1) == Erem
        mask = mask.all(-1)
        non_repeat_mask = ~mask.any(-1)
        self._change_edge_state(edge_type = edge_type, mask = non_repeat_mask)
    
    def delete_nodes(self, node_identifiers: Optional[Tensor] = None, relative_node_indices: Optional[Tensor] = None):
        assert ((node_identifiers != None) ^ (relative_node_indices != None)), "Either node_identifiers, or relative_node_indices have to be provided"
        if node_identifiers != None:
            relative_node_indices = self.get_relative_node_indices(node_identifiers)
        relative_node_indices = cat((relative_node_indices.unsqueeze(0),relative_node_indices.unsqueeze(0)), dim = 0).T
        for key in self.edge_data:
            E = cat((self.edge_data[key].U.unsqueeze(0), self.edge_data[key].V.unsqueeze(0)), dim=0).T
            mask = (E.unsqueeze(1) == relative_node_indices).any(-1).any(-1)
            self._change_edge_state(edge_type = key, mask = mask)
        nodes_with_new_relative_indices = arange(self.node_identifiers.shape[0] - relative_node_indices.shape[0], self.node_identifiers.shape[0])
        for ind, elm in enumerate(relative_node_indices[:,0]):
            self.node_identifiers[elm] = self.node_identifiers[-1]
            self.node_identifiers = self.node_identifiers[:-1]
            for k in self.node_data:
                self.node_data[k][elm] = self.node_data[k][-1]
                self.node_data[k] = self.node_data[k][:-1]
            for k in self.edge_data:
                Umask = self.edge_data[k].U == nodes_with_new_relative_indices[-(ind+1)]
                Vmask = self.edge_data[k].V == nodes_with_new_relative_indices[-(ind+1)]
                self.edge_data[k].U[Umask] = elm
                self.edge_data[k].V[Vmask] = elm

        
    def _assign_processed_edges(self,edge_type:str, row: Tensor, col: Tensor, N: int, v: Optional[Tensor]):
        adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N), value = v).coalesce()
        self.edge_data[edge_type].U = adj.storage.row()
        self.edge_data[edge_type].V = adj.storage.col()
        if v is not None:
            self.edge_data[edge_type].values = adj.storage.value() 
            
    def _change_edge_state(self, edge_type: str, mask: Tensor):
        self.edge_data[edge_type].U = self.edge_data[edge_type].U[mask]
        self.edge_data[edge_type].V = self.edge_data[edge_type].V[mask]
        if self.edge_data[edge_type].values is not None:
            self.edge_data[edge_type].values = self.edge_data[edge_type].values[mask]
