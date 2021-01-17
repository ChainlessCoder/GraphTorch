from torch import Tensor, tensor, empty, cat, arange, logical_and, logical_or, int64
from torch.nn import Parameter
from torch_sparse import SparseTensor
from typing import Optional

class dynamicGraph():
    
    def __init__(self):
        self.node_identifiers = empty(0, dtype = int64)
        self.node_data = {}
        self.edge_data = {}
        self._counter = 0
        
    def __len__(self):
        return self.node_identifiers.shape[0]
        
    def get_nodes_num(self):
        return self.node_identifiers.shape[0]
    
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
        adj = self.edge_data[edge_type]
        U = adj.storage.row()
        V = adj.storage.col()
        values = adj.storage.value()
        return U, V, values
        
                
    def add_nodes(self, nodes_data: dict):
        new_node_identifiers = arange(self._counter, self._counter + len(nodes_data[next(iter(nodes_data))]))
        self.node_identifiers = cat((self.node_identifiers, new_node_identifiers), dim = 0)
        self._counter += len(new_node_identifiers)
        for node_data_type, data in nodes_data.items():
            if node_data_type in self.node_data:
                self.node_data[node_data_type] = Parameter(data = cat((self.node_data[node_data_type], data), dim=0),
                                                           requires_grad = True)
            else:
                if self.node_identifiers.shape[0] != 0:
                    assert (data.shape[0] == self.node_identifiers.shape[0]), "number of nodes must be equal"
                self.node_data[node_data_type] = Parameter(data = data, requires_grad = True)  
    
    def add_edges(self, edge_type: str, U: Tensor, V: Tensor, directed: bool, edge_weights: Optional[Tensor] = None):
        if directed == False:
            a = cat((U,V), dim = 0)
            b = cat((V,U), dim = 0)
            U,V = a, b
        N = self.get_nodes_num()
        if edge_type not in self.edge_data:
            self.edge_data[edge_type] = SparseTensor(row=U, col=V, sparse_sizes=(N, N), value = edge_weights).coalesce()
        else:
            r,c,v = self.get_edges(edge_type)
            if (self.edge_data[edge_type].storage.value() != None) ^ (edge_weights != None):
                raise "edge_weights have to be consistent"
            new_row = cat((r, U), dim = 0)
            new_col = cat((c, V), dim = 0)
            if edge_weights is not None:
                v = cat((v, edge_weights), dim = 0)
            self.edge_data[edge_type] = SparseTensor(row=new_row, 
                                                     col=new_col, 
                                                     sparse_sizes=(N, N), 
                                                     value = v).coalesce()
            
    def delete_edges(self, edge_type: str, U: Tensor, V: Tensor):
        assert (edge_type in self.edge_data), "the given key does not exist"
        N = self.get_nodes_num()
        r,c,v = self.get_edges(edge_type)
        E = cat((r.unsqueeze(0), c.unsqueeze(0)),dim=0).T
        Erem = cat((U.unsqueeze(0), V.unsqueeze(0)), dim=0).T
        mask = E.unsqueeze(1) == Erem
        mask = mask.all(-1)
        non_repeat_mask = ~mask.any(-1)
        if v is not None:
            v = v[non_repeat_mask]
        self.edge_data[edge_type] = SparseTensor(row=r[non_repeat_mask], 
                                                 col=c[non_repeat_mask], 
                                                 sparse_sizes=(N, N), 
                                                 value = v).coalesce()
    
    def delete_nodes(self, node_identifiers: Optional[Tensor] = None, relative_node_indices: Optional[Tensor] = None):
        assert ((node_identifiers != None) ^ (relative_node_indices != None)), "Either node_identifiers, or relative_node_indices have to be provided"
        if node_identifiers != None:
            relative_node_indices = self.get_relative_node_indices(node_identifiers)
        relative_node_indices = cat((relative_node_indices.unsqueeze(0),relative_node_indices.unsqueeze(0)), dim = 0).T
        N = self.get_nodes_num()
        for key in self.edge_data:
            r,c,v = self.get_edges(key)
            E = cat((r.unsqueeze(0), c.unsqueeze(0)), dim=0).T
            mask = (E.unsqueeze(1) == relative_node_indices).any(-1).any(-1)
            if v is not None:
                v = v[mask]
            self.edge_data[key] = SparseTensor(row=r[mask], col=c[mask], sparse_sizes=(N, N), value = v).coalesce()
        nodes_with_new_relative_indices = arange(self.node_identifiers.shape[0] - relative_node_indices.shape[0], N)
        for ind, elm in enumerate(relative_node_indices[:,0]):
            self.node_identifiers[elm] = self.node_identifiers[-1]
            self.node_identifiers = self.node_identifiers[:-1]
            for k in self.node_data:
                self.node_data[k][elm] = self.node_data[k][-1]
                self.node_data[k] = self.node_data[k][:-1]
            for k in self.edge_data:
                r,c,v = self.get_edges(k)
                Umask = r == nodes_with_new_relative_indices[-(ind+1)]
                Vmask = c == nodes_with_new_relative_indices[-(ind+1)]
                r[Umask] = elm
                c[Vmask] = elm
                self.edge_data[k] = SparseTensor(row=r, 
                                                 col=c, 
                                                 sparse_sizes=(N, N), 
                                                 value = v).coalesce()
    
    
    def set_value(self, edge_type: str, value: Tensor):
        self.edge_data[edge_type] = self.edge_data[edge_type].set_value(value, layout = 'coo')
        return
    
    def update_edge_weight(self, edge_type: str, u: int, v: int, value, operation, **kargs):
        internal_U, internal_V, internal_values = self.get_edges(edge_type)
        mask = logical_and(internal_U == u,
                           internal_V == v
                          )
        self.edge_data[edge_type].storage._value[mask] = operation(values = internal_values[mask],
                                                                   value = value,
                                                                   *kargs
                                                                  )
    
    def update_multiple_edge_weights(self, edge_type: str, U: Tensor, V: Tensor, values, operation, **kargs):
        internal_U, internal_V, internal_values = self.get_edges(edge_type)
        internal_edges = cat((internal_U.unsqueeze(1),internal_V.unsqueeze(1)),dim=1)
        edges2change = cat((U.unsqueeze(1),V.unsqueeze(1)),dim=1)
        indices2change = arange(len(internal_U)).repeat(len(U),1)[((internal_edges == edges2change.unsqueeze(1)).sum(2) == 2)]
        self.edge_data[edge_type].storage._value[indices2change] = operation(values = internal_values[indices2change],
                                                                             value = values,
                                                                             *kargs
                                                                            )
        
    def get_outgoing_nodes(self, edge_type: str, node_index):
        U, V, _ = self.get_edges(edge_type)
        if isinstance(node_index, int):
            return V[U == node_index]
        if isinstance(node_index, list):
            node_index = tensor(node_index)
        rU = U.repeat(len(node_index), 1)
        mask = rU == node_index.unsqueeze(1)
        return V.repeat(len(node_index), 1)[mask]
        
    def get_incoming_nodes(self, edge_type: str, node_index):
        U, V, _ = self.get_edges(edge_type)
        if isinstance(node_index, int):
            return U[V == node_index]
        if isinstance(node_index, list):
            node_index = tensor(node_index)
        rV = V.repeat(len(node_index), 1)
        mask = rV == node_index.unsqueeze(1)
        return U.repeat(len(node_index), 1)[mask]
    
    def get_node_edges(self, edge_type: str, node_index):
        U, V, values = self.get_edges(edge_type)
        if isinstance(node_index, int):
            edges = cat((U.unsqueeze(0),V.unsqueeze(0)), dim=0)
            mask = logical_or(U == node_index,
                              V == node_index
                             )
            return U[mask], V[mask]
        if isinstance(node_index, list):
            node_index = tensor(node_index)
        umask = (U == node_index.unsqueeze(1)).sum(0)
        vmask = (V == node_index.unsqueeze(1)).sum(0)
        mask = logical_or((U == node_index.unsqueeze(1)).sum(0), 
                          (V == node_index.unsqueeze(1)).sum(0)
                         )
        if values is not None:
            return U[mask], V[mask], values[mask]
        return U[mask], V[mask], None

    
class edge_updates:
    
    def add2edge_value(self, values, value):
        return values + value
    
    def set_edge_value(self, values, value):
        return value