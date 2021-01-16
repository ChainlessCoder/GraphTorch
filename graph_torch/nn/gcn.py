from torch_sparse import SparseTensor, fill_diag, sum, mul, matmul
from torch.nn import Module, Parameter
from torch import Tensor
from typing import Optional
from graph_torch.nn.init import glorot

def gcn_norm(adj: SparseTensor, add_self_loops: bool = True):
    if adj.has_value() == False:
        adj = adj.fill_value(1.)
    if add_self_loops:
        A_tilde = fill_diag(adj, 1)
    D_tilde = sum(A_tilde, dim = 1)
    D_tilde_inv_sqrt = D_tilde.pow_(-0.5)
    D_tilde_inv_sqrt.masked_fill_(D_tilde_inv_sqrt == float('inf'), 0)
    A_hat = mul(mul(A_tilde, D_tilde_inv_sqrt.unsqueeze(0)), D_tilde_inv_sqrt.unsqueeze(0))
    return A_hat

class GCN_layer(Module):
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 add_self_loops: bool = True,
                 weight_init = glorot
                ):
        super(GCN_layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.weights = Parameter(data = Tensor(in_channels, out_channels))
        weight_init(self.weights)
    
    def forward(self, X: Tensor, edges: SparseTensor):
        
        A_hat = gcn_norm(adj = edges, 
                         add_self_loops = self.add_self_loops
                        )
        out = matmul(A_hat, X) @ self.weights
        return out
