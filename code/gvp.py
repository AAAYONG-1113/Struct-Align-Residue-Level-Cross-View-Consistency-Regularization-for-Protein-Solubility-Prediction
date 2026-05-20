import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class GVP(nn.Module):
    """
    Geometric Vector Perceptron.
    Takes in a tuple of (scalar_features, vector_features) and applies
    equivariant transformations.
    """
    def __init__(self, in_dims, out_dims, h_dim=None, activations=(F.relu, torch.sigmoid)):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.h_dim = h_dim or max(1, self.vo)
        self.activations = activations
        
        self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
        self.ws = nn.Linear(self.h_dim + self.si, self.so)
        if self.vo > 0:
            self.wv = nn.Linear(self.vi, self.vo, bias=False)
            self.wsv = nn.Linear(self.so, self.vo)

    def forward(self, x):
        s, v = x
        
        # v shape: (..., vi, 3) -> transpose to (..., 3, vi) for nn.Linear
        v_trans = v.transpose(-1, -2) 
        v_h_trans = self.wh(v_trans) # (..., 3, h_dim)
        v_h = v_h_trans.transpose(-1, -2) # (..., h_dim, 3)
        
        v_norm = torch.norm(v_h, dim=-1) # (..., h_dim)
        
        s_h = torch.cat([s, v_norm], dim=-1)
        s_out = self.ws(s_h)
        if self.activations[0]:
            s_out = self.activations[0](s_out)
            
        if self.vo == 0:
            return s_out
            
        v_out_trans = self.wv(v_trans) # (..., 3, vo)
        v_out = v_out_trans.transpose(-1, -2) # (..., vo, 3)
        
        if self.activations[1]:
            v_norm_out = torch.norm(v_out, dim=-1, keepdim=True)
            # Safe division
            v_dir = v_out / (v_norm_out + 1e-8)
            v_scale = self.wsv(s_out).unsqueeze(-1)
            v_scale = self.activations[1](v_scale)
            v_out = v_dir * v_scale
            
        return (s_out, v_out)

class GVPConv(MessagePassing):
    """
    A message passing layer using GVP.
    """
    def __init__(self, node_dims, edge_dims, out_dims):
        # explicitly set node_dim=0 to handle vector feature tensors properly
        super(GVPConv, self).__init__(aggr='mean', node_dim=0)
        self.message_func = GVP(
            in_dims=(node_dims[0] * 2 + edge_dims[0], node_dims[1] * 2 + edge_dims[1]),
            out_dims=out_dims,
            activations=(F.relu, torch.sigmoid)
        )
        self.update_func = GVP(
            in_dims=(node_dims[0] + out_dims[0], node_dims[1] + out_dims[1]),
            out_dims=out_dims,
            activations=(F.relu, torch.sigmoid)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: (s, v)
        # edge_attr: (s, v)
        s, v = x
        return self.propagate(edge_index, s=s, v=v, edge_attr=edge_attr)

    def message(self, s_i, s_j, v_i, v_j, edge_attr):
        s_e, v_e = edge_attr
        
        s_msg = torch.cat([s_i, s_j, s_e], dim=-1)
        v_msg = torch.cat([v_i, v_j, v_e], dim=-2)
        
        msg_s, msg_v = self.message_func((s_msg, v_msg))
        
        # Flatten vector messages to pack them into a single tensor for PyG aggregation
        msg_v_flat = msg_v.reshape(msg_v.size(0), -1)
        return torch.cat([msg_s, msg_v_flat], dim=-1)

    def update(self, aggr_out, s, v):
        so, vo = self.message_func.so, self.message_func.vo
        
        # Unpack back to scalar and vector
        s_aggr = aggr_out[:, :so]
        v_aggr = aggr_out[:, so:].reshape(aggr_out.size(0), vo, 3)
        
        s_update = torch.cat([s, s_aggr], dim=-1)
        v_update = torch.cat([v, v_aggr], dim=-2)
        
        return self.update_func((s_update, v_update))
