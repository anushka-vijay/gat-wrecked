# Standard GAT and GATv2 class definitions

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import global_mean_pool
    from torch_geometric.utils import softmax as pyg_softmax
except ImportError:
    global_mean_pool = None
    pyg_softmax = None

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, gain=1.414):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=gain)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=gain)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, return_attn=False):  # Added return_attn, we can change it to true after training to create a visualization.
        Wh = torch.mm(h, self.W)
        N = Wh.size()[0]

        a_input = torch.cat([Wh.repeat_interleave(N, dim=0), Wh.repeat(N, 1)], dim=1)
        a_input = a_input.view(N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -1e9 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            out = F.elu(h_prime)
        else:
            out = h_prime

        if return_attn:
            return out, attention
        return out


class GAT_Model(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT_Model, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, gain=1.414)
            for _ in range(nheads)
        ])

        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False, gain=1.0)

    def forward(self, x, adj, return_attn=False):  # Added return_attn, we can change it to true after training to create a visualization.
        x = F.dropout(x, self.dropout, training=self.training)

        if return_attn:
            head_outs, attn_weights = [], []
            for att in self.attentions:
                out, attn = att(x, adj, return_attn=True)
                head_outs.append(out)
                attn_weights.append(attn)
            x = torch.cat(head_outs, dim=1)
        else:
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.out_att(x, adj)

        out = F.log_softmax(x, dim=1)

        if return_attn:
            return out, attn_weights
        return out

class GATv2Layer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2, concat=True):
        super(GATv2Layer, self).__init__()
        self.dropout = dropout
        self.concat = concat
        
        # GATv2: W interacts with the concatenated hi and hj
        self.W = nn.Linear(2 * in_features, out_features, bias=True)
        # We need a separate linear layer for the features being aggregated
        self.W_prop = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(out_features, 1)))
        
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.W_prop.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        N = h.size()[0]
        h_i = h.repeat_interleave(N, dim=0)
        h_j = h.repeat(N, 1)
        
        # Scoring: a^T * LeakyReLU(W * [hi || hj])
        e = self.leakyrelu(self.W(torch.cat([h_i, h_j], dim=1)))
        e = torch.matmul(e, self.a).squeeze(1).view(N, N)

        # Masking and Softmax
        zero_vec = -1e9 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Use the specific projection for the aggregated values
        Wh = self.W_prop(h)
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime) if self.concat else h_prime
    
class GATv2_Model(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GATv2_Model, self).__init__()
        self.dropout = dropout
        # Multi-head attention implementation [cite: 140, 142, 143]
        self.attentions = nn.ModuleList([
            GATv2Layer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)
        ])
        # Output layer [cite: 144, 145]
        self.out_att = GATv2Layer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)


class GATLayerQM9(nn.Module):
    """
    Sparse GAT layer for molecular graphs (QM9-style), with optional edge attributes.
    Does not use torch_geometric.nn.GATConv.
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True, edge_dim=None):
        super().__init__()
        if pyg_softmax is None:
            raise ImportError("torch_geometric is required for GATLayerQM9.")

        self.dropout = dropout
        self.concat = concat
        self.lin = nn.Linear(in_features, out_features, bias=False)
        self.att = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.edge_att = nn.Linear(edge_dim, 1, bias=False) if edge_dim is not None else None

        nn.init.xavier_uniform_(self.lin.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att.data, gain=1.414)
        if self.edge_att is not None:
            nn.init.xavier_uniform_(self.edge_att.weight, gain=1.414)

    def forward(self, x, edge_index, edge_attr=None):
        src, dst = edge_index[0], edge_index[1]
        wh = self.lin(x)
        wh_src = wh[src]
        wh_dst = wh[dst]

        e = self.leakyrelu(torch.matmul(torch.cat([wh_dst, wh_src], dim=1), self.att).squeeze(-1))
        if self.edge_att is not None and edge_attr is not None:
            e = e + self.edge_att(edge_attr).squeeze(-1)

        alpha = pyg_softmax(e, dst)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros_like(wh)
        out.index_add_(0, dst, wh_src * alpha.unsqueeze(-1))
        return F.elu(out) if self.concat else out


class QM9_GAT_Model(nn.Module):
    """
    Minimal graph-level regressor for QM9 using sparse custom GAT + optional edge_attr.
    """
    def __init__(self, nfeat, nhid, nclass=1, dropout=0.1, alpha=0.2, nheads=4, edge_dim=None):
        super().__init__()
        self.dropout = dropout
        self.conv1_heads = nn.ModuleList([
            GATLayerQM9(
                in_features=nfeat,
                out_features=nhid,
                dropout=dropout,
                alpha=alpha,
                concat=True,
                edge_dim=edge_dim,
            )
            for _ in range(nheads)
        ])
        self.conv2 = GATLayerQM9(
            in_features=nhid * nheads,
            out_features=nhid,
            dropout=dropout,
            alpha=alpha,
            concat=False,
            edge_dim=edge_dim,
        )
        self.out = nn.Linear(nhid, nclass)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if global_mean_pool is None:
            raise ImportError("torch_geometric is required for QM9_GAT_Model.")
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([att(x, edge_index, edge_attr=edge_attr) for att in self.conv1_heads], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)
        return self.out(x)


class GATv2LayerQM9(nn.Module):
    """
    Sparse GATv2 layer for molecular graphs (QM9-style), with optional edge attributes.

    Expected inputs:
      - x: node features [num_nodes, in_features]
      - edge_index: COO edges [2, num_edges]
      - edge_attr: optional edge features [num_edges, edge_dim]
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2,
                 concat=True, edge_dim=None):
        super().__init__()
        if pyg_softmax is None:
            raise ImportError("torch_geometric is required for GATv2LayerQM9.")

        self.dropout = dropout
        self.concat = concat
        # GATv2 scoring: a^T LeakyReLU(W [x_i || x_j])
        self.score_proj = nn.Linear(2 * in_features, out_features, bias=True)
        # Value projection for neighbor aggregation
        self.value_proj = nn.Linear(in_features, out_features, bias=False)
        self.att = nn.Parameter(torch.empty(size=(out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.edge_att = nn.Linear(edge_dim, 1, bias=False) if edge_dim is not None else None

        nn.init.xavier_uniform_(self.score_proj.weight, gain=1.414)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att.data, gain=1.414)
        if self.score_proj.bias is not None:
            nn.init.zeros_(self.score_proj.bias)
        if self.edge_att is not None:
            nn.init.xavier_uniform_(self.edge_att.weight, gain=1.414)

    def forward(self, x, edge_index, edge_attr=None):
        src, dst = edge_index[0], edge_index[1]
        x_src = x[src]
        x_dst = x[dst]

        score_in = torch.cat([x_dst, x_src], dim=1)
        e = self.leakyrelu(self.score_proj(score_in))
        e = torch.matmul(e, self.att).squeeze(-1)
        if self.edge_att is not None and edge_attr is not None:
            e = e + self.edge_att(edge_attr).squeeze(-1)

        alpha = pyg_softmax(e, dst)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        values = self.value_proj(x)[src]
        out = torch.zeros(x.size(0), values.size(1), device=x.device, dtype=values.dtype)
        out.index_add_(0, dst, values * alpha.unsqueeze(-1))
        return F.elu(out) if self.concat else out


class QM9_GATv2_Model(nn.Module):
    """
    Minimal graph-level regressor for QM9 using sparse GATv2 + optional edge_attr.
    """
    def __init__(self, nfeat, nhid, nclass=1, dropout=0.1, alpha=0.2, nheads=4, edge_dim=None):
        super().__init__()
        self.dropout = dropout
        self.conv1_heads = nn.ModuleList([
            GATv2LayerQM9(
                in_features=nfeat,
                out_features=nhid,
                dropout=dropout,
                alpha=alpha,
                concat=True,
                edge_dim=edge_dim,
            )
            for _ in range(nheads)
        ])
        self.conv2 = GATv2LayerQM9(
            in_features=nhid * nheads,
            out_features=nhid,
            dropout=dropout,
            alpha=alpha,
            concat=False,
            edge_dim=edge_dim,
        )
        self.out = nn.Linear(nhid, nclass)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if global_mean_pool is None:
            raise ImportError("torch_geometric is required for QM9_GATv2_Model.")
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([att(x, edge_index, edge_attr=edge_attr) for att in self.conv1_heads], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)
        return self.out(x)


class OGB_GAT_Model(nn.Module):
    """
    Graph-level OGB classifier using custom sparse GAT layers.
    """
    def __init__(self, nfeat, nhid, nclass, num_layers=3, nheads=4, dropout=0.2, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.num_layers = num_layers

        self.input_heads = nn.ModuleList([
            GATLayerQM9(
                in_features=nfeat,
                out_features=nhid,
                dropout=dropout,
                alpha=alpha,
                concat=True,
                edge_dim=None,
            )
            for _ in range(nheads)
        ])

        self.hidden_heads = nn.ModuleList()
        for _ in range(max(num_layers - 2, 0)):
            self.hidden_heads.append(nn.ModuleList([
                GATLayerQM9(
                    in_features=nhid * nheads,
                    out_features=nhid,
                    dropout=dropout,
                    alpha=alpha,
                    concat=True,
                    edge_dim=None,
                )
                for _ in range(nheads)
            ]))

        self.out_conv = GATLayerQM9(
            in_features=nhid * nheads,
            out_features=nhid,
            dropout=dropout,
            alpha=alpha,
            concat=False,
            edge_dim=None,
        )
        self.out = nn.Linear(nhid, nclass)

    def forward(self, x, edge_index, batch):
        if global_mean_pool is None:
            raise ImportError("torch_geometric is required for OGB_GAT_Model.")
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = x.float()

        x = torch.cat([head(x, edge_index) for head in self.input_heads], dim=1)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer_heads in self.hidden_heads:
            x = torch.cat([head(x, edge_index) for head in layer_heads], dim=1)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.out_conv(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.out(x)


class OGB_GATv2_Model(nn.Module):
    """
    Graph-level OGB classifier using custom sparse GATv2 layers.
    """
    def __init__(self, nfeat, nhid, nclass, num_layers=3, nheads=4, dropout=0.2, alpha=0.2,
                 use_residual=False, use_layer_norm=False):
        super().__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        self.input_heads = nn.ModuleList([
            GATv2LayerQM9(
                in_features=nfeat,
                out_features=nhid,
                dropout=dropout,
                alpha=alpha,
                concat=True,
                edge_dim=None,
            )
            for _ in range(nheads)
        ])
        self.input_norm = nn.LayerNorm(nhid * nheads) if use_layer_norm else nn.Identity()

        self.hidden_heads = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        for _ in range(max(num_layers - 2, 0)):
            self.hidden_heads.append(nn.ModuleList([
                GATv2LayerQM9(
                    in_features=nhid * nheads,
                    out_features=nhid,
                    dropout=dropout,
                    alpha=alpha,
                    concat=True,
                    edge_dim=None,
                )
                for _ in range(nheads)
            ]))
            self.hidden_norms.append(nn.LayerNorm(nhid * nheads) if use_layer_norm else nn.Identity())

        self.out_conv = GATv2LayerQM9(
            in_features=nhid * nheads,
            out_features=nhid,
            dropout=dropout,
            alpha=alpha,
            concat=False,
            edge_dim=None,
        )
        self.out_norm = nn.LayerNorm(nhid) if use_layer_norm else nn.Identity()
        self.out = nn.Linear(nhid, nclass)

    def forward(self, x, edge_index, batch):
        if global_mean_pool is None:
            raise ImportError("torch_geometric is required for OGB_GATv2_Model.")
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = x.float()

        x = torch.cat([head(x, edge_index) for head in self.input_heads], dim=1)
        x = F.elu(x)
        x = self.input_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer_heads, layer_norm in zip(self.hidden_heads, self.hidden_norms):
            x_prev = x
            x = torch.cat([head(x, edge_index) for head in layer_heads], dim=1)
            x = F.elu(x)
            if self.use_residual and x.shape == x_prev.shape:
                x = x + x_prev
            x = layer_norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.out_conv(x, edge_index)
        x = F.elu(x)
        x = self.out_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.out(x)

def build_model(model_type, nfeat, nhid, nclass,
                dropout=0.6, alpha=0.2, nheads=8, edge_dim=None):
    model_type = model_type.lower()
    if model_type == "gat":
        return GAT_Model(nfeat=nfeat, nhid=nhid, nclass=nclass,
                         dropout=dropout, alpha=alpha, nheads=nheads)
    elif model_type == "gatv2":
        return GATv2_Model(nfeat=nfeat, nhid=nhid, nclass=nclass,
                           dropout=dropout, alpha=alpha, nheads=nheads)
    elif model_type == "gat_qm9":
        return QM9_GAT_Model(nfeat=nfeat, nhid=nhid, nclass=nclass,
                             dropout=dropout, alpha=alpha, nheads=nheads, edge_dim=edge_dim)
    elif model_type == "gatv2_qm9":
        return QM9_GATv2_Model(nfeat=nfeat, nhid=nhid, nclass=nclass,
                               dropout=dropout, alpha=alpha, nheads=nheads, edge_dim=edge_dim)
    elif model_type == "gat_ogb":
        return OGB_GAT_Model(nfeat=nfeat, nhid=nhid, nclass=nclass,
                             num_layers=3, nheads=nheads, dropout=dropout, alpha=alpha)
    elif model_type == "gatv2_ogb":
        return OGB_GATv2_Model(nfeat=nfeat, nhid=nhid, nclass=nclass,
                               num_layers=3, nheads=nheads, dropout=dropout, alpha=alpha)
    raise ValueError(f"Unknown model_type '{model_type}'. Supported: 'gat', 'gatv2', 'gat_qm9', 'gatv2_qm9', 'gat_ogb', 'gatv2_ogb'")
