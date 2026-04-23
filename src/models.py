# Standard GAT and GATv2 class definitions

import torch
import torch.nn as nn
import torch.nn.functional as F


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

# We can add gatv2 here after the implementation so its seperated out.
def build_model(model_type, nfeat, nhid, nclass,
                dropout=0.6, alpha=0.2, nheads=8):
    if model_type.lower() == "gat":
        return GAT_Model(nfeat=nfeat, nhid=nhid, nclass=nclass,
                         dropout=dropout, alpha=alpha, nheads=nheads)
    raise ValueError(f"Unknown model_type '{model_type}'. Only 'gat' is supported.")