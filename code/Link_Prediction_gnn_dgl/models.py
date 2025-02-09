# models classes : 

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv, GATConv

class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, num_rels):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, h_dim)
        self.conv1 = RelGraphConv(
            h_dim, h_dim, num_rels, regularizer="bdd", num_bases=100, self_loop=True
        )
        self.conv2 = RelGraphConv(
            h_dim, h_dim, num_rels, regularizer="bdd", num_bases=100, self_loop=True
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, nids):
        x = self.emb(nids)
        h = F.relu(self.conv1(g, x, g.edata[dgl.ETYPE], g.edata["norm"]))
        h = self.dropout(h)
        h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata["norm"])
        return self.dropout(h)

class GAT(nn.Module):
    def __init__(self, num_nodes, h_dim, num_rels, num_heads=4):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, h_dim)
        self.gat_conv1 = GATConv(h_dim, h_dim // num_heads, num_heads, allow_zero_in_degree=True)
        self.gat_conv2 = GATConv(h_dim, h_dim // num_heads, num_heads, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, nids):
        x = self.emb(nids)
        h = self.gat_conv1(g, x)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.gat_conv2(g, h)
        return self.dropout(h)

class LinkPredict(nn.Module):
    def __init__(self, model, num_nodes, num_rels, h_dim=500, reg_param=0.01):
        super().__init__()
        self.model = model(num_nodes, h_dim, num_rels * 2)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(
            self.w_relation, gain=nn.init.calculate_gain("relu")
        )

    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, nids):
        return self.model(g, nids)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

