import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch.nn.modules.module import Module
from dgl.nn import EdgeWeightNorm, GraphConv


class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # 使用一个简单的线性层来计算每个图嵌入的注意力得分
        self.attention_heads = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_heads)  # 每个头一个线性变换
        ])

    def forward(self, embeddings):
        # embeddings 的维度是 [batch_size, 3, embed_dim]
        batch_size = embeddings.size(0)  # batch size
        attention_scores = []

        # 计算每个头的注意力得分
        for i in range(self.num_heads):
            scores = self.attention_heads[i](embeddings)  # [batch_size, 3, 1]
            attention_scores.append(scores)

        # 将每个头的得分合并为一个 [batch_size, 3, num_heads]
        attention_scores = torch.cat(attention_scores, dim=-1)  # [batch_size, 3, num_heads]

        # 对每个头的得分进行 softmax，得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)  # 维度 1 是视图维度

        # 扩展 attention_weights 的维度，使其与 embeddings 维度匹配
        attention_weights = attention_weights.unsqueeze(2)  # [batch_size, 3, 1, num_heads]

        # 将 embeddings 的维度扩展为 [batch_size, 3, embed_dim, 1]
        embeddings_expanded = embeddings.unsqueeze(3)  # [batch_size, 3, embed_dim, 1]

        # 对不同的图嵌入进行加权求和
        weighted_sum = torch.sum(attention_weights * embeddings_expanded, dim=2)  # [batch_size, 3, num_heads]

        # 最终将加权结果按头数求和并压缩维度，得到最终的融合嵌入
        node_emb_final = weighted_sum.sum(dim=-1)  # [batch_size, 3]

        return node_emb_final, attention_weights


#
# class GCNConv_dgl(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(GCNConv_dgl, self).__init__()
#         self.linear = nn.Linear(input_size, output_size)
#
#     def forward(self, x, g):
#         with g.local_scope():
#             g.ndata['h'] = self.linear(x)
#             g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
#             return g.ndata['h']


class my_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim):
        """
        初始化 MLP 模型。

        Args:
            input_dim (int): 输入特征的维度。
            output_dim (int): 输出特征的维度。
            num_layers (int): 隐藏层的数量（至少 1）。
            hidden_dim (int): 每个隐藏层的神经元数量。
        """
        super(my_MLP, self).__init__()
        assert num_layers >= 1, "Number of layers must be at least 1."

        # 初始化网络结构
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))  # 输入层到第一层隐藏层
        layers.append(nn.ReLU())  # 激活函数

        # 添加隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            # layers.append(nn.ReLU())

        # 添加输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        # layers.append(torch.sigmoid())
        # 添加 Sigmoid 激活函数
        # if output_dim == 1:  # 仅在输出维度为 1 时添加 Sigmoid
        #     layers.append(nn.Sigmoid())

        # 将所有层合并为一个模块列表
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_dim)。
        """
        return self.model(x)


class my_GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_class=1):
        super(my_GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, 2 * h_feats)
        self.conv2 = GraphConv(2 * h_feats, h_feats)

        self.fc1 = nn.Linear(h_feats, h_feats, bias=False)
        self.fc2 = nn.Linear(h_feats, h_feats, bias=False)

        # self.param_init()
        # self.fc1 = nn.Linear(h_feats, 1, bias=False)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        feat1 = self.fc1(h)
        feat2 = self.fc2(h)
        return h, feat1, feat2

    def get_final_predict(self, h):
        return torch.sigmoid(self.fc1(h))

    def param_init(self):
        nn.init.xavier_normal_(self.conv1.weight, gain=1.414)
        nn.init.xavier_normal_(self.conv2.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits


def get_cos(feature):
    feature = feature / (torch.norm(feature, dim=-1, keepdim=True))
    sim_matrix = torch.mm(feature, feature.T)
    return sim_matrix


def min_max_norm(feature):
    feature = (feature - feature.min()) / (feature.max() - feature.min())
    return feature


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn1 = GCN(n_in, 2 * n_h, activation)
        self.gcn2 = GCN(2 * n_h, n_h, activation)

        self.act = nn.PReLU()
        self.fc1 = nn.Linear(n_h, 2 * n_h, bias=False)
        self.fc2 = nn.Linear(n_h, 2 * n_h, bias=False)

        self.ReLU = nn.ReLU()

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

    def forward(self, seq, adj, sparse=False):

        feat = self.gcn1(seq, adj)
        feat = self.gcn2(feat, adj)
        feat1 = self.fc1(feat)
        feat2 = self.fc2(feat)

        return feat, feat1, feat2


# Graphsage layer
class SageConv(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()

        self.proj = nn.Linear(in_features * 2, out_features, bias=bias)

        self.reset_parameters()

        # print("note: for dense graph in graphsage, require it normalized.")

    def reset_parameters(self):

        nn.init.normal_(self.proj.weight)

        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, features, adj):
        """
        Args:
            adj: can be sparse or dense matrix.
        """

        # fuse info from neighbors. to be added:
        if adj.layout != torch.sparse_coo:
            if len(adj.shape) == 3:
                neigh_feature = torch.bmm(adj, features) / (
                        adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
            else:
                neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        else:
            # print("spmm not implemented for batch training. Note!")

            neigh_feature = torch.spmm(adj, features) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)

        # perform conv
        data = torch.cat([features, neigh_feature], dim=-1)
        combined = self.proj(data)

        return combined


class Sage_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En, self).__init__()

        self.sage1 = SageConv(nfeat, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Sage_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En2, self).__init__()

        self.sage1 = SageConv(nfeat, nhid)
        self.sage2 = SageConv(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class Sage_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Sage_Classifier, self).__init__()

        self.sage1 = SageConv(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


def neighList_to_edgeList(adj):
    edge_list = []
    for i in range(adj.shape[0]):
        for j in torch.argwhere(adj[i, :] > 0):
            edge_list.append((int(i), int(j)))
    return edge_list


from torch_geometric.nn import GINConv


class GIN(torch.nn.Module):
    def __init__(self, ft_size, hidden_dim, num_layers):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(ft_size, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim)))
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, hidden_dim))))

    def forward(self, feat, adj):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        adj = torch.squeeze(adj)
        feat = torch.squeeze(feat)
        edge_index = neighList_to_edgeList(adj)
        edge_index = torch.tensor(np.array(edge_index)).T.cuda()
        x = F.relu(self.conv1(feat, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        return torch.unsqueeze(x, 0)
