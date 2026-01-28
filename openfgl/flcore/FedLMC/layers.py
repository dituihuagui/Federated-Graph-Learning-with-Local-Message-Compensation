import torch
import torch.nn as nn
import torch.nn.functional as F

from openfgl.flcore.ours.metrics import EpsilonNN, KNN, Cosine, WeightedCosine


class NodeGen(nn.Module):
    def __init__(self, input_dim, hid_dim, feat_shape, num_layers_encoder=2, num_layers_decoder=2,
                 dropout=0.5):
        super(NodeGen, self).__init__()
        # input_dim：即embed_dim：随机生成的节点嵌入维度
        # hid_dim：即hid_dim_node，节点特征生成器隐藏层维度
        # feat_shape：节点原始特征维度
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.feat_shape = feat_shape
        self.encoder = nn.ModuleList()
        self.test_emb = nn.Parameter(torch.empty(3, 3))
        # 编码器
        self.encoder.append(nn.Linear(input_dim, hid_dim))
        for _ in range(num_layers_encoder - 1):
            self.encoder.append(nn.Linear(hid_dim, hid_dim))
        # 解码器
        self.decoder = nn.ModuleList()
        for _ in range(num_layers_decoder - 1):
            self.decoder.append(nn.Linear(hid_dim, hid_dim))
        self.decoder.append(nn.Linear(hid_dim, feat_shape))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.encoder:
            layer.reset_parameters()
        for layer in self.decoder:
            layer.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = x
        for i, layer in enumerate(self.decoder[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.decoder[-1](x)
        return logits, embedding


class EdgeGen(nn.Module):
    def __init__(self, input_dim, topk=None, epsilon=None, num_pers=16, weightedcosine=True):
        super(EdgeGen, self).__init__()
        self.topk = topk
        self.epsilon = epsilon
        self.weightedcosine = weightedcosine
        if self.weightedcosine:
            self.metric = WeightedCosine(input_dim, num_pers)
            self.reset_parameters()
        else:
            self.metric = Cosine()
        self.enn = EpsilonNN(epsilon)
        self.knn = KNN(topk)

    def reset_parameters(self):
        if self.weightedcosine:
            self.metric.reset_parameters()

    def forward(self, node_features, anchor=None):
        # return a new adj according to the representation gived

        # expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        # if len(context.shape) == 3:
        #     expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

        adj = self.metric(node_features, y=anchor)

        if self.epsilon is not None:
            adj = self.enn(adj)

        if self.topk is not None:
            adj = self.knn(adj=adj)
        return adj
