import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from typing import List, Optional, Tuple, Union
from torch_geometric.nn import SAGEConv
from torch import Tensor
from torch_geometric.typing import (
    OptPairTensor,
    OptTensor,
    Adj,
    Size
)
from torch_geometric.nn.aggr import Aggregation
from openfgl.flcore.ours.utils import dense_to_coo


class GCN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        if num_layers > 1:
            self.layers.append(GCNConv(input_dim, hid_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hid_dim, hid_dim))
            self.layers.append(GCNConv(hid_dim, output_dim))
        else:
            self.layers.append(GCNConv(input_dim, output_dim))
            self.border_encoder.append(nn.Linear(input_dim, hid_dim))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        if not isinstance(edge_index, SparseTensor) and not edge_index.is_sparse and edge_index.shape[0] != 2:
            edge_index, edge_attr = dense_to_coo(edge_index)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.layers[-1](x, edge_index, edge_attr)
        return x, logits


class BorderGCN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(BorderGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # GCN层
        self.layers = nn.ModuleList()
        self.border_encoder = nn.ModuleList()
        if num_layers > 1:
            self.layers.append(GCNConv(input_dim, hid_dim))
            self.border_encoder.append(nn.Linear(input_dim, hid_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hid_dim, hid_dim))
                self.border_encoder.append(nn.Linear(hid_dim, hid_dim))
            self.layers.append(GCNConv(hid_dim, output_dim))
        else:
            self.layers.append(GCNConv(input_dim, output_dim))
            self.border_encoder.append(nn.Linear(input_dim, hid_dim))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.border_encoder:
            layer.reset_parameters()

    def norm_weight(self, adj, weight_score):
        # 对传给真实节点的边进行归一化
        node_degree = torch.sum(adj, dim=1)  # 形状: (num_node,)

        # 防止度为 0 的节点引发除零错误，添加一个小值
        node_degree = torch.clamp(node_degree, min=1e-12)  # 形状: (num_node,)

        # 扩展节点度以匹配 weight_score 的形状
        node_degree_expanded = node_degree[:, None]  # 形状: (num_node, 1)

        # 对权重矩阵进行归一化
        normalized_TN = weight_score / node_degree_expanded  # 形状: (num_node, num_border)
        # 对传给边界节点的边进行归一化
        weight_score_T = weight_score.T
        node_degree = torch.sum(weight_score_T, dim=1)

        node_degree = torch.clamp(node_degree, min=1e-12)  # 形状: (num_node,)
        node_degree_expanded = node_degree[:, None]
        normalized_BN = weight_score_T / node_degree_expanded
        return normalized_TN,normalized_BN

    def forward(self, x, edge_index, border_node=None, weight_border=None, edge_attr=None):
        if weight_border is not None:
            weight_border_TN,weight_border_BN = self.norm_weight(edge_index, weight_border)
        border_embedding = border_node

        if not isinstance(edge_index, SparseTensor) and not edge_index.is_sparse and edge_index.shape[0] != 2:
            edge_index, edge_attr = dense_to_coo(edge_index)

        for i, layer in enumerate(self.layers[:-1]):
            #第一阶段补偿
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            #第二阶段补偿
            if border_node is not None and weight_border is not None:
                # 计算边界embedding
                border_embedding = self.border_encoder[i](border_embedding)
                # 计算来自邻域节点的传递消息
                message_fromBN = torch.mm(weight_border_TN, border_embedding)

                # 计算来自真实节点的传递消息
                message_fromTN = torch.mm(weight_border_BN, x)
                # 将消息从边界节点传递给真实节点
                x = x + message_fromBN
                border_embedding = border_embedding + message_fromTN
                border_embedding = F.dropout(border_embedding, p=self.dropout, training=self.training)

        logits = self.layers[-1](x, edge_index, edge_attr)
        self.border_embedding = border_embedding
        return x, logits


class GraphSage(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(GraphSage, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        if num_layers > 1:
            self.layers.append(SAGEConvPlus(input_dim, hid_dim))
            for _ in range(num_layers - 2):
                self.layers.append(SAGEConvPlus(hid_dim, hid_dim))
            self.layers.append(SAGEConvPlus(hid_dim, output_dim))
        else:
            self.layers.append(SAGEConvPlus(input_dim, output_dim))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        if not isinstance(edge_index, SparseTensor) and not edge_index.is_sparse:
            edge_index, edge_attr = dense_to_coo(edge_index)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.layers[-1](x, edge_index, edge_attr)
        return x, logits


class SAGEConvPlus(SAGEConv):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            aggr: str = "mean",
            normalize: bool = False,
            root_weight: bool = True,
            bias: bool = True,
            **kwargs
    ):
        super(SAGEConvPlus, self).__init__(
            in_channels, out_channels, aggr, normalize, root_weight, bias, **kwargs
        )

    def forward(
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            edge_weight: OptTensor = None,
            size: Size = None,
    ) -> Tensor:
        # 如果 x 是张量，将其转换为 (x_src, x_dst) 的形式
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # 如果需要投影，则应用线性变换
        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # 将 edge_weight 存储为类的属性，供 message 方法使用
        self._edge_weight = edge_weight

        # 调用 propagate 方法
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        # 处理根节点特征
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        # 如果需要，归一化输出
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        # 使用类属性 _edge_weight 处理边权重
        edge_weight = self._edge_weight
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
