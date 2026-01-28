import torch
import torch.nn as nn
from openfgl.flcore.FedLMC.layers import EdgeGen, NodeGen
from openfgl.flcore.FedLMC.classification import BorderGCN
from openfgl.flcore.ours.utils import symmetry

class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim_cla, num_layers_cla, num_nodes, embed_dim,
                 hid_dim_node, num_layers_node_encoder=2, num_layers_node_decoder=2,
                 dropout_node=0.5, dropout_cla=0.5, alpha_adj=0.5, adj_fuse=True, topk=10, epsilon=None,
                 num_pers_cos=16, weightedcosine=True, num_border=0,node_border_mask=None):
        # input_dim：节点原始特征维度
        # output_dim：输出维度，一般为类别数

        # hid_dim_cla：分类器隐藏层维度
        # num_layers_cla：分类器层数
        # dropout_cla：分类器的dropout率

        # num_nodes：客户端原始节点数
        # embed_dim：随机生成的节点嵌入维度
        # hid_dim_node：节点特征生成器隐藏层维度

        # num_layers_node：节点特征生成器的网络层数
        # dropout_node：节点特征生成器的dropout率

        super(MyModel, self).__init__()

        # 邻域节点生成器
        self.border_gen = BorderNodeGen(num_nodes=num_border, embed_dim=embed_dim, hid_dim_node=hid_dim_node,
                                        feat_shape=input_dim,
                                        num_layers_node_encoder=num_layers_node_encoder,
                                        num_layers_node_decoder=num_layers_node_decoder,
                                        dropout_node=dropout_node)

        # 分类器
        self.local_compensation = LocMesCom(input_dim=input_dim, topk=topk, epsilon=epsilon, num_pers_cos=num_pers_cos,
                                            weightedcosine=weightedcosine, adj_fuse=adj_fuse, alpha_adj=alpha_adj,
                                            hid_dim_cla=hid_dim_cla, output_dim=output_dim, num_layers_cla=num_layers_cla,
                                            dropout_cla=dropout_cla, node_border_mask=node_border_mask)



    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        border_node = self.border_gen()
        embedding, logits = self.local_compensation(x, edge_index, border_node)
        return embedding, logits


class LocMesCom(nn.Module):
    def __init__(self, input_dim, topk, epsilon, num_pers_cos, weightedcosine, adj_fuse, alpha_adj, hid_dim_cla,
                 output_dim, num_layers_cla, dropout_cla,node_border_mask=None):
        super(LocMesCom, self).__init__()
        self.weightedcosine = weightedcosine
        self.edge_gen = EdgeGen(input_dim=input_dim, topk=topk, epsilon=epsilon, num_pers=num_pers_cos,
                                weightedcosine=self.weightedcosine)
        self.node_projection=nn.Linear(input_dim, output_dim)
        self.adj_fuse = adj_fuse
        self.alpha_adj = alpha_adj
        self.hid_dim = hid_dim_cla

        self.classifier = BorderGCN(input_dim=input_dim, hid_dim=hid_dim_cla, output_dim=output_dim,
                              num_layers=num_layers_cla,
                              dropout=dropout_cla)
        self.adj_learn = None
        self.node_border_mask = node_border_mask
        self.reset_parameters()

    def reset_parameters(self):
        self.edge_gen.reset_parameters()
        self.classifier.reset_parameters()

    def segment_compensate(self, node_feature, edge_index=None, border_node=None, border_subgraph_map=None):
        # 将第一阶段新增的语义邻居加入到原始图结构中，共同完成消息传播
        adj_ori = self.edge_gen(node_feature)
        adj_normalize = adj_ori / torch.clamp(torch.sum(adj_ori, dim=-1, keepdim=True),max=1, min=1e-12)  # 归一化
        #adj_normalize=symmetry(adj_normalize)
        if self.adj_fuse and edge_index is not None:
            num_nodes = node_feature.shape[0]
            adj_ini = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(node_feature.device),
                                              torch.Size([num_nodes, num_nodes]))
            adj = adj_normalize * self.alpha_adj + adj_ini #* (1 - self.alpha_adj)
            return adj, adj_normalize
        return adj_normalize, adj_normalize

    def original_graph(self, node_feature, edge_index=None, border_node=None, border_subgraph_map=None):
        # 将第一阶段新增的语义邻居加入到原始图结构中，共同完成消息传播
        num_nodes = node_feature.shape[0]
        adj_ini = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(node_feature.device),
                                          torch.Size([num_nodes, num_nodes])).to_dense()
        return adj_ini, adj_ini


    def cal_border_weight(self, node_feature, border_nodes):
        weight_border = self.edge_gen.metric(node_feature, border_nodes)
        weight_border=torch.clamp(weight_border, max=1, min=1e-12)
        '''weight_border = weight_border / torch.clamp(torch.sum(weight_border, dim=-2, keepdim=True), max=1, min=1e-12)  # 归一化'''
        weight_border = weight_border * self.node_border_mask + 0 * (1 - self.node_border_mask)
        return weight_border

    def forward(self, x, edge_index=None,   border_nodes=None):
        adj, adj_learn = self.segment_compensate(x, edge_index)
        #adj, adj_learn = self.original_graph(x, edge_index)
        if border_nodes is not None and self.node_border_mask is not None:
            weight_border = self.cal_border_weight(x, border_nodes)
            embedding, logits = self.classifier(x, adj, border_nodes, weight_border)
        else:
            embedding, logits = self.classifier(x, adj)
        # self.adj_learn = adj_learn
        self.border_embedding = self.classifier.border_embedding
        return embedding, logits


class BorderNodeGen(nn.Module):
    def __init__(self, num_nodes, embed_dim, hid_dim_node, feat_shape, num_layers_node_encoder, num_layers_node_decoder,
                 dropout_node):
        super(BorderNodeGen, self).__init__()
        # 生成初始节点随机嵌入部分
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.ini_emb = nn.Parameter(torch.zeros(self.num_nodes, self.embed_dim))  # 随机生成的节点embedding
        self.node_gen = NodeGen(input_dim=self.embed_dim, hid_dim=hid_dim_node, feat_shape=feat_shape,
                                num_layers_encoder=num_layers_node_encoder,
                                num_layers_decoder=num_layers_node_decoder,
                                dropout=dropout_node)
        self.node_feat_learn = None
        self.reset_parameters()

    def reset_parameters(self):
        #nn.init.xavier_uniform_(self.ini_emb)
        self.node_gen.reset_parameters()

    def learn_node(self, node_embedding):
        node_feat_learn, embedding = self.node_gen(node_embedding)
        return node_feat_learn, embedding

    def forward(self):
        node_feat_learn, embedding = self.learn_node(self.ini_emb)
        return node_feat_learn


