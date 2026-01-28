import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from openfgl.flcore.FedLMC.config.config import allconfig
from openfgl.flcore.FedLMC.models import MyModel
from sknetwork.clustering import Louvain
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


class FedLMClient(BaseClient):

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedLMClient.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedLMClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.config = allconfig[args.dataset[0]]
        self.border_subgraph_map = self.divide_subgraph(len_max=self.config['K'])
        self.node_border_mask = self.border_mask(self.border_subgraph_map).detach()
        self.temperature = self.config['temperature']

        self.task.load_custom_model(
            MyModel(input_dim=self.task.num_feats,
                    output_dim=self.task.num_global_classes,
                    hid_dim_cla=self.args.hid_dim,
                    num_layers_cla=self.args.num_layers,
                    num_nodes=self.task.num_samples,
                    embed_dim=self.config['embed_dim'],
                    hid_dim_node=self.config['hid_dim_node'],
                    num_layers_node_encoder=self.config['num_layers_node_encoder'],
                    num_layers_node_decoder=self.config['num_layers_node_decoder'],
                    dropout_node=self.config['dropout_node'],
                    dropout_cla=self.args.dropout,
                    alpha_adj=self.config['alpha_adj'],
                    adj_fuse=self.config['adj_fuse'],
                    topk=self.config['topk'],
                    epsilon=self.config['epsilon'],
                    num_pers_cos=self.config['num_pers_cos'],
                    weightedcosine=self.config['weightedcosine'],
                    num_border=len(self.border_subgraph_map),
                    node_border_mask=self.node_border_mask
                    ))
        self.optim1 = Adam(self.task.model.local_compensation.parameters(), lr=self.args.lr,
                           weight_decay=self.args.weight_decay)
        self.optim2 = Adam(self.task.model.border_gen.parameters(), lr=self.args.lr,
                           weight_decay=self.args.weight_decay)

    def execute(self):
        """
        Executes the local training process. This method first synchronizes the local model
        with the global model parameters received from the server, and then trains the model
        on the client's local data.
        """

        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.local_compensation.parameters(),
                                                   self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)
        self.train()

    def send_message(self):
        """
        Sends a message to the server containing the model parameters after training
        and the number of samples in the client's dataset.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.local_compensation.parameters()),
        }

    def train(self):
        data = self.task.processed_data
        class_mask = data["train_mask"]
        class_label = data["data"].y
        adj = self.node_border_mask.T
        node_degree = torch.sum(adj, dim=1)
        border_adj = adj / node_degree[:, None]

        self.task.model.local_compensation.train()
        self.task.model.border_gen.train()
        for _ in range(self.args.num_epochs):
            # 训练生成器
            self.optim1.zero_grad()
            self.optim2.zero_grad()
            border_node = self.task.model.border_gen()
            embedding, logits = self.task.model.local_compensation.forward(data["data"].x,
                                                                           data["data"].edge_index, border_node)
            # 交叉熵损失
            loss_cross = torch.nn.functional.cross_entropy(logits[class_mask], class_label[class_mask])

            # 社区锚点损失
            embedding_true = embedding.detach()
            # 社区锚点embedding
            border_embedding = self.task.model.local_compensation.border_embedding  # num_node*hid_dim
            # 生成社区原型
            subgraph_embedding = torch.mm(border_adj, embedding_true)  # num_node*hid_dim
            # 归一化
            border_embedding = F.normalize(border_embedding, dim=1)
            subgraph_embedding = F.normalize(subgraph_embedding, dim=1)

            # 计算相似性矩阵 (num_border, num_border)，每个边界节点的嵌入与所有边界节点的嵌入之间的相似性
            similarity_matrix = torch.matmul(border_embedding, border_embedding.T)

            # 计算正样本相似性 (num_border,)，计算边界节点和子图embedding的cos相似性
            positive_similarity = torch.sum(border_embedding * subgraph_embedding,
                                            dim=1)

            # 将正样本相似性扩展为 (num_border, 1)，以便与相似性矩阵广播
            positive_similarity = positive_similarity[:, None]

            # 归一化相似性矩阵，将相似性除以温度参数 tau
            similarity_matrix = similarity_matrix / self.temperature
            positive_similarity = positive_similarity / self.temperature
            # 计算 InfoNCE Loss，对每个边界节点，计算负对数似然
            exp_similarity = torch.exp(similarity_matrix)  # 对所有相似性取指数
            exp_positive_similarity = torch.exp(positive_similarity)  # 对正样本相似性取指数
            mask = ~torch.eye(exp_similarity.size(0), device=self.device).bool()  # 生成指示对角线元素的mask
            # 计算分母 (num_border,)
            denominator = torch.sum(exp_similarity * mask, dim=1, keepdim=True)  # 分母是所有边界节点的相似性之和，排除对角线元素
            # 计算对比损失 (num_border,)
            loss_con = -torch.log(exp_positive_similarity / denominator)
            #loss_con = -exp_similarity  # 对齐损失
            #loss_con = torch.log(denominator) # 区分损失
            # 对所有边界节点的损失求平均
            loss_con = torch.mean(loss_con)
            loss = loss_cross + loss_con * self.config['loss_con']
            loss.backward()
            self.optim1.step()
            self.optim2.step()

    def border_mask(self, border_subgraph_map):
        # 创建真实节点和边界节点的mask矩阵num_nude*num_border
        num_nodes = self.task.processed_data['data'].x.shape[0]
        num_border = len(border_subgraph_map)
        node_border_mask = torch.zeros((num_nodes, num_border), dtype=torch.float, device=self.device)

        # 遍历边界节点及其对应的子图节点
        for border_idx, (border_node_id, subgraph_node_ids) in enumerate(border_subgraph_map.items()):
            # 在 node_border_mask 中标记子图节点与边界节点的连接为 1
            node_border_mask[subgraph_node_ids, border_idx] = 1.0
        return node_border_mask

    def divide_subgraph(self, len_max=10):
        """
        将客户端的子图划分为多个小子图，并合并孤立子图使最终子图数量不超过 len_max。

        参数:
        len_max: 最终划分的子图最大数量

        返回:
        partition_groups: 字典，键是社区ID，值是该社区中的所有节点ID
        """
        # 使用 Louvain 算法进行社区划分
        louvain = Louvain(modularity='newman', resolution=1.0, return_aggregate=True)
        num_nodes = self.task.processed_data['data'].x.shape[0]
        adj_csr = to_scipy_sparse_matrix(self.task.processed_data['data'].edge_index)
        fit_result = louvain.fit_predict(adj_csr)

        # 创建 partition：字典，键是节点ID，值是社区ID
        partition = {}
        for node_id, com_id in enumerate(fit_result):
            partition[node_id] = int(com_id)

        # 获取所有社区ID
        groups = list(set(partition.values()))

        # 创建 partition_groups：字典，键是社区ID，值是该社区中的所有节点ID
        partition_groups = {group_id: [] for group_id in groups}
        for node_id, com_id in partition.items():
            partition_groups[com_id].append(node_id)

        # 将所有社区按照大小进行排序
        len_dict = {group_id: len(node_ids) for group_id, node_ids in partition_groups.items()}
        sorted_len_dict = sorted(len_dict.items(), key=lambda item: item[1], reverse=True)

        # 合并孤立子图（节点数少的社区）
        merged_partition_groups = {}
        small_groups = []  # 存储需要合并的小社区
        for group_id, group_size in sorted_len_dict:
            if group_size < 3:  # 这里可以调整阈值，比如小于5个节点的社区需要合并
                small_groups.append(group_id)
            else:
                merged_partition_groups[group_id] = partition_groups[group_id]

        # 将小社区合并到一个大社区
        if small_groups:
            merged_partition_groups['merged_small'] = []
            for group_id in small_groups:
                merged_partition_groups['merged_small'].extend(partition_groups[group_id])

        # 如果合并后社区数量仍然超过 len_max，则继续合并
        while len(merged_partition_groups) > len_max:
            # 按社区大小排序
            sorted_groups = sorted(merged_partition_groups.items(), key=lambda item: len(item[1]), reverse=True)
            # 合并最后两个社区
            smallest_group_id, smallest_group_nodes = sorted_groups[-1]
            second_smallest_group_id, second_smallest_group_nodes = sorted_groups[-2]

            # 合并
            merged_partition_groups[second_smallest_group_id].extend(smallest_group_nodes)
            del merged_partition_groups[smallest_group_id]

        return merged_partition_groups
