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
            self.optim1.zero_grad()
            self.optim2.zero_grad()
            border_node = self.task.model.border_gen()
            embedding, logits = self.task.model.local_compensation.forward(data["data"].x,
                                                                           data["data"].edge_index, border_node)
            loss_cross = torch.nn.functional.cross_entropy(logits[class_mask], class_label[class_mask])

            embedding_true = embedding.detach()
            border_embedding = self.task.model.local_compensation.border_embedding  # num_node*hid_dim
            subgraph_embedding = torch.mm(border_adj, embedding_true)  # num_node*hid_dim
            border_embedding = F.normalize(border_embedding, dim=1)
            subgraph_embedding = F.normalize(subgraph_embedding, dim=1)

            similarity_matrix = torch.matmul(border_embedding, border_embedding.T)

            positive_similarity = torch.sum(border_embedding * subgraph_embedding,
                                            dim=1)

            positive_similarity = positive_similarity[:, None]

            similarity_matrix = similarity_matrix / self.temperature
            positive_similarity = positive_similarity / self.temperature
            exp_similarity = torch.exp(similarity_matrix) 
            exp_positive_similarity = torch.exp(positive_similarity) 
            mask = ~torch.eye(exp_similarity.size(0), device=self.device).bool()  
            denominator = torch.sum(exp_similarity * mask, dim=1, keepdim=True) 
            loss_con = -torch.log(exp_positive_similarity / denominator)
            #loss_con = -exp_similarity  
            #loss_con = torch.log(denominator) 
            loss_con = torch.mean(loss_con)
            loss = loss_cross + loss_con * self.config['loss_con']
            loss.backward()
            self.optim1.step()
            self.optim2.step()

    def border_mask(self, border_subgraph_map):
        num_nodes = self.task.processed_data['data'].x.shape[0]
        num_border = len(border_subgraph_map)
        node_border_mask = torch.zeros((num_nodes, num_border), dtype=torch.float, device=self.device)

        for border_idx, (border_node_id, subgraph_node_ids) in enumerate(border_subgraph_map.items()):
            node_border_mask[subgraph_node_ids, border_idx] = 1.0
        return node_border_mask

    def divide_subgraph(self, len_max=10):

        louvain = Louvain(modularity='newman', resolution=1.0, return_aggregate=True)
        num_nodes = self.task.processed_data['data'].x.shape[0]
        adj_csr = to_scipy_sparse_matrix(self.task.processed_data['data'].edge_index)
        fit_result = louvain.fit_predict(adj_csr)

        partition = {}
        for node_id, com_id in enumerate(fit_result):
            partition[node_id] = int(com_id)

        groups = list(set(partition.values()))

        partition_groups = {group_id: [] for group_id in groups}
        for node_id, com_id in partition.items():
            partition_groups[com_id].append(node_id)

        len_dict = {group_id: len(node_ids) for group_id, node_ids in partition_groups.items()}
        sorted_len_dict = sorted(len_dict.items(), key=lambda item: item[1], reverse=True)

        merged_partition_groups = {}
        small_groups = []  
        for group_id, group_size in sorted_len_dict:
            if group_size < 3:  
                small_groups.append(group_id)
            else:
                merged_partition_groups[group_id] = partition_groups[group_id]

        if small_groups:
            merged_partition_groups['merged_small'] = []
            for group_id in small_groups:
                merged_partition_groups['merged_small'].extend(partition_groups[group_id])

        while len(merged_partition_groups) > len_max:
            sorted_groups = sorted(merged_partition_groups.items(), key=lambda item: len(item[1]), reverse=True)
            smallest_group_id, smallest_group_nodes = sorted_groups[-1]
            second_smallest_group_id, second_smallest_group_nodes = sorted_groups[-2]

            merged_partition_groups[second_smallest_group_id].extend(smallest_group_nodes)
            del merged_partition_groups[smallest_group_id]

        return merged_partition_groups
