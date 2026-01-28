import torch
from openfgl.flcore.base import BaseServer
from openfgl.flcore.FedLMC.models import MyModel
import torch.nn.functional as F
from openfgl.flcore.FedLMC.config.config import allconfig

class FedLMCServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedLMCServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(FedLMCServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.config = allconfig[args.dataset[0]]
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
                    ))

    def execute(self):
        """
        Executes the server-side operations. This method aggregates model updates from the
        clients by computing a weighted average of the model parameters.
        """
        # 计算全局模型参数
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in
                                   self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples

                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"],
                                                       self.task.model.local_compensation.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param

    def send_message(self):
        """
        Sends a message to the clients containing the updated global model parameters after 
        aggregation.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.local_compensation.parameters())
        }

