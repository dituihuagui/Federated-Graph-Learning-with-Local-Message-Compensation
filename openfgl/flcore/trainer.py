import torch
import random
import pandas as pd
import json
import os
from datetime import datetime  # [新增] 用于生成时间戳
from openfgl.data.distributed_dataset_loader import FGLDataset
from openfgl.utils.basic_utils import load_client, load_server
from openfgl.utils.logger import Logger


class FGLTrainer:
    """
    Federated Graph Learning Trainer class to manage the training and evaluation process.
    """

    def __init__(self, args):
        """
        Initialize the FGLTrainer with provided arguments and dataset.
        """
        self.args = args
        self.message_pool = {}
        fgl_dataset = FGLDataset(args)
        self.device = torch.device(f"cuda:{args.gpuid}" if (torch.cuda.is_available() and args.use_cuda) else "cpu")
        self.clients = [load_client(args, client_id, fgl_dataset.local_data[client_id], fgl_dataset.processed_dir,
                                    self.message_pool, self.device) for client_id in range(self.args.num_clients)]
        self.server = load_server(args, fgl_dataset.global_data, fgl_dataset.processed_dir, self.message_pool,
                                  self.device)
        self.evaluation_result = {"best_round": 0}

        self.round_records = []

        if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
            for metric in self.args.metrics:
                self.evaluation_result[f"best_val_{metric}"] = 0
                self.evaluation_result[f"best_test_{metric}"] = 0
                self.evaluation_result[f"most_test_{metric}"] = 0
        elif self.args.task in ["node_clust"]:
            for metric in self.args.metrics:
                self.evaluation_result[f"best_{metric}"] = 0

        self.logger = Logger(args, self.message_pool, fgl_dataset.processed_dir, self.server.personalized)

    def train(self):
        """
        Train the model over a specified number of rounds.
        """
        for round_id in range(self.args.num_rounds):
            sampled_clients = sorted(
                random.sample(list(range(self.args.num_clients)), int(self.args.num_clients * self.args.client_frac)))
            print(f"round # {round_id}\t\tsampled_clients: {sampled_clients}")

            self.message_pool["round"] = round_id
            self.message_pool["sampled_clients"] = sampled_clients
            self.server.send_message()

            for client_id in sampled_clients:
                self.clients[client_id].execute()
                self.clients[client_id].send_message()

            self.server.execute()

            self.evaluate()
            print("-" * 50)

        self.save_metrics_to_excel()
        self.logger.save()

    def save_metrics_to_excel(self):
        try:
            dataset_name = self.args.dataset[0] if isinstance(self.args.dataset, list) else self.args.dataset
            save_dir = os.path.join("result", dataset_name, f"client_{self.args.num_clients}")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir) 
            method_name = getattr(self.args, 'fl_algorithm', 'Method')
            test_result = self.evaluation_result[f"most_test_{self.args.metrics[0]}"]

            file_basename = f"{method_name}_{test_result:.6f}"

            excel_full_path = os.path.join(save_dir, f"{file_basename}.xlsx")
            txt_full_path = os.path.join(save_dir, f"{file_basename}.txt")

            df = pd.DataFrame(self.round_records)
            df.to_excel(excel_full_path, index=False)
            print(f"训练指标已成功保存到: {os.path.abspath(excel_full_path)}")

            if self.args.fl_algorithm == "FedLMC":
                with open(txt_full_path, 'w', encoding='utf-8') as f:
                    from openfgl.flcore.FedLMC.config.config import allconfig
                    config = allconfig[self.args.dataset[0]]
                    json.dump(config, f, indent=4, default=str)

                print(f"训练参数已成功保存到: {os.path.abspath(txt_full_path)}")

        except Exception as e:
            print(f"保存文件失败: {e}")
            import traceback
            traceback.print_exc()

    def evaluate(self):
        """
        Evaluate the model.
        """
        evaluation_result = {"current_round": self.message_pool["round"]}

        if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
            for metric in self.args.metrics:
                evaluation_result[f"current_val_{metric}"] = 0
                evaluation_result[f"current_test_{metric}"] = 0
        elif self.args.task in ["node_clust"]:
            for metric in self.args.metrics:
                evaluation_result[f"current_{metric}"] = 0

        tot_samples = 0
        one_time_infer = False

        for client_id in range(self.args.num_clients):
            if self.args.evaluation_mode == "local_model_on_local_data":
                num_samples = self.clients[client_id].task.num_samples
                result = self.clients[client_id].task.evaluate()
            elif self.args.evaluation_mode == "local_model_on_global_data":
                num_samples = self.server.task.num_samples
                result = self.clients[client_id].task.evaluate(self.server.task.splitted_data)
            elif self.args.evaluation_mode == "global_model_on_local_data":
                num_samples = self.clients[client_id].task.num_samples
                if self.server.personalized:
                    raise ValueError(
                        f"personalized algorithm {self.args.fl_algorithm} doesn't support global model evaluation.")
                result = self.server.task.evaluate(self.clients[client_id].task.splitted_data)
            elif self.args.evaluation_mode == "global_model_on_global_data":
                num_samples = self.server.task.num_samples
                if self.server.personalized:
                    raise ValueError(
                        f"personalized algorithm {self.args.fl_algorithm} doesn't support global model evaluation.")
                one_time_infer = True
                result = self.server.task.evaluate()

            if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
                for metric in self.args.metrics:
                    val_metric, test_metric = result[f"{metric}_val"], result[f"{metric}_test"]
                    evaluation_result[f"current_val_{metric}"] += val_metric * num_samples
                    evaluation_result[f"current_test_{metric}"] += test_metric * num_samples
            elif self.args.task in ["node_clust"]:
                for metric in self.args.metrics:
                    metric_value = result[f"{metric}"]
                    evaluation_result[f"current_{metric}"] += metric_value * num_samples

            if one_time_infer:
                tot_samples = num_samples
                break
            else:
                tot_samples += num_samples

        record_entry = {'round': evaluation_result['current_round']}

        if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
            for metric in self.args.metrics:
                evaluation_result[f"current_val_{metric}"] /= tot_samples
                evaluation_result[f"current_test_{metric}"] /= tot_samples

                record_entry[f'val_{metric}'] = evaluation_result[f"current_val_{metric}"]
                record_entry[f'test_{metric}'] = evaluation_result[f"current_test_{metric}"]

            if evaluation_result[f"current_val_{self.args.metrics[0]}"] > self.evaluation_result[
                f"best_val_{self.args.metrics[0]}"]:
                for metric in self.args.metrics:
                    self.evaluation_result[f"best_val_{metric}"] = evaluation_result[f"current_val_{metric}"]
                    self.evaluation_result[f"best_test_{metric}"] = evaluation_result[f"current_test_{metric}"]
                self.evaluation_result[f"best_round"] = evaluation_result[f"current_round"]

            if evaluation_result[f"current_test_{self.args.metrics[0]}"] > self.evaluation_result[
                f"most_test_{self.args.metrics[0]}"]:
                for metric in self.args.metrics:
                    self.evaluation_result[f"most_test_{metric}"] = evaluation_result[
                        f"current_test_{self.args.metrics[0]}"]

            current_output = f"curr_round: {evaluation_result['current_round']}\t" + \
                             "\t".join([
                                           f"curr_val_{metric}: {evaluation_result[f'current_val_{metric}']:.4f}\tcurr_test_{metric}: {evaluation_result[f'current_test_{metric}']:.4f}"
                                           for metric in self.args.metrics])

            best_output = f"best_round: {self.evaluation_result['best_round']}\t" + \
                          "\t".join([
                                        f"best_val_{metric}: {self.evaluation_result[f'best_val_{metric}']:.4f}\tbest_test_{metric}: {self.evaluation_result[f'best_test_{metric}']:.4f}"
                                        for metric in self.args.metrics])

            most_test = f"".join(
                [f"most_test_{metric}: {self.evaluation_result[f'most_test_{metric}']:.4f}" for metric in
                 self.args.metrics])
            print(current_output)
            print(best_output)
            print(most_test)
        else:
            for metric in self.args.metrics:
                evaluation_result[f"current_{metric}"] /= tot_samples
                record_entry[f'{metric}'] = evaluation_result[f"current_{metric}"]

            if evaluation_result[f"current_{self.args.metrics[0]}"] > self.evaluation_result[
                f"best_{self.args.metrics[0]}"]:
                for metric in self.args.metrics:
                    self.evaluation_result[f"best_{metric}"] = evaluation_result[f"current_{metric}"]
                self.evaluation_result[f"best_round"] = evaluation_result[f"current_round"]

            current_output = f"curr_round: {evaluation_result['current_round']}\t" + \
                             "\t".join([f"curr_{metric}: {evaluation_result[f'current_{metric}']:.4f}" for metric in
                                        self.args.metrics])

            best_output = f"best_round: {self.evaluation_result['best_round']}\t" + \
                          "\t".join([f"best_{metric}: {self.evaluation_result[f'best_{metric}']:.4f}" for metric in
                                     self.args.metrics])

            print(current_output)
            print(best_output)

        self.round_records.append(record_entry)

        self.logger.add_log(evaluation_result)