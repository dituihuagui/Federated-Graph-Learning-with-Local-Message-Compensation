import torch

import openfgl.config as config
import os
from openfgl.flcore.trainer import FGLTrainer
import datetime





args = config.args
#args.seed = 3407
args.root = "dataset"
args.simulation_mode = "subgraph_fl_louvain"  # subgraph_fl_label_skew"subgraph_fl_louvain"
#args.simulation_mode = "subgraph_fl_metis"

args.train_val_test = '0.2-0.4-0.4'
args.model = ["gcn"]
args.metrics = ["accuracy"]
#exit()
times=1
for i in range(times):
    trainer = FGLTrainer(args)
    trainer.train()
#python main.py --gpuid 5 hid_dim 64 --num_layers 2 --dataset "Cora" --train_val_test "0.2-0.4-0.4" --num_epochs 3 --num_rounds 100 --lr 0.01 --fl_algorithm "FedLMC" --num_clients 10
