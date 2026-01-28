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
