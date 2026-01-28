import copy
import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch_geometric.data import Data
from sknetwork.clustering import Louvain
import sys
from sklearn.cluster import KMeans
import pymetis as metis
import torch_geometric.utils
from tqdm import tqdm
import torch_geometric

def get_subgraph_pyg_data(global_dataset, node_list):
    """
    Extract a subgraph from the global dataset given a list of node indices.

    Args:
        global_dataset (Data): The global graph dataset.
        node_list (list): List of node indices to include in the subgraph.

    Returns:
        Data: The subgraph containing the specified nodes and their edges.
    """
    global_edge_index = global_dataset.edge_index
    node_id_set = set(node_list)

    # global_id_to_local_id：字典，键是节点在全局图的ID，值是该节点在本地子图的ID
    # local_id_to_global_id：字典，键是节点在本地子图的ID，值是该节点在全局图的ID
    global_id_to_local_id = {}
    local_id_to_global_id = {}
    local_edge_list = []
    for local_id, global_id in enumerate(node_list):
        global_id_to_local_id[global_id] = local_id
        local_id_to_global_id[local_id] = global_id


    for edge_id in tqdm(range(global_edge_index.shape[1]), desc="Processing Edge Mapping"):
        #src和tgt都是全局图的节点ID，需要通过global_id_to_local_id转换为子图的节点ID
        src = global_edge_index[0, edge_id].item()
        tgt = global_edge_index[1, edge_id].item()
        if src in node_id_set and tgt in node_id_set:
            local_id_src = global_id_to_local_id[src]
            local_id_tgt = global_id_to_local_id[tgt]
            local_edge_list.append((local_id_src, local_id_tgt))

    local_edge_index = torch.tensor(local_edge_list).T

    #生成当前客户端的数据集
    local_subgraph = Data(x=global_dataset.x[node_list], edge_index=local_edge_index, y=global_dataset.y[node_list])
    local_subgraph.global_map = local_id_to_global_id

    if hasattr(global_dataset, "num_classes"):
        local_subgraph.num_global_classes = global_dataset.num_classes
    else:
        local_subgraph.num_global_classes = global_dataset.num_global_classes

    return local_subgraph



def graph_fl_cross_domain(args, global_dataset):
    print("Conducting graph-fl cross domain simulation...")
    local_data = []
    for client_id in range(args.num_clients):
        local_graphs = global_dataset[client_id] # list(InMemoryDataset) -> InMemoryDataset
        local_graphs.num_global_classes = global_dataset[client_id].num_classes
        local_data.append(local_graphs)
    return local_data



def graph_fl_label_skew(args, global_dataset, shuffle=True):
    """
    Simulate cross-domain federated learning for graph data.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (list): List of global graph datasets for each client.

    Returns:
        list: List of local graph datasets for each client.
    """
    print("Conducting graph-fl label skew simulation...")
    
    graph_labels = global_dataset.y.numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(graph_labels, return_counts=True)
    
    
    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")
       
    min_size = 0
    K = len(unique_labels)
    N = graph_labels.shape[0]

    try_cnt = 0
    while min_size < args.least_samples:
        if try_cnt > args.dirichlet_try_cnt:
            print(f"Client data size does not meet the minimum requirement {args.least_samples}. Try 'args.dirichlet_alpha' larger than {args.dirichlet_alpha} /  'args.try_cnt' larger than {args.try_cnt} / 'args.least_sampes' lower than {args.least_samples}.")
            sys.exit(0)
            
        client_indices = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(graph_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,client_indices)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            client_indices = [idx_j + idx.tolist() for idx_j,idx in zip(client_indices,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in client_indices])
        try_cnt += 1
   
    local_data = []
    client_label_counts = [[0] * K for _ in range(args.num_clients)]
    for client_id in range(args.num_clients):
        for class_i in range(K):
            client_label_counts[client_id][class_i] = (graph_labels[client_indices[client_id]] == class_i).sum()
        
        list.sort(client_indices[client_id])
        
        local_id_to_global_id = {}
        for local_id, global_id in enumerate(client_indices[client_id]):
            local_id_to_global_id[local_id] = global_id
        
        local_graphs = global_dataset.copy(client_indices[client_id]) # InMemoryDataset -> deep-copy subset
        local_graphs.num_global_classes = global_dataset.num_classes
        local_graphs.global_map = local_id_to_global_id
        local_data.append(local_graphs)
    
    print(f"label_counts:\n{np.array(client_label_counts)}")
    return local_data
    
    
def graph_fl_topology_skew(args, global_dataset, shuffle=True):
    """
    Simulate topology skew in federated learning for graph data.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.
        shuffle (bool, optional): If True, shuffle the dataset. Defaults to True.

    Returns:
        list: List of local graph datasets for each client with topology skew.
    """
    print("Conducting graph-fl topology skew simulation...")
    
    graph_labels = global_dataset.y.numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(graph_labels, return_counts=True)
    
    
    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")
       
    min_size = 0
    K = len(unique_labels)
    N = graph_labels.shape[0]
    client_indices = [[] for _ in range(num_clients)]
    
    d_list = []
    for graph_id, data in enumerate(global_dataset):
        deg = torch_geometric.utils.degree(data.edge_index[1], num_nodes=data.num_nodes)
        
        d_list.append((graph_id, deg.mean()))
    
    d_list.sort(key= lambda x: x[1])
    
    segment_len = len(d_list) // num_clients
    for client_id in range(num_clients):
        left = client_id * segment_len
        right = segment_len * (client_id + 1)
        if client_id == num_clients - 1:
            right = len(d_list)
        
        segment = d_list[left : right]
        client_indices[client_id] = [x[0] for x in segment]
    
    assert sum([len(x) for x in client_indices]) == N
    
    local_data = []
    client_label_counts = [[0] * K for _ in range(args.num_clients)]
    for client_id in range(args.num_clients):
        for class_i in range(K):
            client_label_counts[client_id][class_i] = (graph_labels[client_indices[client_id]] == class_i).sum()
        
        list.sort(client_indices[client_id])
        
        local_id_to_global_id = {}
        for local_id, global_id in enumerate(client_indices[client_id]):
            local_id_to_global_id[local_id] = global_id
        
        local_graphs = global_dataset.copy(client_indices[client_id]) # InMemoryDataset -> deep-copy subset
        local_graphs.num_global_classes = global_dataset.num_classes
        local_graphs.global_map = local_id_to_global_id
        local_data.append(local_graphs)
    
    print(f"label_counts:\n{np.array(client_label_counts)}")
    return local_data
    
    
    
    
    
  
def graph_fl_feature_skew(args, global_dataset, shuffle=True):
    """
    Simulate feature skew in federated learning for graph data.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.
        shuffle (bool, optional): If True, shuffle the dataset. Defaults to True.

    Returns:
        list: List of local graph datasets for each client with feature skew.
    """
    print("Conducting graph-fl feature skew simulation...")
    
    graph_labels = global_dataset.y.numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(graph_labels, return_counts=True)
    
    
    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")
       
    min_size = 0
    K = len(unique_labels)
    N = graph_labels.shape[0]
    client_indices = [[] for _ in range(num_clients)]
    
    f_list = []
    for graph_id, data in enumerate(global_dataset):
        avg_feature = data.x.mean()
        f_list.append((graph_id, avg_feature))
        # deg = torch_geometric.utils.degree(data.edge_index[1], num_nodes=data.num_nodes)
        # f_list.append((graph_id, deg.mean()))
    
    f_list.sort(key= lambda x: x[1])
    
    segment_len = len(f_list) // num_clients
    for client_id in range(num_clients):
        left = client_id * segment_len
        right = segment_len * (client_id + 1)
        if client_id == num_clients - 1:
            right = len(f_list)
        
        segment = f_list[left : right]
        client_indices[client_id] = [x[0] for x in segment]
    
    assert sum([len(x) for x in client_indices]) == N
    
    local_data = []
    client_label_counts = [[0] * K for _ in range(args.num_clients)]
    for client_id in range(args.num_clients):
        for class_i in range(K):
            client_label_counts[client_id][class_i] = (graph_labels[client_indices[client_id]] == class_i).sum()
        
        list.sort(client_indices[client_id])
        
        local_id_to_global_id = {}
        for local_id, global_id in enumerate(client_indices[client_id]):
            local_id_to_global_id[local_id] = global_id
        
        local_graphs = global_dataset.copy(client_indices[client_id]) # InMemoryDataset -> deep-copy subset
        local_graphs.num_global_classes = global_dataset.num_classes
        local_graphs.global_map = local_id_to_global_id
        local_data.append(local_graphs)
    
    print(f"label_counts:\n{np.array(client_label_counts)}")
    return local_data 
    
    
    
    
    
def subgraph_fl_label_skew(args, global_dataset, shuffle=True):
    """
    Simulate label skew in federated learning for subgraphs.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (list): List of global graph datasets.
        shuffle (bool, optional): If True, shuffle the dataset. Defaults to True.

    Returns:
        list: List of local subgraph datasets for each client with label skew.
    """
    print("Conducting subgraph-fl label skew simulation...")
    node_labels = global_dataset[0].y.numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(node_labels, return_counts=True)
    
    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")
       
    min_size = 0
    K = len(unique_labels)
    N = node_labels.shape[0]

    try_cnt = 0
    while min_size < args.least_samples:
        if try_cnt > args.dirichlet_try_cnt:
            print(f"Client data size does not meet the minimum requirement {args.least_samples}. Try 'args.dirichlet_alpha' larger than {args.dirichlet_alpha} /  'args.try_cnt' larger than {args.try_cnt} / 'args.least_sampes' lower than {args.least_samples}.")
            sys.exit(0)
            
        client_indices = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(node_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,client_indices)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            client_indices = [idx_j + idx.tolist() for idx_j,idx in zip(client_indices,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in client_indices])
        try_cnt += 1
   
    
    local_data = []
    client_label_counts = [[0] * K for _ in range(args.num_clients)]
    for client_id in range(args.num_clients):
        for class_i in range(K):
            client_label_counts[client_id][class_i] = (node_labels[client_indices[client_id]] == class_i).sum()
        local_subgraph = get_subgraph_pyg_data(global_dataset, client_indices[client_id])
        if local_subgraph.edge_index.dim() == 1:
            local_subgraph.edge_index, _ = torch_geometric.utils.add_random_edge(local_subgraph.edge_index.view(2,-1))
        local_data.append(local_subgraph)
    print(f"label_counts:\n{np.array(client_label_counts)}")
    return local_data


def subgraph_fl_louvain_plus(args, global_dataset):
    """
    Simulate subgraph federated learning using the Louvain+ method.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.

    Returns:
        list: List of local subgraph datasets for each client using Louvain+ method.
    """
    print("Conducting subgraph-fl louvain+ simulation...")
    louvain = Louvain(modularity='newman', resolution=args.louvain_resolution, return_aggregate=True) # resolution 越大产生的社区越多, 社区粒度越小
    adj_csr = to_scipy_sparse_matrix(global_dataset[0].edge_index)
    fit_result = louvain.fit_predict(adj_csr)
    communities = {}
    for node_id, com_id in enumerate(fit_result):
        if com_id not in communities:
            communities[com_id] = {"nodes":[], "num_nodes":0, "label_distribution":[0] * global_dataset.num_classes}
        communities[com_id]["nodes"].append(node_id)
        
    for com_id in communities.keys():
        communities[com_id]["num_nodes"] = len(communities[com_id]["nodes"])
        for node in communities[com_id]["nodes"]:
            label = copy.deepcopy(global_dataset[0].y[node])
            communities[com_id]["label_distribution"][label] += 1

    num_communities = len(communities)
    clustering_data = np.zeros(shape=(num_communities, global_dataset.num_classes))
    for com_id in communities.keys():
        for class_i in range(global_dataset.num_classes):
            clustering_data[com_id][class_i] = communities[com_id]["label_distribution"][class_i]
        clustering_data[com_id, :] /= clustering_data[com_id, :].sum()

    kmeans = KMeans(n_clusters=args.num_clients)
    kmeans.fit(clustering_data)

    clustering_labels = kmeans.labels_

    client_indices = {client_id: [] for client_id in range(args.num_clients)}
    
    for com_id in range(num_communities):
        client_indices[clustering_labels[com_id]] += communities[com_id]["nodes"]
    
    
      
    local_data = []
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(global_dataset, client_indices[client_id])
        if local_subgraph.edge_index.dim() == 1:
            local_subgraph.edge_index, _ = torch_geometric.utils.add_random_edge(local_subgraph.edge_index.view(2,-1))
        local_data.append(local_subgraph)

    return local_data


def subgraph_fl_metis_plus(args, global_dataset):
    """
    Simulate subgraph federated learning using the Metis+ method.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.

    Returns:
        list: List of local subgraph datasets for each client using Metis+ method.
    """
    print("Conducting subgraph-fl metis+ simulation...")
    graph_nx = to_networkx(global_dataset[0], to_undirected=True)
    communities = {com_id: {"nodes":[], "num_nodes":0, "label_distribution":[0] * global_dataset.num_classes} 
                            for com_id in range(args.metis_num_coms)}
    n_cuts, membership = metis.part_graph(args.metis_num_coms, graph_nx)
    for com_id in range(args.metis_num_coms):
        com_indices = np.where(np.array(membership) == com_id)[0]
        com_indices = list(com_indices)
        communities[com_id]["nodes"] = com_indices
        communities[com_id]["num_nodes"] = len(com_indices)
        for node in communities[com_id]["nodes"]:
            label = copy.deepcopy(global_dataset[0].y[node])
            communities[com_id]["label_distribution"][label] += 1
    
    num_communities = len(communities)
    clustering_data = np.zeros(shape=(num_communities, global_dataset.num_classes))
    for com_id in communities.keys():
        for class_i in range(global_dataset.num_classes):
            clustering_data[com_id][class_i] = communities[com_id]["label_distribution"][class_i]
        clustering_data[com_id, :] /= clustering_data[com_id, :].sum()

    kmeans = KMeans(n_clusters=args.num_clients)
    kmeans.fit(clustering_data)

    clustering_labels = kmeans.labels_

    client_indices = {client_id: [] for client_id in range(args.num_clients)}
    
    for com_id in range(num_communities):
        client_indices[clustering_labels[com_id]] += communities[com_id]["nodes"]
    
    local_data = []
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(global_dataset, client_indices[client_id])
        if local_subgraph.edge_index.dim() == 1:
            local_subgraph.edge_index, _ = torch_geometric.utils.add_random_edge(local_subgraph.edge_index.view(2,-1))
        local_data.append(local_subgraph)
    
    return local_data
    
    



    
    
    

def subgraph_fl_metis(args, global_dataset):
    """
    Simulate subgraph federated learning using the Metis method.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.

    Returns:
        list: List of local subgraph datasets for each client using Metis method.
    """
    print("Conducting subgraph-fl metis simulation...")
    graph_nx = to_networkx(global_dataset[0], to_undirected=True)
    n_cuts, membership = metis.part_graph(args.num_clients, graph_nx)
    
    client_indices = [None] * args.num_clients
    for client_id in range(args.num_clients):
        client_indices[client_id] = np.where(np.array(membership) == client_id)[0].tolist()
        
    local_data = []
    
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(global_dataset, client_indices[client_id])
        if local_subgraph.edge_index.dim() == 1:
            local_subgraph.edge_index, _ = torch_geometric.utils.add_random_edge(local_subgraph.edge_index.view(2,-1))
        local_data.append(local_subgraph)
    
    return local_data
    
    

def subgraph_fl_louvain(args, global_dataset):
    """
    Simulate subgraph federated learning using the Louvain method.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.

    Returns:
        list: List of local subgraph datasets for each client using Louvain method.
    """
    print("Conducting subgraph-fl louvain simulation...")
    louvain = Louvain(modularity='newman', resolution=args.louvain_resolution, return_aggregate=True)
    num_nodes = global_dataset[0].x.shape[0]
    adj_csr = to_scipy_sparse_matrix(global_dataset[0].edge_index)
    fit_result = louvain.fit_predict(adj_csr)

    #partition：字典，键是节点ID，值是社区ID
    partition = {}
    for node_id, com_id in enumerate(fit_result):
        partition[node_id] = int(com_id)
        #print(node_id,com_id)


    #groups：列表，包含所有社区ID
    groups = []

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    print(groups)

    #partition_groups：字典，键是社区ID，值是该社区中的所有节点ID
    partition_groups = {group_i: [] for group_i in groups}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)

    #for key in partition_groups.keys():
    #    print(key,partition_groups[key])


    #对社区进行划分，确保每个社区的节点数不超过group_len_max，如果超过则将该社区分割成两个社区。
    group_len_max = num_nodes // args.num_clients - args.louvain_delta
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    print(groups)

    #将所有的社区按照大小进行排序
    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    #len_dict：字典，键是社区ID，值是该社区的节点数量，对len_dict进行从大到小排列
    len_dict = {}

    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {
        k: v
        for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)
    }

    #for key in sort_len_dict.keys():
    #   print(key,sort_len_dict[key])



    #将排序后的社区分配给不同的客户端，每个客户端的节点数不超过owner_nodes_len
    #owner_node_ids，字典，键是客户端ID，值是对应客户端包含的节点ID
    owner_node_ids = {owner_id: [] for owner_id in range(args.num_clients)}

    owner_nodes_len = num_nodes // args.num_clients
    owner_list = [i for i in range(args.num_clients)]
    owner_ind = 0

    give_up = 1000

    for group_i in sort_len_dict.keys():
        #对于每个社区，检查当前客户端（由owner_list[owner_ind]指定）的节点数量是否已经达到了owner_nodes_len。如果达到了，那么就将这个客户端从owner_list中移除，并更新当前客户端的索引owner_ind。
        while (
            len(owner_list) >= 2
            and len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len
        ):
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        #检查当前社区的节点数量加上当前客户端已有的节点数量是否超过了owner_nodes_len。
        # 如果超过了，那么就将当前客户端的索引owner_ind加1，即切换到下一个客户端。这个过程会重复进行，直到找到一个客户端，其已有的节点数量加上当前社区的节点数量不超过owner_nodes_len。
        # 为了防止无限循环，设置了一个退出条件，即如果尝试的次数超过了give_up，那么就强制退出循环，并选择当前节点数量最少的客户端作为当前客户端。
        cnt = 0
        while (
            len(owner_node_ids[owner_list[owner_ind]]) +
                len(partition_groups[group_i])
            >= owner_nodes_len + args.louvain_delta
        ):
            owner_ind = (owner_ind + 1) % len(owner_list)
            cnt += 1
            if cnt > give_up:
                cnt = 0
                min_v = 1e15
                for i in range(len(owner_list)):
                    if len(owner_node_ids[owner_list[owner_ind]]) < min_v:
                        min_v = len(owner_node_ids[owner_list[owner_ind]])
                        owner_ind = i
                break
        #将当前社区的节点添加到当前客户端的节点列表中
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
    for key in owner_node_ids.keys():
        #print(key)
        print(key,len(owner_node_ids[key]))

    #with open('client_node_split.txt', 'r') as file:




    local_data = []
    for client_id in range(args.num_clients):
        #根据每个客户端提取当前节点的子图
        local_subgraph = get_subgraph_pyg_data(global_dataset, owner_node_ids[client_id])
        if local_subgraph.edge_index.dim() == 1:
            local_subgraph.edge_index, _ = torch_geometric.utils.add_random_edge(local_subgraph.edge_index.view(2,-1))
        local_data.append(local_subgraph)

    return local_data