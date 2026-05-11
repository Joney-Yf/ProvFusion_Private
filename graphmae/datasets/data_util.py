

from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F
import os
import os.path as osp
import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}

def make_bidirectional_graph(G):
    """
    将DGL有向图转换为双向图，为每条边添加反向边，同时保持节点ID、节点特征和边特征不变。
    
    参数:
    G (dgl.DGLGraph): 输入的DGL有向图
    
    返回:
    dgl.DGLGraph: 包含原始边和反向边的新图
    """
    # 获取原图的节点数量
    num_nodes = G.number_of_nodes()
    
    # 获取原图的所有边，u是源节点列表，v是目标节点列表
    u, v = G.edges()
    
    # 创建双向边：原始边 + 反向边
    src = torch.cat([u, v])  # 拼接原始源节点和反向源节点
    dst = torch.cat([v, u])  # 拼接原始目标节点和反向目标节点
    
    # 创建包含双向边的新图
    new_G = dgl.graph((src, dst), num_nodes=num_nodes)
    
    # 复制节点特征到新图
    for key in G.ndata:
        new_G.ndata[key] = G.ndata[key]
    
    # 复制并扩展边特征到新图（原始边特征 + 原始边特征的副本）
    for key in G.edata:
        # 将原始边特征在第一个维度拼接两次（正向边+反向边）
        new_G.edata[key] = torch.cat([G.edata[key], G.edata[key]], dim=0)
    
    return new_G

def reverse_graph(G):
    """
    反转DGL有向图中所有边的方向，同时保持节点ID、节点特征和边特征不变。
    
    参数:
    G (dgl.DGLGraph): 输入的DGL有向图
    
    返回:
    dgl.DGLGraph: 边方向反转后的新图
    """
    # 获取原图的节点数量
    num_nodes = G.number_of_nodes()
    
    # 获取原图的所有边，u是源节点列表，v是目标节点列表
    u, v = G.edges()
    
    # 创建一个新图，将边的方向反转，即将v作为源节点，u作为目标节点
    new_G = dgl.graph((v, u), num_nodes=num_nodes)
    
    # 复制节点特征到新图
    for key in G.ndata:
        new_G.ndata[key] = G.ndata[key]
    
    # 复制边特征到新图，反转后的边按顺序对应原边的特征
    for key in G.edata:
        new_G.edata[key] = G.edata[key]
    
    return new_G

import dgl
import torch
import math
import networkx as nx
import pymetis
from collections import defaultdict

def separate_connected_components(g):
    """
    分离DGL图的连通组件，并保留原始节点索引。
    
    参数:
    - g: 输入的DGL图
    
    返回:
    - components: 连通组件的列表（每个组件是一个DGL子图）
    - original_node_ids: 每个子图对应的原始节点ID列表
    """
    # 将DGL图转换为NetworkX图（无向图）
    nx_g = dgl.to_networkx(g).to_undirected()
    
    # 获取连通组件
    components = []
    original_node_ids = []
    for cc in nx.connected_components(nx_g):
        # cc 是一个集合，包含原始图中的节点ID
        subgraph = g.subgraph(list(cc))
        components.append(subgraph)
        original_node_ids.append(list(cc))
    return components, original_node_ids

def partition_connected_component(g, original_nodes, initial_index, target_size_range=(10, 20)):
    """
    将一个连通的DGL图分割成多个子图，并生成高层次图。
    
    参数:
    - g: 输入的连通DGL图
    - original_nodes: 子图对应的原始节点ID列表
    - target_size_range: 子图目标大小范围，默认(100, 120)
    
    返回:
    - dict: 包含以下键值对
        - 'subgraphs': 分割后的子图列表（DGL图格式）
        - 'abstract_graph': 高层次图，节点为子图，边表示子图间的信息流
        - 'subgraph_original_nodes': 每个子图对应的原始节点ID列表
    """
    num_nodes = g.number_of_nodes()
    min_size, max_size = target_size_range
    
    # 动态确定分割数量
    num_parts = max(1, num_nodes // max_size)
    if num_nodes / num_parts < min_size:
        num_parts = max(1, num_nodes // min_size)
    
    # 如果num_parts为1，返回原始图和高层次图（只有一个节点）
    if num_parts == 1:
        return {
            'subgraphs': [g],
            'abstract_graph': None,
            'subgraph_original_nodes':[original_nodes]
        }
    
    # 将DGL图转换为NetworkX图
    nx_g = dgl.to_networkx(g).to_undirected()
    
    # 使用pymetis进行图分割
    _, parts = pymetis.part_graph(num_parts, nx_g)
    
    # 提取子图并记录原始节点ID
    subgraphs = []
    subgraph_original_nodes = []
    for i in range(num_parts):
        nodes = [node for node, part in enumerate(parts) if part == i]
        # 映射回原始节点ID
        original_sub_nodes = [original_nodes[node] for node in nodes]
        subgraph = g.subgraph(nodes)
        subgraphs.append(subgraph)
        subgraph_original_nodes.append(original_sub_nodes)
    
    # 构建节点到子图的映射（基于原始节点ID）
    node_to_subgraph = {node: i for i, nodes in enumerate(subgraph_original_nodes) for node in nodes}
    
    # 获取原始图的边
    src, dst = g.edges()
    src = src.numpy()
    dst = dst.numpy()
    
    # 映射回原始节点ID
    original_src = [original_nodes[s] for s in src]
    original_dst = [original_nodes[d] for d in dst]
    
    # 构建高层次图的边
    abstract_edges = set()
    for s, d in zip(original_src, original_dst):
        s_part = node_to_subgraph[s]
        d_part = node_to_subgraph[d]
        if s_part != d_part:
            s_part += initial_index
            d_part += initial_index
            # 添加无向边，确保不重复
            abstract_edges.add((s_part, d_part))
            # abstract_edges.add((d_part, s_part))

    
    return {
        'subgraphs': subgraphs,
        'abstract_graph': abstract_edges,
        'subgraph_original_nodes': subgraph_original_nodes
    }

def partition_graph(g, target_size_range=(500, 2000)):
    """
    将DGL图（可能包含多个连通组件）分割成多个连通子图。
    
    参数:
    - g: 输入的DGL图
    - target_size_range: 子图目标大小范围，默认(500, 2000)
    
    返回:
    - subgraphs: 分割后的子图列表（DGL图格式）
    """
    # 分离连通组件
    components, original_node_ids = separate_connected_components(g)
    print(f"分离出 {len(components)} 个连通组件")
    
    # 对每个连通组件进行分割
    all_subgraphs = []
    subgraph_map_ids = []
    initial_index = 0
    subgraph_edges = []
    for i, component in enumerate(components):
        print(f"处理连通组件 {i+1}/{len(components)}，节点数: {component.number_of_nodes()}, 边数量: {component.number_of_edges()}")
        ret_dict = partition_connected_component(component, original_node_ids[i], initial_index, target_size_range)
        all_subgraphs.extend(ret_dict['subgraphs'])
        subgraph_map_ids.extend(ret_dict['subgraph_original_nodes'])
        initial_index = len(all_subgraphs)
        if ret_dict['abstract_graph'] is not None:
            subgraph_edges.extend(list(ret_dict['abstract_graph']))
    
    return all_subgraphs, subgraph_map_ids, subgraph_edges

def merge_multi_edges_with_reverse(graph):
    """
    将多重有向图中的边合并为单条边，并为每条边添加反向边，形成双向图。
    每条边（包括反向边）都带有 10 维 multi-hot 向量表示边类型，每维度最大值为 1。
    
    参数:
        graph (dgl.DGLGraph): 输入的多重有向图，带有节点特征和边标签。
        
    返回:
        dgl.DGLGraph: 新的双向图，边已压缩并带有 multi-hot 向量特征。
    """
    # 获取边的源节点和目标节点
    u, v = graph.edges()
    # 获取边标签（类型）
    edge_labels = graph.edata['label']
    # 将边组成 (u, v) 对
    edges = torch.stack([u, v], dim=1)
    # 找出唯一的 (u, v) 对及其索引
    unique_edges, inverse_indices = torch.unique(edges, dim=0, return_inverse=True)
    # 将边标签转换为 one-hot 编码
    one_hot_types = torch.nn.functional.one_hot(edge_labels, num_classes=11).float()
    # 初始化 multi-hot 向量张量
    multi_hot = torch.zeros(unique_edges.size(0), 11, device=edge_labels.device)
    
    # 使用 scatter_max 确保每个维度的最大值为 1
    for i in range(one_hot_types.size(0)):
        edge_idx = inverse_indices[i]
        multi_hot[edge_idx] = torch.max(multi_hot[edge_idx], one_hot_types[i])
    
    # 创建新图，保持节点数量
    new_graph = dgl.graph((unique_edges[:,0], unique_edges[:,1]), num_nodes=graph.number_of_nodes())
    
    # 复制节点特征
    for key in graph.ndata:
        new_graph.ndata[key] = graph.ndata[key]
    
    # 设置边的 multi-hot 特征
    new_graph.edata['label'] = multi_hot
    
    # 添加反向边
    # reverse_edges = torch.stack([unique_edges[:,1], unique_edges[:,0]], dim=1)
    # # 反向边的 multi-hot 特征与原始边相同（假设反向边的类型与原始边相同）
    # reverse_multi_hot = multi_hot.clone()
    
    # # 将反向边添加到图中
    # new_graph.add_edges(reverse_edges[:,0], reverse_edges[:,1], {'multi_hot': reverse_multi_hot})
    
    return new_graph

def normalize_l1_torch(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L1 归一化：各元素除以绝对值之和"""
    norm = torch.sum(torch.abs(v), dim=-1, keepdim=True)
    safe_norm = torch.max(norm, torch.tensor(eps, device=v.device))
    return v / safe_norm

def convert_to_dgl_data(g):
    # 定义常量
    emb_dim = 128
    node_type_dim = 3
    edge_type_dim = 10
    
    # 节点去重和映射
    all_nodes, _ = torch.sort(torch.unique(torch.cat([g.src, g.dst])))
    node_to_idx = {node.item(): idx for idx, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    # 批量提取特征
    src_types = g.msg[:, :node_type_dim]
    src_embs = g.msg[:, node_type_dim:node_type_dim + emb_dim]
    dst_types = g.msg[:, node_type_dim + emb_dim + edge_type_dim:node_type_dim + emb_dim + edge_type_dim + node_type_dim]
    dst_embs = g.msg[:, node_type_dim + emb_dim + edge_type_dim + node_type_dim:]
    
    src_features = torch.cat([src_types, src_embs], dim=-1)
    dst_features = torch.cat([dst_types, dst_embs], dim=-1)
    # src_features = src_types
    # dst_features = dst_types
    
    # 初始化节点特征和标签张量
    feature_dim = node_type_dim + emb_dim
    # feature_dim = node_type_dim
    x = torch.zeros(num_nodes, feature_dim)
    y = torch.zeros(num_nodes, dtype=torch.long)
    uuid = torch.zeros(num_nodes, dtype=torch.long)
    
    # 构建节点索引
    src_indices = torch.tensor([node_to_idx[node.item()] for node in g.src])
    dst_indices = torch.tensor([node_to_idx[node.item()] for node in g.dst])
    
    # 赋值源节点特征和标签
    uuid[src_indices] = g.src
    x[src_indices] = src_features
    y[src_indices] = torch.argmax(src_types, dim=1)
    
    # 处理目标节点，仅在未赋值时填充
    mask = (x[dst_indices] == 0).all(dim=1)
    x[dst_indices[mask]] = dst_features[mask]
    y[dst_indices[mask]] = torch.argmax(dst_types[mask], dim=1)
    uuid[dst_indices[mask]] = g.dst[mask]
    
    # 可选：一致性检查
    if not torch.allclose(x[dst_indices[~mask]], dst_features[~mask], atol=1e-6):
        raise ValueError("Some nodes have inconsistent features")
    
    # 构建边索引
    edge_index = torch.stack([src_indices, dst_indices], dim=0).long()
    
    # 构建边特征
    t = g.t.unsqueeze(1)
    edge_types = g.msg[:, node_type_dim + emb_dim:node_type_dim + emb_dim + edge_type_dim]
    edge_attr = torch.cat([t, edge_types], dim=1)
    
    # 创建 DGL 图
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)

    
    # 赋值节点和边特征
    graph.ndata["feat"] = x
    graph.ndata["label"] = y
    graph.ndata['uuid'] = uuid
    graph.edata["label"] = torch.argmax(edge_types, dim=1)
    graph = merge_multi_edges_with_reverse(graph)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    self_edges = torch.nn.functional.one_hot(torch.tensor([10]*len(x)), num_classes=11).float()
    graph.edata['label'][-len(x):] = self_edges
    graph.edata['label'] = normalize_l1_torch(graph.edata['label'])
    # subgraphs, nodes_map_ids,subgraph_edges = partition_graph(graph, target_size_range=(100, 200))

    # # 查看结果

    return graph

def Custimized(path):
    paths= {}
    paths['train'] = osp.join(path, 'train')
    paths['val'] = osp.join(path, 'val')
    paths['test'] = osp.join(path, 'test')

    for path in paths.values():
        if not os.path.exists(path):
            print(f"path {path} doesn't exist")
            return

    graphs = {}
    reverse_graphs = {}
    node_map_ids = {}
    subgraph_edges = {}
    # ground_node_ids = set(torch.load('ground_truth_cadet_v2.pt'))
    # cnt = 1
    edge_counts = torch.zeros(11)
    node_counts = torch.zeros(3)
    for key in paths.keys():
        graphs[key] = []
        reverse_graphs[key] = []
        node_map_ids[key] = []
        subgraph_edges[key] = []
        for root, dirs, files in os.walk(paths[key]):
            for file in tqdm(files,desc=f"preprocessing {key} files"):
                # print(file)
                file_path = os.path.join(root, file)  
                g = torch.load(file_path)
                data_g = convert_to_dgl_data(g)

                num_nodes = data_g.num_nodes()

                # if key=='train':
                #     for label in data_g.edata['label']:
                #         edge_counts += label
                    # for label in data_g.ndata['label']:
                    #     node_counts += label
                    

                # for uuid in data_g.ndata['uuid']:
                #     if uuid.item() in ground_node_ids:
                #         print(f'count {cnt}: uuid: {uuid}')
                #         cnt += 1
                #         ground_node_ids.remove(uuid.item())



                # 根据目录设置掩码
                if key == 'train':
                    data_g.ndata['train_mask'] = torch.ones(num_nodes, dtype=torch.bool)
                    data_g.ndata['val_mask'] = torch.zeros(num_nodes, dtype=torch.bool)
                    data_g.ndata['test_mask'] = torch.zeros(num_nodes, dtype=torch.bool)
                elif key == 'val':
                    data_g.ndata['train_mask'] = torch.zeros(num_nodes, dtype=torch.bool)
                    data_g.ndata['val_mask'] = torch.ones(num_nodes, dtype=torch.bool)
                    data_g.ndata['test_mask'] = torch.zeros(num_nodes, dtype=torch.bool)
                elif key == 'test':
                    data_g.ndata['train_mask'] = torch.zeros(num_nodes, dtype=torch.bool)
                    data_g.ndata['val_mask'] = torch.zeros(num_nodes, dtype=torch.bool)
                    data_g.ndata['test_mask'] = torch.ones(num_nodes, dtype=torch.bool)

                graphs[key].append((data_g))
                reverse_graphs[key].append(reverse_graph(data_g))
                # node_map_ids[key].append(nodes_map)

                # subgraph_edges[key].append(among_edges)
    # print(edge_counts.cpu().tolist())
    # exit()
    return graphs, edge_counts, reverse_graphs

def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name):
    # assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    elif dataset_name not in GRAPH_DICT:
        dataset = None
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
        num_features = graph.ndata["feat"].shape[1]
        num_classes = dataset.num_classes
        weights=None
        node_map_ids=None
        subgraph_edges=None
        reverse_graphs=None
        edge_features = None
    elif dataset_name in (
        'THEIA_E3', 'THEIA_E3_2', 'THEIA_E3_3', 'CADETS_E3', 'CLEARSCOPE_E3',
        'THEIA_E5', 'CLEARSCOPE_E5', 'CADETS_E5', 'CADETS_E3_Mimicry',
        'OPTC_h201', 'OPTC_h051', 'OPTC_h501',
        'THEIA_E3_change_validation_1', 'THEIA_E3_change_validation_2',
        'THEIA_E3_change_validation_3',
    ):
        path = os.path.join('../raw_data', dataset_name)
        graph, weights, reverse_graphs = Custimized(path)
        num_features = graph['train'][0].ndata["feat"].shape[1]
        edge_features = graph['train'][0].edata['label'].shape[1]
        num_classes = 3
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        num_features = graph.ndata["feat"].shape[1]
        num_classes = dataset.num_classes
        weights=None
        node_map_ids=None
        subgraph_edges=None
        reverse_graphs=None
        edge_features = None,
    return (graph, weights), (num_features, num_classes, edge_features), reverse_graphs


def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]
        
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)
