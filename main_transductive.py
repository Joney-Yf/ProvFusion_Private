import logging
import os
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict
import copy
import dgl
import dgl.function as fn
import heapq

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset
from graphmae.models import build_model
from configs.dataset_weights import DATASET_WEIGHTS
from link_prediction_evaluation import link_prediction
from try_different_threshold import method_12_with_different_normalization
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def knn_anomaly_detection(train_graphs, test_graphs, n_neighbors):
    train_features = np.vstack([graph.cpu().numpy() for graph in train_graphs])
    all_test_features = np.vstack([test_graph.cpu().numpy() for test_graph in test_graphs])

    try:
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=100)
        nn.fit(train_features)
        all_distances, _ = nn.kneighbors(all_test_features, n_neighbors=n_neighbors)
        all_scores = all_distances.mean(axis=1)

        result = []
        start_idx = 0
        for test_graph in test_graphs:
            num_nodes = test_graph.size(0)
            score = all_scores[start_idx:start_idx + num_nodes].tolist()
            result.append(score)
            start_idx += num_nodes
    except Exception as e:
        result = None

    return result


def pretrain(model, reverse_graphs, train_graphs, node_map_ids, subgraph_edges, val_graphs, val_feats, optimizer, optimizer_whole_graph, max_epoch, device, scheduler, scheduler_whole_graph, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None, weights=None):
    logging.info("start training..")
    epoch_iter = tqdm(range(max_epoch))
    model.train()
    for epoch in epoch_iter:
        for gi, graph in enumerate(train_graphs):
            graph = graph.to(device)
            x = graph.ndata["feat"].to(device)
            loss, loss_dict = model(graph, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_learning_rate = get_current_lr(optimizer)
            epoch_iter.set_description(
                f"# Epoch {epoch}: train_loss: {loss.item():.4f}, LR: {current_learning_rate:.1e}"
            )

        if scheduler is not None:
            scheduler.step()
    return model


def get_malicious_ranks(malicious_node_ids, loss_dict, method='mean'):
    if method not in ['max', 'mean']:
        raise ValueError("method must be 'max' or 'mean'")

    loss_values = {}
    for uuid, losses in loss_dict.items():
        if losses:
            if method == 'max':
                loss_values[uuid] = max(losses)
            elif method == 'mean':
                loss_values[uuid] = sum(losses) / len(losses)
        else:
            loss_values[uuid] = float('-inf')

    sorted_uuids = sorted(loss_values, key=loss_values.get, reverse=True)
    rank_dict = {uuid: (rank + 1, loss_values[uuid]) for rank, uuid in enumerate(sorted_uuids)}

    malicious_ranks = {}
    for uuid in malicious_node_ids:
        if uuid in rank_dict:
            malicious_ranks[uuid] = rank_dict[uuid][0]
            print(uuid, rank_dict[uuid])
    res = []
    for uuid in rank_dict:
        if rank_dict[uuid][0] < 100 and uuid not in malicious_node_ids:
            res.append(uuid)
    print(res)
    return malicious_ranks


def train(args, graphs):
    device = args.device if args.device >= 0 else "cpu"
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    use_scheduler = args.scheduler

    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(args.optimizer, model, lr, weight_decay)

    if use_scheduler:
        logging.info("Use schedular")
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    val_graphs = graphs['val']
    val_xs = [val_graph.ndata['feat'] for val_graph in val_graphs]

    model.to(device)
    model = pretrain(model, None, graphs['train'], None, None, val_graphs,
                     val_xs, optimizer, None, max_epoch, device, scheduler, None,
                     None, lr_f, weight_decay_f, max_epoch_f, linear_prob, None, None)
    model = model.cpu()
    torch.cuda.empty_cache()

    return model


def compute_top_k_dissimilar_mean(sample_tensors, test_tensors, k, device):
    sample_tensors = [tensor.to(device) for tensor in sample_tensors]
    test_tensors = [tensor.to(device) for tensor in test_tensors]
    S = torch.cat(sample_tensors, dim=0)
    n, d = S.shape
    mean = S.mean(dim=0, keepdim=True)
    S_centered = S - mean
    del S
    torch.cuda.empty_cache()

    S_sq = torch.sum(S_centered ** 2, dim=1)

    node_dissimilarity = []
    for T in test_tensors:
        T_centered = T - mean
        m = T_centered.size(0)
        means_list = []
        t_batch_size = 2048

        for i in range(0, m, t_batch_size):
            t_start, t_end = i, min(i + t_batch_size, m)
            T_batch = T_centered[t_start:t_end]
            T_batch_sq = torch.sum(T_batch ** 2, dim=1, keepdim=True)
            batch_top_k_dists_sq = torch.full((T_batch.size(0), k), float('inf'), device=T.device)

            s_batch_size = 2048
            for j in range(0, n, s_batch_size):
                s_start, s_end = j, min(j + s_batch_size, n)
                S_batch = S_centered[s_start:s_end]
                S_sq_batch = S_sq[s_start:s_end]

                dot_product_chunk = torch.mm(T_batch, S_batch.t())
                dist_sq_chunk = T_batch_sq + S_sq_batch - 2 * dot_product_chunk

                combined_dists = torch.cat([batch_top_k_dists_sq, dist_sq_chunk], dim=1)
                batch_top_k_dists_sq, _ = torch.topk(combined_dists, k, dim=1, largest=False)

            batch_top_k_dists = torch.clamp(batch_top_k_dists_sq, min=0.0)
            means_batch = batch_top_k_dists.mean(dim=1)
            means_list.append(means_batch.cpu())

            del T_batch, T_batch_sq, batch_top_k_dists, means_batch
            torch.cuda.empty_cache()

        means = torch.cat(means_list)
        node_dissimilarity.append(means.cpu().numpy())

    return node_dissimilarity


def calibrate_scores_cpu(benign_scores, test_scores):
    sorted_benign = np.sort(benign_scores)
    ranks = np.searchsorted(sorted_benign, test_scores, side='right')
    calibrated = ranks / len(benign_scores)
    return calibrated


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    optim_type = args.optimizer
    loss_fn = args.loss_fn
    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    if dataset_name not in DATASET_WEIGHTS:
        raise ValueError(
            f"No BCE loss weights configured for dataset '{dataset_name}'. "
            f"Add an entry to configs/dataset_weights.DATASET_WEIGHTS."
        )
    weights = torch.tensor(DATASET_WEIGHTS[dataset_name], dtype=torch.float32)

    (graphs, _, num_features, edge_features, num_classes, reverse_graphs) = torch.load(args.data_path)

    graphs_2 = copy.deepcopy(graphs)
    for split in ('train', 'val', 'test'):
        for graph in graphs[split]:
            graph.ndata['feat'] = graph.ndata['feat'][:, :3]
        for graph in reverse_graphs[split]:
            graph.ndata['feat'] = graph.ndata['feat'][:, :3]

    args.num_features = num_features
    args.edge_in_dim = edge_features
    args.edge_feat_name = 'label'

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        args.num_features = num_features
        model_attribute = train(args, graphs_2)
        args.num_features = 3
        model_pure_structure = train(args, graphs)

        model_pure_structure = model_pure_structure.to(device)
        model_pure_structure.eval()
        model_attribute = model_attribute.to(device)
        model_attribute.eval()

        with torch.no_grad():
            train_x_rep = [model_pure_structure.embed(graph.to(device), graph.ndata['feat'].to(device)).to('cpu') for graph in graphs['train']]
            val_x_rep = [model_pure_structure.embed(graph.to(device), graph.ndata['feat'].to(device)).to('cpu') for graph in graphs['test']]
            real_val_rep = [model_pure_structure.embed(graph.to(device), graph.ndata['feat'].to(device)).to('cpu') for graph in graphs['val']]
            torch.cuda.empty_cache()

            train_x = [graph.ndata['feat'].to('cpu') for graph in graphs_2['train']]
            val_x = [graph.ndata['feat'].to('cpu') for graph in graphs_2['test']]
            real_val_x = [graph.ndata['feat'].to('cpu') for graph in graphs_2['val']]

            train_x_rep_attribute = [model_attribute.embed(graph.to(device), graph.ndata['feat'].to(device)).to('cpu') for graph in graphs_2['train']]
            test_x_rep_attribute = [model_attribute.embed(graph.to(device), graph.ndata['feat'].to(device)).to('cpu') for graph in graphs_2['test']]
            val_x_rep_attribute = [model_attribute.embed(graph.to(device), graph.ndata['feat'].to(device)).to('cpu') for graph in graphs_2['val']]
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        _, _, edge_test_loss, val_test_loss, out_pred, test_edges, val_edges = link_prediction(
            train_x_rep_attribute, reverse_graphs, test_x_rep_attribute, val_x_rep_attribute,
            graphs_2, num_classes, lr_f, weight_decay_f, max_epoch_f, args.device, weights
        )
        torch.cuda.empty_cache()

        print(args)
        node_level_distances_1 = compute_top_k_dissimilar_mean(train_x_rep, val_x_rep, 10, device)
        val_baseline = compute_top_k_dissimilar_mean(train_x_rep, real_val_rep, 10, device)
        node_level_distances_3 = compute_top_k_dissimilar_mean(train_x, val_x, 10, device)
        emb_baseline = compute_top_k_dissimilar_mean(train_x, real_val_x, 10, device)

        total_uuid_to_max = defaultdict(dict)
        val_total_uuid_to_max = defaultdict(dict)

        for gi, graph in enumerate(graphs['test']):
            graph = graph.to(device)
            current_edges = test_edges[gi].T
            src, dst = current_edges
            losses = edge_test_loss[gi]

            num_nodes = graph.num_nodes()
            max_losses_per_node = torch.full((num_nodes,), float('-inf'), device=device)
            max_losses_per_node.scatter_reduce_(dim=0, index=src, src=losses, reduce="amax", include_self=True)
            max_losses_per_node.scatter_reduce_(dim=0, index=dst, src=losses, reduce="amax", include_self=True)

            uuids = graph.ndata['uuid']
            valid_uuids = uuids.cpu().numpy()
            valid_max_losses = max_losses_per_node.cpu().numpy()
            uuid_to_max = dict(zip(valid_uuids, valid_max_losses))
            total_uuid_to_max[gi] = uuid_to_max

        for gi, graph in enumerate(graphs['val']):
            graph = graph.to(device)
            current_edges = val_edges[gi].T
            src, dst = current_edges
            losses = val_test_loss[gi]

            num_nodes = graph.num_nodes()
            edge_loss_baseline = torch.full((num_nodes,), float('-inf'), device=device)
            edge_loss_baseline.scatter_reduce_(dim=0, index=src, src=losses, reduce="amax", include_self=True)
            edge_loss_baseline.scatter_reduce_(dim=0, index=dst, src=losses, reduce="amax", include_self=True)
            edge_loss_baseline = edge_loss_baseline.cpu().numpy()

        A1_benign = calibrate_scores_cpu(val_baseline[0], val_baseline[0])
        A2_benign = calibrate_scores_cpu(emb_baseline[0], emb_baseline[0])
        A3_benign = calibrate_scores_cpu(edge_loss_baseline, edge_loss_baseline)
        max_A3 = np.max(edge_loss_baseline)

        data = (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, max_A3)
        name = f"{dataset_name}_loss_{loss_fn}_dim_{args.num_hidden}_nhd_{args.num_heads}_nh_{args.mask_rate}_nl_{num_layers}_lr_{lr}_lsf_{lr_f}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}"
        os.makedirs('save_middle_results_optc', exist_ok=True)
        results_path = f'save_middle_results_optc/{name}.pt'
        torch.save(data, results_path)

        print(f"\n===== Evaluating with method_12_with_different_normalization =====")
        method_12_with_different_normalization(results_path, args.data_path, args.ground_truth_path)


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    main(args)
