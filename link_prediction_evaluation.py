import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import dgl
from graphmae.utils import create_optimizer, accuracy
from torch.utils.data import Dataset, DataLoader


class EdgeDataset(Dataset):
    def __init__(self, h_src, h_dst, y):
        self.h_src = h_src
        self.h_dst = h_dst
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.h_src[idx], self.h_dst[idx], self.y[idx]


def link_prediction(train_feats, reverse_train_feats, val_feats, real_val_feats, graphs, num_classes, lr_f, weight_decay_f, max_epoch_f, device, weights=None):
    if weights is None:
        raise ValueError(
            "weights must be provided: pass a dataset-specific 11-element tensor "
            "from configs.dataset_weights.DATASET_WEIGHTS."
        )
    weights = weights.to(device)

    in_feat = val_feats[0].shape[1]
    train_graphs = [dgl.remove_self_loop(graph) for graph in graphs['train']]
    test_graphs = [dgl.remove_self_loop(graph) for graph in graphs['test']]
    val_graphs = [dgl.remove_self_loop(graph) for graph in graphs['val']]

    activation = nn.ReLU()
    model = EdgeTypeDecoder(
        in_dim=in_feat,
        num_edge_types=11,
        dropout=0.1,
        num_layers=2,
        activation=activation,
    )

    num_finetune_params = [p.numel() for p in model.parameters() if p.requires_grad]
    print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    model.to(device)
    optimizer = create_optimizer("adam", model, lr_f, weight_decay_f)

    criterion = nn.BCEWithLogitsLoss(reduction='none', weight=weights)
    criterion2 = nn.BCEWithLogitsLoss(reduction='none')

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None
    epoch_iter = tqdm(range(max_epoch_f))

    all_pos_edge_src = []
    all_pos_edge_dst = []
    all_labels_list = []

    print("Preparing data and moving to CPU for parallel loading...")
    for i, graph in enumerate(train_graphs):
        pos_edges = torch.stack(graph.edges()).T
        labels = graph.edata['label']

        h_src_features = train_feats[i][pos_edges[:, 0]].cpu()
        h_dst_features = train_feats[i][pos_edges[:, 1]].cpu()

        all_pos_edge_src.append(h_src_features)
        all_pos_edge_dst.append(h_dst_features)
        all_labels_list.append(labels.cpu())

    h_src_total = torch.cat(all_pos_edge_src, dim=0)
    h_dst_total = torch.cat(all_pos_edge_dst, dim=0)
    labels_total = torch.cat(all_labels_list, dim=0)
    print(f"Total edges to train on: {len(h_src_total)}")

    final_dataset = EdgeDataset(h_src_total, h_dst_total, labels_total)
    final_dataloader = DataLoader(
        final_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
    )

    model.to(device)
    for epoch in epoch_iter:
        model.train()
        for batch in final_dataloader:
            h_src_batch, h_dst_batch, y_batch = [x.to(device) for x in batch]
            out = model(h_src_batch, h_dst_batch)
            loss = criterion(out, y_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}")

    with torch.no_grad():
        model.eval()
        out_test_loss = []
        out_pred = []
        out_edges = []
        out_val_loss = []
        val_edges = []

        for i, val_graph in enumerate(test_graphs):
            val_graph = val_graph.to(device)

            h_src = val_feats[i].cpu()
            h_dst = val_feats[i].cpu()
            pos_edges = torch.stack(val_graph.edges()).T.cpu()
            pos_labels = val_graph.edata['label'].cpu()
            val_dataset = EdgeDataset(h_src[pos_edges[:, 0]], h_dst[pos_edges[:, 1]], pos_labels)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=1024,
                shuffle=False,
                num_workers=32,
                pin_memory=True,
            )
            pred = []
            for batch in val_dataloader:
                val_h_src_batch, val_h_dst_batch, val_y_batch = [x.to(device) for x in batch]
                out = model(val_h_src_batch, val_h_dst_batch)
                pred.append(out)
            pred = torch.cat(pred, dim=0)
            pos_labels = pos_labels.to(device)

            out_pred.append(pred)
            out_test_loss.append(criterion2(pred, pos_labels).mean(dim=-1))
            out_edges.append(pos_edges.to(device))

        for i, val_graph in enumerate(val_graphs):
            val_graph = val_graph.to(device)

            h_src = real_val_feats[i].cpu()
            h_dst = real_val_feats[i].cpu()
            pos_edges = torch.stack(val_graph.edges()).T.cpu()
            pos_labels = val_graph.edata['label'].cpu()
            val_dataset = EdgeDataset(h_src[pos_edges[:, 0]], h_dst[pos_edges[:, 1]], pos_labels)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=1024,
                shuffle=False,
                num_workers=32,
                pin_memory=True,
            )
            pred = []
            for batch in val_dataloader:
                val_h_src_batch, val_h_dst_batch, val_y_batch = [x.to(device) for x in batch]
                out = model(val_h_src_batch, val_h_dst_batch)
                pred.append(out)
            pred = torch.cat(pred, dim=0)
            pos_labels = pos_labels.to(device)

            out_pred.append(pred)
            out_val_loss.append(criterion2(pred, pos_labels).mean(dim=-1))
            val_edges.append(pos_edges.to(device))

    return None, None, out_test_loss, out_val_loss, out_pred, out_edges, val_edges


class EdgeTypeDecoder(nn.Module):
    def __init__(self, in_dim, num_edge_types, dropout, num_layers, activation):
        super(EdgeTypeDecoder, self).__init__()
        if num_layers == 2:
            self.lin_seq = nn.Sequential(
                nn.Linear(in_dim * 2, in_dim * 1),
                nn.Dropout(dropout),
                activation,
                nn.Linear(in_dim * 1, num_edge_types),
            )
        elif num_layers == 3:
            self.lin_seq = nn.Sequential(
                nn.Linear(in_dim * 4, in_dim * 4),
                nn.Dropout(dropout),
                activation,
                nn.Linear(in_dim * 4, in_dim * 2),
                nn.Dropout(dropout),
                activation,
                nn.Linear(in_dim * 2, num_edge_types),
            )
        else:
            raise ValueError(f"Invalid number of layers, found {num_layers}")

    def forward(self, h_src, h_dst, **kwargs):
        h = torch.cat([h_src, h_dst], dim=-1)
        return self.lin_seq(h)
