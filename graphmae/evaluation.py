import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import dgl
from graphmae.utils import create_optimizer, accuracy
from torch.utils.data import Dataset, DataLoader

class EdgeDataset(Dataset):
    def __init__(self, h_src, h_dst, y):
        """
        Args:
            h_src (torch.Tensor): 源节点特征矩阵 [num_edges, feature_dim]
            h_dst (torch.Tensor): 目标节点特征矩阵 [num_edges, feature_dim]
            y (torch.Tensor): 边标签 [num_edges]
        """
        self.h_src = h_src
        self.h_dst = h_dst
        self.y = y

    def __len__(self):
        return self.y.shape[0]  # 边的总数

    def __getitem__(self, idx):
        return (
            self.h_src[idx],  # [feature_dim]
            self.h_dst[idx],  # [feature_dim]
            self.y[idx]       # 标量
        )

def node_classification_evaluation(model, train_graphs, graphs, xs, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False, val=True, weights=None):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            x = [model.embed(graph.to(device), xs[i].to(device)) for i, graph in enumerate(graphs)]
            in_feat = x[0].shape[1]
            train_feats = [model.embed(train_graph.to(device), train_graph.ndata['feat'].to(device)) for train_graph in train_graphs]
            graphs = [dgl.remove_self_loop(graph) for graph in graphs]
            edge_x = [(x[i][graph.edges()[0]], x[i][graph.edges()[1]]) for i,graph in enumerate(graphs)]
            # print(graphs[0].edges())
            edge_in_feat = edge_x[0][0].shape[1]
            train_graphs = [dgl.remove_self_loop(g) for g in train_graphs]
            edge_train_feat = [(train_feats[i][train_graph.edges()[0]], train_feats[i][train_graph.edges()[1]]) for i, train_graph in enumerate(train_graphs)]

        encoder = LogisticRegression(in_feat, num_classes)

        activation = nn.ReLU()
        edge_encoder = EdgeTypeDecoder(
            in_dim=edge_in_feat,
            num_edge_types=10,
            dropout=0.1,
            num_layers=2,
            activation=activation,
        )
            
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    edge_encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    optimizer_e = create_optimizer("adam", edge_encoder, lr_f, weight_decay_f)
    # final_acc, estp_acc, test_loss
    final_acc, estp_acc, test_loss = _linear_probing_full_batch(encoder, train_graphs, train_feats, graphs, x, optimizer_f, max_epoch_f, device, mute, val=val)
    edge_final_acc, edge_estp_acc, edge_test_loss, out_pred, val_x_rep= _linear_probing_full_batch_edge(edge_encoder, train_graphs, edge_train_feat, graphs, edge_x, optimizer_e, 6, device, mute, val=val, weights=weights)

    return final_acc, estp_acc, test_loss,edge_final_acc, edge_estp_acc, edge_test_loss, out_pred, edge_x, x, train_feats

def _linear_probing_full_batch_edge(model, train_graphs, train_feats, val_graphs, val_feats, optimizer, max_epoch, device, mute=False,val=True, weights=None):
    if weights is not None:
        
        epsilon = 1e-5
        weights = 1.0 / (weights + epsilon)
        weights[-1] = 0
        weights = weights / weights.sum()  # 归一化
        weights = weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        criterion_2 = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss()
        criterion_2 = torch.nn.CrossEntropyLoss(reduction='none')


    best_val_acc = 0
    best_val_epoch = 0
    best_model = None
    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        for i, graph in enumerate(train_graphs):
            graph = graph.to(device)
            h_src, h_dst = train_feats[i]
            h_src = h_src.to('cpu')
            h_dst = h_dst.to('cpu')
            train_label = graph.edata['label'].to('cpu')
            dataset = EdgeDataset(h_src, h_dst, train_label)
            dataloader = DataLoader(
                dataset,
                batch_size=1024,      # 根据GPU内存调整
                shuffle=False,        # 训练时建议开启
                num_workers=4,       # 加速数据加载
                pin_memory=True      # 加速GPU传输
            )
            for batch in dataloader:
                h_src_batch, h_dst_batch, y_batch = [x.to(device) for x in batch]
                out = model(h_src_batch, h_dst_batch)
                # print(y_batch)
                loss = criterion(out, y_batch)

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
                optimizer.step()

    with torch.no_grad():
        model.eval()
        out_test_loss = []
        out_pred = []
        for i, val_graph in enumerate(val_graphs):
            val_graph = val_graph.to(device)
            h_src, h_dst = val_feats[i]
            val_h_src = h_src.to('cpu')
            val_h_dst = h_dst.to('cpu')
            # train_mask = val_graph.edata["train_mask"]
            # val_mask = val_graph.edata["val_mask"]
            # test_mask = val_graph.edata["test_mask"]
            labels = val_graph.edata["label"].to('cpu')
            val_dataset = EdgeDataset(val_h_src, val_h_dst, labels)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=1024,      # 根据GPU内存调整
                shuffle=False,        # 训练时建议开启
                num_workers=4,       # 加速数据加载
                pin_memory=True      # 加速GPU传输
            )
            pred = []
            for batch in val_dataloader:
                val_h_src_batch, val_h_dst_batch, val_y_batch = [x.to(device) for x in batch]
                _ = model(val_h_src_batch, val_h_dst_batch)
                pred.append(_)
            pred = torch.cat(pred,dim=0)    
            labels = labels.to(device)
            if val:
                val_acc = accuracy(pred, labels)
                val_loss = criterion(pred, labels)
                test_acc = val_acc
                test_loss = val_loss
                out_test_loss.append(test_loss)
            else:
                test_acc = accuracy(pred, labels)
                out_pred.append(pred)
                out_test_loss.append(criterion_2(pred, labels))
                a = criterion_2(pred, labels)
                for i, item in enumerate(a):
                    if item > 3.8:
                        print(pred[i], labels[i], criterion_2(pred[i].unsqueeze(0), labels[i].unsqueeze(0)))
                test_loss = criterion(pred, labels)
                val_acc = test_acc
                val_loss = test_loss

            # if val_acc >= best_val_acc:
            #     best_val_acc = val_acc
            #     best_val_epoch = epoch
            #     best_model = copy.deepcopy(model)

    if not mute:
        epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, test_loss:{val_loss.item(): .4f}")


   
    return None, None, out_test_loss, out_pred, val_feats

def _linear_probing_full_batch(model, train_graphs, train_feats, val_graphs, val_feats, optimizer, max_epoch, device, mute=False,val=True):
    criterion = torch.nn.CrossEntropyLoss()
    criterion_2 = torch.nn.CrossEntropyLoss(reduction='none')

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None
    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)
    for epoch in epoch_iter:
        model.train()
        for i, graph in enumerate(train_graphs):
            graph = graph.to(device)
            x = train_feats[i].to(device)
            train_mask = graph.ndata["train_mask"]
            train_label = graph.ndata['label']
            out = model(graph, x)
            loss = criterion(out[train_mask], train_label[train_mask])
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

        with torch.no_grad():
            model.eval()
            out_test_loss = []
            for i, val_graph in enumerate(val_graphs):
                val_graph = val_graph.to(device)
                val_x = val_feats[i].to(device)

                train_mask = val_graph.ndata["train_mask"]
                val_mask = val_graph.ndata["val_mask"]
                test_mask = val_graph.ndata["test_mask"]
                labels = val_graph.ndata["label"]
                pred = model(val_graph, val_x)
                if val:
                    val_acc = accuracy(pred[val_mask], labels[val_mask])
                    val_loss = criterion(pred[val_mask], labels[val_mask])
                    test_acc = val_acc
                    test_loss = val_loss
                    out_test_loss = test_loss
                else:
                    test_acc = accuracy(pred[test_mask], labels[test_mask])
                    out_test_loss.append(criterion_2(pred[test_mask], labels[test_mask]))
                    test_loss = out_test_loss[-1].mean()
                    val_acc = test_acc
                    val_loss = test_loss

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{val_loss.item(): .4f}, test_acc:{val_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        for i, val_graph in enumerate(val_graphs):
            val_graph = val_graph.to(device)
            val_x = val_feats[i].to(device)

            train_mask = val_graph.ndata["train_mask"]
            val_mask = val_graph.ndata["val_mask"]
            test_mask = val_graph.ndata["test_mask"]
            labels = val_graph.ndata["label"]
            pred = best_model(val_graph, val_x)
        if val:
            estp_val_acc = accuracy(pred[val_mask], labels[val_mask])
            estp_test_acc = estp_val_acc
        else:
            estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
            estp_val_acc = estp_test_acc

    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    return test_acc, estp_test_acc, out_test_loss

def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc


def linear_probing_for_inductive_node_classiifcation(model, x, labels, mask, optimizer, max_epoch, device, mute=False):
    if len(labels.shape) > 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    train_mask, val_mask, test_mask = mask

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)  

        best_val_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(None, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(None, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(None, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}")

    return test_acc, estp_test_acc


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits


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
        logits = self.lin_seq(h)
        
        return logits