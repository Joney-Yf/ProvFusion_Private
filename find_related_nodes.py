from collections import defaultdict
import os
import torch
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  

def plot_score_distribution(scores, score_2, filename):
    scores = np.array(scores)
    score_2 = np.array(score_2)

    plt.figure(figsize=(12, 6))
    sns.histplot(scores, bins=100, color='blue', stat='percent', label='All Most Similar')
    sns.histplot(score_2, bins=100, color='orange', stat='percent', label='High Distance')
    plt.legend()
    # plt.xlim(0, 20)  # Fixed x-axis range
    plt.xlabel('Kernel Frequency')
    plt.ylabel('Density')
    plt.title('Kernel Frequency Distribution')
    plt.tight_layout()
    plt.savefig(f'cos_similarity_distribution_{filename}.png')
    plt.close()
    # Optional: Return data for Chart.js
    hist, bin_edges = np.histogram(scores, bins=100)
    return {"bin_edges": bin_edges.tolist(), "counts": hist.tolist()}

def find_related_msg(g, malicious_nodes):
    # 将恶意节点转为集合以加速查找
    malicious_set = set(malicious_nodes)
    # 使用defaultdict存储每个恶意节点的msg列表
    related_msg = defaultdict(list)
    
    # 遍历所有边
    for i in range(len(g.src)):
        src = g.src[i].item()
        dst = g.dst[i].item()
        msg = g.msg[i]
        t = g.t[i]
        # 如果src是恶意节点，添加msg
        if src in malicious_set:
            related_msg[src].append((src, dst, msg, t))
        # 如果dst是恶意节点，添加msg
        if dst in malicious_set:
            related_msg[dst].append((src, dst, msg, t))
    
    # 返回普通字典
    return dict(related_msg)

def collect_all_file_nodes(g):
    # 将恶意节点转为集合以加速查找
    res = []
    # 遍历所有边
    for i in range(len(g.src)):
        msg = g.msg[i]
        # 如果src是恶意节点，添加msg
        emb_dim = 128
        node_type_dim = 3
        edge_type_dim = 10
        
        src_types = msg[:node_type_dim]
        src_embs = msg[:node_type_dim + emb_dim]

        if src_types[2] == 1:
            res.append(src_embs)
    
    # 返回普通字典
    return res


def parse_msg(src, dst, msg, t):
    rel2id = {
        1: 'EVENT_CONNECT',
        'EVENT_CONNECT': 1,
        2: 'EVENT_EXECUTE',
        'EVENT_EXECUTE': 2,
        3: 'EVENT_OPEN',
        'EVENT_OPEN': 3,
        4: 'EVENT_READ',
        'EVENT_READ': 4,
        5: 'EVENT_RECVFROM',
        'EVENT_RECVFROM': 5,
        6: 'EVENT_RECVMSG',
        'EVENT_RECVMSG': 6,
        7: 'EVENT_SENDMSG',
        'EVENT_SENDMSG': 7,
        8: 'EVENT_SENDTO',
        'EVENT_SENDTO': 8,
        9: 'EVENT_WRITE',
        'EVENT_WRITE': 9,
        10: 'EVENT_CLONE',
        'EVENT_CLONE': 10,
    }

    ntype2id ={
        1: 'subject',
        'subject': 1,
        2: 'file',
        'file': 2,
        3: 'netflow',
        'netflow': 3,
    }

    emb_dim = 128
    node_type_dim = 3
    edge_type_dim = 10
    
    src_types = msg[:node_type_dim]
    src_embs = msg[:node_type_dim + emb_dim]
    dst_types = msg[node_type_dim + emb_dim + edge_type_dim:node_type_dim + emb_dim + edge_type_dim + node_type_dim]
    dst_embs = msg[node_type_dim + emb_dim + edge_type_dim:]

    src_type = ntype2id[torch.argmax(src_types).item()+1]
    dst_type = ntype2id[torch.argmax(dst_types).item()+1]

    edge_types = msg[node_type_dim + emb_dim:node_type_dim + emb_dim + edge_type_dim]

    edge_type = rel2id[torch.argmax(edge_types).item()+1]
    

    print(f'{src_type} {src} {edge_type} {dst_type} {dst} {t}')
    return src_type, dst_embs


if __name__ == "__main__":
    path = os.path.join('../raw_data', 'OPTC_h501')
    # ground_node_ids = set(torch.load('ground_truth/_cadet_v2.pt'))
    # print(ground_node_ids)
    # ground_node_ids = [1941786, 1894647, 385249, 380218, 385248, 50522, 50523, 20878, 41377, 41381, 155349, 2104622, 183504, 2104617, 177547, 155322, 379589, 177558]
    # ground_node_ids = list(range(896000, 896000 + 100))  # 假设恶意节点ID为896000到896099
    # ground_node_ids = set(torch.load('high_distance_nodes.pt'))
    ground_node_ids =  [60085, 355406] 



    path_test = osp.join(path, 'test')
    path_train = osp.join(path, 'train')
    # all_file_emb = []
    # for root, dirs, files in os.walk(path_train):
    #     for file in tqdm(files,desc=f"preprocessing train files"):
    #         file_path = os.path.join(root, file)  
    #         g = torch.load(file_path)
    #         related_msg = collect_all_file_nodes(g)
    #         all_file_emb.extend(related_msg)
    #         related_msg = find_related_msg(g, ground_node_ids)
    #         print(file)
    #         for node in related_msg:
    #             print('=' * 20)
    #             print(f'This is for node {node}')
    #             print('=' * 20)
    #             cnt = 0
    #             for (src, dst, msg, t) in related_msg[node]:
    #                 # if dst not in ground_node_ids:
    #                 #     continue
    #                 # if len(related_msg[node]) >1:
    #                 #     continue
    #                 src_type, target_emb = parse_msg(src, dst, msg, t)
    #                 cnt += 1
    #             print(f'number of inedges of {node} is {cnt}')

                
    # all_file_emb = torch.stack(all_file_emb, dim=0).to('cuda:0')
    # print(f'Collected {len(all_file_emb)} file embeddings from train set.')
    # unique_train_emd, inverse_indices = torch.unique(
    #     all_file_emb,
    #     dim=0, 
    #     return_inverse=True
    # )
    all_file_emb_test = []
    high_distance_score = []
    for root, dirs, files in os.walk(path_test):
        for file in tqdm(files,desc=f"preprocessing test files"):
            file_path = os.path.join(root, file)  
            g = torch.load(file_path)
            related_msg = find_related_msg(g, ground_node_ids)
            file_nodes_emb = collect_all_file_nodes(g)
            all_file_emb_test.extend(file_nodes_emb)
            print(file)
            for node in related_msg:
                print('=' * 20)
                print(f'This is for node {node}')
                print('=' * 20)
                cnt = 0
                for (src, dst, msg, t) in related_msg[node]:
                    
                    cnt += 1
                    # if len(related_msg[node]) >1:
                    #     continue
                    src_type, target_emb = parse_msg(src, dst, msg, t)
                    # if src_type != 'file':
                    #     continue
                    # target_emb = target_emb.unsqueeze(0).to('cuda:0')
                    # # 计算与所有文件嵌入的余弦相似度
                    # cos_sim = torch.nn.functional.cosine_similarity(target_emb, all_file_emb, dim=1)
                    # # 获取最相似的文件嵌入索引
                    # top_k_indices = torch.topk(cos_sim, k=1).indices
                    # # 打印最相似的嵌入值
                    # print(f'Top 1 similar file embeddings for node {node}:')
                    # for idx in top_k_indices:
                    #     print(cos_sim[idx].item())
                    #     high_distance_score.append(cos_sim[idx].item())
                print(f'number of inedges of {node} is {cnt}')
    exit()
    all_file_emb_test = torch.stack(all_file_emb_test, dim=0).to('cuda:0')
    unique_test_emb, inverse_indices = torch.unique(
        all_file_emb_test,
        dim=0, 
        return_inverse=True
    )
    print(f'Collected {len(all_file_emb_test)} file embeddings from test set.')
    all_most_smilar = []
    for item in unique_test_emb:
        item = item.unsqueeze(0).to('cuda:0')
        # 计算与所有文件嵌入的余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(item, unique_train_emd, dim=1)
        # 获取最相似的文件嵌入索引
        top_k_indices = torch.topk(cos_sim, k=1).indices
        # 打印最相似的嵌入值
        print(f'Top 1 similar file embeddings for test item:')
        for idx in top_k_indices:
            print(cos_sim[idx].item())     
        all_most_smilar.append(cos_sim[top_k_indices].item())
    
    plot_score_distribution(all_most_smilar, high_distance_score, 'theia_test')

    
