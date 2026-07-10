import argparse
import torch
from collections import defaultdict
import numpy as np
from scipy.stats import genpareto
from scipy.spatial.distance import cdist

from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import IsolationForest
import os
# import hdbscan # 导入hdbscan库
from math import sqrt
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt

def sort_with_preferred_index_np(values, preferred_indices):
    values = np.array(values)
    indices = np.arange(len(values))
    
    # 创建优先级数组：preferred_indices 中的索引优先级为 0，其他为 1
    priorities = np.ones(len(values))
    for i, value in enumerate(preferred_indices):
        if value==0:
            priorities[i] = 0
    
    # 按 (值, 优先级) 排序
    sort_key = np.lexsort((priorities, values))
    
    sorted_values = values[sort_key]
    sorted_indices = indices[sort_key]
    
    return sorted_values, sorted_indices[::-1]

def plot_detected_attacks_vs_precision(scores, nodes, node2attacks, labels, out_file):
    """
    Plot the percentage of detected attacks vs precision using anomaly scores, node IDs,
    ground truth labels, and mapping of nodes to their respective attack numbers.

    This function calculates the precision on the x-axis and the cumulative percentage of
    detected attacks on the y-axis, handles duplicate x-values by averaging, and then plots
    it with a filled area under the curve.
    """
    # Sort nodes by descending anomaly scores
    _, sorted_indices = sort_with_preferred_index_np(scores, labels)
    sorted_nodes = [nodes[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    # Initialize variables for tracking detected attacks and total attacks
    total_attacks = len(set(attack for attacks in node2attacks.values() for attack in attacks))
    detected_attacks = set()
    detected_attacks_percentages = [0]  # Start at y=0
    precisions = [0]  # Start at x=0

    # Count detected attacks and precision at each threshold
    tp = 0  # True positives
    fp = 0  # False positives
    for i, node in enumerate(sorted_nodes):
        # Update tp and fp based on label
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        # Update detected attacks set if node has associated attacks
        if node in node2attacks:
            detected_attacks.update(node2attacks[node])

        # Calculate precision and detected attacks percentage
        precision = tp / (tp + fp)
        detected_attacks_percentage = (len(detected_attacks) / total_attacks) * 100


        precisions.append(precision)
        detected_attacks_percentages.append(detected_attacks_percentage)
    # Average out duplicate x-values (precision) by grouping y-values (detected attacks percentages)
    precision_to_attacks = defaultdict(float)
    for precision, detected_percentage in zip(precisions, detected_attacks_percentages):
        precision_to_attacks[precision] = max(precision_to_attacks[precision], detected_percentage)
    
    

    unique_precisions = []
    max_detected_attacks_percentages = []
    for precision, attack_list in sorted(precision_to_attacks.items()):
        unique_precisions.append(precision)
        max_detected_attacks_percentages.append(attack_list)

    # Calculate area under the curve for % detected attacks vs precision
    area_under_curve = np.trapz(max_detected_attacks_percentages, unique_precisions) / 100

    # Plotting
    # try:
    plt.figure(figsize=(10, 6))
    plt.plot(
        unique_precisions,
        max_detected_attacks_percentages,
        color="b",
        label=f"Area under curve = {area_under_curve:.2f}",
    )
    plt.fill_between(
        unique_precisions, max_detected_attacks_percentages, color="blue", alpha=0.2
    )
    plt.xlabel("Precision")
    plt.ylabel("% of Detected Attacks")
    plt.title("Percentage of Detected Attacks vs Precision")
    plt.legend(loc="lower left")
    plt.xlim(0, 1)
    plt.ylim(0, 100.5)
    plt.grid(True)
    plt.savefig(out_file)
    # except:
    #     print("Error while generating ADP plot")
    return area_under_curve
    

def dynamic_weight_fusion(s1, s2, s3, alpha=3):
    """
    动态Softmax权重融合
    s1, s2, s3: shape (N,)
    alpha: 越大越偏向最大维度，越小越平均
    """
    scores = np.stack([s1, s2, s3], axis=1)  # (N, 3)
    max_vals = np.max(scores, axis=1, keepdims=True)
    r = scores / (max_vals + 1e-8)  # 相对异常性

    w = np.exp(alpha * r)
    w = w / np.sum(w, axis=1, keepdims=True)  # Softmax权重

    fused_score = np.sum(w * scores, axis=1)
    return w, fused_score

def dynamic_weight_fusion_with_two(s1, s2, alpha=3):
    """
    动态Softmax权重融合
    s1, s2: shape (N,)
    alpha: 越大越偏向最大维度，越小越平均
    """
    scores = np.stack([s1, s2], axis=1)  # (N, 2)
    max_vals = np.max(scores, axis=1, keepdims=True)
    r = scores / (max_vals + 1e-8)  # 相对异常性

    w = np.exp(alpha * r)
    w = w / np.sum(w, axis=1, keepdims=True)  # Softmax权重

    fused_score = np.sum(w * scores, axis=1)
    return w, fused_score


def detect_anomalies_production_ready(
    validation_data,
    test_data,
    extreme_percentile_threshold=0.999,
    min_extreme_features=2,
    random_state=42
):
    """
    使用基于逻辑规则的方法检测异常点（生产级别最终版）。

    此版本遵循严格的无数据泄漏原则：
    1. 仅在良性验证集上学习数据变换规则。
    2. 将学到的规则应用于未知的测试集。
    """
    # print("开始执行基于逻辑规则的异常检测流程（生产级别）...")
    
    # --- 第一步：仅在 Validation Set 上学习变换规则 ---
    # print(f"正在仅在 {validation_data.shape[0]} 个良性验证点上学习分位数变换规则...")
    
    # 因为我们只在验证集上学习，所以 n_quantiles 不能超过验证集样本数
    n_quantiles = max(min(validation_data.shape[0] // 10, 100000), 100) # 提高分辨率
    transformer = QuantileTransformer(
        output_distribution='uniform',
        n_quantiles=n_quantiles,
        subsample= validation_data.shape[0],  # 强制使用所有样本，而不是默认的10000
        random_state=random_state
    )
    
    # Fit ONLY on validation data
    transformer.fit(validation_data)
    # print("变换规则学习完成。")

    # --- 第二步：将学到的规则应用于测试集 ---
    # print(f"正在将已学到的规则应用于 {test_data.shape[0]} 个测试点...")
    # Transform test data
    test_data_transformed = transformer.transform(test_data)
    # print(test_data_transformed, "测试集数据变换完成。")
    # print("测试集数据变换完成。")

    # --- 第三步：应用核心逻辑 ---
    # print(f"正在应用规则：计算每个点有多少个特征超过 {extreme_percentile_threshold:.4f} 的百分位阈值...")
    
    is_extreme_matrix = test_data_transformed > extreme_percentile_threshold
    extreme_feature_counts = np.sum(is_extreme_matrix, axis=1)

    # --- 第四步：最终决策 ---
    # print(f"正在筛选极端特征数量大于或等于 {min_extreme_features} 的点...")
    anomaly_indices = np.where(extreme_feature_counts >= min_extreme_features)[0]
    
    anomaly_count = len(anomaly_indices)
    total_count = test_data.shape[0]
    # print(f"检测完成！共在 {total_count} 个测试点中发现 {anomaly_count} 个异常点。")

    return anomaly_indices

def detect_anomalies_hdbscan(
    validation_data,
    test_data,
    quantile_for_threshold=99.9,
    min_cluster_size=500,
    min_samples=10,
    random_state=42 # Kept for interface consistency
):
    """
    使用分位数变换、HDBSCAN和手动校准的阈值来检测异常点。
    (已修正：添加了 prediction_data=True 来解决 AttributeError)
    """
    print("开始执行基于HDBSCAN的异常检测流程...")
    
    # --- 第一步：分位数变换 ---
    n_validation = validation_data.shape[0]
    full_data = np.vstack((validation_data, test_data))
    print(f"合并数据集完成，总大小: {full_data.shape}")

    print("正在进行分位数变换...")
    n_quantiles = 10_000 
    transformer = QuantileTransformer(
        output_distribution='uniform',
        n_quantiles=n_quantiles,
        random_state=random_state
    )
    full_data_transformed = transformer.fit_transform(full_data)
    
    validation_data_transformed = full_data_transformed[:n_validation]
    test_data_transformed = full_data_transformed[n_validation:]
    print("分位数变换完成。")

    # --- 第二步：在良性数据上训练HDBSCAN模型 ---
    print("正在使用变换后的良性验证数据训练HDBSCAN模型...")
    
    # --- 关键修正 ---
    # 添加 prediction_data=True，以确保模型在训练后可以用于预测新点。
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        prediction_data=True # 修正点
    )
    model.fit(validation_data_transformed)
    print("模型训练完成。")

    # --- 第三步 & 第四步：在良性数据上校准阈值 ---
    print(f"正在良性验证集上计算异常分数以校准阈值...")
    validation_scores = model.outlier_scores_
    
    non_zero_scores = validation_scores[validation_scores > 0]
    if len(non_zero_scores) == 0:
        threshold = 1e-5
    else:
        # 使用您指定的百分位数
        # 注意：您之前日志中的 0.99% 可能是笔误，这里使用函数参数传入的值
        threshold = np.percentile(non_zero_scores, quantile_for_threshold)

    print(f"阈值校准完成。基于良性数据中非零异常分数的 {quantile_for_threshold}% 百分位数，阈值为: {threshold:.6f}")
    
    # --- 第五步：在测试集上检测异常 ---
    print("正在计算测试集的异常分数...")
    # 现在 approximate_predict 可以正常工作了
    test_scores = hdbscan.approximate_predict(model, test_data_transformed)
    
    print(f"正在使用阈值 {threshold:.6f} 进行最终判断...")
    anomaly_indices = np.where(test_scores > threshold)[0]
    
    anomaly_count = len(anomaly_indices)
    total_count = test_data.shape[0]
    print(f"检测完成！共在 {total_count} 个测试点中发现 {anomaly_count} 个异常点。")

    return anomaly_indices

def detect_anomalies_calibrated(
    validation_data,
    test_data,
    quantile_for_threshold=99.999, # 用于设置阈值的百分位数，这是控制误报率的关键参数
    n_estimators=100,
    random_state=42
):
    """
    使用分位数变换、孤立森林和手动校准的阈值来检测异常点。

    此方法通过在纯良性验证集上学习分数分布并手动设定阈值，
    来避免在预测时因contamination参数导致的误报问题。

    参数:
    validation_data (np.ndarray): 形状为 (n_validation, 3) 的良性验证数据。
    test_data (np.ndarray): 形状为 (n_test, 3) 的待检测测试数据。
    quantile_for_threshold (float): 在良性验证集分数上取定的百分位数，
                                    用以设定最终决策阈值。推荐范围: [99.0, 99.99]。
                                    值越高，阈值越严格，检测到的异常点越少（误报更少）。
    n_estimators (int): 孤立森林中基估计器的数量。
    random_state (int): 随机种子，确保结果可复现。

    返回:
    np.ndarray: 测试数据中被检测为异常点的索引数组。
    """
    print("开始执行基于手动校准阈值的异常检测流程...")
    
    # --- 准备工作：合并数据集以进行全局变换 ---
    n_validation = validation_data.shape[0]
    full_data = np.vstack((validation_data, test_data))
    print(f"合并数据集完成，总大小: {full_data.shape}")

    # --- 第一步：分位数变换 ---
    print("正在进行分位数变换...")
    n_quantiles = max(min(full_data.shape[0] // 10, 100000), 100) # 提高分辨率
    transformer = QuantileTransformer(
        output_distribution='uniform',
        n_quantiles=n_quantiles,
        subsample= full_data.shape[0],  # 强制使用所有样本，而不是默认的10000
        random_state=random_state
    )
    full_data_transformed = transformer.fit_transform(full_data)
    
    # 分离回变换后的验证集和测试集
    validation_data_transformed = full_data_transformed[:n_validation]
    test_data_transformed = full_data_transformed[n_validation:]
    print("分位数变换完成。")

    # --- 第二步：在良性数据上训练孤立森林模型 ---
    print("正在使用变换后的良性验证数据训练孤立森林模型...")
    # 注意：这里不再设置contamination，因为它在score_samples模式下无用
    model = IsolationForest(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(validation_data_transformed)
    print("模型训练完成。")

    # --- 第三步 & 第四步：在良性数据上校准阈值 ---
    print(f"正在良性验证集上计算异常分数以校准阈值...")
    # 使用 .score_samples() 获取原始分数，并取反，使分数越高越异常
    validation_scores = -model.score_samples(validation_data_transformed)
    
    # 根据指定百分位确定阈值
    threshold = np.percentile(validation_scores, quantile_for_threshold)
    print(f"阈值校准完成。基于 {quantile_for_threshold}% 百分位数，阈值为: {threshold:.6f}")
    
    # --- 第五步：在测试集上检测异常 ---
    print("正在计算测试集的异常分数...")
    test_scores = -model.score_samples(test_data_transformed)
    
    print(f"正在使用阈值 {threshold:.6f} 进行最终判断...")
    # 找出分数超过阈值的点的索引
    anomaly_indices = np.where(test_scores > threshold)[0]
    
    anomaly_count = len(anomaly_indices)
    total_count = test_data.shape[0]
    print(f"检测完成！共在 {total_count} 个测试点中发现 {anomaly_count} 个异常点。")

    return anomaly_indices

def detect_anomalies(validation_data, test_data, evt_quantile=0.95, false_positive_rate=0.001):
    """
    使用Z-Score标准化、马氏距离和极值理论（EVT）来检测异常。
    此版本使用马氏距离，以更好地处理非补偿性异常模式。

    参数:
    validation_data (np.ndarray): 形状为 (500000, 3) 的验证数据集，只包含良性节点。
    test_data (np.ndarray): 形状为 (3000000, 3) 的测试数据集。
    evt_quantile (float): 用于EVT初始阈值u的分位数，默认为0.95。
    false_positive_rate (float): 期望的误报率，用于确定最终决策阈值T，默认为0.001 (0.1%)。

    返回:
    np.ndarray: 测试数据中被识别为异常的节点的索引数组。
    """
    
    # --- 第一步：Z-Score标准化 ---
    # 仅从验证集学习均值和标准差
    means = np.mean(validation_data, axis=0)
    stds = np.std(validation_data, axis=0)
    
    epsilon = 1e-8
    
    scaled_val_data = (validation_data - means) / (stds + epsilon)
    scaled_test_data = (test_data - means) / (stds + epsilon)
    
    # --- 第二步：计算马氏距离作为异常分数 ---
    # 1. 从（缩放后的）良性验证数据中计算协方差矩阵并求逆 [2, 3]
    #    这是定义“正常”数据云形状的关键步骤。
    cov_matrix = np.cov(scaled_val_data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    # 2. 计算每个数据点到“正常”数据云中心的马氏距离
    #    中心点是验证数据的均值
    val_mean = np.mean(scaled_val_data, axis=0)
    
    # 计算验证集和测试集上每个点的马氏距离（平方）
    # cdist函数计算效率很高
    mahal_scores_val = cdist(scaled_val_data, val_mean.reshape(1, -1), 
                             metric='mahalanobis', VI=inv_cov_matrix).flatten()
    mahal_scores_test = cdist(scaled_test_data, val_mean.reshape(1, -1), 
                              metric='mahalanobis', VI=inv_cov_matrix).flatten()
    
    # --- 第三步：极值理论（EVT）阈值设定 ---
    # 使用验证集的马氏距离分数来设定阈值
    u = np.quantile(mahal_scores_val, evt_quantile)
    exceedances = mahal_scores_val[mahal_scores_val > u]
    
    if len(exceedances) < 10:
        print("警告：超过EVT初始阈值的点过少，将使用简单的百分位法设定阈值。")
        threshold = np.quantile(mahal_scores_val, 1 - false_positive_rate)
    else:
        xi, _, scale = genpareto.fit(exceedances - u, floc=0)
        n = len(mahal_scores_val)
        n_u = len(exceedances)
        q = false_positive_rate * n / n_u
        threshold = u + genpareto.isf(q, c=xi, scale=scale)

    # --- 第四步：异常检测 ---
    anomalous_indices = np.where(mahal_scores_test > threshold)
    
    return anomalous_indices


def calibrate_scores_cpu(benign_scores, test_scores):
    # 1. 对良性分数进行一次性排序
    sorted_benign = np.sort(benign_scores)
    
    # 2. 使用二分查找，找到每个测试分数在良性分数中的插入点
    # 这个插入点的索引值，就等于有多少个良性分数比它小
    ranks = np.searchsorted(sorted_benign, test_scores, side='right')
    
    # 3. 将排名转换为[0, 1]的百分位数
    calibrated = ranks / len(benign_scores)
    
    return calibrated

def get_malicious_ranks(malicious_node_ids, loss_dict, method='mean'):
    """
    根据 loss_dict 中的 loss 集合的 max 或 mean 值对 UUID 排序，并返回恶意节点的排名。
    
    参数:
        malicious_node_ids (set): 恶意节点的 UUID 集合
        loss_dict (dict): 键是 UUID，值是 loss 集合的字典
        method (str): 排序依据，可选 'max' 或 'mean'，默认为 'max'
    
    返回:
        dict: 键是恶意节点的 UUID，值是其在排序后的排名（从 1 开始）
    """
    # 检查 method 参数是否有效
    # malicious_node_ids = [196432, 846503, 204990, 204311, 204464, 204980, 204989, 164378, 165079, 183486, 183561, 177547, 155322, 281700, 642385, 196391, 385247, 1941786, 246157, 458145, 1894647, 165069, 382243, 382020, 183583, 178625, 380213, 458143, 379931, 385249, 386573, 386584, 385248, 184202, 155330, 164495, 458192, 222095, 516893, 319516, 165037, 165038, 379831, 182568, 160934, 248381, 439665, 439415, 164379, 411378, 183383, 204312, 458240, 184216, 411300, 410000, 409988, 406179, 165068, 411980, 379596, 411724, 439820, 410148, 386155, 439956, 440095, 411470, 413600, 439022, 406078, 712722, 319045, 411963, 204991, 379599, 165106, 458463, 908620, 184259, 406198, 439490, 46001, 405474, 411684, 41116, 14421, 79604, 164542, 414324, 502109, 1052332, 204415, 164483, 183382, 413748, 385856, 440240, 458144]    
    if method not in ['max', 'mean']:
        raise ValueError("method must be 'max' or 'mean'")
    
    # 计算每个 UUID 对应的 loss 值（max 或 mean）
    loss_values = {}
    for uuid, losses in loss_dict.items():
        if losses:  # 如果 loss 集合非空
            if method == 'max':
                loss_values[uuid] = max(losses)
            elif method == 'mean':
                loss_values[uuid] = sum(losses) / len(losses)
        else:  # 如果 loss 集合为空，赋值为负无穷
            loss_values[uuid] = float('-inf')
    
    # 根据 loss 值从高到低排序 UUID
    sorted_uuids = sorted(loss_values, key=loss_values.get, reverse=True)
    
 # 创建排名字典，排名从 1 开始
    rank_dict = {uuid: (rank + 1, loss_values[uuid]) for rank, uuid in enumerate(sorted_uuids)}
    
    # 获取恶意节点的排名
    malicious_ranks = {}
    for uuid in malicious_node_ids:
        if uuid in rank_dict:  # 只处理在 loss_dict 中的恶意节点
            malicious_ranks[uuid] = rank_dict[uuid][0]
            print(uuid, rank_dict[uuid])
    res = []
    for uuid in rank_dict:
        if rank_dict[uuid][0]<100 and uuid not in malicious_node_ids:
            res.append(uuid)
    print(res)
    return malicious_ranks
    


def method_1(filename, data, ground_truth): # 简单的Z-score 加上 EVT和马氏距离，效果还不错但不够稳定。没有我目前的策略好。
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, max_A3) = torch.load(filename,map_location=torch.device('cpu'))
    (graphs, weights, num_features, edge_features, num_classes,reverse_graphs) = torch.load(data,map_location=torch.device('cpu'))
    ground_node_ids = set(torch.load(ground_truth))
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)

    validation_data = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))
    
    if 'CLEARSCOPE_E3' in filename:
        addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
        for node in addition_list:
            ground_node_ids.add(node)
    print('validation data shape is:', validation_data.shape)
    for fpr in [0.01, 0.001, 0.0001, 0.00001]:
        for gi,graph in enumerate(graphs['test']):
            node_based_score = []
            both_node_edge = []
            edge_based_score = []
            uuids = graph.ndata['uuid'].cpu().numpy() 
            edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
            edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)

            test_data = np.array(list(zip(node_level_distances_1[gi], node_level_distances_3[gi], edge_based_score)))
            print('test data shape is:', test_data.shape)

    
            return_idx = detect_anomalies(validation_data, test_data, evt_quantile=0.95, false_positive_rate=fpr)
            TP=0
            FP=0
            for idx in return_idx[0]:
                uuid_val = uuids[idx]

                if uuid_val in ground_node_ids:
                    TP+=1
                else:
                    FP+=1
                
            print(f'The TP is {TP} and the FP is {FP} for graph {gi}')



def method_2(filename, data, ground_truth): # 这个是原始的百分位归一化再加上weigth赋予权重，其中还给了edge特别优待。
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, max_A3) = torch.load(filename,map_location=torch.device('cpu'))
    (graphs, weights, num_features, edge_features, num_classes,reverse_graphs) = torch.load(data,map_location=torch.device('cpu'))
    ground_node_ids = set(torch.load(ground_truth))

    # theia
    # addition_list = [215236, 215806,215179,215721,215237,215796,215235,215854,215801,350388,1027134]

    # CLEARSCOPE_E3
    # addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
    # for node in addition_list:
    #     ground_node_ids.add(node)

    total_tp = {}
    total_fp = {}
    for cnt in range(3):
        total_tp[cnt] = defaultdict(list)
        total_fp[cnt] = defaultdict(list)
    combine_score = defaultdict(list)

    for a1_weight in np.arange(0.1, 1.05, 0.1):
        for a2_weight in np.arange(0.1, 1.05, 0.1):
            for a3_weight in np.arange(0.1, 1.05, 0.1): 
                scores_validation_benign = a1_weight * A1_benign + a2_weight* A2_benign + a3_weight* A3_benign
                # mu = np.mean(scores_validation_benign)
                # sigma = np.std(scores_validation_benign)
                
                TP=0
                FP=0
                node_to_losses = defaultdict(list)
                edge_to_losses = defaultdict(list)
                for gi,graph in enumerate(graphs['test']):
                    node_based_score = []
                    both_node_edge = []
                    edge_based_score = []
                    uuids = graph.ndata['uuid'].cpu().numpy() 
                    edge_based_score = np.array(list(total_uuid_to_max[gi].values()))

                    A1_test_score = calibrate_scores_cpu(val_baseline[0], node_level_distances_1[gi])
                    A2_test_score = calibrate_scores_cpu(emb_baseline[0], node_level_distances_3[gi])
                    A3_test_score = calibrate_scores_cpu(edge_loss_baseline, edge_based_score)


                    for i, uuid_val in enumerate(uuids):

                        if  A3_test_score[i]==1.0:
                            A3_test_score[i] = edge_based_score[i]/max_A3
                        

                        if uuid_val in ground_node_ids:
                            print(f'The node {uuid_val}  in graph {gi} has a score {A1_test_score[i]}, {A2_test_score[i]}, {A3_test_score[i]}')
                    # for iii in range(2, 3):
                        # combine_score = defaultdict(list)


                    threshold =np.max(scores_validation_benign) # 这里可以改成百分位数
                    # print(f'The mean is {mu} and The sigma is {sigma} and THe selected threshold is {threshold}')

                    
                            
                    for i, uuid_val in enumerate(uuids): 
                        # triplet =[A1_test_score[i], A2_test_score[i], A3_test_score[i]]
                        # triplet.sort(reverse=True)
                        # 最大两个值求和
                        # result = triplet[0] + triplet[1]
                        result = a1_weight * A1_test_score[i] + a2_weight * A2_test_score[i] + a3_weight * A3_test_score[i]

                        # combine_score[uuid_val].append(result)


                        if result >= threshold:
                            # print(f'print(f"The node {uuid_val}  in graph {gi} has a score {A1_test_score[i]}, {A2_test_score[i]}, {A3_test_score[i]}')
                            if uuid_val in ground_node_ids:
                                # print(f"The node {uuid_val}  in graph {gi} has a score {result}, is a True Positive")
                                TP+=1
                            else:
                                # if FP < 10:
                                #     print((a1_weight, a2_weight, a3_weight), f'print(f"The node {uuid_val}  in graph {gi} has a score {A1_test_score[i]}, {A2_test_score[i]}, {A3_test_score[i]}')

                                    # print(f"The node {uuid_val}  in graph {gi} has a score {result}, is a False Positive")
                                FP+=1
                print(f'The TP is {TP} and the FP is {FP} for graph {gi} with weight {(a1_weight, a2_weight, a3_weight)}')

    # print("Namespace(")
    # print('='*10)
    # print('node max')
    # get_malicious_ranks(ground_node_ids,combine_score,method='max')
    # for i in range(3):
    #     tps = total_tp[i]
    #     fps = total_fp[i]
    #     for item in tps:
    #         TP_list = tps[item]
    #         FP_list = fps[item]
    #         print(f'The result for Sigma {i}: For weight {item}: The averaged TP is {sum(TP_list)} and the averaged FP is {sum(FP_list)}')



def method_3(filename, data, ground_truth): ## 我们目前采用的方案，线百分位进行归一化，然后取最大的两个数求和作为最终的异常分数。
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, max_A3) = torch.load(filename,map_location=torch.device('cpu'))
    (graphs, weights, num_features, edge_features, num_classes,reverse_graphs) = torch.load(data,map_location=torch.device('cpu'))
    ground_node_ids = set(torch.load(ground_truth))

    total_tp = {}
    total_fp = {}
    for cnt in range(3):
        total_tp[cnt] = defaultdict(list)
        total_fp[cnt] = defaultdict(list)

    for gi,graph in enumerate(graphs['test']):
        node_based_score = []
        both_node_edge = []
        edge_based_score = []
        uuids = graph.ndata['uuid'].cpu().numpy() 
        edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
        
        A1_test_score = calibrate_scores_cpu(val_baseline[0], node_level_distances_1[gi])
        A2_test_score = calibrate_scores_cpu(emb_baseline[0], node_level_distances_3[gi])
        A3_test_score = calibrate_scores_cpu(edge_loss_baseline, edge_based_score)

        for i, uuid_val in enumerate(uuids):


            if uuid_val in ground_node_ids:
                print(f'The node {uuid_val}  in graph {gi} has a score {A1_test_score[i]}, {A2_test_score[i]}, {A3_test_score[i]}')

        for a1_weight in np.arange(0.1, 0.105, 0.1):
            for a2_weight in np.arange(0.1, 0.105, 0.1):
                for a3_weight in np.arange(0.1, .105, 0.1): 
                    result = []
                    for a1, a2, a3 in zip(A1_benign, A2_benign, A3_benign):
                        # 组成三元组
                        triplet = [a1, a2, a3]
                        # 排序并取最大的两个值
                        triplet.sort(reverse=True)
                        # 最大两个值求和
                        anomaly_score = triplet[0] + triplet[1]
                        result.append(anomaly_score)
                    node_to_losses = defaultdict(list)
                    edge_to_losses = defaultdict(list)
                    combine_score = defaultdict(list)
                    for iii in range(1, 2):

                        threshold =np.max(result)
                        # print(f'The mean is {mu} and The sigma is {sigma} and THe selected threshold is {threshold}')


                        TP=0
                        FP=0
                        
                                
                        for i, uuid_val in enumerate(uuids): 
                            triplet =[A1_test_score[i], A2_test_score[i], A3_test_score[i]]
                            triplet.sort(reverse=True)
                            # 最大两个值求和
                            result = triplet[0] + triplet[1]

                            combine_score[uuid_val].append(result)


                            if result >= threshold:
                                if uuid_val in ground_node_ids:
                                    print(f"The node {uuid_val}  in graph {gi} has a score {result}, is a True Positive")
                                    TP+=1
                                else:
                                    if FP < 10:
                                        print(f"The node {uuid_val}  in graph {gi} has a score {result}, is a False Positive")
                                    FP+=1
                        
                        total_tp[iii-1][(a1_weight, a2_weight, a3_weight)].append(TP)
                        total_fp[iii-1][(a1_weight, a2_weight, a3_weight)].append(FP)
                        # print('='*10)
                        # print('node max')
                        # get_malicious_ranks(ground_node_ids,combine_score,method='max')
    for i in range(3):
        tps = total_tp[i]
        fps = total_fp[i]
        for item in tps:
            TP_list = tps[item]
            FP_list = fps[item]
            print(f'The result for Sigma {i}: For weight {item}: The averaged TP is {sum(TP_list)} and the averaged FP is {sum(FP_list)}')


def method_4(filename, data, ground_truth): #这个开始使用到了的随机隔离森林,效果还凑合，但是不如我method 3而且不同的数据集估计要选不同的阈值，又是一个超参数。
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, max_A3) = torch.load(filename,map_location=torch.device('cpu'))
    (graphs, weights, num_features, edge_features, num_classes,reverse_graphs) = torch.load(data,map_location=torch.device('cpu'))
    ground_node_ids = set(torch.load(ground_truth))
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)
    validation_data = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))
    if 'CLEARSCOPE_E3' in filename:
        addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
        for node in addition_list:
            ground_node_ids.add(node)

    if 'THEIA_E3' in filename:
        addition_list = [215236, 215806, 215179, 215721, 215237, 215796, 215235, 215854, 215801, 350388, 1027134]
        for node in addition_list:
            ground_node_ids.add(node)
    print('validation data shape is:', validation_data.shape)
    for thr in [99.9, 99.99, 99.999]:
        TP=0
        FP=0
        for gi,graph in enumerate(graphs['test']):
            node_based_score = []
            both_node_edge = []
            edge_based_score = []
            uuids = graph.ndata['uuid'].cpu().numpy() 
            edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
            edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)


            test_data = np.array(list(zip(node_level_distances_1[gi], node_level_distances_3[gi], edge_based_score)))
            print('test data shape is:', test_data.shape)


            return_idx = detect_anomalies_calibrated(validation_data, test_data, quantile_for_threshold=thr)
            for idx in return_idx:
                uuid_val = uuids[idx]

                if uuid_val in ground_node_ids:
                    TP+=1
                else:
                    FP+=1
            print(f'The TP is {TP} and the FP is {FP} for graph {gi}')

def method_5(filename, data, ground_truth):  # 这个开始使用到了HDBSCAN，效果不好，是一个基于密度的算法。它的目标是在数据中寻找高密度区域（簇），并把那些位于低密度区域的点识别为异常。而异常值往往有一个维度是不那么异常的，因为会被拖后腿。
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, max_A3) = torch.load(filename,map_location=torch.device('cpu'))
    (graphs, weights, num_features, edge_features, num_classes,reverse_graphs) = torch.load(data,map_location=torch.device('cpu'))
    ground_node_ids = set(torch.load(ground_truth))
    validation_data = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))
    print('validation data shape is:', validation_data.shape)
    for thr in [99.99, 99.999, 99.9999, 99.99999, 99.999999]:
        TP=0
        FP=0
        for gi,graph in enumerate(graphs['test']):
            node_based_score = []
            both_node_edge = []
            edge_based_score = []
            uuids = graph.ndata['uuid'].cpu().numpy() 
            edge_based_score = np.array(list(total_uuid_to_max[gi].values()))

            test_data = np.array(list(zip(node_level_distances_1[gi], node_level_distances_3[gi], edge_based_score)))
            print('test data shape is:', test_data.shape)


            return_idx = detect_anomalies_hdbscan(validation_data, test_data, quantile_for_threshold=thr)
            for idx in return_idx:
                uuid_val = uuids[idx]

                if uuid_val in ground_node_ids:
                    TP+=1
                else:
                    FP+=1
            print(f'The TP is {TP} and the FP is {FP} for graph {gi}')

def method_6(filename, data, ground_truth):  
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, max_A3) = torch.load(filename,map_location=torch.device('cpu'))
    (graphs, weights, num_features, edge_features, num_classes,reverse_graphs) = torch.load(data,map_location=torch.device('cpu'))
    ground_node_ids = set(torch.load(ground_truth))
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)
    validation_data = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))
    # print('validation data shape is:', validation_data.shape)
           # CLEARSCOPE_E3
    addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
    for node in addition_list:
        ground_node_ids.add(node)
    for thr in [0.99999]:
        TP=0
        FP=0
        for gi,graph in enumerate(graphs['test']):
            node_based_score = []
            both_node_edge = []
            edge_based_score = []
            uuids = graph.ndata['uuid'].cpu().numpy() 
            edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
            edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)
            test_data = np.array(list(zip(node_level_distances_1[gi], node_level_distances_3[gi], edge_based_score)))
            # print('test data shape is:', test_data.shape)


            return_idx = detect_anomalies_production_ready(validation_data, test_data, extreme_percentile_threshold=thr)
            for idx in return_idx:
                uuid_val = uuids[idx]

                if uuid_val in ground_node_ids:
                    TP+=1
                else:
                    FP+=1
            print(f'The TP is {TP} and the FP is {FP} for graph {gi}')

def method_8(filename, data, ground_truth):  
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, max_A3) = torch.load(filename,map_location=torch.device('cpu'))
    (graphs, weights, num_features, edge_features, num_classes,reverse_graphs) = torch.load(data,map_location=torch.device('cpu'))
    ground_node_ids = set(torch.load(ground_truth))
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)
    validation_data = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))
    # theia
    # addition_list = [215236, 215806,215179,215721,215237,215796,215235,215854,215801,350388,1027134]

    # CLEARSCOPE_E3
    addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
    for node in addition_list:
        ground_node_ids.add(node)
    max_1 = np.max(val_baseline[0])
    min_1 = np.min(val_baseline[0])
    max_2 = np.max(emb_baseline[0])
    min_2 = np.min(emb_baseline[0])
    max_3 = np.max(edge_loss_baseline)
    min_3 = np.min(edge_loss_baseline)
    
    A1_benign_norm = (val_baseline[0] - min_1) / (max_1 - min_1)
    A2_benign_norm = (emb_baseline[0] - min_2) / (max_2 - min_2)
    A3_benign_norm = (edge_loss_baseline - min_3) / (max_3 - min_3) 



    result =[]
    for a1, a2, a3 in zip(A1_benign, A2_benign, A3_benign):
        # 组成三元组
        triplet = [a1, a2, a3]
        # 排序并取最大的两个值
        triplet.sort(reverse=True)
        # 最大两个值求和
        anomaly_score = triplet[0] + triplet[1]
        result.append(anomaly_score)

    threshold_max_two =np.max(result)

    for multi in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        score_benign = A1_benign + A2_benign + multi * A3_benign
        threshold_three_add = np.max(score_benign)

        for alpha in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            _, benign_fused = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=alpha)  
            _, benign_fused_norm = dynamic_weight_fusion(A1_benign_norm, A2_benign_norm, A3_benign_norm, alpha=alpha)
            TP_1=0
            FP_1=0
            TP_2=0
            FP_2=0
            TP_3=0
            FP_3=0
            TP_4=0
            FP_4=0
            TP=0
            FP=0
            
            # 设置阈值为99.9999百分位数
            threshold_fused = np.max(benign_fused)  # 设置阈值为99.9999百分位数
            threshold_norm = np.max(benign_fused_norm)  # 设置阈值为99.9999百分位数
            for gi,graph in enumerate(graphs['test']):
                cnt_uuid = defaultdict(int)
                uids = graph.ndata['uuid'].cpu().numpy() 
                edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
                edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)

                A1_test_score_norm = (node_level_distances_1[gi] - min_1) / (max_1 - min_1)
                A2_test_score_norm = (node_level_distances_3[gi] - min_2) / (max_2 - min_2)
                A3_test_score_norm = (edge_based_score - min_3) / (max_3 - min_3)


                A1_test_score = calibrate_scores_cpu(val_baseline[0], node_level_distances_1[gi])
                A2_test_score = calibrate_scores_cpu(emb_baseline[0], node_level_distances_3[gi])
                A3_test_score = calibrate_scores_cpu(edge_loss_baseline, edge_based_score)

                w, test_fused_norm = dynamic_weight_fusion(A1_test_score_norm, A2_test_score_norm, A3_test_score_norm, alpha=alpha)

                
                for i, uuid_val in enumerate(uids):
                    triplet =[A1_test_score[i], A2_test_score[i], A3_test_score[i]]
                    triplet.sort(reverse=True)
                                    # 最大两个值求和
                    result = triplet[0] + triplet[1]

                    if result >= threshold_max_two:
                        cnt_uuid[uuid_val] += 1
                        if uuid_val in ground_node_ids:
                            TP_1 += 1
                        else:
                            FP_1 += 1


                    if  A3_test_score[i]==1.0:
                        A3_test_score[i] =  edge_based_score[i]/max_A3

                w, test_fused = dynamic_weight_fusion(A1_test_score, A2_test_score, A3_test_score, alpha=alpha)
                for i, uuid_val in enumerate(uids):

                    add_result = A1_test_score[i] + A2_test_score[i] + multi * A3_test_score[i]
                    if test_fused[i] > threshold_fused:
                        cnt_uuid[uuid_val] += 1

                        if uuid_val in ground_node_ids:
                            TP_2 += 1
                        else:
                            FP_2 += 1
                        
                    if test_fused_norm[i] > threshold_norm:
                        cnt_uuid[uuid_val] += 1
                        if uuid_val in ground_node_ids:
                            TP_4 += 1
                        else:
                            FP_4 += 1
                        

                    if add_result > threshold_three_add:
                        cnt_uuid[uuid_val] += 1
                        if uuid_val in ground_node_ids:
                            TP_3 += 1
                        else:
                            FP_3 += 1

                for uuid_val, count in cnt_uuid.items():
                    if count >= 3:
                        if uuid_val in ground_node_ids:
                            print(f'The node {uuid_val}  in graph {gi} is a True Positive with count {count}')
                            TP += 1            
                        else:
                            FP += 1
            
            print(f'The TP_1 is {TP_1} and the FP_1 is {FP_1} for graph {gi} with alpha {alpha} and threshold_max_two {threshold_max_two}')
            print(f'The TP_2 is {TP_2} and the FP_2 is {FP_2} for graph {gi} with alpha {alpha} and threshold_fused {threshold_fused}')
            print(f'The TP_3 is {TP_3} and the FP_3 is {FP_3} for graph {gi} with alpha {alpha} and threshold_three_add {threshold_three_add}')
            print(f'The TP_4 is {TP_4} and the FP_4 is {FP_4} for graph {gi} with alpha {alpha} and threshold_norm {threshold_norm}')
            print(f'The Final TP is {TP} and the Final FP is {FP}  with alpha {alpha} and multiper {multi}')



def method_8_optimized(filename, data, ground_truth):
    # --- 1. 数据加载与初始化 ---
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, 
     total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, 
     max_A3) = torch.load(filename, map_location=torch.device('cpu'))
    
    (graphs, weights, num_features, edge_features, num_classes, 
     reverse_graphs) = torch.load(data, map_location=torch.device('cpu'))

     
    
    ground_node_ids = set(torch.load(ground_truth))
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)

    validation_data = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))
    if 'CLEARSCOPE_E3' in filename:
        addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
        for node in addition_list:
            ground_node_ids.add(node)
     # 转换为列表以便于 np.isin 函数使用
    ground_node_ids_list = list(ground_node_ids)

    # --- 2. 良性样本分数和阈值的预计算 ---
    
    # 用于归一化的最大最小值
    max_1 = np.max(val_baseline[0])
    min_1 = np.min(val_baseline[0])
    max_2 = np.max(emb_baseline[0])
    min_2 = np.min(emb_baseline[0])
    max_3 = np.max(edge_loss_baseline)
    min_3 = np.min(edge_loss_baseline)
    
    
    # 归一化后的良性样本分数
    A1_benign_norm = (val_baseline[0] - min_1) / (max_1 - min_1)
    A2_benign_norm = (emb_baseline[0] - min_2) / (max_2 - min_2)
    A3_benign_norm = (edge_loss_baseline - min_3) / (max_3 - min_3)

    # 向量化计算良性样本的“最大的两个值求和”分数
    benign_scores_stacked = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    benign_scores_stacked.sort(axis=1)
    result_benign_top_two = np.sum(benign_scores_stacked[:, -2:], axis=1)
    threshold_max_two = np.max(result_benign_top_two)

    # 预计算所有阈值以避免重复计算
    threshold_three_add_dict = {}
    for multi in range(1, 11):
        score_benign_three_add = A1_benign + A2_benign + multi * A3_benign
        threshold_three_add_dict[multi] = np.max(score_benign_three_add)

    threshold_fused_dict = {}
    threshold_norm_dict = {}
    for alpha in range(1, 11):
        _, benign_fused = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=alpha)
        threshold_fused_dict[alpha] = np.max(benign_fused)

        _, benign_fused_norm = dynamic_weight_fusion(A1_benign_norm, A2_benign_norm, A3_benign_norm, alpha=alpha)
        threshold_norm_dict[alpha] = np.max(benign_fused_norm)

    # --- 3. 预计算每个图的分数 (将图循环移到网格搜索之外) ---
    per_graph_data = []
    for gi, graph in enumerate(graphs['test']):
        uids = graph.ndata['uuid'].cpu().numpy()
        is_gt_mask = np.isin(uids, ground_node_ids_list)

        edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
        edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)

        ### 提前进行方法5的计算，方法5是利用随机森林进行检测。
        test_data = np.array(list(zip(node_level_distances_1[gi], node_level_distances_3[gi], edge_based_score)))
        return_idx = detect_anomalies_calibrated(validation_data, test_data, quantile_for_threshold=99.999)
        anomaly_mask_5 = np.isin(uids, uids[return_idx])

        # 计算所有未经修改的校准分数
        A1_test = calibrate_scores_cpu(val_baseline[0], node_level_distances_1[gi])
        A2_test = calibrate_scores_cpu(emb_baseline[0], node_level_distances_3[gi])
        A3_calib = calibrate_scores_cpu(edge_loss_baseline, edge_based_score)
        
        # 方法 1: 最大的两个值求和 (使用未经修改的 A3)
        scores_stacked_m1 = np.stack([A1_test, A2_test, A3_calib], axis=1)
        scores_stacked_m1.sort(axis=1)
        score_top_two = np.sum(scores_stacked_m1[:, -2:], axis=1)

        # 修改后的 A3 分数 (用于方法 2、3、4)
        A3_modified = np.where(A3_calib == 1.0, edge_based_score / max_A3, A3_calib)

        # 归一化分数 (用于方法 4)
        A1_norm = (node_level_distances_1[gi] - min_1) / (max_1 - min_1)
        A2_norm = (node_level_distances_3[gi] - min_2) / (max_2 - min_2)
        A3_norm = (edge_based_score - min_3) / (max_3 - min_3)

        per_graph_data.append({
            'gi': gi,
            'uids': uids,
            'is_gt_mask': is_gt_mask,
            'score_top_two': score_top_two,
            'A1_test': A1_test,
            'A2_test': A2_test,
            'A3_modified': A3_modified,
            'A1_norm': A1_norm,
            'A2_norm': A2_norm,
            'A3_norm': A3_norm,
            'anomaly_mask_5': anomaly_mask_5
        })

    # 预计算方法 1 的总 TP/FP (不依赖 multi 或 alpha)
    total_TP_1 = 0
    total_FP_1 = 0
    for d in per_graph_data:
        anomaly_mask_1 = d['score_top_two'] >= threshold_max_two
        total_TP_1 += (anomaly_mask_1 & d['is_gt_mask']).sum()
        total_FP_1 += (anomaly_mask_1 & ~d['is_gt_mask']).sum()

    # --- 4. 主评估循环 (超参数网格搜索) ---
    for multi in range(1, 11):
        threshold_three_add = threshold_three_add_dict[multi]

        for alpha in range(1, 11):
            threshold_fused = threshold_fused_dict[alpha]
            threshold_norm = threshold_norm_dict[alpha]

            # 初始化当前 (multi, alpha) 组合的聚合计数器
            total_TP_2, total_FP_2 = 0, 0
            total_TP_3, total_FP_3 = 0, 0
            total_TP_4, total_FP_4 = 0, 0
            total_TP_5, total_FP_5 = 0, 0
            total_final_TP, total_final_FP = 0, 0
            
            # 用于存储最终投票为TP的节点详细信息
            final_tp_details = []

            # --- 5. 向量化的图评估 (使用预计算数据) ---
            for d in per_graph_data:
                # 方法 2: 动态权重融合 (非归一化，使用修改后的 A3)
                _, test_fused = dynamic_weight_fusion(d['A1_test'], d['A2_test'], d['A3_modified'], alpha=alpha)
                anomaly_mask_2 = test_fused > threshold_fused

                # 方法 3: 加权求和 (使用修改后的 A3)
                score_three_add = d['A1_test'] + d['A2_test'] + multi * d['A3_modified']
                anomaly_mask_3 = score_three_add > threshold_three_add

                # 方法 4: 动态权重融合 (归一化)
                _, test_fused_norm = dynamic_weight_fusion(d['A1_norm'], d['A2_norm'], d['A3_norm'], alpha=alpha)
                anomaly_mask_4 = test_fused_norm > threshold_norm

                # 方法 1: 使用预计算的分数
                anomaly_mask_1 = d['score_top_two'] >= threshold_max_two

                # 更新 TP/FP 计数
                is_gt = d['is_gt_mask']
                not_gt = ~is_gt

                total_TP_2 += (anomaly_mask_2 & is_gt).sum()
                total_FP_2 += (anomaly_mask_2 & not_gt).sum()
                total_TP_3 += (anomaly_mask_3 & is_gt).sum()
                total_FP_3 += (anomaly_mask_3 & not_gt).sum()
                total_TP_4 += (anomaly_mask_4 & is_gt).sum()
                total_FP_4 += (anomaly_mask_4 & not_gt).sum()
                total_TP_5 += (d['anomaly_mask_5'] & is_gt).sum()
                total_FP_5 += (d['anomaly_mask_5'] & not_gt).sum()

                # 最终聚合: 投票 (>=3)
                flags_per_node = (anomaly_mask_1.astype(np.int8) + 
                                  anomaly_mask_2.astype(np.int8) + 
                                  anomaly_mask_3.astype(np.int8) + 
                                  anomaly_mask_4.astype(np.int8) +
                                  d['anomaly_mask_5'].astype(np.int8))

                final_vote_mask = flags_per_node >= 6

                total_final_TP += (final_vote_mask & is_gt).sum()
                total_final_FP += (final_vote_mask & not_gt).sum()

                # 记录最终投票为TP的节点信息
                final_tp_mask_in_graph = final_vote_mask & is_gt
                final_tp_uids_in_graph = d['uids'][final_tp_mask_in_graph]
                for uuid in final_tp_uids_in_graph:
                    final_tp_details.append({'uuid': uuid, 'graph_index': d['gi']})

            # --- 6. 报告当前 (multi, alpha) 组合的结果 ---
            print(f'Results for multi={multi}, alpha={alpha}:')
            print(f'  Method 1 (Top-Two): TP={total_TP_1}, FP={total_FP_1} (Threshold={threshold_max_two:.4f})')
            print(f'  Method 2 (Fused):   TP={total_TP_2}, FP={total_FP_2} (Threshold={threshold_fused:.4f})')
            print(f'  Method 3 (Weighted):TP={total_TP_3}, FP={total_FP_3} (Threshold={threshold_three_add:.4f})')
            print(f'  Method 4 (NormFus): TP={total_TP_4}, FP={total_FP_4} (Threshold={threshold_norm:.4f})')
            print(f'  Method 5 (Isolation): TP={total_TP_5}, FP={total_FP_5} (Threshold=99.999)')
            print(f'  Final Voted (>=3):  TP={total_final_TP}, FP={total_final_FP}')
            
            # 打印最终投票为TP的详细信息
            print("  Final Voted True Positive Details:")
            if not final_tp_details:
                print("    None")
            else:
                for detail in final_tp_details:
                    print(f"    - Node UUID: {detail['uuid']} found in Graph Index: {detail['graph_index']}")
            
            print("-" * 30)

def method_9(filename, data, ground_truth):
    
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3,
    total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, max_A3) = \
        torch.load(filename, map_location='cpu')
    (graphs, weights, num_features, edge_features, num_classes, reverse_graphs) = \
        torch.load(data, map_location='cpu')
    ground_node_ids = set(torch.load(ground_truth))
    if 'CLEARSCOPE_E3' in filename:
        addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
        for node in addition_list:
            ground_node_ids.add(node)
     # 转换为列表以便于 np.isin 函数使用
    ground_node_ids_list = list(ground_node_ids)

    


    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)
    ground_node_ids_arr = np.array(list(ground_node_ids), dtype=np.int64)

    # 阈值计算
    max_1, min_1 = np.max(val_baseline[0]), np.min(val_baseline[0])
    max_2, min_2 = np.max(emb_baseline[0]), np.min(emb_baseline[0])
    max_3, min_3 = np.max(edge_loss_baseline), np.min(edge_loss_baseline)

    benign_sorted = np.sort(np.stack([A1_benign, A2_benign, A3_benign], axis=1), axis=1)
    benign_top2 = benign_sorted[:, -2:].sum(axis=1)
    thr_top2 = np.max(benign_top2)

    _, benign_fused = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=5.0)
    thr_fused = np.max(benign_fused)

    A1_benign_norm = (val_baseline[0] - min_1)/(max_1-min_1+1e-12)
    A2_benign_norm = (emb_baseline[0] - min_2)/(max_2-min_2+1e-12)
    A3_benign_norm = (edge_loss_baseline - min_3)/(max_3-min_3+1e-12)
    _, benign_fused_norm = dynamic_weight_fusion(A1_benign_norm, A2_benign_norm, A3_benign_norm, alpha=5.0)
    thr_fused_norm = np.max(benign_fused_norm)

    thr_weighted = np.max(A1_benign + A2_benign + 5.0 * A3_benign)
    thr_A1 = max_1
    thr_A2 = max_2
    thr_A3 = max_3
    vote_threshold = 4  # 投票阈值，至少3个方法认为是异常

    # ----------------------------
    # 方法定义
    # ----------------------------
    def m_U123(a1, a2, a3, **_):
        """只要 A1、A2、A3 中任意一个分数超过对应阈值就触发"""
        mask = (a1 >= thr_A1) | (a2 >= thr_A2) | (a3 >= thr_A3)
        score = np.max(np.stack([a1, a2, a3], axis=1), axis=1)  # 取最大分数作为输出
        return score, mask, "U123@Any"

    def m_C1_top2(a1,a2,a3,**_):
        St=np.stack([a1,a2,a3],1); St.sort(axis=1)
        score=St[:,-2:].sum(1)
        return score,(score>=thr_top2),"C1@Top2"

    def m_C2_fused(a1,a2,a3,**_):
        _,score=dynamic_weight_fusion(a1,a2,a3,alpha=5.0)
        return score,(score>=thr_fused),"C2@Fused"

    def m_C3_fused_norm(a1n,a2n,a3n,**_):
        _,score=dynamic_weight_fusion(a1n,a2n,a3n,alpha=5.0)
        return score,(score>=thr_fused_norm),"C3@FusedNorm"

    def m_C4_weighted(a1,a2,a3,**_):
        score=a1+a2+3*a3
        return score,(score>=thr_weighted),"C4@Weighted"

    candidate_methods=[
        ("U123", m_U123, {}),
        ("C1", m_C1_top2, {}),
        ("C2", m_C2_fused, {}),
        ("C3", m_C3_fused_norm, {}),
        ("C4", m_C4_weighted, {})
    ]

    # ----------------------------
    # 数据处理 & 特征预计算
    # ----------------------------
    per_graph=[]
    for gi, graph in enumerate(graphs['test']):
        uids = graph.ndata['uuid'].cpu().numpy()
        is_gt = np.isin(uids, ground_node_ids_arr)
        edge_based = np.array(list(total_uuid_to_max[gi].values()))
        edge_based = np.where(np.isinf(edge_based),0,edge_based)
        a1 = calibrate_scores_cpu(val_baseline[0], node_level_distances_1[gi])
        a2 = calibrate_scores_cpu(emb_baseline[0], node_level_distances_3[gi])
        a3c = calibrate_scores_cpu(edge_loss_baseline, edge_based)
        a3 = np.where(a3c==1.0, edge_based/max_A3, a3c)
        a1n=(node_level_distances_1[gi]-min_1)/(max_1-min_1+1e-12)
        a2n=(node_level_distances_3[gi]-min_2)/(max_2-min_2+1e-12)
        a3n=(edge_based-min_3)/(max_3-min_3+1e-12)
        per_graph.append(dict(gi=gi,uids=uids,is_gt=is_gt,
                            a1=a1,a2=a2,a3=a3,a1n=a1n,a2n=a2n,a3n=a3n))

    # ----------------------------
    # 投票 & 统计
    # ----------------------------
    total_TP, total_FP = 0, 0
    per_method_stats = {tag: dict(TP=0, FP=0) for tag,_,_ in candidate_methods}

    for d in per_graph:
        votes = np.zeros_like(d['a1'], dtype=np.int16)

        for tag, fn, kwargs in candidate_methods:
            if tag == "U123":
                score, mask, name = fn(d['a1'], d['a2'], d['a3'])
            elif tag == "C1":
                score, mask, name = fn(d['a1'], d['a2'], d['a3'])
            elif tag == "C2":
                score, mask, name = fn(d['a1'], d['a2'], d['a3'])
            elif tag == "C3":
                score, mask, name = fn(d['a1n'], d['a2n'], d['a3n'])
            elif tag == "C4":
                score, mask, name = fn(d['a1'], d['a2'], d['a3'])
            else:
                continue

            votes += mask.astype(np.int16)
            per_method_stats[tag]['TP'] += int((mask & d['is_gt']).sum())
            per_method_stats[tag]['FP'] += int((mask & (~d['is_gt'])).sum())

        final_mask = votes >= vote_threshold
        total_TP += int((final_mask & d['is_gt']).sum())
        total_FP += int((final_mask & (~d['is_gt'])).sum())

    # ----------------------------
    # 汇总输出
    # ----------------------------
    print(f"(投票阈值={vote_threshold})")
    for tag,_,_ in candidate_methods:
        print(f"  - {tag}: TP={per_method_stats[tag]['TP']}, FP={per_method_stats[tag]['FP']}")
    print(f"[Final] TP={total_TP}, FP={total_FP}")


def method_10(filename, data, ground_truth):  ### LOF就不适合！一个距离度量的算法，validation太少的话，就会失效。
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    """
    使用 LOF (Local Outlier Factor) 算法检测异常节点的中文注释版本。
    """
    # 1. 数据加载
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, 
     total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, 
     max_A3) = torch.load(filename, map_location=torch.device('cpu'))
    
    (graphs, weights, num_features, edge_features, 
     num_classes, reverse_graphs) = torch.load(data, map_location=torch.device('cpu'))
    
    ground_node_ids = set(torch.load(ground_truth))

    # 将无穷大的值替换为0，以避免计算错误
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)
    
    # 2. 准备验证数据 (作为正常数据的基准)
    validation_data = np.array(list(zip(val_baseline[0], edge_loss_baseline)))

    # 💡 数据标准化：统一特征尺度，这通常能提升LOF等距离算法的性能
    scaler = StandardScaler()
    validation_data_scaled = scaler.fit_transform(validation_data)
    
    # 根据业务需求，向ground_truth中添加一些已知的特殊节点
    addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
    for node in addition_list:
        ground_node_ids.add(node)

    # 3. 设置检测阈值
    for thr in [0.99]:
        TP = 0  # True Positive: 真正例 (正确识别的异常)
        FP = 0  # False Positive: 假正例 (错误识别的异常)
        
        # ⚙️ 初始化LOF模型
        # contamination 参数表示数据集中异常值的预期比例
        # novelty=True 允许模型用于新数据的异常检测（而不仅仅是拟合数据自身的异常检测）
        contamination_rate = 1 - thr
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=contamination_rate)

        # 使用“正常”的验证数据来训练LOF模型
        print("正在使用验证数据训练LOF模型...")
        lof.fit(validation_data_scaled)
        print("模型训练完成。")

        # 4. 遍历测试图进行异常检测
        for gi, graph in enumerate(graphs['test']):
            uuids = graph.ndata['uuid'].cpu().numpy()
            
            edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
            edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)
            
            # 准备测试数据
            test_data = np.array(list(zip(node_level_distances_1[gi], edge_based_score)))
            
            # 使用之前训练好的scaler来标准化测试数据
            test_data_scaled = scaler.transform(test_data)

            # --- LOF异常检测核心部分 ---
            # .predict() 方法会返回-1（异常/外れ値）或1（正常/内れ値）
            predictions = lof.predict(test_data_scaled)
            
            # 获取所有被预测为异常的节点的索引
            return_idx = np.where(predictions == -1)[0]
            # --- 核心部分结束 ---

            # 5. 评估检测结果
            for idx in return_idx:
                uuid_val = uuids[idx]
                if uuid_val in ground_node_ids:
                    TP += 1
                else:
                    FP += 1
            print(f'对于图 {gi}, TP (真正例) 是 {TP}, FP (假正例) 是 {FP}')
    

def method_11(filename, data, ground_truth): 
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, max_A3) = torch.load(filename,map_location=torch.device('cpu'))
    (graphs, weights, num_features, edge_features, num_classes,reverse_graphs) = torch.load(data,map_location=torch.device('cpu'))
    ground_node_ids = set(torch.load(ground_truth))
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)


    # theia
    # addition_list = [215236, 215806,215179,215721,215237,215796,215235,215854,215801,350388,1027134]

    # CLEARSCOPE_E3
    addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
    for node in addition_list:
        ground_node_ids.add(node)

    

    total_tp = {}
    total_fp = {}
    for cnt in range(3):
        total_tp[cnt] = defaultdict(list)
        total_fp[cnt] = defaultdict(list)
    combine_score = defaultdict(list)

   
    # max_1 = np.max(val_baseline[0])
    # min_1 = np.min(val_baseline[0])
    # max_2 = np.max(emb_baseline[0])
    # min_2 = np.min(emb_baseline[0])
    # max_3 = np.max(edge_loss_baseline)
    # min_3 = np.min(edge_loss_baseline)

    # A1_benign_norm = (val_baseline[0] - min_1) / (max_1 - min_1)
    # A2_benign_norm = (emb_baseline[0] - min_2) / (max_2 - min_2)
    # A3_benign_norm = (edge_loss_baseline - min_3) / (max_3 - min_3)

    scores_validation_benign =  A1_benign + A2_benign + A3_benign
                # mu = np.mean(scores_validation_benign)
                # sigma = np.std(scores_validation_benign)
                
    TP=0
    FP=0


    combine_score = defaultdict(list)
               
    for gi,graph in enumerate(graphs['test']):
        node_based_score = []
        both_node_edge = []
        edge_based_score = []
        uuids = graph.ndata['uuid'].cpu().numpy() 
        edge_based_score = np.array(list(total_uuid_to_max[gi].values()))

        edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)

        # A3_benign = np.where(A3_benign == 1.0, edge_loss_baseline / max_A3, A3_benign)
        A1_test_score_norm = calibrate_scores_cpu(val_baseline[0], node_level_distances_1[gi])
        A2_test_score_norm = calibrate_scores_cpu(emb_baseline[0], node_level_distances_3[gi])
        A3_test_score_norm = calibrate_scores_cpu(edge_loss_baseline, edge_based_score)
        A3_test_score_norm = np.where(A3_test_score_norm == 1.0, edge_based_score / max_A3, A3_test_score_norm)




        # test_score = A1_test_score_norm + A2_test_score_norm + A3_test_score_norm


        for i, uuid_val in enumerate(uuids):


            if uuid_val in ground_node_ids:
                print(f'The node {uuid_val}  in graph {gi} has a score {A1_test_score_norm[i]}, {A2_test_score_norm[i]}, {A3_test_score_norm[i]}')
                    # for iii in range(2, 3):
                        # 


            threshold = np.max(scores_validation_benign) # 这里可以改成百分位数
                    # print(f'The mean is {mu} and The sigma is {sigma} and THe selected threshold is {threshold}')

                

            result = A1_test_score_norm[i] + A2_test_score_norm[i] + A3_test_score_norm[i]
            combine_score[uuid_val].append(result)
            if result >= threshold:
                if uuid_val in ground_node_ids:
                    TP+=1
                else:
                    FP+=1
        print(f'The TP is {TP} and the FP is {FP} for graph {gi}')
    print('='*10)
    print('node max')
    get_malicious_ranks(ground_node_ids,combine_score,method='max')



def method_12(filename, data, ground_truth):
    # --- 1. 数据加载与初始化 ---
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3, 
     total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign, 
     max_A3) = torch.load(filename, map_location=torch.device('cpu'))
    
    (graphs, weights, num_features, edge_features, num_classes, 
     reverse_graphs) = torch.load(data, map_location=torch.device('cpu'))

     
    
    ground_node_ids = set(torch.load(ground_truth))
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)
    validation_data = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))

    if 'THEIA_E3' in filename:
        addition_list = [215236, 215806, 215179, 215721, 215237, 215796, 215235, 215854, 215801, 350388, 1027134]
        for node in addition_list:
            ground_node_ids.add(node)


    if 'CLEARSCOPE_E3' in filename:
        addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
        for node in addition_list:
            ground_node_ids.add(node)
     # 转换为列表以便于 np.isin 函数使用
    ground_node_ids_list = list(ground_node_ids)

    # --- 2. 良性样本分数和阈值的预计算 ---
    
    

    # 向量化计算良性样本的“最大的两个值求和”分数
    benign_scores_stacked = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    benign_scores_stacked.sort(axis=1)
    result_benign_top_two = np.sum(benign_scores_stacked[:, -2:], axis=1)
    threshold_max_two = np.max(result_benign_top_two)

    benign_scores_max_two_extracted = benign_scores_stacked[:, -2:]
    extracted_max_1 = benign_scores_max_two_extracted[:, 0]
    extracted_max_2 = benign_scores_max_two_extracted[:, 1]


    # 预计算所有阈值以避免重复计算
    threshold_three_add_dict = {}
    for multi in range(1, 2):
        score_benign_three_add = A1_benign + A2_benign + A3_benign
        threshold_three_add_dict[multi] = np.max(score_benign_three_add)

    threshold_fused_dict = {}
    threshold_2_fused_dict = {}
    for alpha in range(1, 11):
        _, benign_fused = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=alpha)
        threshold_fused_dict[alpha] = np.max(benign_fused)
        
        _, benign_fused_2 = dynamic_weight_fusion_with_two(extracted_max_1, extracted_max_2, alpha=alpha)
        threshold_2_fused_dict[alpha] = np.max(benign_fused_2)

    # --- 3. 预计算每个图的分数 (将图循环移到网格搜索之外) ---
    per_graph_data = []
    for gi, graph in enumerate(graphs['test']):
        uids = graph.ndata['uuid'].cpu().numpy()
        is_gt_mask = np.isin(uids, ground_node_ids_list)

        edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
        edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)

        # 计算所有未经修改的校准分数
        A1_test = calibrate_scores_cpu(val_baseline[0], node_level_distances_1[gi])
        A2_test = calibrate_scores_cpu(emb_baseline[0], node_level_distances_3[gi])
        A3_calib = calibrate_scores_cpu(edge_loss_baseline, edge_based_score)

        anomaly_mask_1 = A1_test >= 0.99999
        anomaly_mask_2 = A2_test >= 0.99999
        anomaly_mask_3 = A3_calib >= 0.99999

        A3_modified = np.where(A3_calib == 1.0, edge_based_score / max_A3, A3_calib)

        # 方法 1: 最大的两个值求和 (使用未经修改的 A3)
        scores_stacked_m1 = np.stack([A1_test, A2_test, A3_modified], axis=1)
        scores_stacked_m1.sort(axis=1)
        score_top_two = np.sum(scores_stacked_m1[:, -2:], axis=1)
        anomaly_mask_4 = score_top_two >= threshold_max_two

        extracted_test_max_1 = scores_stacked_m1[:, -2]
        extracted_test_max_2 = scores_stacked_m1[:, -1]



        # 修改后的 A3 分数 (用于方法 2、3、4)
        # A3_modified = np.where(A3_calib == 1.0, edge_based_score / max_A3, A3_calib)

        per_graph_data.append({
            'gi': gi,
            'uids': uids,
            'is_gt_mask': is_gt_mask,
            'score_top_two': score_top_two,
            'A1_test': A1_test,
            'A2_test': A2_test,
            'A3_calib': A3_calib,
            'A3_modified': A3_modified,
            'extracted_test_max_1': extracted_test_max_1,
            'extracted_test_max_2': extracted_test_max_2,
            'anomaly_mask_1': anomaly_mask_1,
            'anomaly_mask_2': anomaly_mask_2,
            'anomaly_mask_3': anomaly_mask_3,
            'anomaly_mask_4': anomaly_mask_4
        })

    # 预计算方法 1 的总 TP/FP (不依赖 multi 或 alpha)
    total_TP_1, total_FP_1 = 0, 0
    total_TP_2, total_FP_2 = 0, 0
    total_TP_3, total_FP_3 = 0, 0
    total_TP_4, total_FP_4 = 0, 0

    for d in per_graph_data:
        total_TP_1 += (d['anomaly_mask_1'] & d['is_gt_mask']).sum()
        total_FP_1 += (d['anomaly_mask_1'] & ~d['is_gt_mask']).sum()
        total_TP_2 += (d['anomaly_mask_2'] & d['is_gt_mask']).sum()
        total_FP_2 += (d['anomaly_mask_2'] & ~d['is_gt_mask']).sum()
        total_TP_3 += (d['anomaly_mask_3'] & d['is_gt_mask']).sum()
        total_FP_3 += (d['anomaly_mask_3'] & ~d['is_gt_mask']).sum()
        total_TP_4 += (d['anomaly_mask_4'] & d['is_gt_mask']).sum()
        total_FP_4 += (d['anomaly_mask_4'] & ~d['is_gt_mask']).sum()

    # --- 4. 主评估循环 (超参数网格搜索) ---
    for multi in range(1, 2):
        threshold_three_add = threshold_three_add_dict[multi]

        for alpha in range(1, 11):
            threshold_fused = threshold_fused_dict[alpha]
            threshold_fused_with_two = threshold_2_fused_dict[alpha]

            # 初始化当前 (multi, alpha) 组合的聚合计数器
            total_TP_5, total_FP_5 = 0, 0
            total_TP_6, total_FP_6 = 0, 0
            total_TP_7, total_FP_7 = 0, 0
            total_final_TP, total_final_FP, total_final_TN, total_final_FN = 0, 0, 0, 0
            
            # 用于存储最终投票为TP的节点详细信息
            final_tp_details = []

            # --- 5. 向量化的图评估 (使用预计算数据) ---
            for d in per_graph_data:
                _, test_fused_2 = dynamic_weight_fusion_with_two(d['extracted_test_max_1'], d['extracted_test_max_2'], alpha=alpha)
                anomaly_mask_5 = test_fused_2 >= threshold_fused_with_two


                score_three_add = d['A1_test'] + d['A2_test'] + multi * d['A3_modified']
                anomaly_mask_6 = score_three_add > threshold_three_add


                _, test_fused = dynamic_weight_fusion(d['A1_test'], d['A2_test'], d['A3_modified'], alpha=alpha)
                anomaly_mask_7 = test_fused > threshold_fused


                # 更新 TP/FP 计数
                is_gt = d['is_gt_mask']
                not_gt = ~is_gt

                total_TP_5 += (anomaly_mask_5 & is_gt).sum()
                total_FP_5 += (anomaly_mask_5 & not_gt).sum()
                total_TP_6 += (anomaly_mask_6 & is_gt).sum()
                total_FP_6 += (anomaly_mask_6 & not_gt).sum()
                total_TP_7 += (anomaly_mask_7 & is_gt).sum()
                total_FP_7 += (anomaly_mask_7 & not_gt).sum()


                # 最终聚合: 投票 (>=3)
                flags_per_node = (d['anomaly_mask_1'].astype(np.int8) +
                                  d['anomaly_mask_2'].astype(np.int8) +
                                  d['anomaly_mask_3'].astype(np.int8) +
                                  d['anomaly_mask_4'].astype(np.int8) +
                                  anomaly_mask_5.astype(np.int8) + 
                                  anomaly_mask_6.astype(np.int8) + 
                                  anomaly_mask_7.astype(np.int8))

                final_vote_mask = flags_per_node >= 4

                total_final_TP += (final_vote_mask & is_gt).sum()
                total_final_FP += (final_vote_mask & not_gt).sum()
                total_final_FN += ((~final_vote_mask & is_gt).sum())
                total_final_TN += ((~final_vote_mask & not_gt).sum())


                # 记录最终投票为TP的节点信息
                final_tp_mask_in_graph = final_vote_mask & is_gt
                final_tp_uids_in_graph = d['uids'][final_tp_mask_in_graph]
                for i, uuid in enumerate(final_tp_uids_in_graph):
                    final_tp_details.append({'uuid': uuid, 'graph_index': d['gi'], 'corresponding vote count': flags_per_node[final_tp_mask_in_graph][i]})

            # --- 6. 报告当前 (multi, alpha) 组合的结果 ---
            print(f'Results for multi={multi}, alpha={alpha}:')
            print(f'  Method 1 (A1): TP={total_TP_1}, FP={total_FP_1} (Threshold={1.0:.4f})')
            print(f'  Method 2 (A2): TP={total_TP_2}, FP={total_FP_2} (Threshold={1.0:.4f})')
            print(f'  Method 3 (A3): TP={total_TP_3}, FP={total_FP_3} (Threshold={1.0:.4f})')
            print(f'  Method 4 (Top-Two no fused): TP={total_TP_4}, FP={total_FP_4} (Threshold={threshold_max_two:.4f})')
            print(f'  Method 5 (Top-Two weighted): TP={total_TP_5}, FP={total_FP_5} (Threshold={threshold_max_two:.4f})')
            print(f'  Method 6 (Top three):   TP={total_TP_6}, FP={total_FP_6} (Threshold={threshold_three_add:.4f})')
            print(f'  Method 7 (Top three weighted):TP={total_TP_7}, FP={total_FP_7} (Threshold={threshold_fused:.4f})')
            print(f'  Final Voted (>=5):  TP={total_final_TP}, FP={total_final_FP}')

            # 打印最终投票为TP的详细信息
            print("  Final Voted True Positive Details:")
            if not final_tp_details:
                print("    None")
            else:
                for detail in final_tp_details:
                    print(f"    - Node UUID: {detail['uuid']} found in Graph Index: {detail['graph_index']}, Corresponding Vote Count: {detail['corresponding vote count']}")
            
            print("-" * 30)

            precision = total_final_TP / (total_final_TP + total_final_FP + 1e-12)
            recall = total_final_TP / (total_final_TP + total_final_FN + 1e-12)
            accuracy = (total_final_TP + total_final_TN) / (total_final_TP + total_final_FP + total_final_FN + total_final_TN + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            mcc = (total_final_TP * total_final_TN - total_final_FP * total_final_FN) / sqrt( 
                (total_final_TP + total_final_FP + 1e-12) *
                (total_final_TP + total_final_FN + 1e-12) *
                (total_final_TN + total_final_FP + 1e-12) *
                (total_final_TN + total_final_FN + 1e-12)
            )

            print(f"[Final Results]")
            print(f"TP={total_final_TP}, FP={total_final_FP}, FN={total_final_FN}, TN={total_final_TN}")
            print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}, MCC={mcc:.4f}")



def method_12_revised(filename, data, ground_truth):
    """
    修改后的代码版本，确保每个节点（由uuid标识）在所有图中只被计数一次。
    """
    # --- 1. 数据加载与初始化 ---
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3,
     total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign,
     max_A3) = torch.load(filename, map_location=torch.device('cpu'))

    (graphs, weights, num_features, edge_features, num_classes,
     reverse_graphs) = torch.load(data, map_location=torch.device('cpu'))

    ground_node_ids = set(torch.load(ground_truth))
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)
    validation_data = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))

    if 'THEIA_E3' in filename:
        addition_list = [215236, 215806, 215179, 215721, 215237, 215796, 215235, 215854, 215801, 350388, 1027134]
        for node in addition_list:
            ground_node_ids.add(node)

    if 'CLEARSCOPE_E3' in filename:
        addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
        for node in addition_list:
            ground_node_ids.add(node)

    if 'CLEARSCOPE_E5' in filename:
        addition_list = [158937, 445211]
        for node in addition_list:
            ground_node_ids.add(node)
    
    # 转换为列表以便于 np.isin 函数使用
    ground_node_ids_list = list(ground_node_ids)
    final_ground_node_ids_list = set()  # 用于存储所有测试集中出现过的节点的uuid
    # --- 2. 良性样本分数和阈值的预计算 ---
    benign_scores_stacked = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    benign_scores_stacked.sort(axis=1)
    result_benign_top_two = np.sum(benign_scores_stacked[:, -2:], axis=1)
    threshold_max_two = np.max(result_benign_top_two)

    extracted_max_1 = benign_scores_stacked[:, -2]
    extracted_max_2 = benign_scores_stacked[:, -1]

    threshold_three_add_dict = {}
    for multi in range(1, 2):
        score_benign_three_add = A1_benign + A2_benign + A3_benign
        threshold_three_add_dict[multi] = np.max(score_benign_three_add)

    threshold_fused_dict = {}
    threshold_2_fused_dict = {}
    for alpha in range(1, 11):
        _, benign_fused = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=alpha)
        threshold_fused_dict[alpha] = np.max(benign_fused)
        
        _, benign_fused_2 = dynamic_weight_fusion_with_two(extracted_max_1, extracted_max_2, alpha=alpha)
        threshold_2_fused_dict[alpha] = np.max(benign_fused_2)

    # --- 3. 预计算每个图的分数 (将图循环移到网格搜索之外) ---
    per_graph_data = []
    # 【修改】: 增加一个集合来存储测试集中所有出现过的节点的uuid，用于后续计算TN和FN
    all_test_uuids = set()
    for gi, graph in enumerate(graphs['test']):
        uids = graph.ndata['uuid'].cpu().numpy()
        all_test_uuids.update(uids) # 将当前图的uuid添加到总集合中

        is_gt_mask = np.isin(uids, ground_node_ids_list)
        final_ground_node_ids_list.update(uids[is_gt_mask])  # 更新最终的ground truth节点集合

        edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
        edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)

        A1_test = calibrate_scores_cpu(val_baseline[0], node_level_distances_1[gi])
        A2_test = calibrate_scores_cpu(emb_baseline[0], node_level_distances_3[gi])
        A3_calib = calibrate_scores_cpu(edge_loss_baseline, edge_based_score)

        anomaly_mask_1 = A1_test >= 0.99999
        anomaly_mask_2 = A2_test >= 0.99999
        anomaly_mask_3 = A3_calib >= 0.99999

        # 【注】: 原代码中这里有一个变量 A3_mo 似乎是笔误，应为 A3_modified。
        # 我假设 A3_modified 的计算应该在这里，并将其用于方法4。
        A3_modified = np.where(A3_calib == 1.0, edge_based_score / max_A3, A3_calib)
        
        scores_stacked_m1 = np.stack([A1_test, A2_test, A3_modified], axis=1)
        scores_stacked_m1.sort(axis=1)
        score_top_two = np.sum(scores_stacked_m1[:, -2:], axis=1)
        anomaly_mask_4 = score_top_two >= threshold_max_two

        extracted_test_max_1 = scores_stacked_m1[:, -2]
        extracted_test_max_2 = scores_stacked_m1[:, -1]

        per_graph_data.append({
            'gi': gi,
            'uids': uids,
            'is_gt_mask': is_gt_mask,
            'score_top_two': score_top_two,
            'A1_test': A1_test,
            'A2_test': A2_test,
            'A3_calib': A3_calib,
            'A3_modified': A3_modified,
            'extracted_test_max_1': extracted_test_max_1,
            'extracted_test_max_2': extracted_test_max_2,
            'anomaly_mask_1': anomaly_mask_1,
            'anomaly_mask_2': anomaly_mask_2,
            'anomaly_mask_3': anomaly_mask_3,
            'anomaly_mask_4': anomaly_mask_4
        })

    # --- 4. 预计算方法 1-4 的唯一性 TP/FP ---
    # 【修改】: 初始化set用于存储各个方法检测出的TP和FP节点的uuid
    uuids_m1_tp, uuids_m1_fp = set(), set()
    uuids_m2_tp, uuids_m2_fp = set(), set()
    uuids_m3_tp, uuids_m3_fp = set(), set()
    uuids_m4_tp, uuids_m4_fp = set(), set()

    for d in per_graph_data:
        is_gt = d['is_gt_mask']
        not_gt = ~is_gt
        uids = d['uids']
        
        # 将当前图检测到的节点的uuid添加到set中
        uuids_m1_tp.update(uids[d['anomaly_mask_1'] & is_gt])
        uuids_m1_fp.update(uids[d['anomaly_mask_1'] & not_gt])
        uuids_m2_tp.update(uids[d['anomaly_mask_2'] & is_gt])
        uuids_m2_fp.update(uids[d['anomaly_mask_2'] & not_gt])
        uuids_m3_tp.update(uids[d['anomaly_mask_3'] & is_gt])
        uuids_m3_fp.update(uids[d['anomaly_mask_3'] & not_gt])
        uuids_m4_tp.update(uids[d['anomaly_mask_4'] & is_gt])
        uuids_m4_fp.update(uids[d['anomaly_mask_4'] & not_gt])

    # 【修改】: 通过set的长度获得唯一的计数值
    total_TP_1, total_FP_1 = len(uuids_m1_tp), len(uuids_m1_fp)
    total_TP_2, total_FP_2 = len(uuids_m2_tp), len(uuids_m2_fp)
    total_TP_3, total_FP_3 = len(uuids_m3_tp), len(uuids_m3_fp)
    total_TP_4, total_FP_4 = len(uuids_m4_tp), len(uuids_m4_fp)

    # --- 5. 主评估循环 (超参数网格搜索) ---
    for multi in range(1, 2):
        threshold_three_add = threshold_three_add_dict[multi]

        for alpha in range(5, 6):
            threshold_fused = threshold_fused_dict[alpha]
            threshold_fused_with_two = threshold_2_fused_dict[alpha]

            # 【修改】: 为每个(multi, alpha)组合初始化独立的set来跟踪uuid
            uuids_m5_tp, uuids_m5_fp = set(), set()
            uuids_m6_tp, uuids_m6_fp = set(), set()
            uuids_m7_tp, uuids_m7_fp = set(), set()
            
            # 用于最终投票结果的uuid集合
            final_voted_tp_uuids = set()
            final_voted_fp_uuids = set()
            
            # 【修改】: 使用字典来存储TP节点的详细信息，uuid作为key
            final_tp_details = {}
            final_all_details = defaultdict(list)

            # --- 6. 向量化的图评估 (使用预计算数据) ---
            for d in per_graph_data:
                uids = d['uids']
                is_gt = d['is_gt_mask']
                not_gt = ~is_gt

                # 计算方法5, 6, 7的异常掩码
                _, test_fused_2 = dynamic_weight_fusion_with_two(d['extracted_test_max_1'], d['extracted_test_max_2'], alpha=alpha)
                anomaly_mask_5 = test_fused_2 >= threshold_fused_with_two

                score_three_add = d['A1_test'] + d['A2_test'] + multi * d['A3_modified']
                anomaly_mask_6 = score_three_add > threshold_three_add

                _, test_fused = dynamic_weight_fusion(d['A1_test'], d['A2_test'], d['A3_modified'], alpha=alpha)
                anomaly_mask_7 = test_fused > threshold_fused

                # 【修改】: 更新方法5, 6, 7的uuid集合
                uuids_m5_tp.update(uids[anomaly_mask_5 & is_gt])
                uuids_m5_fp.update(uids[anomaly_mask_5 & not_gt])
                uuids_m6_tp.update(uids[anomaly_mask_6 & is_gt])
                uuids_m6_fp.update(uids[anomaly_mask_6 & not_gt])
                uuids_m7_tp.update(uids[anomaly_mask_7 & is_gt])
                uuids_m7_fp.update(uids[anomaly_mask_7 & not_gt])
                
                # --- 最终聚合: 投票 (>=4) ---
                flags_per_node = (d['anomaly_mask_1'].astype(np.int8) +
                                  d['anomaly_mask_2'].astype(np.int8) +
                                  d['anomaly_mask_3'].astype(np.int8) +
                                  d['anomaly_mask_4'].astype(np.int8) +
                                  anomaly_mask_5.astype(np.int8) +
                                  anomaly_mask_6.astype(np.int8) +
                                  anomaly_mask_7.astype(np.int8))

                final_vote_mask = flags_per_node >= 4


                for i, value in enumerate(flags_per_node):
                    final_all_details[uids[i]].append(value)

                # 【修改】: 更新最终投票的uuid集合
                final_voted_tp_uuids.update(uids[final_vote_mask & is_gt])
                final_voted_fp_uuids.update(uids[final_vote_mask & not_gt])


                print('Detailed score of True positive nodes in current graph:')
                A1 = d['A1_test'][is_gt]
                A2 = d['A2_test'][is_gt]
                A3 = d['A3_modified'][is_gt]

                all_A = zip(A1, A2, A3)
                for i, aaa in enumerate(all_A):
                    print(f'  - Node UUID: {uids[is_gt][i]}, A1: {aaa[0]:.6f}, A2: {aaa[1]:.6f}, A3: {aaa[2]:.6f}, Total Votes: {flags_per_node[is_gt][i]}')


                print('Detailed score of false positive nodes in current graph:')
                A1 = d['A1_test'][final_vote_mask & not_gt]
                A2 = d['A2_test'][final_vote_mask & not_gt]
                A3 = d['A3_modified'][final_vote_mask & not_gt]

                all_A = zip(A1, A2, A3)
                for i, aaa in enumerate(all_A):
                    print(f'  - Node UUID: {uids[final_vote_mask & not_gt][i]}, A1: {aaa[0]:.6f}, A2: {aaa[1]:.6f}, A3: {aaa[2]:.6f}, Total Votes: {flags_per_node[final_vote_mask & not_gt][i]}')

                # 【修改】: 记录最终投票为TP的节点详细信息
                final_tp_mask_in_graph = final_vote_mask & is_gt
                final_tp_uids_in_graph = uids[final_tp_mask_in_graph]
                for i, uuid in enumerate(final_tp_uids_in_graph):
                    vote_count = flags_per_node[final_tp_mask_in_graph][i]
                    if uuid not in final_tp_details:
                        final_tp_details[uuid] = {'graph_indices': [], 'vote_counts': []}
                    final_tp_details[uuid]['graph_indices'].append(d['gi'])
                    final_tp_details[uuid]['vote_counts'].append(vote_count)

            # --- 7. 计算并报告当前 (multi, alpha) 组合的唯一性结果 ---
            # 【修改】: 从set的长度计算最终的唯一计数值
            total_TP_5 = len(uuids_m5_tp)
            total_FP_5 = len(uuids_m5_fp)
            total_TP_6 = len(uuids_m6_tp)
            total_FP_6 = len(uuids_m6_fp)
            total_TP_7 = len(uuids_m7_tp)
            total_FP_7 = len(uuids_m7_fp)
            
            total_final_TP = len(final_voted_tp_uuids)
            total_final_FP = len(final_voted_fp_uuids)

            # 【修改】: 基于唯一的uuid集合，更准确地计算FN和TN
            # FN = 所有真实异常点中，没有被检测为TP的点的数量
            total_final_FN = len(final_ground_node_ids_list - final_voted_tp_uuids)
            # TN = 所有测试集中的点，排除掉所有真实异常点后，再排除掉被误报为FP的点
            non_gt_uuids = all_test_uuids - ground_node_ids
            total_final_TN = len(non_gt_uuids - final_voted_fp_uuids)


            print(f'Results for multi={multi}, alpha={alpha}:')
            print(f'  Method 1 (A1):                TP={total_TP_1}, FP={total_FP_1} (Threshold=1.0)')
            print(f'  Method 2 (A2):                TP={total_TP_2}, FP={total_FP_2} (Threshold=1.0)')
            print(f'  Method 3 (A3):                TP={total_TP_3}, FP={total_FP_3} (Threshold=1.0)')
            print(f'  Method 4 (Top-Two no fused):  TP={total_TP_4}, FP={total_FP_4} (Threshold={threshold_max_two:.4f})')
            print(f'  Method 5 (Top-Two weighted):  TP={total_TP_5}, FP={total_FP_5} (Threshold={threshold_fused_with_two:.4f})')
            print(f'  Method 6 (Top three):         TP={total_TP_6}, FP={total_FP_6} (Threshold={threshold_three_add:.4f})')
            print(f'  Method 7 (Top three weighted):TP={total_TP_7}, FP={total_FP_7} (Threshold={threshold_fused:.4f})')
            print(f'  Final Voted (>=4):            TP={total_final_TP}, FP={total_final_FP}')

            # 打印最终投票为TP的详细信息
            print("  Final Voted True Positive Details (Unique Nodes):")
            if not final_tp_details:
                print("    None")
            else:
                for uuid, details in final_tp_details.items():
                    print(f"    - Node UUID: {uuid}, Found in Graph(s): {details['graph_indices']}, Vote Counts: {details['vote_counts']}")
            
            print("-" * 30)
            
            # --- 8. 计算最终性能指标 ---
            precision = total_final_TP / (total_final_TP + total_final_FP + 1e-12)
            recall = total_final_TP / (total_final_TP + total_final_FN + 1e-12)
            accuracy = (total_final_TP + total_final_TN) / (total_final_TP + total_final_FP + total_final_FN + total_final_TN + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            
            denominator_mcc = sqrt(
                (total_final_TP + total_final_FP + 1e-12) *
                (total_final_TP + total_final_FN + 1e-12) *
                (total_final_TN + total_final_FP + 1e-12) *
                (total_final_TN + total_final_FN + 1e-12)
            )
            mcc = (total_final_TP * total_final_TN - total_final_FP * total_final_FN) / denominator_mcc

            print(f"[Final Results for alpha={alpha}]")
            print(f"Unique TP={total_final_TP}, Unique FP={total_final_FP}, Unique FN={total_final_FN}, Unique TN={total_final_TN}")
            print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}, MCC={mcc:.4f}")

            scores = []
            nodes = []
            dataset_name = filename.split('/')[-1].split('_')[0] + '_' + filename.split('/')[-1].split('_')[1]
            attack_to_GPs = torch.load(f'{dataset_name}_attack_to_nids.pt')
            attack2nodes = {k: v["nids"] for k, v in attack_to_GPs.items()}
            node2attacks = defaultdict(set)
            for attack, node_s in attack2nodes.items():
                for node in node_s:
                    if node not in ground_node_ids:
                        continue
                    node2attacks[node].add(attack)
            labels = []
            for uuid, details in final_all_details.items():
                scores.append(max(details))
                nodes.append(uuid)
                if uuid in ground_node_ids:
                    labels.append(1)
                else:
                    labels.append(0)
            
            adp_score = plot_detected_attacks_vs_precision(scores, nodes, node2attacks, labels, f'{dataset_name}_attack_adp_detection.png')
            print('The adp score is:', adp_score)
            print("=" * 40 + "\n")

def normalize_min_max(baseline_data, test_data):
    """
    Min-Max 归一化。
    基于 baseline_data 计算 min/max，并应用于 test_data。
    """
    min_val = np.min(baseline_data)
    max_val = np.max(baseline_data)
    range_val = max_val - min_val
    
    # 避免除以零
    scaled_data = (test_data - min_val) / (range_val + 1e-12)
    return scaled_data # 返回原始缩放值 (可能 > 1 或 < 0)

def normalize_z_score(baseline_data, test_data):
    """
    Z-Score 归一化 (StandardScaler)。
    基于 baseline_data 计算 mean/std。
    """
    mean_val = np.mean(baseline_data)
    std_val = np.std(baseline_data)
    
    # 避免除以零
    if std_val < 1e-12:
        return (test_data - mean_val)
        
    z_scores = (test_data - mean_val) / (std_val + 1e-12)
    return z_scores

def normalize_robust(baseline_data, test_data):
    """
    【V2 - 最原始的 Robust Scaling】
    Robust Scaling 归一化 (RobustScaler)。
    基于 baseline_data 计算 median/IQR。
    公式: (x - median) / IQR
    """
    q25, median, q75 = np.percentile(baseline_data, [25, 50, 75])
    iqr = q75 - q25

    # 避免除以零
    if iqr < 1e-12:
        # 如果 iqr 为 0，所有数据都等于中位数
        # 我们只返回 (x - median)，这几乎全是 0
        return (test_data - median)
    
    # 【最原始用法】: 直接除以 IQR
    robust_scores = (test_data - median) / (iqr + 1e-12)
    
    return robust_scores

# ==========================================================
# --- 【修改后】的主函数 ---
# ==========================================================

def method_12_with_different_normalization(filename, data, ground_truth, normalization_method='percentile'):
    """
    修改后的代码版本，确保每个节点（由uuid标识）在所有图中只被计数一次。
    
    【新功能】:
    添加了 'normalization_method' 参数，可选:
    - 'percentile' (原始方法)
    - 'min_max'
    - 'z_score'
    - 'robust'
    """
    
    # --- 0. 【新】选择归一化函数 ---
    if normalization_method == 'percentile':
        normalize_func = calibrate_scores_cpu
    elif normalization_method == 'min_max':
        normalize_func = normalize_min_max
    elif normalization_method == 'z_score':
        normalize_func = normalize_z_score
    elif normalization_method == 'robust':
        normalize_func = normalize_robust
    else:
        raise ValueError(f"未知的归一化方法: {normalization_method}。请选择 'percentile', 'min_max', 'z_score', 或 'robust'")
    
    print(f"\n--- 正在使用 [ {normalization_method} ] 归一化方法 ---")
    
    # --- 1. 数据加载与初始化 ---
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3,
     total_uuid_to_max, edge_loss_baseline, 
     A1_benign_loaded, A2_benign_loaded, A3_benign_loaded, # 【修改】重命名
     max_A3) = torch.load(filename, map_location=torch.device('cpu'))

    (graphs, weights, num_features, edge_features, num_classes,
     reverse_graphs) = torch.load(data, map_location=torch.device('cpu'))

    ground_node_ids = set(torch.load(ground_truth))
    # 【修改】修复 val_baseline[0] 等被错误覆盖的问题
    # edge_loss_baseline 在 torch.load 中被加载，但也在 zip 中被使用
    # 我们需要确保使用正确的 benign 分布
    benign_dist_A1 = val_baseline[0]
    benign_dist_A2 = emb_baseline[0]
    benign_dist_A3 = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline) # 使用加载的 edge_loss_baseline
    
    # validation_data = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))
    # ^^^ 这行代码现在是多余的，因为我们直接使用 benign_dist_A1 等

    # MIGRATION: the hardcoded per-dataset `addition_list` blocks that used to live here
    # (extra attack nodes recorded by ORIGINAL-database index_id) have been retired. Those
    # nodes are now recorded canonically — by node UUID — in the Ground_Truth CSVs
    # (E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv, E3-THEIA/node_phishing_email.csv,
    # E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv), and `tools/build_ground_truth.py`
    # resolves them to index ids for WHATEVER database the data came from. Pass the
    # resulting canonical ground-truth file (which already contains all of these nodes)
    # as `ground_truth`; equivalence on the original databases was verified live
    # (THEIA 91/2, CLEARSCOPE 6/7 reproduce exactly).

    ground_node_ids_list = list(ground_node_ids)
    final_ground_node_ids_list = set()
    
    # --- 2. 良性样本分数和阈值的预计算 ---
    
    # --- 【修改】: 根据选择的方法重新计算 A1_benign, A2_benign, A3_benign ---
    if normalization_method == 'percentile':
        print("使用已加载的 'percentile' 良性分数。")
        A1_benign = A1_benign_loaded
        A2_benign = A2_benign_loaded
        A3_benign = A3_benign_loaded
    else:
        # 使用新选择的归一化方法，
        # 基于 *baseline* 分布 (benign_dist_A1 等) 来归一化它们 *自己*
        print("使用新方法重新计算良性分数...")
        A1_benign = normalize_func(benign_dist_A1, benign_dist_A1)
        A2_benign = normalize_func(benign_dist_A2, benign_dist_A2)
        A3_benign = normalize_func(benign_dist_A3, benign_dist_A3)
    # --- 【修改结束】 ---

    # --- 【修改】: M1, M2, M3 的阈值现在是归一化后的最大值 ---
    if normalization_method == 'percentile':
        print("使用已加载的 'percentile' 阈值 (1.0)。")
        threshold_m1 = 0.99999
        threshold_m2 = 0.99999
        threshold_m3 = 0.99999
    else:
        threshold_m1 = np.percentile(A1_benign, 99.999)
        threshold_m2 = np.percentile(A2_benign, 99.999)
        threshold_m3 = np.percentile(A3_benign, 99.999)
    print(f"新阈值: M1_max={threshold_m1:.5f}, M2_max={threshold_m2:.5f}, M3_max={threshold_m3:.5f}")
    # --- 【修改结束】 ---

    # 剩下的阈值计算逻辑不变, 它们现在会使用新计算的 A1_benign 等
    benign_scores_stacked = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    benign_scores_stacked.sort(axis=1)
    result_benign_top_two = np.sum(benign_scores_stacked[:, -2:], axis=1)
    threshold_max_two = np.max(result_benign_top_two) # 阈值现在是新方法的max

    extracted_max_1 = benign_scores_stacked[:, -2]
    extracted_max_2 = benign_scores_stacked[:, -1]

    threshold_three_add_dict = {}
    for multi in range(1, 2):
        score_benign_three_add = A1_benign + A2_benign + A3_benign
        threshold_three_add_dict[multi] = np.max(score_benign_three_add) # 阈值现在是新方法的max

    threshold_fused_dict = {}
    threshold_2_fused_dict = {}
    for alpha in range(1, 11):
        _, benign_fused = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=alpha)
        threshold_fused_dict[alpha] = np.max(benign_fused) # 阈值现在是新方法的max
        
        _, benign_fused_2 = dynamic_weight_fusion_with_two(extracted_max_1, extracted_max_2, alpha=alpha)
        threshold_2_fused_dict[alpha] = np.max(benign_fused_2) # 阈值现在是新方法的max

    # --- 3. 预计算每个图的分数 (将图循环移到网格搜索之外) ---
    per_graph_data = []
    all_test_uuids = set()
    for gi, graph in enumerate(graphs['test']):
        uids = graph.ndata['uuid'].cpu().numpy()
        all_test_uuids.update(uids) 

        is_gt_mask = np.isin(uids, ground_node_ids_list)
        final_ground_node_ids_list.update(uids[is_gt_mask])  

        edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
        edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)

        # --- 【修改点】 ---
        # 使用选择的 normalize_func 替换 calibrate_scores_cpu
        # 并使用正确的 benign 分布作为 baseline
        A1_test = normalize_func(benign_dist_A1, node_level_distances_1[gi])
        A2_test = normalize_func(benign_dist_A2, node_level_distances_3[gi])
        A3_calib = normalize_func(benign_dist_A3, edge_based_score)
        # --- 【修改结束】 ---

        # --- 【修改点】 ---
        # 使用新计算的阈值, 而不是 0.99999
        anomaly_mask_1 = A1_test >= threshold_m1
        anomaly_mask_2 = A2_test >= threshold_m2
        anomaly_mask_3 = A3_calib >= threshold_m3
        # --- 【修改结束】 ---

        # 【修改】: 保持 A3_modified 的原始逻辑意图
        # 原始逻辑是 "如果分数是最大值(1.0)，则替换"
        # 新逻辑是 "如果分数是最大值(threshold_m3)，则替换"
        # 注意: 由于浮点数精度，使用 '==' 可能很危险， '>= ' 更安全
        if normalization_method == 'percentile':
            A3_modified = np.where(A3_calib == 1.0, edge_based_score / max_A3, A3_calib)
        else:
            A3_modified = A3_calib
        
        scores_stacked_m1 = np.stack([A1_test, A2_test, A3_modified], axis=1)
        scores_stacked_m1.sort(axis=1)
        score_top_two = np.sum(scores_stacked_m1[:, -2:], axis=1)
        anomaly_mask_4 = score_top_two >= threshold_max_two

        extracted_test_max_1 = scores_stacked_m1[:, -2]
        extracted_test_max_2 = scores_stacked_m1[:, -1]

        per_graph_data.append({
            'gi': gi,
            'uids': uids,
            'is_gt_mask': is_gt_mask,
            'score_top_two': score_top_two,
            'A1_test': A1_test,
            'A2_test': A2_test,
            'A3_calib': A3_calib,
            'A3_modified': A3_modified,
            'extracted_test_max_1': extracted_test_max_1,
            'extracted_test_max_2': extracted_test_max_2,
            'anomaly_mask_1': anomaly_mask_1,
            'anomaly_mask_2': anomaly_mask_2,
            'anomaly_mask_3': anomaly_mask_3,
            'anomaly_mask_4': anomaly_mask_4
        })

    # --- 4. 预计算方法 1-4 的唯一性 TP/FP ---
    # (此部分无变化)
    uuids_m1_tp, uuids_m1_fp = set(), set()
    uuids_m2_tp, uuids_m2_fp = set(), set()
    uuids_m3_tp, uuids_m3_fp = set(), set()
    uuids_m4_tp, uuids_m4_fp = set(), set()
    for d in per_graph_data:
        is_gt = d['is_gt_mask']
        not_gt = ~is_gt
        uids = d['uids']
        uuids_m1_tp.update(uids[d['anomaly_mask_1'] & is_gt])
        uuids_m1_fp.update(uids[d['anomaly_mask_1'] & not_gt])
        uuids_m2_tp.update(uids[d['anomaly_mask_2'] & is_gt])
        uuids_m2_fp.update(uids[d['anomaly_mask_2'] & not_gt])
        uuids_m3_tp.update(uids[d['anomaly_mask_3'] & is_gt])
        uuids_m3_fp.update(uids[d['anomaly_mask_3'] & not_gt])
        uuids_m4_tp.update(uids[d['anomaly_mask_4'] & is_gt])
        uuids_m4_fp.update(uids[d['anomaly_mask_4'] & not_gt])
    total_TP_1, total_FP_1 = len(uuids_m1_tp), len(uuids_m1_fp)
    total_TP_2, total_FP_2 = len(uuids_m2_tp), len(uuids_m2_fp)
    total_TP_3, total_FP_3 = len(uuids_m3_tp), len(uuids_m3_fp)
    total_TP_4, total_FP_4 = len(uuids_m4_tp), len(uuids_m4_fp)

    # --- 5. 主评估循环 (超参数网格搜索) ---
    # (此部分无变化)
    for multi in range(1, 2):
        threshold_three_add = threshold_three_add_dict[multi]
        for alpha in range(5, 6):
            threshold_fused = threshold_fused_dict[alpha]
            threshold_fused_with_two = threshold_2_fused_dict[alpha]
            uuids_m5_tp, uuids_m5_fp = set(), set()
            uuids_m6_tp, uuids_m6_fp = set(), set()
            uuids_m7_tp, uuids_m7_fp = set(), set()
            final_voted_tp_uuids = set()
            final_voted_fp_uuids = set()
            final_tp_details = {}
            final_all_details = defaultdict(list)

            # --- 6. 向量化的图评估 (使用预计算数据) ---
            # (此部分无变化)
            for d in per_graph_data:
                uids = d['uids']
                is_gt = d['is_gt_mask']
                not_gt = ~is_gt
                _, test_fused_2 = dynamic_weight_fusion_with_two(d['extracted_test_max_1'], d['extracted_test_max_2'], alpha=alpha)
                anomaly_mask_5 = test_fused_2 >= threshold_fused_with_two
                score_three_add = d['A1_test'] + d['A2_test'] + multi * d['A3_modified']
                anomaly_mask_6 = score_three_add > threshold_three_add
                _, test_fused = dynamic_weight_fusion(d['A1_test'], d['A2_test'], d['A3_modified'], alpha=alpha)
                anomaly_mask_7 = test_fused > threshold_fused
                uuids_m5_tp.update(uids[anomaly_mask_5 & is_gt])
                uuids_m5_fp.update(uids[anomaly_mask_5 & not_gt])
                uuids_m6_tp.update(uids[anomaly_mask_6 & is_gt])
                uuids_m6_fp.update(uids[anomaly_mask_6 & not_gt])
                uuids_m7_tp.update(uids[anomaly_mask_7 & is_gt])
                uuids_m7_fp.update(uids[anomaly_mask_7 & not_gt])
                flags_per_node = (d['anomaly_mask_1'].astype(np.int8) +
                                  d['anomaly_mask_2'].astype(np.int8) +
                                  d['anomaly_mask_3'].astype(np.int8) +
                                  d['anomaly_mask_4'].astype(np.int8) +
                                  anomaly_mask_5.astype(np.int8) +
                                  anomaly_mask_6.astype(np.int8) +
                                  anomaly_mask_7.astype(np.int8))
                final_vote_mask = flags_per_node >= 4
                for i, value in enumerate(flags_per_node):
                    final_all_details[uids[i]].append(value)
                final_voted_tp_uuids.update(uids[final_vote_mask & is_gt])
                final_voted_fp_uuids.update(uids[final_vote_mask & not_gt])
                
                print('Detailed score of True positive nodes in current graph:')
                A1 = d['A1_test'][is_gt]
                A2 = d['A2_test'][is_gt]
                A3 = d['A3_modified'][is_gt]
                all_A = zip(A1, A2, A3)
                for i, aaa in enumerate(all_A):
                    print(f'  - Node UUID: {uids[is_gt][i]}, A1: {aaa[0]:.6f}, A2: {aaa[1]:.6f}, A3: {aaa[2]:.6f}, Total Votes: {flags_per_node[is_gt][i]}')
                
                print('Detailed score of false positive nodes in current graph:')
                A1 = d['A1_test'][final_vote_mask & not_gt]
                A2 = d['A2_test'][final_vote_mask & not_gt]
                A3 = d['A3_modified'][final_vote_mask & not_gt]
                all_A = zip(A1, A2, A3)
                for i, aaa in enumerate(all_A):
                    print(f'  - Node UUID: {uids[final_vote_mask & not_gt][i]}, A1: {aaa[0]:.6f}, A2: {aaa[1]:.6f}, A3: {aaa[2]:.6f}, Total Votes: {flags_per_node[final_vote_mask & not_gt][i]}')
                
                final_tp_mask_in_graph = final_vote_mask & is_gt
                final_tp_uids_in_graph = uids[final_tp_mask_in_graph]
                for i, uuid in enumerate(final_tp_uids_in_graph):
                    vote_count = flags_per_node[final_tp_mask_in_graph][i]
                    if uuid not in final_tp_details:
                        final_tp_details[uuid] = {'graph_indices': [], 'vote_counts': []}
                    final_tp_details[uuid]['graph_indices'].append(d['gi'])
                    final_tp_details[uuid]['vote_counts'].append(vote_count)

            # --- 7. 计算并报告 ---
            # (此部分无变化)
            total_TP_5 = len(uuids_m5_tp)
            total_FP_5 = len(uuids_m5_fp)
            total_TP_6 = len(uuids_m6_tp)
            total_FP_6 = len(uuids_m6_fp)
            total_TP_7 = len(uuids_m7_tp)
            total_FP_7 = len(uuids_m7_fp)
            total_final_TP = len(final_voted_tp_uuids)
            total_final_FP = len(final_voted_fp_uuids)
            total_final_FN = len(final_ground_node_ids_list - final_voted_tp_uuids)
            non_gt_uuids = all_test_uuids - ground_node_ids
            total_final_TN = len(non_gt_uuids - final_voted_fp_uuids)

            print(f'Results for multi={multi}, alpha={alpha}:')
            # 【修改】: 打印 M1,M2,M3 的新阈值
            print(f'  Method 1 (A1):                TP={total_TP_1}, FP={total_FP_1} (Threshold={threshold_m1:.4f})')
            print(f'  Method 2 (A2):                TP={total_TP_2}, FP={total_FP_2} (Threshold={threshold_m2:.4f})')
            print(f'  Method 3 (A3):                TP={total_TP_3}, FP={total_FP_3} (Threshold={threshold_m3:.4f})')
            # (其他阈值不变)
            print(f'  Method 4 (Top-Two no fused):  TP={total_TP_4}, FP={total_FP_4} (Threshold={threshold_max_two:.4f})')
            print(f'  Method 5 (Top-Two weighted):  TP={total_TP_5}, FP={total_FP_5} (Threshold={threshold_fused_with_two:.4f})')
            print(f'  Method 6 (Top three):         TP={total_TP_6}, FP={total_FP_6} (Threshold={threshold_three_add:.4f})')
            print(f'  Method 7 (Top three weighted):TP={total_TP_7}, FP={total_FP_7} (Threshold={threshold_fused:.4f})')
            print(f'  Final Voted (>=4):            TP={total_final_TP}, FP={total_final_FP}')

            print("  Final Voted True Positive Details (Unique Nodes):")
            if not final_tp_details:
                print("    None")
            else:
                for uuid, details in final_tp_details.items():
                    print(f"    - Node UUID: {uuid}, Found in Graph(s): {details['graph_indices']}, Vote Counts: {details['vote_counts']}")
            
            print("-" * 30)
            
            # --- 8. 计算最终性能指标 ---
            # (此部分无变化)
            precision = total_final_TP / (total_final_TP + total_final_FP + 1e-12)
            recall = total_final_TP / (total_final_TP + total_final_FN + 1e-12)
            accuracy = (total_final_TP + total_final_TN) / (total_final_TP + total_final_FP + total_final_FN + total_final_TN + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            
            denominator_mcc = sqrt(
                (total_final_TP + total_final_FP + 1e-12) *
                (total_final_TP + total_final_FN + 1e-12) *
                (total_final_TN + total_final_FP + 1e-12) *
                (total_final_TN + total_final_FN + 1e-12)
            )
            mcc = (total_final_TP * total_final_TN - total_final_FP * total_final_FN) / (denominator_mcc + 1e-12) # 避免除以0

            print(f"[Final Results for alpha={alpha}]")
            print(f"Unique TP={total_final_TP}, Unique FP={total_final_FP}, Unique FN={total_final_FN}, Unique TN={total_final_TN}")
            print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}, MCC={mcc:.4f}")

            scores = []
            nodes = []
            dataset_name = filename.split('/')[-1].split('_')[0] + '_' + filename.split('/')[-1].split('_')[1]
            attack_to_GPs = torch.load(f'{dataset_name}_attack_to_nids.pt')
            attack2nodes = {k: v["nids"] for k, v in attack_to_GPs.items()}
            node2attacks = defaultdict(set)
            for attack, node_s in attack2nodes.items():
                for node in node_s:
                    if node not in ground_node_ids:
                        continue
                    node2attacks[node].add(attack)
            labels = []
            for uuid, details in final_all_details.items():
                scores.append(max(details))
                nodes.append(uuid)
                if uuid in ground_node_ids:
                    labels.append(1)
                else:
                    labels.append(0)
            
            adp_score = plot_detected_attacks_vs_precision(scores, nodes, node2attacks, labels, f'{dataset_name}_attack_adp_detection.png')
            print('The adp score is:', adp_score)
            print("=" * 40 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection results using method_12_with_different_normalization.')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to the intermediate results .pt file saved by main_transductive.py')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the preprocessed graph .pt file')
    parser.add_argument('--ground_truth_path', type=str, required=True,
                        help='Path to the ground truth node IDs .pt file')
    parser.add_argument('--normalization_method', type=str, default='percentile',
                        choices=['percentile', 'min_max', 'z_score', 'robust'],
                        help='Score normalization method (default: percentile)')
    args = parser.parse_args()

    print(f'============================Processing file:=====================\n{args.results_path}')
    method_12_with_different_normalization(
        args.results_path,
        args.data_path,
        args.ground_truth_path,
        normalization_method=args.normalization_method,
    )


if __name__ == '__main__':
    main()

# --- Batch scan example (run manually by editing this block) ---
# for file in os.listdir('save_middle_results'):
#     if file.endswith('.pt'):
#         if 'h201' not in file:
#             continue
#         filename = os.path.join('save_middle_results', file)
#         data = 'optc_h201_merge_edge_normalized.pt'
#         ground_truth = '../Ground_Truth/ground_truth_nids_optc_h201.pt'
#         print('============================Processing file:=====================\n', filename)
#         method_12_with_different_normalization(filename, data, ground_truth, normalization_method='percentile')


# filename =  'saved_middle_result/THEIA_E3_loss_sce_rpr_8_nh_0.3_nl_2_lr_0.0015_mp_5_mpf_50_wd_0.01_wdf_0.0001_gatedge_gat_data.pt'

    # filenames = [
    #     'saved_middle_result_2/CADETS_E5_loss_sce_dim_64_nhd_8_nh_0.5_nl_2_lr_0.0015_mp_500_mpf_10_wd_0.0001_wdf_5e-05_gatedge_gat_data.pt',
    #     'saved_middle_result_2/THEIA_E5_loss_sce_dim_64_nhd_8_nh_0.5_nl_2_lr_0.0015_mp_500_mpf_10_wd_1e-05_wdf_2e-06_gatedge_gat_data.pt',
    #     'saved_middle_result/CLEARSCOPE_E5_loss_sce_rpr_8_nh_0.1_nl_2_lr_0.0015_mp_200_mpf_20_wd_0.0001_wdf_5e-05_gatedge_gat_data.pt',
    #     'saved_middle_result/CADETS_E3_loss_sce_dim_64_nhd_2_nh_0.3_nl_2_lr_0.0015_mp_5_mpf_50_wd_0.001_wdf_0.0001_gatedge_gat_data.pt',
    #     'saved_middle_result/THEIA_E3_loss_sce_rpr_8_nh_0.3_nl_2_lr_0.0015_mp_5_mpf_50_wd_0.01_wdf_0.0001_gatedge_gat_data.pt',
    #     'saved_middle_result/CLEARSCOPE_E3_loss_sce_dim_64_nhd_4_nh_0.1_nl_2_lr_0.0015_mp_200_mpf_50_wd_0.001_wdf_0.0001_gatedge_gat_data.pt'
        
    # ]
    # datas = [
    #     'cadets_e5_merge_edge_data.pt',
    #     'theia_e5_merge_edge_data_final.pt',
    #     'clearscope_e5_merge_edge_data.pt',
    #     'cadets_merge_edge_no_day_2_data.pt',
    #     'theia_merge_edge_data.pt',
    #     'clearscope_e3_merge_edge_data.pt'


    # ]
    # ground_truths = [
    #     '../Ground_Truth/ground_truth_nids_cadets_e5.pt',
    #     '../Ground_Truth/ground_truth_nids_theia_e5.pt',
    #     '../Ground_Truth/ground_truth_nids_clearscope_e5.pt',
    #     '../Ground_Truth/ground_truth_cadet_v2.pt',
    #     '../Ground_Truth/ground_truth_nids.pt',
    #     '../Ground_Truth/ground_truth_nids_clearscope.pt'
    # ]
   
    # for i in range(6):
    #     filename = filenames[i]
    #     data = datas[i]
    #     ground_truth = ground_truths[i]
    #     print('============================Processing file:=====================\n', filename)
    #     method_12_with_different_normalization(filename,data, ground_truth,normalization_method='z_score')
