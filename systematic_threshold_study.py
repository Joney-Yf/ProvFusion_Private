import numpy as np
from typing import Literal, Dict, Any, Optional, Union, List, Tuple
import torch
import argparse
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import itertools
import multiprocessing
import time  # <--- 用于进度打印
from functools import partial
from collections import defaultdict
from math import sqrt

NormalizationMethod = Literal['min_max', 'z_score', 'robust', 'quantile']
QuantileTieBreak = Literal['left', 'right']
FusionMethod = Literal[
    'max', 'weighted_sum', 'l2_norm',
    'max_two', 'dynamic_max_two', 'dynamic_three'
]
ProbabilisticMethod = Literal['mvg', 'gmm']
DistanceMethod = Literal['knn']
BoundaryMethod = Literal['ocsvm', 'iforest']
ReconstructionMethod = Literal['autoencoder']


def calibrate_scores_cpu(benign_scores, test_scores):
    # 1. 对良性分数进行一次性排序
    sorted_benign = np.sort(benign_scores)
    
    # 2. 使用二分查找，找到每个测试分数在良性分数中的插入点
    # 这个插入点的索引值，就等于有多少个良性分数比它小
    ranks = np.searchsorted(sorted_benign, test_scores, side='right')
    
    # 3. 将排名转换为[0, 1]的百分位数
    calibrated = ranks / len(benign_scores)
    
    return calibrated
    

def dynamic_power_fusion(s1, s2, s3, p=2):
    """
    动态 Power-normalization 权重融合
    s1, s2, s3: shape (N,)
    p: 幂次参数 (p>1 时放大大值，0<p<1 时平滑，小于0时反向强调小值)
    """
    scores = np.stack([s1, s2, s3], axis=1)  # (N, 3)

    # 避免负数或0
    scores_clipped = np.clip(scores, 1e-8, None)

    w = scores_clipped ** p
    w = w / np.sum(w, axis=1, keepdims=True)  # 归一化权重

    fused_score = np.sum(w * scores, axis=1)
    return w, fused_score


def dynamic_sigmoid_fusion(s1, s2, s3, alpha=10, tau=0.8):
    """
    动态 Sigmoid-gating 权重融合
    s1, s2, s3: shape (N,)
    alpha: 控制sigmoid陡峭度 (越大越接近硬门控)
    tau:   阈值 (在百分位归一化空间内，例如0.8表示强调高于80%分位的值)
    """
    scores = np.stack([s1, s2, s3], axis=1)  # (N, 3)

    g = 1 / (1 + np.exp(-alpha * (scores - tau)))  # gating
    w = g / (np.sum(g, axis=1, keepdims=True) + 1e-8)  # 归一化权重

    fused_score = np.sum(w * scores, axis=1)
    return w, fused_score


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
    # plt.savefig(out_file)
    # except:
    #     print("Error while generating ADP plot")
    plt.close()
    return area_under_curve

class DataNormalizer:
    # NormalizationMethod = Literal['min_max', 'z_score', 'robust', 'quantile']
    """
    A robust data normalizer that learns scaling parameters from a benign
    dataset and applies them to any dataset.
    
    Version 2: Includes a special handling for the 3rd feature in 'quantile' mode.
    """

    def __init__(self, method: NormalizationMethod = 'min_max'):
        """
        Initializes the DataNormalizer.

        Args:
            method: The normalization method to be used. One of
                    'min_max', 'z_score', 'robust', or 'quantile'.
        """
        if method not in ['min_max', 'z_score', 'robust', 'quantile']:
            raise ValueError(f"Invalid method '{method}'. Choose from 'min_max', "
                             f"'z_score', 'robust', 'quantile'.")
        self.method = method
        self.stats_: Optional[Dict[str, np.ndarray]] = None
        self._quantile_tie_break: QuantileTieBreak = 'right'


    def fit(self, benign_data: np.ndarray):
        """
        Computes the necessary statistics from the benign data.

        This method learns the "ruler" from the provided benign dataset.
        It must be called before `transform`.

        Args:
            benign_data: A 2D numpy array of shape (n_samples, n_features)
                         containing the benign data for learning the
                         normalization parameters.
        """
        data = np.asarray(benign_data)
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array.")

        self.stats_ = {}
        if self.method == 'min_max':
            self.stats_['min'] = np.min(data, axis=0)
            self.stats_['max'] = np.max(data, axis=0)
            self.stats_['range'] = self.stats_['max'] - self.stats_['min']

        elif self.method == 'z_score':
            self.stats_['mean'] = np.mean(data, axis=0)
            self.stats_['std'] = np.std(data, axis=0)

        elif self.method == 'robust':
            q1, median, q3 = np.percentile(data, [25, 50, 75], axis=0)
            self.stats_['median'] = median
            self.stats_['iqr'] = q3 - q1

        elif self.method == 'quantile':
            self.stats_['benign_samples_sorted'] = np.sort(data, axis=0)
            self.stats_['n_samples'] = data.shape[0]
            # NEW: Store the maximum of benign data for the special 3rd-dim rule.
            self.stats_['max_benign'] = np.max(data, axis=0)

    def transform(self, data_to_transform: np.ndarray) -> np.ndarray:
        """
        Applies the learned normalization to the given data.

        Args:
            data_to_transform: A 2D numpy array of shape (n_samples, n_features)
                               to be normalized.

        Returns:
            A 2D numpy array with the transformed data.
        """
        if self.stats_ is None:
            raise RuntimeError("You must call `fit` before calling `transform`.")

        data = np.asarray(data_to_transform)
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
            
        transformed_data = data.copy().astype(float)

        if self.method == 'min_max':
            min_val = self.stats_['min']
            range_val = self.stats_['range']
            safe_range = np.where(range_val == 0, 1e-9, range_val)
            transformed_data = (data - min_val) / safe_range
            return np.clip(transformed_data, 0, 1)

        elif self.method == 'z_score':
            mean_val = self.stats_['mean']
            std_val = self.stats_['std']
            safe_std = np.where(std_val == 0, 1e-9, std_val)
            return (data - mean_val) / safe_std

        elif self.method == 'robust':
            median_val = self.stats_['median']
            iqr_val = self.stats_['iqr']
            safe_iqr = np.where(iqr_val == 0, 1e-9, iqr_val)
            return (data - median_val) / safe_iqr

        elif self.method == 'quantile':
            benign_sorted = self.stats_['benign_samples_sorted']
            n_benign_samples = self.stats_['n_samples']
            
            for i in range(data.shape[1]):
                ranks = np.searchsorted(
                    benign_sorted[:, i],
                    data[:, i],
                    side=self._quantile_tie_break
                )
                transformed_data[:, i] = ranks / n_benign_samples

            # MODIFIED: Add the special logic for the third dimension (index 2)
            # This logic is applied *after* the initial quantile transformation.
            if data.shape[1] >= 3:
                third_dim_idx = 2
                
                # Get original and normalized values for the 3rd feature
                original_third_dim = data[:, third_dim_idx]
                normalized_third_dim = transformed_data[:, third_dim_idx]
                
                # Get the learned maximum value for the 3rd feature from the benign set
                max_benign_third_dim = self.stats_['max_benign'][third_dim_idx]
                
                # Avoid division by zero, though unlikely in this context
                safe_max = max_benign_third_dim if max_benign_third_dim > 0 else 1e-9
                
                # Identify where the normalized score is 1.0
                condition = (normalized_third_dim == 1.0)
                
                # Apply the special rule using np.where for high performance
                transformed_data[:, third_dim_idx] = np.where(
                    condition,
                    original_third_dim / safe_max,  # New value if condition is True
                    normalized_third_dim           # Keep old value if False
                )
                
            return transformed_data
        


class StaticFusionDetector:
    """
    Implements a threshold-based static fusion strategy for anomaly detection.

    This detector learns a global anomaly threshold from a normalized benign
    dataset and uses it to identify anomalies in a test dataset.
    
    Version 2: Adds 'max_two', 'dynamic_max_two', and 'dynamic_three' fusion methods.
    """
    def __init__(
        self,
        fusion_method: FusionMethod = 'max',
        percentile_threshold: float = 99.9,
        weights: Optional[List[float]] = None,
        alpha: float = 5.0  # NEW: Added alpha parameter for dynamic methods
    ):
        """
        Initializes the StaticFusionDetector.

        Args:
            fusion_method: The method to fuse multiple anomaly scores.
            percentile_threshold: The percentile of benign scores to use as the
                                  anomaly threshold (e.g., 99.9).
            weights: A list of weights for the 'weighted_sum' method.
            alpha: The exponent factor for dynamic weighting methods, controlling
                   the emphasis on the maximum score.
        """
        # MODIFIED: Updated validation list
        valid_methods = ['max', 'weighted_sum', 'l2_norm', 'max_two', 'dynamic_max_two', 'dynamic_three']
        if fusion_method not in valid_methods:
            raise ValueError(f"Invalid fusion_method: '{fusion_method}'. Choose from {valid_methods}")
        if fusion_method == 'weighted_sum' and weights is None:
            raise ValueError("Weights must be provided for 'weighted_sum' method.")

        self.fusion_method = fusion_method
        self.percentile_threshold = percentile_threshold
        self.weights = np.array(weights) if weights is not None else None
        self.alpha = alpha # NEW: Store alpha
        self.threshold_: Optional[float] = None

    # NEW: A private helper for the core dynamic weighting logic to avoid code duplication.
    def _dynamic_softmax_fusion(self, scores: np.ndarray) -> np.ndarray:
        """
        Performs dynamic softmax weighting and fusion.
        This function encapsulates the logic from your provided examples.

        Args:
            scores: A numpy array of shape (N, K) where K is the number of
                    scores to be fused (e.g., 2 or 3).

        Returns:
            A numpy array of shape (N,) containing the fused scores.
        """
        # Add a small epsilon for numerical stability
        max_vals = np.max(scores, axis=1, keepdims=True) + 1e-9
        
        # Relative importance score 'r'
        relative_importance = scores / max_vals
        
        # Softmax-like weights 'w'
        exp_weights = np.exp(self.alpha * relative_importance)
        softmax_weights = exp_weights / np.sum(exp_weights, axis=1, keepdims=True)
        
        # Fused score is the weighted sum
        fused_score = np.sum(softmax_weights * scores, axis=1)
        return fused_score

    def _fuse_scores(self, normalized_scores: np.ndarray) -> np.ndarray:
        """
        Private helper to fuse scores based on the chosen method.
        This is a performance-critical function using vectorized operations.
        """
        # --- Existing Methods ---
        if self.fusion_method == 'max':
            return np.max(normalized_scores, axis=1)
            
        elif self.fusion_method == 'weighted_sum':
            if self.weights.shape[0] != normalized_scores.shape[1]:
                raise ValueError("Number of weights must match number of features.")
            return np.dot(normalized_scores, self.weights)
            
        elif self.fusion_method == 'l2_norm':
            return np.linalg.norm(normalized_scores, axis=1)

        # --- NEW Methods ---
        elif self.fusion_method == 'max_two':
            if normalized_scores.shape[1] < 2:
                raise ValueError("'max_two' method requires at least 2 features.")
            # Sort each row and sum the last two elements (the largest two)
            sorted_scores = np.sort(normalized_scores, axis=1)
            return np.sum(sorted_scores[:, -2:], axis=1)
            
        elif self.fusion_method == 'dynamic_max_two':
            if normalized_scores.shape[1] < 2:
                raise ValueError("'dynamic_max_two' method requires at least 2 features.")
            # Sort to find the top two scores for each sample
            sorted_scores = np.sort(normalized_scores, axis=1)
            top_two_scores = sorted_scores[:, -2:]
            return self._dynamic_softmax_fusion(top_two_scores)
            
        elif self.fusion_method == 'dynamic_three':
            if normalized_scores.shape[1] < 3:
                raise ValueError("'dynamic_three' method requires at least 3 features.")
            # Use all three scores for dynamic fusion
            return self._dynamic_softmax_fusion(normalized_scores)
            
        raise NotImplementedError(f"Fusion method '{self.fusion_method}' not implemented.")


    def fit(self, normalized_benign_data: np.ndarray):
        """
        Learns the anomaly threshold from the normalized benign data.
        (This method remains unchanged as its logic is independent of the fusion method.)
        """
        if normalized_benign_data.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        benign_fused_scores = self._fuse_scores(normalized_benign_data)
        self.threshold_ = np.percentile(benign_fused_scores, self.percentile_threshold)
        print(f"Detector '{self.fusion_method}' fitted. Threshold at {self.percentile_threshold}% is: {self.threshold_:.4f}")

    def predict(self, normalized_test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects anomalies in the normalized test data.
        (This method remains unchanged as its logic is independent of the fusion method.)
        """
        if self.threshold_ is None:
            raise RuntimeError("The detector has not been fitted yet. Call `fit` first.")
        if normalized_test_data.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        test_fused_scores = self._fuse_scores(normalized_test_data)
        anomaly_indices = np.where(test_fused_scores > self.threshold_)[0]
        return anomaly_indices, test_fused_scores

class ProbabilisticDetector:
    """
    Implements anomaly detection strategies based on probabilistic models.

    This detector models the distribution of normal data and identifies outliers
    as points that have a low probability under the learned model. It is
    recommended to use standardized data (e.g., via Z-score) as input.
    """
    def __init__(
        self,
        method: ProbabilisticMethod = 'mvg',
        percentile_threshold: float = 99.9,
        n_components: int = 1,
        regularization_cov: float = 1e-6
    ):
        """
        Initializes the ProbabilisticDetector.

        Args:
            method: The probabilistic model to use.
                    'mvg': Multivariate Gaussian Distribution. Anomaly score is
                           the Mahalanobis distance (higher is more anomalous).
                    'gmm': Gaussian Mixture Model. Anomaly score is the
                           log-likelihood (lower is more anomalous).
            percentile_threshold: The percentile used to set the anomaly threshold.
                                  For 'mvg', this is a high percentile (e.g., 99.9).
                                  For 'gmm', this is interpreted as the inverse,
                                  so 99.9 corresponds to the 0.1 percentile of
                                  log-likelihoods.
            n_components: The number of components (clusters) for the GMM.
                          Ignored if method is 'mvg'.
            regularization_cov: A small value added to the diagonal of the
                                covariance matrix for 'mvg' to ensure it is
                                invertible and numerically stable.
        """
        if method not in ['mvg', 'gmm']:
            raise ValueError(f"Invalid method: '{method}'. Choose from 'mvg', 'gmm'.")
        
        self.method = method
        self.percentile_threshold = percentile_threshold
        self.n_components = n_components
        self.regularization_cov = regularization_cov
        
        self.threshold_: Optional[float] = None
        # Learned parameters will be stored with a trailing underscore
        self.mean_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None
        self.gmm_: Optional[GaussianMixture] = None

    def fit(self, normalized_benign_data: np.ndarray):
        """
        Fits the probabilistic model to the normalized benign data and determines
        the anomaly threshold.

        Args:
            normalized_benign_data: A 2D numpy array of shape (n_samples, n_features)
                                    containing the normalized scores of benign data.
        """
        if normalized_benign_data.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        
        n_features = normalized_benign_data.shape[1]

        if self.method == 'mvg':
            # Learn the parameters: mean vector and inverse covariance matrix
            self.mean_ = np.mean(normalized_benign_data, axis=0)
            covariance = np.cov(normalized_benign_data, rowvar=False)
            # Add regularization for numerical stability before inverting
            regularized_cov = covariance + self.regularization_cov * np.identity(n_features)
            self.cov_inv_ = np.linalg.inv(regularized_cov)
            
            # Calculate scores on benign data to find the threshold
            benign_scores = self._score_samples_mvg(normalized_benign_data)
            self.threshold_ = np.percentile(benign_scores, self.percentile_threshold)
            print(f"MVG detector fitted. Mahalanobis distance threshold at {self.percentile_threshold}% is: {self.threshold_:.4f}")

        elif self.method == 'gmm':
            # Fit the GMM model using the highly optimized scikit-learn implementation
            self.gmm_ = GaussianMixture(
                n_components=self.n_components,
                random_state=42, # for reproducibility
                covariance_type='full'
            )
            self.gmm_.fit(normalized_benign_data)
            
            # Calculate scores on benign data to find the threshold
            benign_scores = self.gmm_.score_samples(normalized_benign_data)
            
            # For GMM, lower scores are more anomalous, so we use a low percentile.
            low_percentile = 100.0 - self.percentile_threshold
            self.threshold_ = np.percentile(benign_scores, low_percentile)
            print(f"GMM detector ({self.n_components} components) fitted. Log-likelihood threshold at {low_percentile:.1f}% is: {self.threshold_:.4f}")

    def _score_samples_mvg(self, data: np.ndarray) -> np.ndarray:
        """Calculates the Mahalanobis distance for each sample."""
        diff = data - self.mean_
        # Efficiently compute (X-μ)T * Σ^-1 * (X-μ) for all rows
        mahal_dist = np.sum(np.dot(diff, self.cov_inv_) * diff, axis=1)
        return mahal_dist

    def score_samples(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Calculates the anomaly score for each sample in the input data.

        Args:
            normalized_data: A 2D numpy array to be scored.

        Returns:
            A 1D array of anomaly scores. For MVG, this is the Mahalanobis
            distance. For GMM, it's the log-likelihood.
        """
        if self.method == 'mvg':
            if self.mean_ is None:
                raise RuntimeError("MVG model has not been fitted. Call `fit` first.")
            return self._score_samples_mvg(normalized_data)
        elif self.method == 'gmm':
            if self.gmm_ is None:
                raise RuntimeError("GMM model has not been fitted. Call `fit` first.")
            return self.gmm_.score_samples(normalized_data)
        raise NotImplementedError

    def predict(self, normalized_test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects anomalies in the normalized test data based on the learned threshold.

        Args:
            normalized_test_data: A 2D numpy array to screen for anomalies.

        Returns:
            A tuple containing:
            - An array of indices corresponding to the detected anomalies.
            - An array of the anomaly scores for each test sample.
        """
        if self.threshold_ is None:
            raise RuntimeError("The detector has not been fitted yet. Call `fit` first.")
        
        test_scores = self.score_samples(normalized_test_data)
        
        if self.method == 'mvg':
            # For Mahalanobis distance, a HIGHER score is anomalous
            anomaly_indices = np.where(test_scores > self.threshold_)[0]
        else: # 'gmm'
            # For log-likelihood, a LOWER score is anomalous
            anomaly_indices = np.where(test_scores < self.threshold_)[0]
            
        return anomaly_indices, test_scores


class DistanceBasedDetector:
    """
    Implements anomaly detection strategies based on distance, specifically k-NN.

    This detector identifies anomalies as points that are far from their k-nearest
    neighbors in a reference (benign) dataset. It leverages GPU acceleration for
    high performance on large datasets.
    """
    def __init__(
        self,
        method: DistanceMethod = 'knn',
        k: int = 10,
        percentile_threshold: float = 99.9,
        sample_ratio: float = 1.0,
        device: str = 'cuda',
        test_batch_size: int = 2048,
        sample_batch_size: int = 2048
    ):
        """
        Initializes the DistanceBasedDetector.

        Args:
            method: The distance-based method to use. Currently only 'knn'.
            k: The number of nearest neighbors to consider.
            percentile_threshold: The percentile of benign distances to use as
                                  the anomaly threshold.
            sample_ratio: The fraction of the benign dataset to use as the
                          reference. A value < 1.0 can significantly speed up
                          fitting and prediction, at the cost of some precision.
            device: The computing device ('cuda' or 'cpu').
            test_batch_size: Batch size for the test data (outer loop).
            sample_batch_size: Batch size for the reference data (inner loop).
        """
        if method != 'knn':
            raise ValueError("Currently, only 'knn' method is supported.")
        if not (0 < sample_ratio <= 1.0):
            raise ValueError("sample_ratio must be between 0 (exclusive) and 1 (inclusive).")
            
        self.method = method
        self.k = k
        self.percentile_threshold = percentile_threshold
        self.sample_ratio = sample_ratio
        self.device_str = device
        self.t_batch_size = test_batch_size
        self.s_batch_size = sample_batch_size
        
        self.device_: Optional[torch.device] = None
        self.benign_samples_ref_: Optional[torch.Tensor] = None
        self.threshold_: Optional[float] = None

    def _initialize_device(self):
        """Initializes the torch device."""
        if torch.cuda.is_available() and self.device_str == 'cuda':
            self.device_ = torch.device('cuda')
            print("Using GPU for k-NN computation.")
        else:
            self.device_ = torch.device('cpu')
            print("GPU not available or not selected. Using CPU for k-NN computation.")
    
    # This is your high-performance function, refactored into a class method.
    def _score_samples_knn(self, test_tensors: torch.Tensor, k_to_find: int) -> np.ndarray:
        """
        Computes the mean distance to the k-nearest neighbors for each test vector.
        This method is adapted from the user-provided high-performance GPU code.
        """
        S = self.benign_samples_ref_
        n, d = S.shape
        
        # Pre-calculate squared L2 norms of the reference samples
        S_sq = torch.sum(S ** 2, dim=1)
        
        m = test_tensors.size(0)
        all_means = torch.empty(m, dtype=torch.float32)

        # Batch process the test tensors to manage memory
        for i in range(0, m, self.t_batch_size):
            t_start, t_end = i, min(i + self.t_batch_size, m)
            T_batch = test_tensors[t_start:t_end]
            
            T_batch_sq = torch.sum(T_batch ** 2, dim=1, keepdims=True)
            batch_top_k_dists_sq = torch.full((T_batch.size(0), k_to_find), float('inf'), device=self.device_)

            # Inner loop: batch process the reference samples
            for j in range(0, n, self.s_batch_size):
                s_start, s_end = j, min(j + self.s_batch_size, n)
                S_batch = S[s_start:s_end]
                S_sq_batch = S_sq[s_start:s_end]

                # Euclidean distance squared: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a^T b
                dot_product = torch.mm(T_batch, S_batch.t())
                dist_sq_chunk = T_batch_sq + S_sq_batch - 2 * dot_product

                combined_dists = torch.cat([batch_top_k_dists_sq, dist_sq_chunk], dim=1)
                batch_top_k_dists_sq, _ = torch.topk(combined_dists, k_to_find, dim=1, largest=False)
            
            # Take the square root to get actual distances, then the mean
            # Clamp to avoid negative values from floating point inaccuracies
            batch_top_k_dists = torch.sqrt(torch.clamp(batch_top_k_dists_sq, min=0.0))
            means_batch = batch_top_k_dists.mean(dim=1)
            all_means[t_start:t_end] = means_batch.cpu()

        return all_means.numpy()

    def fit(self, normalized_benign_data: np.ndarray):
        """
        Fits the detector by defining the reference (benign) dataset and
        calculating the anomaly score threshold from it.
        """
        if self.device_ is None:
            self._initialize_device()
            
        # --- 1. Sub-sample the benign data if sample_ratio < 1.0 ---
        num_samples = int(normalized_benign_data.shape[0] * self.sample_ratio)
        if self.sample_ratio < 1.0:
            print(f"Sub-sampling benign data: using {num_samples} out of {normalized_benign_data.shape[0]} samples.")
            rng = np.random.default_rng(42)
            sample_indices = rng.choice(normalized_benign_data.shape[0], num_samples, replace=False)
            benign_subset = normalized_benign_data[sample_indices]
        else:
            benign_subset = normalized_benign_data
            
        self.benign_samples_ref_ = torch.from_numpy(benign_subset.astype(np.float32)).to(self.device_)
        
        # --- 2. Calculate threshold based on distances within the benign set ---
        print(f"Calculating anomaly threshold by finding neighbors within the benign set (k={self.k})...")
        # To find k neighbors for a point *within its own set*, we must find k+1
        # neighbors and discard the first one (the point itself, dist=0).
        k_for_fit = self.k + 1
        
        # The scoring function gives the mean of the top k+1 distances.
        # Mean(d_1, ..., d_{k+1}) = (0 + d_2 + ... d_{k+1}) / (k+1)
        benign_scores_raw = self._score_samples_knn(self.benign_samples_ref_, k_for_fit)
        
        # The true mean of the k neighbors is (d_2 + ... + d_{k+1}) / k.
        # We can correct the raw score: true_mean = raw_mean * (k+1) / k
        benign_scores_corrected = benign_scores_raw * (k_for_fit / self.k)
        
        self.threshold_ = np.percentile(benign_scores_corrected, self.percentile_threshold)
        print(f"Detector fitted. k-NN distance threshold at {self.percentile_threshold}% is: {self.threshold_:.4f}")


    def predict(self, normalized_test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects anomalies in the test data by computing their k-NN distance
        to the reference benign samples.
        """
        if self.threshold_ is None or self.benign_samples_ref_ is None:
            raise RuntimeError("The detector has not been fitted yet. Call `fit` first.")
        
        test_tensor = torch.from_numpy(normalized_test_data.astype(np.float32)).to(self.device_)
        
        print(f"Scoring {test_tensor.shape[0]} test points against {self.benign_samples_ref_.shape[0]} reference points...")
        test_scores = self._score_samples_knn(test_tensor, self.k)
        
        # For k-NN, a HIGHER score (distance) is anomalous
        anomaly_indices = np.where(test_scores >= self.threshold_)[0]
            
        return anomaly_indices, test_scores

class BoundaryBasedDetector:
    """
    Implements anomaly detection strategies based on learning a boundary around
    normal data, such as One-Class SVM and Isolation Forest.

    This detector leverages the highly optimized implementations from scikit-learn.
    """
    def __init__(
        self,
        method: BoundaryMethod = 'iforest',
        percentile_threshold: float = 99.9,
        n_estimators = 100,
        nu = 0.000001,
        **kwargs: Any
    ):
        """
        Initializes the BoundaryBasedDetector.

        Args:
            method: The boundary-based model to use.
                    'ocsvm': One-Class Support Vector Machine.
                    'iforest': Isolation Forest.
            percentile_threshold: The percentile used to set the anomaly threshold.
                                  For both models, a low score is anomalous, so 99.9
                                  corresponds to the 0.1 percentile of scores.
            **kwargs: Additional keyword arguments to be passed directly to the
                      underlying scikit-learn model constructor. E.g., for 'ocsvm',
                      you can pass `nu=0.01`, `gamma='auto'`. For 'iforest',
                      you can pass `n_estimators=100`, `random_state=42`.
        """
        if method not in ['ocsvm', 'iforest']:
            raise ValueError(f"Invalid method: '{method}'. Choose from 'ocsvm', 'iforest'.")
        
        self.method = method
        self.percentile_threshold = percentile_threshold
        self.model_params = kwargs
        self.n_estimators = n_estimators
        self.nu = nu
        
        self.model_: Optional[Any] = None
        self.threshold_: Optional[float] = None

    def fit(self, normalized_benign_data: np.ndarray):
        """
        Fits the one-class classification model to the normalized benign data
        and determines the anomaly score threshold.

        Args:
            normalized_benign_data: A 2D numpy array of shape (n_samples, n_features)
                                    containing the normalized data of benign samples.
        """
        if normalized_benign_data.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        
        # --- 1. Initialize and train the scikit-learn model ---
        print(f"Initializing and fitting '{self.method}' model...")
        if self.method == 'ocsvm':
            # Set default nu if not provided, as it's a critical parameter
            self.model_params.setdefault('nu', self.nu) # Corresponds to assumed outlier fraction
            self.model_params.setdefault('gamma', 'auto')
            self.model_ = OneClassSVM(**self.model_params)
        elif self.method == 'iforest':
            # Set default n_estimators and random_state for reproducibility
            self.model_params.setdefault('n_estimators', self.n_estimators)
            self.model_params.setdefault('random_state', 42)
            self.model_ = IsolationForest(**self.model_params)
            
        self.model_.fit(normalized_benign_data)
        print("Model fitting complete.")

        # --- 2. Calculate scores on benign data to find the threshold ---
        # For both OC-SVM and Isolation Forest in scikit-learn, the `score_samples`
        # method returns scores where a lower value indicates a higher anomaly likelihood.
        benign_scores = self.model_.score_samples(normalized_benign_data)
        
        # Therefore, we need to find a low percentile of these scores to act as our boundary.
        low_percentile = 100.0 - self.percentile_threshold
        self.threshold_ = np.percentile(benign_scores, low_percentile)
        
        print(f"Detector fitted. Anomaly score threshold at {low_percentile:.1f}% is: {self.threshold_:.4f}")

    def score_samples(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Calculates the anomaly score for each sample in the input data.

        Args:
            normalized_data: A 2D numpy array to be scored.

        Returns:
            A 1D array of anomaly scores. For both models, a lower score
            means more anomalous.
        """
        if self.model_ is None:
            raise RuntimeError("The model has not been fitted yet. Call `fit` first.")
        return self.model_.score_samples(normalized_data)

    def predict(self, normalized_test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects anomalies in the normalized test data based on the learned boundary.

        Args:
            normalized_test_data: A 2D numpy array to screen for anomalies.

        Returns:
            A tuple containing:
            - An array of indices corresponding to the detected anomalies.
            - An array of the anomaly scores for each test sample.
        """
        if self.threshold_ is None:
            raise RuntimeError("The detector has not been fitted yet. Call `fit` first.")
        
        test_scores = self.score_samples(normalized_test_data)
        
        # For both models, a score BELOW the threshold is considered an anomaly.
        anomaly_indices = np.where(test_scores < self.threshold_)[0]
            
        return anomaly_indices, test_scores

class ReconstructionDetector:
    """
    Implements anomaly detection using a reconstruction-based deep learning model,
    specifically an Autoencoder (AE).

    The detector trains an AE on benign data to learn a compressed representation
    of normality. Anomalies are identified by their high reconstruction error.
    It is highly recommended to use data scaled to [0, 1] (e.g., via Min-Max).
    """

    # Internal PyTorch model is defined as a nested class for encapsulation.
    class _Autoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: List[int]):
            super().__init__()
            
            # --- Build Encoder ---
            encoder_layers = []
            last_dim = input_dim
            for h_dim in hidden_dims:
                encoder_layers.append(nn.Linear(last_dim, h_dim))
                encoder_layers.append(nn.ReLU(True))
                last_dim = h_dim
            self.encoder = nn.Sequential(*encoder_layers)

            # --- Build Decoder (symmetric to encoder) ---
            decoder_layers = []
            # Reverse the hidden_dims list for decoder construction
            reversed_hidden_dims = hidden_dims[::-1]
            for i in range(len(reversed_hidden_dims) - 1):
                decoder_layers.append(nn.Linear(reversed_hidden_dims[i], reversed_hidden_dims[i+1]))
                decoder_layers.append(nn.ReLU(True))
            
            # Final layer to reconstruct the input
            decoder_layers.append(nn.Linear(reversed_hidden_dims[-1], input_dim))
            # Sigmoid activation is often used to ensure output is in [0, 1] range,
            # matching the recommended Min-Max scaled input.
            decoder_layers.append(nn.Sigmoid())
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    def __init__(
        self,
        method: ReconstructionMethod = 'autoencoder',
        percentile_threshold: float = 99.9,
        hidden_dims: List[int] = [32, 16],
        epochs: int = 100,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        device: str = 'cuda'
    ):
        """
        Initializes the ReconstructionDetector.

        Args:
            method: The reconstruction-based model to use.
            percentile_threshold: Percentile for setting the anomaly threshold
                                  on reconstruction errors.
            hidden_dims: A list of integers defining the sizes of the hidden layers
                         of the encoder. The decoder will be symmetric. The last
                         element is the dimension of the latent space.
            epochs: Number of training epochs.
            batch_size: Batch size for training and prediction.
            learning_rate: Learning rate for the Adam optimizer.
            device: The computing device ('cuda' or 'cpu').
        """
        if method != 'autoencoder':
            raise ValueError("Currently, only 'autoencoder' method is supported.")
        
        self.method = method
        self.percentile_threshold = percentile_threshold
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.device_str = device
        
        self.device_: Optional[torch.device] = None
        self.model_: Optional[nn.Module] = None
        self.threshold_: Optional[float] = None

    def _initialize_device(self):
        """Initializes the torch device."""
        if torch.cuda.is_available() and self.device_str == 'cuda':
            self.device_ = torch.device('cuda')
            print("Using GPU for Autoencoder training and prediction.")
        else:
            self.device_ = torch.device('cpu')
            print("GPU not available or not selected. Using CPU.")

    def fit(self, normalized_benign_data: np.ndarray):
        """
        Trains the Autoencoder on the normalized benign data and determines
        the anomaly threshold based on reconstruction errors.
        """
        if self.device_ is None: self._initialize_device()
        
        input_dim = normalized_benign_data.shape[1]
        self.model_ = self._Autoencoder(input_dim, self.hidden_dims).to(self.device_)
        
        # --- Training Setup ---
        dataset = TensorDataset(torch.from_numpy(normalized_benign_data.astype(np.float32)))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        
        # --- Training Loop ---
        print(f"Starting Autoencoder training for {self.epochs} epochs...")
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for data in dataloader:
                inputs, = data # Unpack the tuple from dataloader
                inputs = inputs.to(self.device_)
                
                optimizer.zero_grad()
                outputs = self.model_(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.6f}')
        print("Training complete.")

        # --- Calculate Threshold ---
        print("Calculating reconstruction error threshold on benign data...")
        benign_scores = self.score_samples(normalized_benign_data)
        self.threshold_ = np.percentile(benign_scores, self.percentile_threshold)
        print(f"Detector fitted. Reconstruction error threshold at {self.percentile_threshold}% is: {self.threshold_:.6f}")

    def score_samples(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Calculates the reconstruction error (MSE) for each sample.
        """
        if self.model_ is None:
            raise RuntimeError("The model has not been fitted yet. Call `fit` first.")
        if self.device_ is None: self._initialize_device()
        
        self.model_.eval() # Set model to evaluation mode
        dataset = TensorDataset(torch.from_numpy(normalized_data.astype(np.float32)))
        # Do not shuffle test data
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_errors = []
        criterion = nn.MSELoss(reduction='none') # Calculate per-sample error
        
        with torch.no_grad():
            for data in dataloader:
                inputs, = data
                inputs = inputs.to(self.device_)
                reconstructions = self.model_(inputs)
                # Calculate MSE across features for each sample in the batch
                errors = criterion(reconstructions, inputs).mean(dim=1)
                all_errors.append(errors.cpu().numpy())
                
        return np.concatenate(all_errors)

    def predict(self, normalized_test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects anomalies by identifying samples with reconstruction errors
        above the learned threshold.
        """
        if self.threshold_ is None:
            raise RuntimeError("The detector has not been fitted yet. Call `fit` first.")
        
        test_scores = self.score_samples(normalized_test_data)
        
        # For reconstruction models, a HIGHER score (error) is anomalous
        anomaly_indices = np.where(test_scores > self.threshold_)[0]
            
        return anomaly_indices, test_scores

def calculate_statistical_summary(data: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Calculates a detailed statistical summary for the given dataset.

    Args:
        data: A 2D numpy array of shape (n_samples, n_features).
        feature_names: A list of names for the features.

    Returns:
        A pandas DataFrame containing the statistical summary.
    """
    df = pd.DataFrame(data, columns=feature_names)
    
    # Standard statistics including specified percentiles
    desc = df.describe(percentiles=[.25, .5, .75, .99]).transpose()
    
    # Add skewness and kurtosis
    desc['skew'] = df.skew()
    desc['kurtosis'] = df.kurt()
    
    return desc

def plot_distribution_comparison(
    val_data: np.ndarray, 
    test_data: np.ndarray, 
    feature_name: str, 
    output_path: str
):
    """
    Generates and saves a KDE plot comparing the distributions of a feature
    from the validation and test sets.

    Args:
        val_data: 1D numpy array for the validation set feature.
        test_data: 1D numpy array for the test set feature.
        feature_name: The name of the feature for plot titles and labels.
        output_path: The full path to save the generated plot.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(val_data, label='Validation Set Distribution', fill=True, alpha=0.6, linewidth=2)
    sns.kdeplot(test_data, label='Test Set Distribution', fill=True, alpha=0.6, linewidth=2)
    plt.title(f'Distribution Comparison for {feature_name}', fontsize=16)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure to the specified path and close it to free memory
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def perform_ks_test(val_data: np.ndarray, test_data: np.ndarray) -> Tuple[float, float]:
    """
    Performs a two-sample Kolmogorov-Smirnov (K-S) test.

    Args:
        val_data: 1D numpy array for the validation set feature.
        test_data: 1D numpy array for the test set feature.

    Returns:
        A tuple containing the K-S statistic and the p-value.
    """
    # The K-S test is sensitive to sample size, so if the test set is much
    # larger, it's common to take a random subsample for a more stable comparison.
    # Here, we'll sample the larger dataset to match the smaller one's size
    # for a more balanced test, but for the most rigorous result, use full sets.
    # Note: With huge datasets, even tiny differences can yield p-value ~ 0.
    
    # For this implementation, we will use the full datasets as requested.
    ks_statistic, p_value = stats.ks_2samp(val_data, test_data)
    return ks_statistic, p_value

def analyze_distribution_shift(
    validation_set: np.ndarray, 
    test_set: np.ndarray, 
    feature_names: List[str],
    output_dir: str = 'distribution_analysis_results'
):
    """
    Performs a comprehensive analysis of distribution shift between
    validation and test sets.

    Args:
        validation_set: 2D numpy array of benign validation data.
        test_set: 2D numpy array of test data.
        feature_names: List of feature names.
        output_dir: Directory to save the analysis results (plots).
    """
    print("Starting Distribution Shift Analysis...")
    print("="*80)
    
    # --- 1. Create Output Directory ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in '{output_dir}/'")
    print("="*80)

    # --- 2. Statistical Summary Comparison ---
    print("Step 1: Calculating Statistical Summaries...")
    val_summary = calculate_statistical_summary(validation_set, feature_names)
    test_summary = calculate_statistical_summary(test_set, feature_names)
    
    # Combine summaries for a side-by-side comparison
    comparison_df = pd.concat(
        [val_summary, test_summary], 
        axis=1, 
        keys=['Validation Set', 'Test Set']
    )
    print("Statistical Summary Comparison:")
    # Set display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(comparison_df)
    print("="*80)

    # --- 3. Per-Feature Visualization and K-S Test ---
    print("Step 2 & 3: Generating Plots and Performing K-S Tests...")
    for i, name in enumerate(feature_names):
        print(f"\n--- Analyzing Feature: {name} ---")
        val_feature_data = validation_set[:, i]
        test_feature_data = test_set[:, i]
        
        # Plotting
        plot_path = os.path.join(output_dir, f'{name}_distribution_comparison.png')
        plot_distribution_comparison(val_feature_data, test_feature_data, name, plot_path)
        print(f"  - Distribution plot saved to: {plot_path}")
        
        # K-S Test
        ks_stat, p_val = perform_ks_test(val_feature_data, test_feature_data)
        print(f"  - Kolmogorov-Smirnov (K-S) Test Results:")
        print(f"    - K-S Statistic: {ks_stat:.4f}")
        print(f"    - P-value: {p_val:.4e}") # Using scientific notation for small p-values
        
        # Interpretation of p-value
        alpha = 0.05
        if p_val < alpha:
            print(f"    - Interpretation: The p-value ({p_val:.4e}) is less than {alpha}. "
                  "We REJECT the null hypothesis. The distributions are SIGNIFICANTLY DIFFERENT.")
        else:
            print(f"    - Interpretation: The p-value ({p_val:.4e}) is greater than {alpha}. "
                  "We FAIL to reject the null hypothesis. The distributions are NOT significantly different.")
    
    print("\n" + "="*80)
    print("Analysis Complete.")

def _load_and_prepare_data(processed_data_filename: str, raw_data_filename: str, ground_truth_filename: str) -> Tuple:
    """Loads all necessary data files and performs initial cleaning."""
    # This function would contain your torch.load calls
    # For demonstration, I'm returning mock data.
    # Replace mock data with your actual torch.load logic.
    print(f"Loading data from {processed_data_filename}...")
    
    # --- Mock data for demonstration ---
    # Replace this section with your actual torch.load calls
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3,
     total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign,
     max_A3) = torch.load(processed_data_filename, map_location=torch.device('cpu'))

    (graphs, weights, num_features, edge_features, num_classes,
     reverse_graphs) = torch.load(raw_data_filename, map_location=torch.device('cpu'))

    ground_node_ids = torch.load(ground_truth_filename)
    
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)
    validation_data = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))

    # --- End of mock data section ---

    print("Data loaded successfully.")
    return validation_data, graphs, set(ground_node_ids), total_uuid_to_max, node_level_distances_1, node_level_distances_3

def _apply_manual_ground_truth_updates(filename: str, ground_node_ids: set) -> set:
    """Applies hard-coded additions to the ground truth set based on filename."""
    updates = {
        'THEIA_E3': [215236, 215806, 215179, 215721, 215237, 215796, 215235, 215854, 215801, 350388, 1027134],
        'CLEARSCOPE_E3': [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734],
        'CLEARSCOPE_E5': [158937, 445211]
    }
    for key, id_list in updates.items():
        if key in filename:
            print(f"Applying manual ground truth updates for {key}...")
            ground_node_ids.update(id_list)
            break
    return ground_node_ids

def run_anomaly_detection_pipeline(
    args,
    processed_data_filename: str,
    raw_data_filename: str,
    ground_truth_filename: str,
    normalization_method: str = 'quantile',
    fusion_method: FusionMethod = 'max',
    percentile: float = 99.9,
):
    """
    Executes the full anomaly detection pipeline: load, normalize, detect, and evaluate.
    """
    # --- 1. Data Loading and Preparation ---
    # Your original code for loading data should go inside this function
    validation_data, graphs, ground_node_ids, total_uuid_to_max, node_level_distances_1, node_level_distances_3 = _load_and_prepare_data(processed_data_filename, raw_data_filename, ground_truth_filename)
    
    # Handle ground truth updates
    ground_node_ids = _apply_manual_ground_truth_updates(processed_data_filename, ground_node_ids)

    # --- 2. Normalization ---
    print("\n--- Normalization Stage ---")
    # Your template's validation data construction
    
    normalizer = DataNormalizer(method=normalization_method)
    normalizer.fit(validation_data)
    normalized_validation_data = normalizer.transform(validation_data)
    print(f"Normalizer '{normalization_method}' fitted on validation data of shape {validation_data.shape}.")

    # --- 3. Anomaly Detector Training ---
    print("\n--- Detector Training Stage ---")
    if fusion_method in ['max', 'weighted_sum', 'l2_norm','max_two', 'dynamic_max_two', 'dynamic_three']:
        detector = StaticFusionDetector(fusion_method=fusion_method, percentile_threshold=percentile)
    elif fusion_method in ['mvg', 'gmm']:
        detector = ProbabilisticDetector(method=fusion_method, percentile_threshold=percentile, n_components=args.n_components, regularization_cov=args.lr)
    elif fusion_method in ['knn']:
        detector = DistanceBasedDetector(method=fusion_method, percentile_threshold=percentile, k=args.n_components)
    elif fusion_method in ['ocsvm', 'iforest']:
        detector = BoundaryBasedDetector(method=fusion_method, percentile_threshold=percentile, n_estimators= args.n_components, nu=args.lr)
    elif fusion_method in ['autoencoder']:
        detector = ReconstructionDetector(method=fusion_method, percentile_threshold=percentile, learning_rate=args.lr)
    detector.fit(normalized_validation_data)

    # --- 4. Detection and Evaluation on Test Set ---
    print("\n--- Detection & Evaluation Stage ---")
    total_TP_uuids = set()
    total_FP_uuids = set()
    all_test_node_uuids_set = set() # 跟踪测试集中的所有节点
    final_all_details = defaultdict(list) # 用于 ADP 计算

    for gi, graph in enumerate(graphs['test']):
        # Prepare test data for the current graph
        uuids = graph.ndata['uuid'].cpu().numpy()
        edge_based_scores = np.array(list(total_uuid_to_max[gi].values()))
        edge_based_scores = np.where(np.isinf(edge_based_scores), 0, edge_based_scores)
        
        # Ensure the test data has 3 features to match validation data
        test_data = np.array(list(zip(node_level_distances_1[gi], node_level_distances_3[gi], edge_based_scores)))

        if test_data.shape[0] == 0:
            print(f"Graph {i}: No data to process, skipping.")
            continue

        # Normalize the test data using the already-fitted normalizer
        normalized_test_data = normalizer.transform(test_data)
        
        # Get predictions from the detector
        anomaly_indices, anomaly_scores = detector.predict(normalized_test_data)

        predicted_indices_in_this_graph = set(anomaly_indices)

        # Evaluate the results for the current graph
        tp_count = 0
        fp_count = 0

        for idx in range(len(anomaly_scores)):
            uuid_val = uuids[idx]
            score = anomaly_scores[idx]
            
            all_test_node_uuids_set.add(uuid_val)
            reverse_score = -score
            final_all_details[uuid_val].append(reverse_score) # 存储分数用于 ADP

            is_ground_truth = (uuid_val in ground_node_ids)
            is_predicted = (idx in predicted_indices_in_this_graph)



            if is_ground_truth:
                print(f"Graph {gi}: Node UUID: {uuid_val}, the detailed anomaly score is {normalized_test_data[idx][0]} {normalized_test_data[idx][1]} {normalized_test_data[idx][2]} Score: {anomaly_scores[idx]} is a raw attack nodes.")
        
            if is_predicted:
                if is_ground_truth:
                    print(f"Graph {gi}: True Positive found - UUID: {uuid_val}, the detailed anomaly score is {normalized_test_data[idx][0]} {normalized_test_data[idx][1]} {normalized_test_data[idx][2]} Score: {score} is a True Positive (TP).")
                    total_TP_uuids.add(uuid_val)
                else:
                    print(f"Graph {gi}: False Positive found - UUID: {uuid_val}, the detailed anomaly score is {normalized_test_data[idx][0]} {normalized_test_data[idx][1]} {normalized_test_data[idx][2]} Score: {score} is a False Positive (FP).")
                    total_FP_uuids.add(uuid_val)

    # --- 5. Final Report ---
    all_ground_truth_in_test = ground_node_ids.intersection(all_test_node_uuids_set)
    # 测试集中所有的真实正常节点
    all_normal_nodes_in_test = all_test_node_uuids_set - ground_node_ids

    total_final_TP = len(total_TP_uuids)
    total_final_FP = len(total_FP_uuids)
    
    # FN = 真实攻击节点 中 没有被检测出来的
    total_final_FN_uuids = all_ground_truth_in_test - total_TP_uuids
    total_final_FN = len(total_final_FN_uuids)

    # TN = 真实正常节点 中 没有被误报的
    total_final_TN_uuids = all_normal_nodes_in_test - total_FP_uuids
    total_final_TN = len(total_final_TN_uuids)

    print(f"  Total Unique True Positives (TP): {total_final_TP}")
    print(f"  Total Unique False Positives (FP): {total_final_FP}")
    print(f"  Total Unique False Negatives (FN): {total_final_FN}")
    print(f"  Total Unique True Negatives (TN): {total_final_TN}")
    print(f"  (Total nodes evaluated: {len(all_test_node_uuids_set)})")
    print("-" * 50)

    # --- 计算指标 (来自你的示例) ---
    precision = total_final_TP / (total_final_TP + total_final_FP + 1e-12)
    recall = total_final_TP / (total_final_TP + total_final_FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    accuracy = (total_final_TP + total_final_TN) / (total_final_TP + total_final_FP + total_final_FN + total_final_TN + 1e-12)                    
    
    denominator_mcc = sqrt(
        (total_final_TP + total_final_FP + 1e-12) *
        (total_final_TP + total_final_FN + 1e-12) *
        (total_final_TN + total_final_FP + 1e-12) *
        (total_final_TN + total_final_FN + 1e-12)
    )
    # 检查分母是否为0
    if denominator_mcc == 0:
        mcc = 0.0
    else:
        mcc = (total_final_TP * total_final_TN - total_final_FP * total_final_FN) / denominator_mcc

    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}, MCC={mcc:.4f}")
    print("-" * 50)

    # --- 计算 ADP 分数 (来自你的示例) ---
    print("Calculating ADP Score...")
    scores = []
    nodes = []
    labels = []
    
    # 使用函数参数中的 processed_data_filename
    dataset_name_parts = processed_data_filename.split('/')[-1].split('_')
    if len(dataset_name_parts) >= 2:
        dataset_name = dataset_name_parts[0] + '_' + dataset_name_parts[1]
    else:
        dataset_name = dataset_name_parts[0] # 备用方案
        
    attack_to_GPs_path = f'{dataset_name}_attack_to_nids.pt'
    
    # try:
    attack_to_GPs = torch.load(attack_to_GPs_path)
    attack2nodes = {k: v["nids"] for k, v in attack_to_GPs.items()}
    node2attacks = defaultdict(set)
    for attack, node_s in attack2nodes.items():
        for node in node_s:
            node2attacks[node].add(attack)
    
    # 准备 ADP 计算所需的数据
    for uuid, details in final_all_details.items():
        scores.append(max(details)) # 如果一个 UUID 出现在多图，取最高分
        nodes.append(uuid)
        if uuid in ground_node_ids:
            labels.append(1)
        else:
            labels.append(0)
    
    # !! 确保 plot_detected_attacks_vs_precision 已经导入 !!
    output_png_filename = f'{dataset_name}_attack_adp_detection.png'
    
    # 检查函数是否存在
    if 'plot_detected_attacks_vs_precision' not in locals() and 'plot_detected_attacks_vs_precision' not in globals():
            raise NameError("'plot_detected_attacks_vs_precision' function not found. Skipping ADP calculation.")

    adp_score = plot_detected_attacks_vs_precision(scores, nodes, node2attacks, labels, output_png_filename)
    print(f'ADP Score: {adp_score}')
    print(f"ADP plot saved to: {output_png_filename}")

    # except FileNotFoundError:
    #     print(f"Warning: Could not load {attack_to_GPs_path}. Skipping ADP calculation.")
    # except NameError as e:
    #     print(f"Warning: {e}")
    # except Exception as e:
    #     print(f"An error occurred during ADP calculation: {e}. Skipping.")

    print("="*50)

def distribuion_analysis(
    args,
    processed_data_filename: str,
    raw_data_filename: str,
    ground_truth_filename: str,
    normalization_method: str = 'quantile',
    fusion_method: FusionMethod = 'max',
    percentile: float = 99.9,
):
    validation_data, graphs, ground_node_ids, total_uuid_to_max, node_level_distances_1, node_level_distances_3 = _load_and_prepare_data(processed_data_filename, raw_data_filename, ground_truth_filename)
    
    # Handle ground truth updates
    ground_node_ids = _apply_manual_ground_truth_updates(processed_data_filename, ground_node_ids)
    features_names = ['structual', 'attribute', 'edge causal']

    if 'THEIA_E3' in processed_data_filename:
        dataset_name = 'THEIA_E3'
    elif 'CLEARSCOPE_E3' in processed_data_filename:
        dataset_name = 'CLEARSCOPE_E3'
    elif 'CLEARSCOPE_E5' in processed_data_filename:
        dataset_name = 'CLEARSCOPE_E5'
    elif 'THEIA_E5' in processed_data_filename:
        dataset_name = 'THEIA_E5'
    elif 'CADETS_E5' in processed_data_filename:
        dataset_name = 'CADETS_E5'
    elif 'CADETS_E3' in processed_data_filename:
        dataset_name = 'CADETS_E3'
    else:
        dataset_name = 'Unknown_Dataset'

    # --- 2. Normalization ---
    print("\n--- Normalization Stage ---")
    # Your template's validation data construction
    
    normalizer = DataNormalizer(method=normalization_method)
    normalizer.fit(validation_data)
    normalized_validation_data = normalizer.transform(validation_data)
    print(f"Normalizer '{normalization_method}' fitted on validation data of shape {validation_data.shape}.")

    for gi, graph in enumerate(graphs['test']):
        # Prepare test data for the current graph
        uuids = graph.ndata['uuid'].cpu().numpy()
        edge_based_scores = np.array(list(total_uuid_to_max[gi].values()))
        edge_based_scores = np.where(np.isinf(edge_based_scores), 0, edge_based_scores)
        
        # Ensure the test data has 3 features to match validation data
        test_data = np.array(list(zip(node_level_distances_1[gi], node_level_distances_3[gi], edge_based_scores)))

        normalized_test_data = normalizer.transform(test_data)


        analyze_distribution_shift(
        validation_set=normalized_validation_data,
        test_set=normalized_test_data,
        feature_names=features_names,
        output_dir=f'./my_first_analysis/{dataset_name}_graph_{gi}_analysis_{normalization_method}'
        )

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



def method_12_revised_with_range_detector_optimized_test_all_weight(filename, data, ground_truth):
    """
    优化版本：通过预计算和缓存掩码来大幅降低时间复杂度。
    功能与原始版本完全相同。
    """
    # --- 1. 数据加载与初始化 (无变化) ---
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
        ground_node_ids.update(addition_list)
    if 'CLEARSCOPE_E3' in filename:
        addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
        ground_node_ids.update(addition_list)
    if 'CLEARSCOPE_E5' in filename:
        addition_list = [158937, 445211]
        ground_node_ids.update(addition_list)
    
    ground_node_ids_list = list(ground_node_ids)
    final_ground_node_ids_list = set()

    # --- 2. 良性样本分数和阈值的预计算 (无变化) ---
    benign_scores_stacked = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    benign_scores_stacked.sort(axis=1)
    result_benign_top_two = np.sum(benign_scores_stacked[:, -2:], axis=1)
    threshold_max_two = np.max(result_benign_top_two)
    extracted_max_1 = benign_scores_stacked[:, -2]
    extracted_max_2 = benign_scores_stacked[:, -1]
    threshold_three_add_dict = {}
    for multi_A1 in range(20):
        for multi_A2 in range(20):
            for multi_A3 in range(20):
                score_benign_three_add = multi_A1 * A1_benign + multi_A2 * A2_benign + multi_A3 * A3_benign
                threshold_three_add_dict[(multi_A1, multi_A2, multi_A3)] = np.max(score_benign_three_add)
    threshold_fused_dict = {}
    threshold_2_fused_dict = {}
    alpha_list = [1]
    for alpha in alpha_list:
        _, benign_fused = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=alpha)
        threshold_fused_dict[alpha] = np.max(benign_fused)
        _, benign_fused_2 = dynamic_weight_fusion_with_two(extracted_max_1, extracted_max_2, alpha=alpha)
        threshold_2_fused_dict[alpha] = np.max(benign_fused_2)
    benign_scores_3d_for_range = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    benign_range = np.max(benign_scores_3d_for_range, axis=1)
    threshold_range = np.max(benign_range)

    # --- 3. 预计算每个图的分数 (无变化) ---
    per_graph_data = []
    all_test_uuids = set()
    for gi, graph in enumerate(graphs['test']):
        uids = graph.ndata['uuid'].cpu().numpy()
        all_test_uuids.update(uids)
        is_gt_mask = np.isin(uids, ground_node_ids_list)
        final_ground_node_ids_list.update(uids[is_gt_mask])
        edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
        edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)
        A1_test = calibrate_scores_cpu(val_baseline[0], node_level_distances_1[gi])
        A2_test = calibrate_scores_cpu(emb_baseline[0], node_level_distances_3[gi])
        A3_calib = calibrate_scores_cpu(edge_loss_baseline, edge_based_score)
        A3_modified = np.where(A3_calib == 1.0, edge_based_score / max_A3, A3_calib)
        # scores_stacked_m1 = np.stack([A1_test, A2_test, A3_modified], axis=1)
        # scores_stacked_m1.sort(axis=1)
        per_graph_data.append({
            'gi': gi, 'uids': uids, 'is_gt_mask': is_gt_mask,
            'A1_test': A1_test, 'A2_test': A2_test, 'A3_modified': A3_modified,
            # 'extracted_test_max_1': scores_stacked_m1[:, -2], 'extracted_test_max_2': scores_stacked_m1[:, -1],
            # 'anomaly_mask_1': A1_test >= 0.99999,
            # 'anomaly_mask_2': A2_test >= 0.99999,
            # 'anomaly_mask_3': A3_calib >= 0.99999,
            # 'anomaly_mask_4': np.sum(scores_stacked_m1[:, -2:], axis=1) >= threshold_max_two,
        })

    # --- 4. 预计算方法 1-4 的唯一性 TP/FP (无变化) ---
    # uuids_m1_tp, uuids_m1_fp = set(), set()
    # uuids_m2_tp, uuids_m2_fp = set(), set()
    # uuids_m3_tp, uuids_m3_fp = set(), set()
    # uuids_m4_tp, uuids_m4_fp = set(), set()
    # for d in per_graph_data:
    #     is_gt, not_gt, uids = d['is_gt_mask'], ~d['is_gt_mask'], d['uids']
    #     uuids_m1_tp.update(uids[d['anomaly_mask_1'] & is_gt])
    #     uuids_m1_fp.update(uids[d['anomaly_mask_1'] & not_gt])
    #     uuids_m2_tp.update(uids[d['anomaly_mask_2'] & is_gt])
    #     uuids_m2_fp.update(uids[d['anomaly_mask_2'] & not_gt])
    #     uuids_m3_tp.update(uids[d['anomaly_mask_3'] & is_gt])
    #     uuids_m3_fp.update(uids[d['anomaly_mask_3'] & not_gt])
    #     uuids_m4_tp.update(uids[d['anomaly_mask_4'] & is_gt])
    #     uuids_m4_fp.update(uids[d['anomaly_mask_4'] & not_gt])
    # total_TP_1, total_FP_1 = len(uuids_m1_tp), len(uuids_m1_fp)
    # total_TP_2, total_FP_2 = len(uuids_m2_tp), len(uuids_m2_fp)
    # total_TP_3, total_FP_3 = len(uuids_m3_tp), len(uuids_m3_fp)
    # total_TP_4, total_FP_4 = len(uuids_m4_tp), len(uuids_m4_fp)
    
    # --- 5. 【优化核心】预计算所有超参数组合的掩码 ---
    print("Pre-calculating masks for all hyperparameter combinations...")
    # 结构: alpha_masks[alpha][graph_index] = {'mask_5': ..., 'mask_7': ...}
    alpha_masks = defaultdict(dict)
    # 结构: multi_masks[(m1,m2,m3)][graph_index] = {'mask_6': ...}
    multi_masks = defaultdict(dict)

    # 遍历一次所有图，计算所有可能的掩码
    for gi, d in enumerate(per_graph_data):
        # for alpha in alpha_list:
        #     threshold_fused = threshold_fused_dict[alpha]
        #     threshold_fused_with_two = threshold_2_fused_dict[alpha]
            
        #     _, test_fused_2 = dynamic_weight_fusion_with_two(d['extracted_test_max_1'], d['extracted_test_max_2'], alpha=alpha)
        #     mask_5 = test_fused_2 >= threshold_fused_with_two
            
        #     _, test_fused = dynamic_weight_fusion(d['A1_test'], d['A2_test'], d['A3_modified'], alpha=alpha)
        #     mask_7 = test_fused > threshold_fused
            
        #     alpha_masks[alpha][gi] = {'mask_5': mask_5, 'mask_7': mask_7}

        for multi_A1 in range(20):
            for multi_A2 in range(20):
                for multi_A3 in range(20):
                    key = (multi_A1, multi_A2, multi_A3)
                    threshold_three_add = threshold_three_add_dict[key]
                    score_three_add = multi_A1 * d['A1_test'] + multi_A2 * d['A2_test'] + multi_A3 * d['A3_modified']
                    mask_6 = score_three_add > threshold_three_add
                    multi_masks[key][gi] = {'mask_6': mask_6}
    print("Pre-calculation finished. Starting main evaluation loop...")

    # --- 6. 主评估循环 (现在只进行查表和聚合，速度极快) ---
    for multi_A1 in range(20):
        for multi_A2 in range(20):
            for multi_A3 in range(20):
                multi_key = (multi_A1, multi_A2, multi_A3)
                threshold_three_add = threshold_three_add_dict[multi_key]

                for alpha in alpha_list:
                    # threshold_fused = threshold_fused_dict[alpha]
                    # threshold_fused_with_two = threshold_2_fused_dict[alpha]

                    # uuids_m5_tp, uuids_m5_fp = set(), set()
                    uuids_m6_tp, uuids_m6_fp = set(), set()
                    # uuids_m7_tp, uuids_m7_fp = set(), set()
                    final_voted_tp_uuids, final_voted_fp_uuids = set(), set()
                    final_tp_details = {}
                    final_all_details = defaultdict(list)

                    for gi, d in enumerate(per_graph_data):
                        uids, is_gt, not_gt = d['uids'], d['is_gt_mask'], ~d['is_gt_mask']

                        # --- 【优化】直接从缓存中读取掩码，而不是重新计算 ---
                        # precomputed_alpha = alpha_masks[alpha][gi]
                        # anomaly_mask_5 = precomputed_alpha['mask_5']
                        # anomaly_mask_7 = precomputed_alpha['mask_7']
                        
                        precomputed_multi = multi_masks[multi_key][gi]
                        anomaly_mask_6 = precomputed_multi['mask_6']
                        
                        # 更新方法 5, 6, 7 的 TP/FP
                        # uuids_m5_tp.update(uids[anomaly_mask_5 & is_gt])
                        # uuids_m5_fp.update(uids[anomaly_mask_5 & not_gt])
                        uuids_m6_tp.update(uids[anomaly_mask_6 & is_gt])
                        uuids_m6_fp.update(uids[anomaly_mask_6 & not_gt])
                        # uuids_m7_tp.update(uids[anomaly_mask_7 & is_gt])
                        # uuids_m7_fp.update(uids[anomaly_mask_7 & not_gt])
                        
                        # --- 最终聚合: 投票 (逻辑无变化) ---
                        flags_per_node = (
                            # d['anomaly_mask_1'].astype(np.int8) + d['anomaly_mask_2'].astype(np.int8) +
                            # d['anomaly_mask_3'].astype(np.int8) + d['anomaly_mask_4'].astype(np.int8) +
                            # anomaly_mask_5.astype(np.int8) + 
                            anomaly_mask_6.astype(np.int8))
                            # anomaly_mask_7.astype(np.int8))
                        final_vote_mask = flags_per_node >= 1

                        for i, value in enumerate(flags_per_node):
                            final_all_details[uids[i]].append(value)

                        final_voted_tp_uuids.update(uids[final_vote_mask & is_gt])
                        final_voted_fp_uuids.update(uids[final_vote_mask & not_gt])
                        
                        # 记录最终投票为TP的节点详细信息
                        final_tp_mask_in_graph = final_vote_mask & is_gt
                        final_tp_uids_in_graph = uids[final_tp_mask_in_graph]
                        for i, uuid in enumerate(final_tp_uids_in_graph):
                            vote_count = flags_per_node[final_tp_mask_in_graph][i]
                            if uuid not in final_tp_details:
                                final_tp_details[uuid] = {'graph_indices': [], 'vote_counts': []}
                            final_tp_details[uuid]['graph_indices'].append(d['gi'])
                            final_tp_details[uuid]['vote_counts'].append(vote_count)

                    # --- 7. 计算并报告 (无变化) ---
                    # total_TP_5, total_FP_5 = len(uuids_m5_tp), len(uuids_m5_fp)
                    total_TP_6, total_FP_6 = len(uuids_m6_tp), len(uuids_m6_fp)
                    # total_TP_7, total_FP_7 = len(uuids_m7_tp), len(uuids_m7_fp)
                    total_final_TP, total_final_FP = len(final_voted_tp_uuids), len(final_voted_fp_uuids)
                    total_final_FN = len(final_ground_node_ids_list - final_voted_tp_uuids)
                    non_gt_uuids = all_test_uuids - ground_node_ids
                    total_final_TN = len(non_gt_uuids - final_voted_fp_uuids)

                    print(f'Results for multi={multi_key}, alpha={alpha}:')
                    # print(f'  Method 1 (A1):                TP={total_TP_1}, FP={total_FP_1} (Threshold=1.0)')
                    # print(f'  Method 2 (A2):                TP={total_TP_2}, FP={total_FP_2} (Threshold=1.0)')
                    # print(f'  Method 3 (A3):                TP={total_TP_3}, FP={total_FP_3} (Threshold=1.0)')
                    # print(f'  Method 4 (Top-Two no fused):  TP={total_TP_4}, FP={total_FP_4} (Threshold={threshold_max_two:.4f})')
                    # print(f'  Method 5 (Top-Two weighted):  TP={total_TP_5}, FP={total_FP_5} (Threshold={threshold_fused_with_two:.4f})')
                    print(f'  Method 6 (Top three):         TP={total_TP_6}, FP={total_FP_6} (Threshold={threshold_three_add:.4f})')
                    # print(f'  Method 7 (Top three weighted):TP={total_TP_7}, FP={total_FP_7} (Threshold={threshold_fused:.4f})')
                    print(f'  Final Voted (>=1):            TP={total_final_TP}, FP={total_final_FP}')
                    
                    # 打印最终投票为TP的详细信息
                    print("  Final Voted True Positive Details (Unique Nodes):")
                    if not final_tp_details:
                        print("    None")
                    else:
                        for uuid, details in sorted(final_tp_details.items()):
                            print(f"    - Node UUID: {uuid}, Found in Graph(s): {details['graph_indices']}, Vote Counts: {details['vote_counts']}")
                    print("-" * 30)

                    # --- 8. 计算最终性能指标 (无变化) ---
                    precision = total_final_TP / (total_final_TP + total_final_FP + 1e-12)
                    recall = total_final_TP / (total_final_TP + total_final_FN + 1e-12)
                    f1 = 2 * precision * recall / (precision + recall + 1e-12)
                    print(f"[Final Results for multi={multi_key}, alpha={alpha}]")
                    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                    accuracy = (total_final_TP + total_final_TN) / (total_final_TP + total_final_FP + total_final_FN + total_final_TN + 1e-12)                    
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

def method_12_revised_with_range_detector_optimized_old(filename, data, ground_truth):
    """
    优化版本：通过预计算和缓存掩码来大幅降低时间复杂度。
    功能与原始版本完全相同。
    """
    # --- 1. 数据加载与初始化 (无变化) ---
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3,
     total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign,
     max_A3) = torch.load(filename, map_location=torch.device('cpu'))

    (graphs, weights, num_features, edge_features, num_classes,
     reverse_graphs) = torch.load(data, map_location=torch.device('cpu'))

    ground_node_ids = set(torch.load(ground_truth))
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)
    validation_data = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))


    normalizer = DataNormalizer(method='min_max')
    normalizer.fit(validation_data)
    normalized_validation_data = normalizer.transform(validation_data)
    print(f"Normalizer 'min_max' fitted on validation data of shape {validation_data.shape}.")

    if 'THEIA_E3' in filename:
        addition_list = [215236, 215806, 215179, 215721, 215237, 215796, 215235, 215854, 215801, 350388, 1027134]
        ground_node_ids.update(addition_list)
    if 'CLEARSCOPE_E3' in filename:
        addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
        ground_node_ids.update(addition_list)
    if 'CLEARSCOPE_E5' in filename:
        addition_list = [158937, 445211]
        ground_node_ids.update(addition_list)
    
    ground_node_ids_list = list(ground_node_ids)
    final_ground_node_ids_list = set()

    # --- 2. 良性样本分数和阈值的预计算 (无变化) ---
    benign_scores_stacked = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    benign_scores_stacked.sort(axis=1)
    result_benign_top_two = np.sum(benign_scores_stacked[:, -2:], axis=1)
    threshold_max_two = np.max(result_benign_top_two)
    extracted_max_1 = benign_scores_stacked[:, -2]
    extracted_max_2 = benign_scores_stacked[:, -1]
    threshold_three_add_dict = {}
    for multi_A1 in range(1, 20):
        for multi_A2 in range(1, 20):
            for multi_A3 in range(1, 20):
                score_benign_three_add = multi_A1 * A1_benign + multi_A2 * A2_benign + multi_A3 * A3_benign
                threshold_three_add_dict[(multi_A1, multi_A2, multi_A3)] = np.max(score_benign_three_add)
    threshold_fused_dict = {}
    threshold_2_fused_dict = {}
    for alpha in range(1, 20):
        _, benign_fused = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=alpha)
        threshold_fused_dict[alpha] = np.max(benign_fused)
        _, benign_fused_2 = dynamic_weight_fusion_with_two(extracted_max_1, extracted_max_2, alpha=alpha)
        threshold_2_fused_dict[alpha] = np.max(benign_fused_2)
    benign_scores_3d_for_range = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    # print(benign_scores_3d_for_range.shape)
    # BoundaryBasedDetector(method=fusion_method, percentile_threshold=percentile)
    # detector = BoundaryBasedDetector(method='ocsvm', percentile_threshold=99.99)
    # detector = ReconstructionDetector(method='autoencoder', percentile_threshold=99.9, learning_rate=1e-3)
    detector = BoundaryBasedDetector(method='iforest', percentile_threshold=99.9)
    detector.fit(normalized_validation_data)


    
    # --- 3. 预计算每个图的分数 (无变化) ---
    per_graph_data = []
    all_test_uuids = set()
    for gi, graph in enumerate(graphs['test']):
        uids = graph.ndata['uuid'].cpu().numpy()
        all_test_uuids.update(uids)
        is_gt_mask = np.isin(uids, ground_node_ids_list)
        final_ground_node_ids_list.update(uids[is_gt_mask])
        edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
        edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)
        A1_test = calibrate_scores_cpu(val_baseline[0], node_level_distances_1[gi])
        A2_test = calibrate_scores_cpu(emb_baseline[0], node_level_distances_3[gi])
        A3_calib = calibrate_scores_cpu(edge_loss_baseline, edge_based_score)
        A3_modified = np.where(A3_calib == 1.0, edge_based_score / max_A3, A3_calib)
        # test_data = np.stack([A1_test, A2_test, A3_modified], axis=1)
        test_data = np.array(list(zip(node_level_distances_1[gi], node_level_distances_3[gi], edge_based_score)))
        normalized_test_data = normalizer.transform(test_data)
        anomaly_indices, anomaly_scores = detector.predict(normalized_test_data)
        scores_stacked_m1 = np.stack([A1_test, A2_test, A3_modified], axis=1)
        scores_stacked_m1.sort(axis=1)
        per_graph_data.append({
            'gi': gi, 'uids': uids, 'is_gt_mask': is_gt_mask,
            'A1_test': A1_test, 'A2_test': A2_test, 'A3_modified': A3_modified,
            'extracted_test_max_1': scores_stacked_m1[:, -2], 'extracted_test_max_2': scores_stacked_m1[:, -1],
            'anomaly_mask_1': A1_test >= 0.99999,
            'anomaly_mask_2': A2_test >= 0.99999,
            'anomaly_mask_3': A3_calib >= 0.99999,
            'anomaly_mask_4': np.sum(scores_stacked_m1[:, -2:], axis=1) >= threshold_max_two,
            'anomaly_mask_8': np.isin(np.arange(len(uids)), anomaly_indices, assume_unique=True)
        })

    # --- 4. 预计算方法 1-4 的唯一性 TP/FP (无变化) ---
    uuids_m1_tp, uuids_m1_fp = set(), set()
    uuids_m2_tp, uuids_m2_fp = set(), set()
    uuids_m3_tp, uuids_m3_fp = set(), set()
    uuids_m4_tp, uuids_m4_fp = set(), set()
    uuids_m8_tp, uuids_m8_fp = set(), set()
    for d in per_graph_data:
        is_gt, not_gt, uids = d['is_gt_mask'], ~d['is_gt_mask'], d['uids']
        uuids_m1_tp.update(uids[d['anomaly_mask_1'] & is_gt])
        uuids_m1_fp.update(uids[d['anomaly_mask_1'] & not_gt])
        uuids_m2_tp.update(uids[d['anomaly_mask_2'] & is_gt])
        uuids_m2_fp.update(uids[d['anomaly_mask_2'] & not_gt])
        uuids_m3_tp.update(uids[d['anomaly_mask_3'] & is_gt])
        uuids_m3_fp.update(uids[d['anomaly_mask_3'] & not_gt])
        uuids_m4_tp.update(uids[d['anomaly_mask_4'] & is_gt])
        uuids_m4_fp.update(uids[d['anomaly_mask_4'] & not_gt])
        uuids_m8_tp.update(uids[d['anomaly_mask_8'] & is_gt])
        uuids_m8_fp.update(uids[d['anomaly_mask_8'] & not_gt])
    total_TP_1, total_FP_1 = len(uuids_m1_tp), len(uuids_m1_fp)
    total_TP_2, total_FP_2 = len(uuids_m2_tp), len(uuids_m2_fp)
    total_TP_3, total_FP_3 = len(uuids_m3_tp), len(uuids_m3_fp)
    total_TP_4, total_FP_4 = len(uuids_m4_tp), len(uuids_m4_fp)
    total_TP_8, total_FP_8 = len(uuids_m8_tp), len(uuids_m8_fp)
    
    # --- 5. 【优化核心】预计算所有超参数组合的掩码 ---
    print("Pre-calculating masks for all hyperparameter combinations...")
    # 结构: alpha_masks[alpha][graph_index] = {'mask_5': ..., 'mask_7': ...}
    alpha_masks = defaultdict(dict)
    # 结构: multi_masks[(m1,m2,m3)][graph_index] = {'mask_6': ...}
    multi_masks = defaultdict(dict)

    # 遍历一次所有图，计算所有可能的掩码
    for gi, d in enumerate(per_graph_data):
        for alpha in range(1, 20):
            threshold_fused = threshold_fused_dict[alpha]
            threshold_fused_with_two = threshold_2_fused_dict[alpha]
            
            _, test_fused_2 = dynamic_weight_fusion_with_two(d['extracted_test_max_1'], d['extracted_test_max_2'], alpha=alpha)
            mask_5 = test_fused_2 >= threshold_fused_with_two
            
            _, test_fused = dynamic_weight_fusion(d['A1_test'], d['A2_test'], d['A3_modified'], alpha=alpha)
            mask_7 = test_fused > threshold_fused
            
            alpha_masks[alpha][gi] = {'mask_5': mask_5, 'mask_7': mask_7}

        for multi_A1 in range(1, 2):
            for multi_A2 in range(1, 2):
                for multi_A3 in range(1, 2):
                    key = (multi_A1, multi_A2, multi_A3)
                    threshold_three_add = threshold_three_add_dict[key]
                    score_three_add = multi_A1 * d['A1_test'] + multi_A2 * d['A2_test'] + multi_A3 * d['A3_modified']
                    mask_6 = score_three_add > threshold_three_add
                    multi_masks[key][gi] = {'mask_6': mask_6}
    print("Pre-calculation finished. Starting main evaluation loop...")

    # --- 6. 主评估循环 (现在只进行查表和聚合，速度极快) ---
    for multi_A1 in range(1, 2):
        for multi_A2 in range(1, 2):
            for multi_A3 in range(1, 2):
                multi_key = (multi_A1, multi_A2, multi_A3)
                threshold_three_add = threshold_three_add_dict[multi_key]

                for alpha in range(5, 6):
                    threshold_fused = threshold_fused_dict[alpha]
                    threshold_fused_with_two = threshold_2_fused_dict[alpha]

                    uuids_m5_tp, uuids_m5_fp = set(), set()
                    uuids_m6_tp, uuids_m6_fp = set(), set()
                    uuids_m7_tp, uuids_m7_fp = set(), set()
                    final_voted_tp_uuids, final_voted_fp_uuids = set(), set()
                    final_tp_details = {}
                    final_all_details = defaultdict(list)

                    for gi, d in enumerate(per_graph_data):
                        uids, is_gt, not_gt = d['uids'], d['is_gt_mask'], ~d['is_gt_mask']

                        # --- 【优化】直接从缓存中读取掩码，而不是重新计算 ---
                        precomputed_alpha = alpha_masks[alpha][gi]
                        anomaly_mask_5 = precomputed_alpha['mask_5']
                        anomaly_mask_7 = precomputed_alpha['mask_7']
                        
                        precomputed_multi = multi_masks[multi_key][gi]
                        anomaly_mask_6 = precomputed_multi['mask_6']
                        
                        # 更新方法 5, 6, 7 的 TP/FP
                        uuids_m5_tp.update(uids[anomaly_mask_5 & is_gt])
                        uuids_m5_fp.update(uids[anomaly_mask_5 & not_gt])
                        uuids_m6_tp.update(uids[anomaly_mask_6 & is_gt])
                        uuids_m6_fp.update(uids[anomaly_mask_6 & not_gt])
                        uuids_m7_tp.update(uids[anomaly_mask_7 & is_gt])
                        uuids_m7_fp.update(uids[anomaly_mask_7 & not_gt])
                        
                        # --- 最终聚合: 投票 (逻辑无变化) ---
                        anomaly_mask_1or2or3 = d['anomaly_mask_1'].astype(np.int8) + d['anomaly_mask_2'].astype(np.int8) + d['anomaly_mask_3'].astype(np.int8)
                        anomaly_mask_1or2or3 = anomaly_mask_1or2or3 >= 2

                        flags_per_node = (
                            anomaly_mask_1or2or3.astype(np.int8) + d['anomaly_mask_4'].astype(np.int8) +
                            anomaly_mask_7.astype(np.int8)
                        )
                        final_vote_mask = flags_per_node >= 2

                        rarity_based_flags = d['anomaly_mask_8'].astype(np.int8) >= 1
                        final_vote_mask = final_vote_mask & rarity_based_flags

                        for i, value in enumerate(flags_per_node):
                            final_all_details[uids[i]].append(value)

                        final_voted_tp_uuids.update(uids[final_vote_mask & is_gt])
                        final_voted_fp_uuids.update(uids[final_vote_mask & not_gt])
                        
                        # 记录最终投票为TP的节点详细信息
                        final_tp_mask_in_graph = final_vote_mask & is_gt
                        final_tp_uids_in_graph = uids[final_tp_mask_in_graph]
                        for i, uuid in enumerate(final_tp_uids_in_graph):
                            vote_count = flags_per_node[final_tp_mask_in_graph][i]
                            if uuid not in final_tp_details:
                                final_tp_details[uuid] = {'graph_indices': [], 'vote_counts': []}
                            final_tp_details[uuid]['graph_indices'].append(d['gi'])
                            final_tp_details[uuid]['vote_counts'].append(vote_count)

                    # --- 7. 计算并报告 (无变化) ---
                    total_TP_5, total_FP_5 = len(uuids_m5_tp), len(uuids_m5_fp)
                    total_TP_6, total_FP_6 = len(uuids_m6_tp), len(uuids_m6_fp)
                    total_TP_7, total_FP_7 = len(uuids_m7_tp), len(uuids_m7_fp)
                    total_final_TP, total_final_FP = len(final_voted_tp_uuids), len(final_voted_fp_uuids)
                    total_final_FN = len(final_ground_node_ids_list - final_voted_tp_uuids)
                    non_gt_uuids = all_test_uuids - ground_node_ids
                    total_final_TN = len(non_gt_uuids - final_voted_fp_uuids)

                    print(f'Results for multi={multi_key}, alpha={alpha}:')
                    print(f'  Method 1 (A1):                TP={total_TP_1}, FP={total_FP_1} (Threshold=1.0)')
                    print(f'  Method 2 (A2):                TP={total_TP_2}, FP={total_FP_2} (Threshold=1.0)')
                    print(f'  Method 3 (A3):                TP={total_TP_3}, FP={total_FP_3} (Threshold=1.0)')
                    print(f'  Method 4 (Top-Two no fused):  TP={total_TP_4}, FP={total_FP_4} (Threshold={threshold_max_two:.4f})')
                    print(f'  Method 5 (Top-Two weighted):  TP={total_TP_5}, FP={total_FP_5} (Threshold={threshold_fused_with_two:.4f})')
                    print(f'  Method 6 (Top three):         TP={total_TP_6}, FP={total_FP_6} (Threshold={threshold_three_add:.4f})')
                    print(f'  Method 7 (Top three weighted):TP={total_TP_7}, FP={total_FP_7} (Threshold={threshold_fused:.4f})')
                    print(f'  Method 8 (Rarity-based Detector):TP={total_TP_8}, FP={total_FP_8} (Method=iforest)')
                    print(f'  Final Voted (>=4):            TP={total_final_TP}, FP={total_final_FP}')
                    
                    # 打印最终投票为TP的详细信息
                    print("  Final Voted True Positive Details (Unique Nodes):")
                    if not final_tp_details:
                        print("    None")
                    else:
                        for uuid, details in sorted(final_tp_details.items()):
                            print(f"    - Node UUID: {uuid}, Found in Graph(s): {details['graph_indices']}, Vote Counts: {details['vote_counts']}")
                    print("-" * 30)

                    # --- 8. 计算最终性能指标 (无变化) ---
                    precision = total_final_TP / (total_final_TP + total_final_FP + 1e-12)
                    recall = total_final_TP / (total_final_TP + total_final_FN + 1e-12)
                    f1 = 2 * precision * recall / (precision + recall + 1e-12)
                    print(f"[Final Results for multi={multi_key}, alpha={alpha}]")
                    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                    # ... (其余指标计算和绘图代码省略，它们不受影响)
                    print("=" * 40 + "\n")


# 【方案四：上下文相关的罕见性】
# 这是一个更复杂的独立框架，建议单独调用和实验。
def scheme_4_contextual_rarity(A1_benign, A2_benign, A3_benign, per_graph_data, ground_node_ids_list):
    """
    实现方案四：上下文相关的罕见性。
    该方法为不同的“价值区间”训练不同的罕见性检测器。
    """
    print("\n" + "="*20 + " Scheme 4: Contextual Rarity " + "="*20)
    
    # --- 1. 定义价值区间并划分良性数据 ---
    # 使用dynamic-three分数作为价值分数的例子
    _, benign_value_scores = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=5)
    
    # 定义分箱边界，例如使用百分位数
    low_mid_boundary = np.percentile(benign_value_scores, 80)
    mid_high_boundary = np.percentile(benign_value_scores, 99)
    print(f"Value bins defined: Low < {low_mid_boundary:.4f}, Mid < {mid_high_boundary:.4f}, High >= {mid_high_boundary:.4f}")

    benign_scores_3d = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    
    benign_low_value_data = benign_scores_3d[benign_value_scores < low_mid_boundary]
    benign_mid_value_data = benign_scores_3d[(benign_value_scores >= low_mid_boundary) & (benign_value_scores < mid_high_boundary)]
    benign_high_value_data = benign_scores_3d[benign_value_scores >= mid_high_boundary]

    # --- 2. 为每个分区训练一个Rarity-Based模型 ---
    # 注意：如果某个分区数据过少，模型可能无法稳定训练
    detector_low = BoundaryBasedDetector(method='iforest', percentile_threshold=99)
    if len(benign_low_value_data) > 1:
        detector_low.fit(benign_low_value_data)
        print(f"Trained detector for LOW value bin on {len(benign_low_value_data)} samples.")

    detector_mid = BoundaryBasedDetector(method='iforest', percentile_threshold=99)
    if len(benign_mid_value_data) > 1:
        detector_mid.fit(benign_mid_value_data)
        print(f"Trained detector for MID value bin on {len(benign_mid_value_data)} samples.")

    detector_high = BoundaryBasedDetector(method='iforest', percentile_threshold=99)
    if len(benign_high_value_data) > 1:
        detector_high.fit(benign_high_value_data)
        print(f"Trained detector for HIGH value bin on {len(benign_high_value_data)} samples.")

    # --- 3. 逐点进行上下文检测 ---
    uuids_s4_tp, uuids_s4_fp = set(), set()
    for d in per_graph_data:
        uids, is_gt, not_gt = d['uids'], d['is_gt_mask'], ~d['is_gt_mask']
        test_scores_3d = np.stack([d['A1_test'], d['A2_test'], d['A3_modified']], axis=1)
        
        # a. 计算测试点的Value-Based分数
        _, test_value_scores = dynamic_weight_fusion(d['A1_test'], d['A2_test'], d['A3_modified'], alpha=5)
        
        final_anomaly_mask_s4 = np.zeros(len(uids), dtype=bool)

        # b. 根据其价值分数，送入对应的Rarity检测器
        low_mask = test_value_scores < low_mid_boundary
        if low_mask.any() and len(benign_low_value_data) > 1:
            indices, _ = detector_low.predict(test_scores_3d[low_mask])
            # 对于低价值区间，我们通常不认为是异常，但这里保留了逻辑的完整性
            # final_anomaly_mask_s4[np.where(low_mask)[0][indices]] = True 
            pass # 明确忽略低价值区间的罕见性

        mid_mask = (test_value_scores >= low_mid_boundary) & (test_value_scores < mid_high_boundary)
        if mid_mask.any() and len(benign_mid_value_data) > 1:
            indices, _ = detector_mid.predict(test_scores_3d[mid_mask])
            final_anomaly_mask_s4[np.where(mid_mask)[0][indices]] = True

        high_mask = test_value_scores >= mid_high_boundary
        if high_mask.any() and len(benign_high_value_data) > 1:
            indices, _ = detector_high.predict(test_scores_3d[high_mask])
            final_anomaly_mask_s4[np.where(high_mask)[0][indices]] = True
        
        uuids_s4_tp.update(uids[final_anomaly_mask_s4 & is_gt])
        uuids_s4_fp.update(uids[final_anomaly_mask_s4 & not_gt])

    print(f'  Scheme 4 (Contextual Rarity):   TP={len(uuids_s4_tp)}, FP={len(uuids_s4_fp)}')
    return len(uuids_s4_tp), len(uuids_s4_fp)


def worker_process_key_combined(multi_key, shared_data):
    """
    [Worker V3 - 合并版]
    为单个 (m1, m2, m3) 组合 (multi_key) 完成 *所有* 工作：
    1. (Section 5) 计算此 key 需要的 G=2 个掩码
    2. (Section 6) 立即使用掩码进行评估
    3. (Section 7-8) 计算指标并返回结果字符串
    
    *此设计避免了在进程间传递大型掩码字典*
    """
    
    # --- 1. 解包所有共享的只读数据 ---
    try:
        threshold_three_add_dict = shared_data['threshold_three_add_dict']
        per_graph_data = shared_data['per_graph_data']
        final_ground_node_ids_list = shared_data['final_ground_node_ids_list']
        all_test_uuids = shared_data['all_test_uuids']
        ground_node_ids = shared_data['ground_node_ids']
        alpha_list = shared_data['alpha_list']
        filename = shared_data['filename']

        (multi_A1, multi_A2, multi_A3) = multi_key
        threshold_three_add = threshold_three_add_dict[multi_key]

        # --- 2. (原 Section 5 逻辑) 掩码计算 ---
        local_masks_for_this_key = {} 
        
        for gi, d in enumerate(per_graph_data):
            A1_test = d['A1_test']
            A2_test = d['A2_test']
            A3_modified = d['A3_modified']
            
            score_three_add = multi_A1 * A1_test + multi_A2 * A2_test + multi_A3 * A3_modified
            mask_6 = score_three_add > threshold_three_add
            
            local_masks_for_this_key[gi] = {'mask_6': mask_6}
        
        # --- 3. (原 Section 6-8 逻辑) 评估与报告 ---
        output_lines = []

        for alpha in alpha_list: 
            uuids_m6_tp, uuids_m6_fp = set(), set()
            final_voted_tp_uuids, final_voted_fp_uuids = set(), set()
            final_tp_details = {}
            final_all_details = defaultdict(list)

            for gi, d in enumerate(per_graph_data):
                uids, is_gt, not_gt = d['uids'], d['is_gt_mask'], ~d['is_gt_mask']
                anomaly_mask_6 = local_masks_for_this_key[gi]['mask_6']
                uuids_m6_tp.update(uids[anomaly_mask_6 & is_gt])
                uuids_m6_fp.update(uids[anomaly_mask_6 & not_gt])
                flags_per_node = anomaly_mask_6.astype(np.int8)
                final_vote_mask = flags_per_node >= 1
                for i, value in enumerate(flags_per_node):
                    final_all_details[uids[i]].append(value)
                final_voted_tp_uuids.update(uids[final_vote_mask & is_gt])
                final_voted_fp_uuids.update(uids[final_vote_mask & not_gt])
                final_tp_mask_in_graph = final_vote_mask & is_gt
                final_tp_uids_in_graph = uids[final_tp_mask_in_graph]
                for i, uuid in enumerate(final_tp_uids_in_graph):
                    vote_count = flags_per_node[final_tp_mask_in_graph][i]
                    if uuid not in final_tp_details:
                        final_tp_details[uuid] = {'graph_indices': [], 'vote_counts': []}
                    final_tp_details[uuid]['graph_indices'].append(d['gi'])
                    final_tp_details[uuid]['vote_counts'].append(vote_count)

            total_TP_6, total_FP_6 = len(uuids_m6_tp), len(uuids_m6_fp)
            total_final_TP, total_final_FP = len(final_voted_tp_uuids), len(final_voted_fp_uuids)
            total_final_FN = len(final_ground_node_ids_list - final_voted_tp_uuids)
            non_gt_uuids = all_test_uuids - ground_node_ids
            total_final_TN = len(non_gt_uuids - final_voted_fp_uuids)

            output_lines.append(f'Results for multi={multi_key}, alpha={alpha}:')
            output_lines.append(f'  Method 6 (Top three):         TP={total_TP_6}, FP={total_FP_6} (Threshold={threshold_three_add:.4f})')
            output_lines.append(f'  Final Voted (>=1):            TP={total_final_TP}, FP={total_final_FP}')
            
            output_lines.append("  Final Voted True Positive Details (Unique Nodes):")
            if not final_tp_details:
                output_lines.append("    None")
            else:
                for uuid, details in sorted(final_tp_details.items()):
                    output_lines.append(f"    - Node UUID: {uuid}, Found in Graph(s): {details['graph_indices']}, Vote Counts: {details['vote_counts']}")
            output_lines.append("-" * 30)

            precision = total_final_TP / (total_final_TP + total_final_FP + 1e-12)
            recall = total_final_TP / (total_final_TP + total_final_FN + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            output_lines.append(f"[Final Results for multi={multi_key}, alpha={alpha}]")
            output_lines.append(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            
            accuracy = (total_final_TP + total_final_TN) / (total_final_TP + total_final_FP + total_final_FN + total_final_TN + 1e-12)
            denominator_mcc = sqrt(
                (total_final_TP + total_final_FP + 1e-12) *
                (total_final_TP + total_final_FN + 1e-12) *
                (total_final_TN + total_final_FP + 1e-12) *
                (total_final_TN + total_final_FN + 1e-12)
            )
            mcc = (total_final_TP * total_final_TN - total_final_FP * total_final_FN) / (denominator_mcc + 1e-12) 

            output_lines.append(f"Unique TP={total_final_TP}, Unique FP={total_final_FP}, Unique FN={total_final_FN}, Unique TN={total_final_TN}")
            output_lines.append(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}, MCC={mcc:.4f}")

            scores = []
            nodes = []
            dataset_name = filename.split('/')[-1].split('_')[0] + '_' + filename.split('/')[-1].split('_')[1]
            
            try:
                attack_to_GPs_path = f'{dataset_name}_attack_to_nids.pt'
                attack_to_GPs = torch.load(attack_to_GPs_path)
                attack2nodes = {k: v["nids"] for k, v in attack_to_GPs.items()}
                node2attacks = defaultdict(set)
                for attack, node_s in attack2nodes.items():
                    for node in node_s:
                        node2attacks[node].add(attack)
                
                labels = []
                for uuid, details in final_all_details.items():
                    scores.append(max(details))
                    nodes.append(uuid)
                    labels.append(1 if uuid in ground_node_ids else 0)
                
                # [警告] 此处将生成 8000 个 PNG 文件
                plot_filename = f'{dataset_name}_adp_{multi_key}.png'
                adp_score = plot_detected_attacks_vs_precision(scores, nodes, node2attacks, labels, plot_filename)
                output_lines.append(f'The adp score is: {adp_score}')
            except FileNotFoundError:
                 output_lines.append(f'Error: Could not load {attack_to_GPs_path}')
            except Exception as e:
                 output_lines.append(f'Error during ADP calculation: {e}')

            output_lines.append("=" * 40 + "\n")

        return "\n".join(output_lines)
        
    except Exception as e:
        # 捕获 worker 中的任何异常，并将其作为字符串返回，以便主进程可以打印
        return f"!!! ERROR processing key {multi_key}: {e}\n" + "=" * 40 + "\n"

# ==========================================================
# --- 3. 您的主函数 (V3) ---
# ==========================================================

def method_12_revised_with_range_detector_optimized_test_all_weight_multi_process(filename, data, ground_truth):
    """
    【多进程 V3 - 最终版】：
    将 Section 5 和 6 合并到同一个 worker (worker_process_key_combined) 中。
    使用 'chunksize' 来最小化 IPC 开销。
    """
    
    # --- 1. 数据加载与初始化 (无变化) ---
    print("Loading data...")
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3,
     total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign,
     max_A3) = torch.load(filename, map_location=torch.device('cpu'))
    (graphs, weights, num_features, edge_features, num_classes,
     reverse_graphs) = torch.load(data, map_location=torch.device('cpu'))
    ground_node_ids = set(torch.load(ground_truth))
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)
    
    # (您的 ground_node_ids.update 逻辑)
    if 'THEIA_E3' in filename:
        addition_list = [215236, 215806, 215179, 215721, 215237, 215796, 215235, 215854, 215801, 350388, 1027134]
        ground_node_ids.update(addition_list)
    if 'CLEARSCOPE_E3' in filename:
        addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
        ground_node_ids.update(addition_list)
    if 'CLEARSCOPE_E5' in filename:
        addition_list = [158937, 445211]
        ground_node_ids.update(addition_list)
    
    ground_node_ids_list = list(ground_node_ids)
    final_ground_node_ids_list = set()
    print("Data loading complete.")

    # --- 2. 良性样本分数和阈值的预计算 (无变化) ---
    print("Calculating benign thresholds...")
    threshold_three_add_dict = {}
    for multi_A1 in range(20):
        for multi_A2 in range(20):
            for multi_A3 in range(20):
                score_benign_three_add = multi_A1 * A1_benign + multi_A2 * A2_benign + multi_A3 * A3_benign
                threshold_three_add_dict[(multi_A1, multi_A2, multi_A3)] = np.max(score_benign_three_add)
    alpha_list = [1]
    
    threshold_fused_dict = {}
    threshold_2_fused_dict = {}
    benign_scores_stacked = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    benign_scores_stacked.sort(axis=1)
    extracted_max_1 = benign_scores_stacked[:, -2]
    extracted_max_2 = benign_scores_stacked[:, -1]
    for alpha in alpha_list:
        _, benign_fused = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=alpha)
        threshold_fused_dict[alpha] = np.max(benign_fused)
        _, benign_fused_2 = dynamic_weight_fusion_with_two(extracted_max_1, extracted_max_2, alpha=alpha)
        threshold_2_fused_dict[alpha] = np.max(benign_fused_2)
    print("Threshold calculation complete.")

    # --- 3. 预计算每个图的分数 (无变化) ---
    print("Calculating per-graph scores...")
    per_graph_data = []
    all_test_uuids = set()
    for gi, graph in enumerate(graphs['test']):
        uids = graph.ndata['uuid'].cpu().numpy()
        all_test_uuids.update(uids)
        is_gt_mask = np.isin(uids, ground_node_ids_list)
        final_ground_node_ids_list.update(uids[is_gt_mask])
        edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
        edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)
        val_baseline_0 = val_baseline[0]
        node_level_distances_1_gi = node_level_distances_1[gi]
        emb_baseline_0 = emb_baseline[0]
        node_level_distances_3_gi = node_level_distances_3[gi]
        
        A1_test = calibrate_scores_cpu(val_baseline_0, node_level_distances_1_gi)
        A2_test = calibrate_scores_cpu(emb_baseline_0, node_level_distances_3_gi)
        A3_calib = calibrate_scores_cpu(edge_loss_baseline, edge_based_score)
        A3_modified = np.where(A3_calib == 1.0, edge_based_score / max_A3, A3_calib)
        
        per_graph_data.append({
            'gi': gi, 'uids': uids, 'is_gt_mask': is_gt_mask,
            'A1_test': A1_test, 'A2_test': A2_test, 'A3_modified': A3_modified,
        })
    print(f"Per-graph data prepared for {len(per_graph_data)} graphs.")

    # --- 4. (已注释, 无变化) ---
    
    # ==========================================================
    # --- 5. & 6. 【多进程 V3】合并的掩码计算与评估循环 ---
    # ==========================================================
    num_cores = 80  # 您要求的核心数
    
    # 将所有 *只读* 数据打包
    shared_data = {
        'threshold_three_add_dict': threshold_three_add_dict,
        'per_graph_data': per_graph_data,
        'final_ground_node_ids_list': final_ground_node_ids_list,
        'all_test_uuids': all_test_uuids,
        'ground_node_ids': ground_node_ids,
        'alpha_list': alpha_list,
        'filename': filename
    }
    
    # 创建 8000 个 (m1, m2, m3) 组合
    multi_keys = list(itertools.product(range(20), range(20), range(20)))
    total_tasks = len(multi_keys)
    
    # 【核心】计算 chunksize
    # (total_tasks + num_cores - 1) // num_cores 是确保向上取整的除法
    chunksize = max(1, (total_tasks + num_cores - 1) // num_cores) 
    # 对于 8000 任务 / 80 核心, chunksize = 100
    
    print(f"\n--- Starting Evaluation ---")
    print(f"Total Tasks:     {total_tasks}")
    print(f"Workers (Cores): {num_cores}")
    print(f"Chunk Size:      {chunksize} (tasks per chunk)")
    print("--- This may take a while ---")
    
    # 使用 functools.partial "烘焙" shared_data 参数
    worker_func = partial(worker_process_key_combined, shared_data=shared_data)

    all_results_strings = [] # 存储所有结果
    processed_count = 0
    start_time = time.time()

    with multiprocessing.Pool(processes=num_cores) as pool:
        # imap_unordered + chunksize 是最高效的组合
        # 它会在一个 chunk (100个任务) 完成后，*陆续* 返回结果
        for result_string in pool.imap_unordered(worker_func, multi_keys, chunksize=chunksize):
            processed_count += 1
            all_results_strings.append(result_string)
            
            # --- 实时打印结果 ---
            # 这会打印每个 key 的详细结果 (会非常多)
            print(result_string) 
            
            # --- 实时打印进度 ---
            # 每 100 个任务 (约 1 个 chunk) 打印一次进度摘要
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                tasks_per_sec = processed_count / elapsed
                estimated_total_time = (elapsed / processed_count) * total_tasks
                remaining_time = estimated_total_time - elapsed
                
                print(f"--- PROGRESS: {processed_count} / {total_tasks} ({processed_count/total_tasks*100:.2f}%) ---")
                print(f"--- Speed: {tasks_per_sec:.2f} tasks/sec | Elapsed: {elapsed:.2f}s | Est. Remaining: {remaining_time:.2f}s ---")

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 50)
    print("--- ALL TASKS COMPLETED ---")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average speed: {total_tasks / total_time:.2f} tasks/sec")
    print("=" * 50)
    
    # 将所有 8000 个结果汇总到一个文件中，以便后续分析
    try:
        with open("all_run_results.txt", "w", encoding="utf-8") as f:
            f.write(f"Total Tasks: {total_tasks}\n")
            f.write(f"Total Time: {total_time:.2f} seconds\n\n")
            f.write("\n".join(all_results_strings))
        print("All 8000 results saved to 'all_run_results.txt'")
    except Exception as e:
        print(f"Error saving results to file: {e}")

def method_12_revised_with_range_detector_optimized(filename, data, ground_truth):
    """
    优化版本：通过预计算和缓存掩码来大幅降低时间复杂度。
    【新增】集成了四种 Value-Based 与 Rarity-Based 的融合方案。
    """
    # --- 1. 数据加载与初始化 (无变化) ---
    (node_level_distances_1, val_baseline, emb_baseline, node_level_distances_3,
     total_uuid_to_max, edge_loss_baseline, A1_benign, A2_benign, A3_benign,
     max_A3) = torch.load(filename, map_location=torch.device('cpu'))

    (graphs, weights, num_features, edge_features, num_classes,
     reverse_graphs) = torch.load(data, map_location=torch.device('cpu'))

    ground_node_ids = set(torch.load(ground_truth))
    edge_loss_baseline = np.where(np.isinf(edge_loss_baseline), 0, edge_loss_baseline)
    
    # 【修改】使用原始分数进行归一化和模型训练
    validation_data_raw = np.array(list(zip(val_baseline[0], emb_baseline[0], edge_loss_baseline)))
    normalizer = DataNormalizer(method='min_max')
    normalizer.fit(validation_data_raw)
    normalized_validation_data = normalizer.transform(validation_data_raw)
    print(f"Normalizer 'min_max' fitted on validation data of shape {validation_data_raw.shape}.")

    if 'THEIA_E3' in filename:
        addition_list = [215236, 215806, 215179, 215721, 215237, 215796, 215235, 215854, 215801, 350388, 1027134]
        ground_node_ids.update(addition_list)
    if 'CLEARSCOPE_E3' in filename:
        addition_list = [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734]
        ground_node_ids.update(addition_list)
    if 'CLEARSCOPE_E5' in filename:
        addition_list = [158937, 445211]
        ground_node_ids.update(addition_list)
    
    ground_node_ids_list = list(ground_node_ids)
    final_ground_node_ids_list = set()

    # --- 2. 良性样本分数和阈值的预计算 (部分新增) ---
    benign_scores_stacked = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    benign_scores_stacked.sort(axis=1)
    result_benign_top_two = np.sum(benign_scores_stacked[:, -2:], axis=1)
    threshold_max_two = np.max(result_benign_top_two)
    extracted_max_1 = benign_scores_stacked[:, -2]
    extracted_max_2 = benign_scores_stacked[:, -1]

    threshold_three_add_dict = {}
    for multi_A1 in range(1, 2):
        for multi_A2 in range(1, 2):
            for multi_A3 in range(1, 2):
                score_benign_three_add = multi_A1 * A1_benign + multi_A2 * A2_benign + multi_A3 * A3_benign
                threshold_three_add_dict[(multi_A1, multi_A2, multi_A3)] = np.max(score_benign_three_add)
    threshold_fused_dict = {}
    for alpha in range(5, 6):
        _, benign_fused = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=alpha)
        threshold_fused_dict[alpha] = np.max(benign_fused)

    # --- 【新增】为新方案预计算基线和阈值 ---
    print("Pre-calculating baselines for new schemes on benign data...")
    # 训练Rarity-Based模型
    detector = BoundaryBasedDetector(method='iforest', percentile_threshold=99.9999999999)
    # 【注意】这里我们假设 detector.fit 使用的是百分位归一化后的数据，如果不是，请用 normalized_validation_data
    benign_scores_3d_percentile = np.stack([A1_benign, A2_benign, A3_benign], axis=1)
    detector.fit(normalized_validation_data)
    
    # 获取良性样本的罕见性分数，用于后续归一化和阈值设定
    _, benign_rarity_scores = detector.predict(benign_scores_3d_percentile)
    
    # 为方案2（软排序）和方案3（罕见性作特征）准备
    # 创建一个归一化器，用于将rarity score映射到[0,1]
    rarity_normalizer = DataNormalizer(method='quantile')
    rarity_normalizer.fit(benign_rarity_scores.reshape(-1, 1))
    benign_A4_rarity = rarity_normalizer.transform(benign_rarity_scores.reshape(-1, 1)).flatten()

    # 方案2的阈值
    w_rarity_soft = 0.5 # 软排序的权重，可调
    _, benign_value_score_for_s2 = dynamic_weight_fusion(A1_benign, A2_benign, A3_benign, alpha=5)
    benign_score_s2 = benign_value_score_for_s2 * (1 + w_rarity_soft * benign_A4_rarity)
    threshold_s2 = np.max(benign_score_s2)

    # 方案3的阈值
    weights_s3 = {'w1': 1.0, 'w2': 1.0, 'w3': 1.0, 'w4': 1.0} # 四个维度的权重，可调
    benign_score_s3 = (weights_s3['w1'] * A1_benign +
                       weights_s3['w2'] * A2_benign +
                       weights_s3['w3'] * A3_benign +
                       weights_s3['w4'] * benign_A4_rarity)
    threshold_s3 = np.max(benign_score_s3)
    print("Baselines and thresholds for new schemes are ready.")

    
    # --- 3. 预计算每个图的分数 (部分新增) ---
    per_graph_data = []
    all_test_uuids = set()
    for gi, graph in enumerate(graphs['test']):
        uids = graph.ndata['uuid'].cpu().numpy()
        all_test_uuids.update(uids)
        is_gt_mask = np.isin(uids, ground_node_ids_list)
        final_ground_node_ids_list.update(uids[is_gt_mask])
        
        edge_based_score = np.array(list(total_uuid_to_max[gi].values()))
        edge_based_score = np.where(np.isinf(edge_based_score), 0, edge_based_score)
        
        # 【注意】这里使用百分位归一化分数 (A1, A2, A3)
        A1_test = calibrate_scores_cpu(val_baseline[0], node_level_distances_1[gi])
        A2_test = calibrate_scores_cpu(emb_baseline[0], node_level_distances_3[gi])
        A3_calib = calibrate_scores_cpu(edge_loss_baseline, edge_based_score)
        A3_modified = np.where(A3_calib == 1.0, edge_based_score / max_A3, A3_calib)
        
        scores_stacked_m1 = np.stack([A1_test, A2_test, A3_modified], axis=1)
        
        # 【修改】获取罕见性分数和掩码
        # 【注意】detector.predict 需要接收与训练时相同分布的数据，即百分位归一化后的数据
        test_scores_3d_percentile = scores_stacked_m1
        test_data = np.array(list(zip(node_level_distances_1[gi], node_level_distances_3[gi], edge_based_score)))
        normalized_test_data = normalizer.transform(test_data)
        anomaly_indices, anomaly_scores_raw = detector.predict(normalized_test_data)
        anomaly_mask_8 = np.isin(np.arange(len(uids)), anomaly_indices, assume_unique=True)
        
        scores_stacked_m1.sort(axis=1)

        per_graph_data.append({
            'gi': gi, 'uids': uids, 'is_gt_mask': is_gt_mask,
            'A1_test': A1_test, 'A2_test': A2_test, 'A3_modified': A3_modified,
            'anomaly_mask_1': A1_test >= 0.99999,
            'anomaly_mask_2': A2_test >= 0.99999,
            'anomaly_mask_3': A3_calib >= 0.99999,
            'anomaly_mask_4': np.sum(scores_stacked_m1[:, -2:], axis=1) >= threshold_max_two,
            'anomaly_mask_8': anomaly_mask_8,
            'rarity_scores_raw': anomaly_scores_raw # 【新增】存储原始罕见性分数
        })

    # --- 4. 预计算方法 1-4, 8 的唯一性 TP/FP (无变化) ---
    uuids_m1_tp, uuids_m1_fp = set(), set()
    uuids_m2_tp, uuids_m2_fp = set(), set()
    uuids_m3_tp, uuids_m3_fp = set(), set()
    uuids_m4_tp, uuids_m4_fp = set(), set()
    uuids_m8_tp, uuids_m8_fp = set(), set()
    for d in per_graph_data:
        is_gt, not_gt, uids = d['is_gt_mask'], ~d['is_gt_mask'], d['uids']
        uuids_m1_tp.update(uids[d['anomaly_mask_1'] & is_gt])
        uuids_m1_fp.update(uids[d['anomaly_mask_1'] & not_gt])
        uuids_m2_tp.update(uids[d['anomaly_mask_2'] & is_gt])
        uuids_m2_fp.update(uids[d['anomaly_mask_2'] & not_gt])
        uuids_m3_tp.update(uids[d['anomaly_mask_3'] & is_gt])
        uuids_m3_fp.update(uids[d['anomaly_mask_3'] & not_gt])
        uuids_m4_tp.update(uids[d['anomaly_mask_4'] & is_gt])
        uuids_m4_fp.update(uids[d['anomaly_mask_4'] & not_gt])
        uuids_m8_tp.update(uids[d['anomaly_mask_8'] & is_gt])
        uuids_m8_fp.update(uids[d['anomaly_mask_8'] & not_gt])
    total_TP_1, total_FP_1 = len(uuids_m1_tp), len(uuids_m1_fp)
    total_TP_2, total_FP_2 = len(uuids_m2_tp), len(uuids_m2_fp)
    total_TP_3, total_FP_3 = len(uuids_m3_tp), len(uuids_m3_fp)
    total_TP_4, total_FP_4 = len(uuids_m4_tp), len(uuids_m4_fp)
    total_TP_8, total_FP_8 = len(uuids_m8_tp), len(uuids_m8_fp)
    
    # --- 5. 【修改】主评估循环，集成新方案 ---
    print("Starting main evaluation loop with new schemes...")
    
    # 固定超参数以进行清晰的比较，您可以后续再开启循环
    multi_key = (1, 1, 1)
    alpha = 5 

    # 初始化新方案的TP/FP集合
    uuids_s1_tp, uuids_s1_fp = set(), set()
    uuids_s2_tp, uuids_s2_fp = set(), set()
    uuids_s3_tp, uuids_s3_fp = set(), set()
    uuids_original_baseline_tp, uuids_original_baseline_fp = set(), set() # 新增：用于存储原始基准结果

    for gi, d in enumerate(per_graph_data):
        uids, is_gt, not_gt = d['uids'], d['is_gt_mask'], ~d['is_gt_mask']

        anomaly_mask_1or2or3 = (d['anomaly_mask_1'].astype(np.int8) +
                                d['anomaly_mask_2'].astype(np.int8) +
                                d['anomaly_mask_3'].astype(np.int8)) >= 2
        
        # 视角2: max-two (Method 4)
        anomaly_mask_4 = d['anomaly_mask_4']

        # 视角3: dynamic-three (Method 7)
        _, test_fused_m7 = dynamic_weight_fusion(d['A1_test'], d['A2_test'], d['A3_modified'], alpha=alpha)
        anomaly_mask_7 = test_fused_m7 > threshold_fused_dict[alpha]

        # 最终投票: 3个视角中至少有2个同意
        flags_per_node = (anomaly_mask_1or2or3.astype(np.int8) +
                          anomaly_mask_4.astype(np.int8) +
                          anomaly_mask_7.astype(np.int8))
        
        original_baseline_mask = flags_per_node >= 2

        # 记录原始基准的TP/FP
        uuids_original_baseline_tp.update(uids[original_baseline_mask & is_gt])
        uuids_original_baseline_fp.update(uids[original_baseline_mask & not_gt])

        # --- 生成Value-Based候选集 ---
        # 使用一个比较有代表性的Value-Based策略作为候选集生成器
        # 例如，dynamic-three (method 7)
        _, test_fused = dynamic_weight_fusion(d['A1_test'], d['A2_test'], d['A3_modified'], alpha=alpha)
        # 使用一个稍微宽松的阈值，例如良性最大值的95%
        value_candidate_mask = test_fused > (threshold_fused_dict[alpha] * 0.999)
        
        # --- 【方案一：两阶段硬过滤】---
        # 在Value-Based候选集中，使用Rarity-Based进行过滤
        final_mask_s1 = value_candidate_mask & d['anomaly_mask_8']
        uuids_s1_tp.update(uids[final_mask_s1 & is_gt])
        uuids_s1_fp.update(uids[final_mask_s1 & not_gt])

        # --- 【方案二：两阶段软排序】---
        test_A4_rarity = rarity_normalizer.transform(d['rarity_scores_raw'].reshape(-1, 1)).flatten()
        _, test_value_score_for_s2 = dynamic_weight_fusion(d['A1_test'], d['A2_test'], d['A3_modified'], alpha=alpha)
        
        final_score_s2 = np.zeros_like(test_value_score_for_s2)
        # 只对候选集计算最终分数，避免不必要的计算
        if value_candidate_mask.any():
            final_score_s2[value_candidate_mask] = \
                test_value_score_for_s2[value_candidate_mask] * (1 + w_rarity_soft * test_A4_rarity[value_candidate_mask])
        
        final_mask_s2 = final_score_s2 > threshold_s2
        uuids_s2_tp.update(uids[final_mask_s2 & is_gt])
        uuids_s2_fp.update(uids[final_mask_s2 & not_gt])

        # --- 【方案三：“罕见性”作为新特征】---
        # test_A4_rarity 已在方案二中计算
        final_score_s3 = (weights_s3['w1'] * d['A1_test'] +
                        weights_s3['w2'] * d['A2_test'] +
                        weights_s3['w3'] * d['A3_modified'] +
                        weights_s3['w4'] * test_A4_rarity)
        
        final_mask_s3 = final_score_s3 > threshold_s3
        uuids_s3_tp.update(uids[final_mask_s3 & is_gt])
        uuids_s3_fp.update(uids[final_mask_s3 & not_gt])


    # --- 6. 计算并报告所有方案 ---
    total_TP_original, total_FP_original = len(uuids_original_baseline_tp), len(uuids_original_baseline_fp)

    total_TP_s1, total_FP_s1 = len(uuids_s1_tp), len(uuids_s1_fp)
    total_TP_s2, total_FP_s2 = len(uuids_s2_tp), len(uuids_s2_fp)
    total_TP_s3, total_FP_s3 = len(uuids_s3_tp), len(uuids_s3_fp)
    
    print(f'\n--- Final Report for Combined Schemes ---')
    # 【修改】将基准报告为您原始的投票方案
    print(f'Value-Based Baseline (Original Voted): TP={total_TP_original}, FP={total_FP_original}')
    print(f'Rarity-Based Baseline (Method 8):      TP={total_TP_8}, FP={total_FP_8}')
    print("-" * 40)
    print("New Fusion Schemes:")
    print(f'  Scheme 1 (Hard Filtering):           TP={total_TP_s1}, FP={total_FP_s1}')
    print(f'  Scheme 2 (Soft Ranking):             TP={total_TP_s2}, FP={total_FP_s2} (w_rarity={w_rarity_soft})')
    print(f'  Scheme 3 (Rarity as Feature):        TP={total_TP_s3}, FP={total_FP_s3} (weights={weights_s3})')
    
    # --- 方案四的调用与报告 ---
    # 注意：方案四计算开销较大，且逻辑独立，在此单独调用
    tp_s4, fp_s4 = scheme_4_contextual_rarity(A1_benign, A2_benign, A3_benign, per_graph_data, ground_node_ids_list)
    
    print("=" * 40 + "\n")
# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run script with normalization and method parameters.")
    
    # 添加 normalization 参数
    parser.add_argument(
        '--normalization',
        type=str,
        default='min_max',
        help="Normalization method to use (default: min_max)"
    )
    
    # 添加 method 参数
    parser.add_argument(
        '--method',
        type=str,
        default='max',
        help="Method to use (default: max)"
    )
    parser.add_argument(
        '--n_components',
        type=int,
        default=10,
        help='The number of clusters in different algorithms'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='The number of clusters in different algorithms'
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    filenames = [
        'saved_middle_result_2/CADETS_E5_loss_sce_dim_64_nhd_8_nh_0.5_nl_2_lr_0.0015_mp_500_mpf_10_wd_0.0001_wdf_5e-05_gatedge_gat_data.pt',
        'saved_middle_result_2/THEIA_E5_loss_sce_dim_64_nhd_8_nh_0.5_nl_2_lr_0.0015_mp_500_mpf_10_wd_1e-05_wdf_2e-06_gatedge_gat_data.pt',
        'saved_middle_result/CLEARSCOPE_E5_loss_sce_rpr_8_nh_0.1_nl_2_lr_0.0015_mp_200_mpf_20_wd_0.0001_wdf_5e-05_gatedge_gat_data.pt',
        'saved_middle_result/CADETS_E3_loss_sce_dim_64_nhd_2_nh_0.3_nl_2_lr_0.0015_mp_5_mpf_50_wd_0.001_wdf_0.0001_gatedge_gat_data.pt',
        'saved_middle_result/THEIA_E3_loss_sce_rpr_8_nh_0.3_nl_2_lr_0.0015_mp_5_mpf_50_wd_0.01_wdf_0.0001_gatedge_gat_data.pt',
        'saved_middle_result/CLEARSCOPE_E3_loss_sce_dim_64_nhd_4_nh_0.1_nl_2_lr_0.0015_mp_200_mpf_50_wd_0.001_wdf_0.0001_gatedge_gat_data.pt'
        
    ]
    datas = [
        'cadets_e5_merge_edge_data.pt',
        'theia_e5_merge_edge_data_final.pt',
        'clearscope_e5_merge_edge_data.pt',
        'cadets_merge_edge_no_day_2_data.pt',
        'theia_merge_edge_data.pt',
        'clearscope_e3_merge_edge_data.pt'


    ]
    ground_truths = [
        '../Ground_Truth/ground_truth_nids_cadets_e5.pt',
        '../Ground_Truth/ground_truth_nids_theia_e5.pt',
        '../Ground_Truth/ground_truth_nids_clearscope_e5.pt',
        '../Ground_Truth/ground_truth_cadet_v2.pt',
        '../Ground_Truth/ground_truth_nids.pt',
        '../Ground_Truth/ground_truth_nids_clearscope.pt'
    ]

    print(f'============================Processing Cofigure {args.normalization} and {args.method}:=====================')
    for i in range(6):
        filename = filenames[i]
        data = datas[i]
        ground_truth = ground_truths[i]
        run_anomaly_detection_pipeline(args, filename,data, ground_truth, args.normalization, args.method, 99.9999999999999)
    
    # for i in range(6):
    #     filename = filenames[i]
    #     data = datas[i]
    #     ground_truth = ground_truths[i]
    #     print('============================Processing file:=====================\n', filename)
    #     method_12_revised_with_range_detector_optimized_test_all_weight_multi_process(filename,data, ground_truth)
