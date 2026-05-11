import os
import re
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ==================== 配置区 ====================
# 数据集顺序（固定）
DATASETS = [
    "cadets-e5",
    "theia-e5",
    "clearscope-e5",
    "cadets-e3",
    "theia-e3",
    "clearscope-e3"
]

# 支持的 normalization
NORM_METHODS = {"z_score", "quantile", "min_max", "robust"}


# 日志目录
LOG_DIR = "./my_ablation_analysis"  

# 输出 CSV 文件名
OUTPUT_CSV = "best_mcc_per_method_per_dataset.csv"
# =================================================

# 正则表达式
FILE_PATTERN = re.compile(
    r"(?P<norm>(z_score|quantile|min_max|robust))_"
    r"(?P<method>\w+)_"
    r"(?P<n_comp>\d+)_"
    r"(?P<lr>[\deE\.\-+]+)\.log$"
)

# 日志块分隔线
BLOCK_SEPARATOR = "=" * 50  # Change to this; count confirmed as 50 '=' in the log separators

# 结果提取正则
TP_FP_FN_TN_PATTERN = re.compile(
    r"Total Unique True Positives \(TP\): (\d+)\s+"
    r"Total Unique False Positives \(FP\): (\d+)\s+"
    r"Total Unique False Negatives \(FN\): (\d+)\s+"
    r"Total Unique True Negatives \(TN\): (\d+)"
)
METRICS_PATTERN = re.compile(
    r"Precision=([\d\.]+), Recall=([\d\.]+), F1=([\d\.]+), Accuracy=([\d\.]+), MCC=([\d\.\-]+)"
)
ADP_PATTERN = re.compile(r"ADP Score: ([\d\.e\+\-]+)")

def parse_log_file(filepath):
    """解析单个 .log 文件，提取 6 个数据集的结果"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"无法读取 {filepath}: {e}")
        return None

    # Split on the dataset separator
    blocks = content.split(BLOCK_SEPARATOR)
    if len(blocks) < 6:
        print(f"警告: {filepath} 块数不足，期望至少6个，实际{len(blocks)}")
        return None

    results = []
    for idx, block in enumerate(blocks[:6]):  # Take first 6 blocks (handles header in block[0])
        if idx >= 6:
            break

        dataset_name = DATASETS[idx]

        # 提取 TP/FP/FN/TN
        tp_match = TP_FP_FN_TN_PATTERN.search(block)
        if not tp_match:
            print(f"警告: {filepath} 数据集 {dataset_name} 缺少 TP/FP/FN/TN")
            continue
        tp, fp, fn, tn = map(int, tp_match.groups())

        # 提取 Metrics
        metrics_match = METRICS_PATTERN.search(block)
        if not metrics_match:
            print(f"警告: {filepath} 数据集 {dataset_name} 缺少 Metrics")
            continue
        prec, rec, f1, acc, mcc = map(float, metrics_match.groups())

        # 提取 ADP
        adp_match = ADP_PATTERN.search(block)
        adp = float(adp_match.group(1)) if adp_match else None

        results.append({
            "dataset": dataset_name,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": prec, "Recall": rec, "F1": f1, "Accuracy": acc,
            "MCC": mcc, "ADP": adp
        })

    if len(results) != 6:
        print(f"警告: {filepath} 只提取到 {len(results)} 个数据集")
        return None if len(results) == 0 else results  # Allow partial if some missing

    return results


def extract_config_from_filename(filename):
    """从文件名提取 norm, method, n_comp, lr"""
    match = FILE_PATTERN.match(filename)
    if not match:
        return None
    d = match.groupdict()
    return {
        "normalization": d["norm"],
        "method": d["method"],
        "n_components": int(d["n_comp"]),
        "lr": d["lr"]
    }


def main():
    log_dir = Path(LOG_DIR)
    if not log_dir.exists():
        print(f"目录不存在: {LOG_DIR}")
        return

    all_results = []

    print("正在扫描日志文件...")
    for file_path in log_dir.glob("*.log"):
        filename = file_path.name
        config = extract_config_from_filename(filename)
        if not config:
            print(f"跳过无效文件名: {filename}")
            continue

        print(f"解析: {filename}")
        dataset_results = parse_log_file(file_path)
        if not dataset_results:
            continue

        for res in dataset_results:
            record = {
                "file": filename,
                "normalization": config["normalization"],
                "method": config["method"],
                "n_components": config["n_components"],
                "lr": config["lr"],
                **res
            }
            all_results.append(record)

    if not all_results:
        print("没有解析到任何有效结果！")
        return

    df = pd.DataFrame(all_results)
    print(f"\n共解析 {len(df)} 条记录（6数据集 × 文件数）")

    # 按 method + dataset 分组，找出 MCC 最高的配置
    best_records = []
    grouped = df.groupby(['method', 'dataset'])

    for (method, dataset), group in grouped:
        if group.empty:
            continue
        best_idx = group['MCC'].idxmax()
        best = group.loc[best_idx].copy()
        best_records.append(best)

    best_df = pd.DataFrame(best_records)

    # 排序：先按 method，再按 dataset
    best_df = best_df.sort_values(['method', 'dataset']).reset_index(drop=True)

    # 选择输出列（包含所有关键指标）
    cols = [
        'method', 'dataset', 'normalization', 'n_components', 'lr',
        'MCC', 'F1', 'Precision', 'Recall', 'Accuracy', 'ADP',
        'TP', 'FP', 'FN', 'TN'
    ]
    output_df = best_df[cols].copy()

    # 保存 CSV
    output_df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
    print(f"\n最佳配置已保存至: {OUTPUT_CSV}")

    # 打印详细表格
    print("\n" + "="*180)
    print("每个方法+数据集的最佳配置 (按 MCC 排序)".center(180))
    print("="*180)
    print(f"{'方法':<12} {'数据集':<15} {'Norm':<8} {'Comp':>4} {'LR':>8} {'MCC':>7} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Acc':>7} {'ADP':>9} {'TP':>4} {'FP':>4} {'FN':>5} {'TN':>10}")
    print("-"*180)
    
    for _, row in output_df.iterrows():
        print(f"{row['method']:<12} {row['dataset']:<15} {row['normalization']:<8} "
              f"{row['n_components']:>4} {row['lr']:>8} {row['MCC']:>7.4f} {row['F1']:>6.4f} "
              f"{row['Precision']:>6.4f} {row['Recall']:>6.4f} {row['Accuracy']:>7.4f} "
              f"{row['ADP']:>9.4f} {row['TP']:>4} {row['FP']:>4} {row['FN']:>5} {row['TN']:>10,}")
    print("-"*180)
    
    # 按 MCC 总体排序的汇总
    print("\n" + "="*180)
    print("按 MCC 总体排序的最佳结果".center(180))
    print("="*180)
    mcc_sorted = output_df.sort_values('MCC', ascending=False)
    for _, row in mcc_sorted.iterrows():
        print(f"{row['method']:<12} {row['dataset']:<15} {row['normalization']:<8} "
              f"{row['n_components']:>4} {row['lr']:>8} {row['MCC']:>7.4f} {row['F1']:>6.4f} "
              f"{row['Precision']:>6.4f} {row['Recall']:>6.4f} {row['Accuracy']:>7.4f} "
              f"{row['ADP']:>9.4f} | TP:{row['TP']:>3} FP:{row['FP']:>3} FN:{row['FN']:>3} TN:{row['TN']:>8,}")
    print("-"*180)

    # 统计信息
    print(f"\n统计摘要:")
    print(f"- 总方法数: {best_df['method'].nunique()}")
    print(f"- 总数据集数: {len(best_df)} (期望 {len(DATASETS) * best_df['method'].nunique()})")
    print(f"- 最佳 MCC: {best_df['MCC'].max():.6f}")
    print(f"- 最差 MCC: {best_df['MCC'].min():.6f}")
    print(f"- 平均 MCC: {best_df['MCC'].mean():.6f}")


if __name__ == "__main__":
    main()