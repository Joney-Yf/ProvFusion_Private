#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
from collections import defaultdict
import csv
from pathlib import Path


# ==================== 配置区 ====================
# 用 total (TP+FP+FN+TN) 匹配数据集 ID
DATASET_TOTALS = {
    0: 3136090,   # cadets-e5
    1: 1858987,   # theia-e5
    2: 150964,    # clearscope-e5
    3: 281585,    # cadets-e3
    4: 701622,    # theia-e3
    5: 111403,    # clearscope-e3
}

DATASET_NAMES = [
    "cadets-e5", "theia-e5", "clearscope-e5",
    "cadets-e3", "theia-e3", "clearscope-e3"
]

TOLERANCE = 50
CSV_OUTPUT = "best_mcc_results.csv"  # 输出文件名
# =================================================

HEADER_PATTERN = re.compile(r"\[Final Results for multi=\((\d+),\s*(\d+),\s*(\d+)\),\s*alpha=(\d+)\]")
METRICS_PATTERN = re.compile(r"Precision=([\d\.]+),\s*Recall=([\d\.]+),\s*F1=([\d\.]+),\s*Accuracy=([\d\.]+),\s*MCC=([\d\.\-]+)")
COUNTS_PATTERN = re.compile(r"Unique TP=(\d+),\s*Unique FP=(\d+),\s*Unique FN=(\d+),\s*Unique TN=(\d+)")
ADP_PATTERN = re.compile(r"The adp score is:\s*([\d\.e\+\-]+)")


def parse_log_file(filepath):
    records = []
    current = {}
    try:
        lines = [line.strip() for line in open(filepath, 'r', encoding='utf-8') if line.strip()]
    except Exception as e:
        print(f"读取失败: {e}", file=sys.stderr)
        return records

    i = 0
    while i < len(lines):
        line = lines[i]

        # 新配置开始
        m_header = HEADER_PATTERN.search(line)
        if m_header:
            # 保存上一个配置
            if 'TP' in current and 'MCC' in current:
                tp, fp, fn, tn = current['TP'], current['FP'], current['FN'], current['TN']
                total = tp + fp + fn + tn
                dataset_id = None
                for did, expected in DATASET_TOTALS.items():
                    if abs(total - expected) <= TOLERANCE:
                        dataset_id = did
                        break
                if dataset_id is not None:
                    record = {
                        'dataset_id': dataset_id,
                        'dataset': DATASET_NAMES[dataset_id],
                        'A': current['A'], 'B': current['B'], 'C': current['C'], 'D': current['D'],
                        'MCC': current['MCC'], 'F1': current['F1'], 'Precision': current['Precision'],
                        'Recall': current['Recall'], 'Accuracy': current['Accuracy'],
                        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                        'ADP': current.get('ADP')
                    }
                    records.append(record)

            # 新配置
            A, B, C, D = map(int, m_header.groups())
            current = {'A': A, 'B': B, 'C': C, 'D': D}
            i += 1
            continue

        # 指标行
        m_metrics = METRICS_PATTERN.search(line)
        if m_metrics:
            prec, rec, f1, acc, mcc = map(float, m_metrics.groups())
            current.update({'Precision': prec, 'Recall': rec, 'F1': f1, 'Accuracy': acc, 'MCC': mcc})
            i += 1
            continue

        # 计数行
        m_counts = COUNTS_PATTERN.search(line)
        if m_counts:
            tp, fp, fn, tn = map(int, m_counts.groups())
            current.update({'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn})
            i += 1
            continue

        # ADP
        m_adp = ADP_PATTERN.search(line)
        if m_adp:
            current['ADP'] = float(m_adp.group(1))
            i += 1
            continue

        i += 1

    # 最后一个配置
    if 'TP' in current and 'MCC' in current:
        tp, fp, fn, tn = current['TP'], current['FP'], current['FN'], current['TN']
        total = tp + fp + fn + tn
        dataset_id = None
        for did, expected in DATASET_TOTALS.items():
            if abs(total - expected) <= TOLERANCE:
                dataset_id = did
                break
        if dataset_id is not None:
            record = {
                'dataset_id': dataset_id,
                'dataset': DATASET_NAMES[dataset_id],
                'A': current['A'], 'B': current['B'], 'C': current['C'], 'D': current['D'],
                'MCC': current['MCC'], 'F1': current['F1'], 'Precision': current['Precision'],
                'Recall': current['Recall'], 'Accuracy': current['Accuracy'],
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                'ADP': current.get('ADP')
            }
            records.append(record)

    return records


def find_best_mcc_per_dataset(records):
    best = defaultdict(lambda: {'MCC': -float('inf')})
    for rec in records:
        did = rec['dataset_id']
        if rec['MCC'] > best[did]['MCC']:
            best[did] = rec.copy()
    return dict(best)


def save_to_csv(best_results, csv_path):
    """保存最佳结果到 CSV"""
    fieldnames = [
        'dataset', 'A', 'B', 'C', 'D',
        'MCC', 'F1', 'Precision', 'Recall', 'Accuracy',
        'TP', 'FP', 'FN', 'TN', 'ADP'
    ]
    rows = []
    for r in best_results.values():
        row = {k: r.get(k) for k in fieldnames}
        rows.append(row)

    # 按 MCC 降序
    rows.sort(key=lambda x: x['MCC'], reverse=True)

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV 已保存: {csv_path}")
    except Exception as e:
        print(f"CSV 保存失败: {e}", file=sys.stderr)


def main(log_filepath):
    print(f"正在解析日志: {log_filepath}")
    records = parse_log_file(log_filepath)
    if not records:
        print("未解析到任何记录！")
        return

    print(f"解析到 {len(records)} 条记录")

    best_results = find_best_mcc_per_dataset(records)

    # === 控制台输出 ===
    print("\n最佳 MCC 配置（按 MCC 降序）")
    print("=" * 170)
    header = (
        f"{'数据集':<15} {'A':>3} {'B':>3} {'C':>3} {'D':>3} "
        f"{'MCC':>10} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8} "
        f"{'TP':>6} {'FP':>6} {'FN':>6} {'TN':>10} {'ADP':>12}"
    )
    print(header)
    print("-" * 170)

    sorted_best = sorted(best_results.values(), key=lambda x: x['MCC'], reverse=True)
    for r in sorted_best:
        adp_str = f"{r['ADP']:.6f}" if r['ADP'] is not None else "N/A"
        print(
            f"{r['dataset']:<15} {r['A']:>3} {r['B']:>3} {r['C']:>3} {r['D']:>3} "
            f"{r['MCC']:>10.6f} {r['F1']:>8.6f} {r['Precision']:>8.6f} {r['Recall']:>8.6f} {r['Accuracy']:>8.6f} "
            f"{r['TP']:>6} {r['FP']:>6} {r['FN']:>6} {r['TN']:>10,} {adp_str:>12}"
        )
    print("-" * 170)

    # === 保存 CSV ===
    save_to_csv(best_results, CSV_OUTPUT)

    # === 验证 ===
    max_mcc = max(r['MCC'] for r in records)
    print(f"\n全局最高 MCC: {max_mcc:.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python analyze_mcc.py <log_file>")
        sys.exit(1)
    main(sys.argv[1])