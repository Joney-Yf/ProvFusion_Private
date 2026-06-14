# OpenProvFusion

基于 provenance graph 的 APT 检测框架（GraphMAE 自监督编码 + 边重构，percentile 阈值投票）。

> **本分支 `no_data_prepration_version`**：接入 `data_preparation/`（原始日志 / Postgres → edge_embeds）
> **之前**的版本快照，仅含检测 / 训练框架本体。带数据准备模块的新版见 `main`。

## E3 数据集：已验证的最佳结果（如实记录）

下表如实记录 E3 三个数据集**已验证最佳结果**对应的：结果中间文件名、训练数据文件名、
ground truth 文件名、超参数与指标。文件名均与服务器上实际文件逐字符核对过；
`--data_path` 取自各 `*_e3.sh` 扫描脚本实际加载的文件（注意 THEIA / CLEARSCOPE 用的是
`*_merge_edge_data.pt`，并非 `theia_e3.pt` / `clearscope_e3.pt`）。
评估统一使用 `method_12_with_different_normalization(..., normalization_method='percentile')`。

### 数据 / Ground Truth 文件（脚本实际加载）

| Dataset       | `--data_path`                      | `--ground_truth_path`                             |
|---------------|------------------------------------|---------------------------------------------------|
| THEIA_E3      | `theia_merge_edge_data.pt`         | `../Ground_Truth/ground_truth_nids.pt`            |
| CADETS_E3     | `cadets_e3.pt`                     | `../Ground_Truth/ground_truth_cadet_v2.pt`        |
| CLEARSCOPE_E3 | `clearscope_e3_merge_edge_data.pt` | `../Ground_Truth/ground_truth_nids_clearscope.pt` |

### 最佳结果中间文件（`saved_middle_result/`）与指标

| Dataset       | 最佳结果文件名 | TP | FP |
|---------------|----------------|----|----|
| THEIA_E3      | `THEIA_E3_loss_sce_rpr_8_nh_0.3_nl_2_lr_0.0015_mp_5_mpf_50_wd_0.01_wdf_0.0001_gatedge_gat_data.pt`        | 91 | 2 |
| CADETS_E3     | `CADETS_E3_loss_sce_dim_64_nhd_2_nh_0.3_nl_2_lr_0.0015_mp_5_mpf_50_wd_0.001_wdf_0.0001_gatedge_gat_data.pt` | 24 | 1 |
| CLEARSCOPE_E3 | `CLEARSCOPE_E3_loss_sce_dim_64_nhd_4_nh_0.1_nl_2_lr_0.0015_mp_200_mpf_50_wd_0.001_wdf_0.0001_gatedge_gat_data.pt` | 6 | 7 |

### 最佳超参数

| Dataset       | num_hidden | num_heads | mask_rate | num_layers | lr     | max_epoch | max_epoch_f | weight_decay | weight_decay_f |
|---------------|------------|-----------|-----------|------------|--------|-----------|-------------|--------------|----------------|
| THEIA_E3      | 默认（文件名无 `dim`） | 8 | 0.3 | 2 | 0.0015 | 5   | 50 | 0.01  | 0.0001 |
| CADETS_E3     | 64         | 2         | 0.3       | 2          | 0.0015 | 5         | 50          | 0.001        | 0.0001         |
| CLEARSCOPE_E3 | 64         | 4         | 0.1       | 2          | 0.0015 | 200       | 50          | 0.001        | 0.0001         |

> **CLEARSCOPE_E3 复现说明**：基线 TP=6/FP=7 已在本框架复现（seed=1 同配置重跑，未改代码、未换 seed）：
> `save_middle_results/CLEARSCOPE_E3_RERUN_4.pt` → TP=6, FP=4，命中与基线完全相同的 6 个攻击节点。
> 复现差距集中在 A3 边重构头（`shuffle=True + num_workers=32` 本质不可复现），需同配置多跑捕捉训练方差。
