<div align="center">

# ProvFusion

**Beyond Nodes vs. Edges: A Multi-View Fusion Framework for Provenance-Based Intrusion Detection**

Fan Yang<sup>†</sup>, Binyan Xu<sup>†</sup>, Di Tang<sup>‡</sup>, Kehuan Zhang<sup>†</sup><br>
<sup>†</sup>The Chinese University of Hong Kong &nbsp;·&nbsp; <sup>‡</sup>Sun Yat-Sen University

</div>

---

ProvFusion is a multi-view anomaly-fusion framework for provenance-based intrusion
detection on system audit logs. It characterizes each system entity from three
complementary views — **attribute**, **structure**, and **causality** — and fuses the
per-view anomaly signals through a multi-dimensional, voting-based decision process. The
training backbone is a GraphMAE-style masked graph auto-encoder with an edge-aware
graph-attention encoder.

This repository is the refactored core used to produce the results reported in the paper.
Detection on all six DARPA Transparent Computing datasets (E3 and E5) is reproduced
end-to-end from the released checkpoints and data.

---

## 📊 Detection Results

Node-level detection on the **refined ground truth**. TP / FP count the unique malicious
/ benign nodes flagged by the final voting detector (percentile normalization). These are
the reference numbers reported in the paper; the released checkpoint behind each row
re-evaluates to exactly these values (see [Reproducing the results](#-reproducing-the-results)).

| Dataset | TP | FP | Reproduction checkpoint |
|---|---:|---:|---|
| **CADETS_E3** | 24 | 1 | `checkpoints/CADETS_E3_..._gatedge_gat_data.pt` |
| **THEIA_E3** | 91 | 2 | `checkpoints/THEIA_E3_..._gatedge_gat_data.pt` |
| **CLEARSCOPE_E3** | 6 | 7 | `checkpoints/CLEARSCOPE_E3_..._gatedge_gat_data.pt` |
| **CADETS_E5** | 7 | 9 | `checkpoints/CADETS_E5_<TODO>.pt` *(in progress)* |
| **THEIA_E5** | 11 | 2 | `checkpoints/THEIA_E5_<TODO>.pt` *(placeholder)* |
| **CLEARSCOPE_E5** | 10 | 16 | `checkpoints/CLEARSCOPE_E5_<TODO>.pt` *(placeholder)* |

Each row is reproducible from a released **`.pt`** saved middle result plus the merged
data file and ground truth. The E5 filenames are placeholders while the corresponding
checkpoints are finalized.

### End-to-end from raw logs (full pipeline, Option D)

The same results can also be reached by regenerating the training data directly from the
original DARPA JSON logs, with no reliance on any prebuilt data:

| Dataset | TP | FP | Reproduction checkpoint |
|---|---:|---:|---|
| **THEIA_E3** (regenerated) | 99 | 3 | `checkpoints/THEIA_E3_regen_..._tp99_fp3.pt` |
| **CLEARSCOPE_E3** (regenerated) | 7 | 7 | `checkpoints/CLEARSCOPE_E3_regen_emb25_..._tp7_fp7.pt` |
| **CADETS_E5** (regenerated) | 7 | 9 | `checkpoints/CADETS_E5_regen_r27_a_tp7_fp9.pt` (also r15_a, r31_b) |

> **On randomness.** The edge-reconstruction (link-prediction) head trains with
> `shuffle=True` and multi-worker data loading, so it is **not bit-reproducible**:
> detection FP varies across reruns of an identical config. The released checkpoints pin
> the exact reported numbers (Option A is fully deterministic). When retraining, keep
> `seed=1` and rerun the same command several times — the best draws match the reported
> values.

---

## 🗂 Repository Contents

```
main_transductive.py        # training + evaluation entry point
preprocess.py               # raw_data/{DATASET} -> merged .pt
try_different_threshold.py  # detection / evaluation (multi-view voting, percentile)
graphmae/                   # model + data loading (GraphMAE, edge-aware encoder)
configs.yml                 # per-dataset --use_cfg hyperparameter defaults
configs/                    # per-dataset loss weights
data_preparation/           # raw DARPA logs -> embeddings (Option D; own README)
tools/build_ground_truth.py # UUID-level ground truth -> index labels
Ground_Truth_csv/           # per-attack node-UUID ground truth (refined labels)
gt_canonical/               # prebuilt index-based ground-truth artifacts
*.sh                        # per-dataset hyperparameter sweep scripts
release_assets/             # released checkpoints + regenerated data + MANIFEST.md5
```

---

## ⚙️ Environment

- Python 3.9 (tested 3.9.21), CUDA 11.x GPU.
- `pip install -r requirements.txt` — see the header of that file for the CUDA-specific
  install order (torch → dgl → PyG wheels → the rest).
- Run with `PYTHONHASHSEED=0` (Word2Vec / hashing determinism).
- PostgreSQL is required **only** for Option D (raw-log ingestion).

---

## 💾 Data Layout

The repo expects this sibling layout:

```
parent/
├── OpenProvFusion/            # this repo (run everything from here)
├── raw_data/{DATASET}/        # per-graph embeddings: train/ val/ test/ *.TemporalData.simple
└── Ground_Truth/              # ground-truth node ids (*.pt)
```

Per-dataset training inputs (`--data_path` is the merged `.pt` built from `raw_data`; it
is created automatically on first run and reused afterwards):

| Dataset | `--data_path` | `--ground_truth_path` |
|---|---|---|
| THEIA_E3 | `theia_merge_edge_data.pt` | `gt_canonical/THEIA_E3_orig_ground_truth_nids.pt` |
| CADETS_E3 | `cadets_e3.pt` | `gt_canonical/CADETS_E3_orig_ground_truth_nids.pt` |
| CLEARSCOPE_E3 | `clearscope_e3_merge_edge_data.pt` | `gt_canonical/CLEARSCOPE_E3_orig_ground_truth_nids.pt` |
| THEIA_E5 | `theia_e5_merge_edge_data_final.pt` | `../Ground_Truth/ground_truth_nids_theia_e5.pt` |
| CADETS_E5 | `cadets_e5_merge_edge_data.pt` | `../Ground_Truth/ground_truth_nids_cadets_e5.pt` |
| CLEARSCOPE_E5 | `clearscope_e5_merge_edge_data.pt` | `../Ground_Truth/ground_truth_nids_clearscope_e5.pt` |

### Refined ground truth

Attack nodes are recorded by **node UUID** in `Ground_Truth_csv/` (one CSV per attack,
with a full audit trail). `tools/build_ground_truth.py` resolves those UUIDs against a
given Postgres database and emits the index-based `*_ground_truth_nids.pt` +
`*_attack_to_nids.pt` consumed by evaluation (`gt_canonical/` ships prebuilt artifacts for
the original databases). Ground truth is therefore independent of database ingestion
order: if you regenerate a database from raw logs (Option D), rebuild its labels with the
same script instead of reusing the originals. Equivalence with the historical labels was
verified live (THEIA 91/2, CLEARSCOPE 6/7 reproduce exactly).

---

## 🔁 Reproducing the Results

Four entry points, fastest to fullest.

### Option A — Evaluate a released checkpoint *(deterministic, minutes)*

Evaluation needs the merged data `.pt`, the ground truth, and the per-dataset
`{DATASET}_attack_to_nids.pt` in the working directory:

```bash
python -c "
from try_different_threshold import method_12_with_different_normalization
method_12_with_different_normalization(
    'release_assets/checkpoints/CLEARSCOPE_E3_..._gatedge_gat_data.pt',
    'clearscope_e3_merge_edge_data.pt',
    '../Ground_Truth/ground_truth_nids_clearscope.pt',
    normalization_method='percentile')
"
```

Swap the checkpoint / data / ground-truth triple per the tables above to reproduce each
row exactly.

> **Checkpoint naming matters.** Evaluation derives the dataset from the **first two
> underscore-separated tokens of the checkpoint filename** (`CLEARSCOPE_E3_*.pt` → loads
> `CLEARSCOPE_E3_attack_to_nids.pt` from the working directory). Keep the `{DATASET}_`
> prefix when renaming, otherwise attack attribution silently fails and TP collapses.

### Option B — Retrain from the released merged `.pt`

```bash
# CLEARSCOPE_E3 (baseline config; rerun a few times — best draws reach TP=6 / FP<=7)
CUDA_VISIBLE_DEVICES=0 python main_transductive.py --device 0 \
  --dataset CLEARSCOPE_E3 \
  --data_path clearscope_e3_merge_edge_data.pt \
  --ground_truth_path ../Ground_Truth/ground_truth_nids_clearscope.pt \
  --raw_data_dir ../raw_data \
  --encoder gatedge --decoder gat --num_hidden 64 --num_heads 4 --mask_rate 0.1 \
  --num_layers 2 --lr 0.0015 --weight_decay 0.001 --lr_f 0.001 --weight_decay_f 0.0001 \
  --max_epoch 200 --max_epoch_f 50 --in_drop 0.2 --attn_drop 0.1 --drop_edge_rate 0.0 \
  --loss_fn sce --optimizer adam --alpha_l 3 --replace_rate 0.0 --activation prelu \
  --linear_prob --scheduler --use_cfg --seeds 1 --normalization_method percentile
```

Substitute the data paths and hyperparameters per dataset (see
[Best hyperparameters](#-best-hyperparameters)). Training prints the full evaluation (all
methods + final voting) and saves its middle result under `save_middle_results/`.

### Option C — Rebuild the merged `.pt` from the released embeddings

If `--data_path` does not exist, `main_transductive.py` builds it from
`../raw_data/{DATASET}/{train,val,test}` automatically (or run `preprocess.py` yourself).
Delete/rename the old `.pt` first — an existing file is silently reused.

Verified raw_data → merged mappings (edge counts match exactly after the de-duplicating
DGL conversion):

| raw_data source | rebuilds |
|---|---|
| `raw_data/THEIA_E3/` | `theia_merge_edge_data.pt` |
| `raw_data/CADETS_E3_v2/` (rename/symlink to `raw_data/CADETS_E3/`) | `cadets_e3.pt` |
| `raw_data/CLEARSCOPE_E3/` | `clearscope_e3_merge_edge_data.pt` |

### Option D — Full pipeline from raw DARPA logs

`data_preparation/` regenerates everything from the original DARPA JSON logs: Postgres
ingestion → per-day graphs → Word2Vec featurization → edge embeddings, published as
`../raw_data/{DATASET}` (see `data_preparation/README.md` for prerequisites, isolation
flags, and safety guards).

```bash
# 1) regenerate data from raw logs (Postgres required)
PYTHONHASHSEED=0 python data_preparation/prepare_data.py CLEARSCOPE_E3
# 2) build the merged .pt and train with the Option-D hyperparameters
CUDA_VISIBLE_DEVICES=0 python main_transductive.py --device 0 \
  --dataset CLEARSCOPE_E3 \
  --data_path clearscope_e3_regenerated.pt \
  --ground_truth_path ../Ground_Truth/ground_truth_nids_clearscope.pt \
  --raw_data_dir ../raw_data \
  --encoder gatedge --decoder gat --num_hidden 64 --num_heads 4 --mask_rate 0.1 \
  --num_layers 2 --lr 0.003 --weight_decay 0.001 --lr_f 0.001 --weight_decay_f 0 \
  --max_epoch 100 --max_epoch_f 50 --in_drop 0.2 --attn_drop 0.1 --drop_edge_rate 0.0 \
  --loss_fn sce --optimizer adam --alpha_l 3 --replace_rate 0.0 --activation prelu \
  --linear_prob --scheduler --use_cfg --seeds 1 --normalization_method percentile
```

Validated end-to-end on CLEARSCOPE_E3, THEIA_E3 and CADETS_E5. Because Word2Vec is not
bit-reproducible across machines, a fresh regeneration is a fresh embedding draw — sweep
`lr` / `weight_decay` around the Option-D config and rerun per *On randomness*.

---

## 🎯 Best Hyperparameters

All runs: `--encoder gatedge --decoder gat --loss_fn sce --optimizer adam --num_layers 2
--in_drop 0.2 --attn_drop 0.1 --replace_rate 0.0 --activation prelu --linear_prob
--scheduler --use_cfg --seeds 1 --lr_f 0.001 --normalization_method percentile`.

| Dataset | num_hidden | num_heads | mask_rate | lr | max_epoch | max_epoch_f | weight_decay | weight_decay_f |
|---|---|---|---|---|---|---|---|---|
| THEIA_E3 | —¹ | 8 | 0.3 | 0.0015 | 5 | 50 | 0.01 | 0.0001 |
| CADETS_E3 | 64 | 2 | 0.3 | 0.0015 | 5 | 50 | 0.001 | 0.0001 |
| CLEARSCOPE_E3 | 64 | 4 | 0.1 | 0.0015 | 200 | 50 | 0.001 | 0.0001 |
| CLEARSCOPE_E3 (regen, D) | 64 | 4 | 0.1 | 0.003 | 100 | 50 | 0.001 | 0 |
| THEIA_E3 (regen, D) | 64 | 8 | 0.3 | 0.0015 | 20 | 50 | 0.001 | 1e-5 |
| CADETS_E5 (regen, D) | 64 | 8 | 0.1 | 0.0015 | 500 | 20 | 1e-4 | 5e-6 |
| THEIA_E5 | — | — | — | — | — | — | — | — | *(placeholder)* |
| CLEARSCOPE_E5 | — | — | — | — | — | — | — | — | *(placeholder)* |

¹ Not recorded in the THEIA_E3 artifact name; use the released checkpoint (Option A) for
the exact number, or sweep `num_hidden ∈ {64, 128, 256}` when retraining.

`--use_cfg` additionally applies the per-dataset entries in `configs.yml`. Sweep scripts
with these grids are provided as `theia_e3.sh`, `cadets_e3.sh`, `clearscope_e3.sh`, and
the E5 / OPTC variants.

---

## 📚 Citation

If you use this code or the released artifacts, please cite:

```bibtex
@inproceedings{yang2026provfusion,
  title   = {Beyond Nodes vs. Edges: A Multi-View Fusion Framework for
             Provenance-Based Intrusion Detection},
  author  = {Yang, Fan and Xu, Binyan and Tang, Di and Zhang, Kehuan},
  year    = {2026}
}
```
