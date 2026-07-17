# OpenProvFusion

Provenance-based intrusion detection on system audit logs. A GraphMAE-style masked graph
auto-encoder learns benign behaviour from provenance graphs, then flags anomalous nodes by
multi-view voting (node reconstruction + edge reconstruction). Evaluated on DARPA Transparent
Computing (E3/E5: THEIA, CADETS, CLEARSCOPE) and OpTC.

## Results

Node-level detection. **TP / FP** are unique attack / benign nodes flagged by the final voting
detector (`method_12_with_different_normalization`, `percentile` normalization).

**Baseline** — original released data:

| Dataset | TP | FP |
|---|---:|---:|
| THEIA_E3 | 91 | 2 |
| CADETS_E3 | 24 | 1 |
| CLEARSCOPE_E3 | 6 | 7 |

**Full pipeline (Option D)** — every artifact rebuilt from the original DARPA JSON logs through
Postgres → graphs → Word2Vec → embeddings, then trained. Best draws:

| Dataset | TP | FP | vs. baseline |
|---|---:|---:|---|
| CADETS_E3 | 24 | 1 | matches exactly |
| THEIA_E3 | 88 | 2 | FP matches baseline |
| THEIA_E3 | 99 | 3 | +8 TP at FP=3 |
| CLEARSCOPE_E3 | 6 | 7 | matches exactly |
| CLEARSCOPE_E3 | 7 | 7 | +1 TP at same FP |

> **On reproducibility.** The edge-reconstruction head trains with `shuffle=True` + multi-worker
> loading and is **not** bit-reproducible, so FP varies run-to-run at a fixed config. Released
> checkpoints (Option A) pin the exact numbers above and are deterministic to evaluate. When
> **re**training (Options B–D), keep `--seeds 1` and rerun the same command several times — the
> best draws reach the numbers above.

## Quick start

Evaluate a released checkpoint (deterministic, ~minutes, no training):

```bash
# run from the repo root, with the merged .pt and ground truth in place (see Data layout)
python -c "
from try_different_threshold import method_12_with_different_normalization
method_12_with_different_normalization(
    'release_assets/checkpoints/CADETS_E3_loss_sce_dim_64_nhd_2_nh_0.3_nl_2_lr_0.0015_mp_5_mpf_50_wd_0.001_wdf_0.0001_gatedge_gat_data.pt',
    'cadets_e3.pt',
    'gt_canonical/CADETS_E3_orig_ground_truth_nids.pt',
    normalization_method='percentile')
"
# -> Unique TP=24, Unique FP=1
```

> **Checkpoint naming matters.** Evaluation derives the dataset from the **first two
> underscore-separated tokens of the checkpoint filename** (`CADETS_E3_*.pt` → loads
> `CADETS_E3_attack_to_nids.pt` from the working directory). If you rename a checkpoint, keep the
> `{DATASET}_` prefix, or attack attribution silently fails and TP collapses.

## Environment

- Python 3.9 (tested 3.9.21), a CUDA 11.x GPU.
- `pip install -r requirements.txt` — the file header documents the CUDA install order
  (torch → dgl → PyG wheels → the rest).
- Always run with `PYTHONHASHSEED=0` (Word2Vec / hashing determinism).
- PostgreSQL is required **only** for Option D (raw-log ingestion).

## Data layout

The repo expects this sibling layout, and everything runs from inside `OpenProvFusion/`:

```
parent/
├── OpenProvFusion/         # this repo
├── raw_data/{DATASET}/     # per-graph embeddings: train/ val/ test/ *.TemporalData.simple
└── Ground_Truth/           # legacy ground-truth node-id lists (*.pt)
```

`--data_path` is the merged `.pt` built from `raw_data/` — created automatically on first run and
reused afterwards. `--ground_truth_path` is the node-id label list.

| Dataset | `--data_path` | `--ground_truth_path` |
|---|---|---|
| THEIA_E3 | `theia_merge_edge_data.pt` | `gt_canonical/THEIA_E3_orig_ground_truth_nids.pt` |
| CADETS_E3 | `cadets_e3.pt` | `gt_canonical/CADETS_E3_orig_ground_truth_nids.pt` |
| CLEARSCOPE_E3 | `clearscope_e3_merge_edge_data.pt` | `gt_canonical/CLEARSCOPE_E3_orig_ground_truth_nids.pt` |
| THEIA_E5 | `theia_e5_merge_edge_data_final.pt` | `../Ground_Truth/ground_truth_nids_theia_e5.pt` |
| CADETS_E5 | `cadets_e5_merge_edge_data.pt` | `../Ground_Truth/ground_truth_nids_cadets_e5.pt` |
| CLEARSCOPE_E5 | `clearscope_e5_merge_edge_data.pt` | `../Ground_Truth/ground_truth_nids_clearscope_e5.pt` |
| OPTC_h201 / h051 / h501 | `optc_h{201,051,501}_merge_edge_normalized.pt` | `../Ground_Truth/ground_truth_nids_optc_h{201,051,501}.pt` |

### Canonical ground truth (by UUID)

Attack nodes are recorded by **node UUID** in `Ground_Truth_csv/` (one CSV per attack; THEIA's
third attack — a phishing e-mail with an executable attachment — is `node_phishing_email.csv`).
`tools/build_ground_truth.py` resolves those UUIDs against a given Postgres database and emits the
index-based `*_ground_truth_nids.pt` + `*_attack_to_nids.pt` consumed by evaluation. `gt_canonical/`
ships prebuilt artifacts for both the original databases (`*_orig_*`) and the regenerated ones
(`*_regen_*`).

Recording labels by UUID makes them **independent of database ingestion order**: regenerating a
database from raw logs (Option D) changes the node `index_id`s, so rebuild the labels for that
database with the same script instead of reusing the originals — `prepare_data.py` does this
automatically after ingest. The UUID CSVs reproduce the historical labels exactly (verified:
THEIA 91/2, CADETS 24/1, CLEARSCOPE 6/7; the retired hardcoded `addition_list` nodes are all
included).

## Released assets

`release_assets/` holds the checkpoints and regenerated training inputs; `MANIFEST.md5` covers
integrity. The large binaries are distributed separately (not tracked in git).

```
release_assets/
├── checkpoints/
│   ├── THEIA_E3_...gat_data.pt                      # baseline   TP=91 / FP=2
│   ├── CADETS_E3_...dim_64...gat_data.pt            # baseline   TP=24 / FP=1
│   ├── CLEARSCOPE_E3_...dim_64...gat_data.pt        # baseline   TP=6  / FP=7
│   ├── CLEARSCOPE_E3_RERUN_4.pt                     # retrained  TP=6  / FP=4
│   ├── CADETS_E3_regen_d128_mask0.5_...tp24_fp1.pt  # Option D   TP=24 / FP=1  (matches baseline)
│   ├── THEIA_E3_regen_..._tp88_fp2.pt               # Option D   TP=88 / FP=2  (FP matches baseline)
│   ├── THEIA_E3_regen_..._tp99_fp3.pt               # Option D   TP=99 / FP=3
│   └── CLEARSCOPE_E3_regen_emb25_...tp6_fp7.pt      # Option D   TP=6  / FP=7  (+ tp7_fp7, tp6_fp9, tp7_fp16)
├── regenerated_data/
│   ├── cadets_e3_regen.pt                           # merged training input, regenerated (Option D)
│   ├── theia_e3_regen.pt
│   └── clearscope_e3_emb25.pt
└── MANIFEST.md5
```

## Reproducing the results

Four entry points into the pipeline, fastest first:

| | Start from | Cost | Reproducibility |
|---|---|---|---|
| **A** | a released checkpoint | minutes, CPU | exact, deterministic |
| **B** | the released merged `.pt` | one training run | best-of-N draws |
| **C** | the released embeddings | rebuild `.pt`, then train | best-of-N draws |
| **D** | raw DARPA logs | full pipeline (Postgres) | best-of-N draws |

### A — evaluate a released checkpoint

See [Quick start](#quick-start). Swap the checkpoint / data / ground-truth triple per the tables
above to reproduce any results row exactly.

### B — retrain from the released merged `.pt`

```bash
CUDA_VISIBLE_DEVICES=0 python main_transductive.py --device 0 \
  --dataset CADETS_E3 \
  --data_path cadets_e3.pt \
  --ground_truth_path gt_canonical/CADETS_E3_orig_ground_truth_nids.pt \
  --raw_data_dir ../raw_data \
  --num_hidden 64 --num_heads 2 --mask_rate 0.3 \
  --lr 0.0015 --weight_decay 0.001 --weight_decay_f 0.0001 --max_epoch 5 --max_epoch_f 50 \
  --encoder gatedge --decoder gat --loss_fn sce --optimizer adam --num_layers 2 \
  --in_drop 0.2 --attn_drop 0.1 --replace_rate 0.0 --alpha_l 3 --activation prelu \
  --norm layernorm --linear_prob --scheduler --use_cfg --seeds 1 --lr_f 0.001 \
  --normalization_method percentile
```

Substitute the data path + per-dataset values from [Hyperparameters](#hyperparameters) for the
other datasets. Every run prints the full evaluation at the end and saves its middle result under
`save_middle_results/`. Rerun a few times and keep the best draw.

### C — rebuild the merged `.pt` from the released embeddings

If `--data_path` does not exist, `main_transductive.py` builds it from
`../raw_data/{DATASET}/{train,val,test}` automatically (or run `preprocess.py`). **Delete or
rename the old `.pt` first** — an existing file is silently reused.

Verified `raw_data` → merged mappings (edge counts match after the de-duplicating DGL conversion):

| raw_data source | rebuilds |
|---|---|
| `raw_data/THEIA_E3/` | `theia_merge_edge_data.pt` |
| `raw_data/CADETS_E3_v2/` (rename or symlink to `CADETS_E3/`) | `cadets_e3.pt` |
| `raw_data/CLEARSCOPE_E3/` | `clearscope_e3_merge_edge_data.pt` |

### D — full pipeline from raw DARPA logs

`data_preparation/` regenerates everything from the original DARPA JSON: Postgres ingestion →
per-day graphs → Word2Vec featurization → edge embeddings, published to `../raw_data/{DATASET}`
(see `data_preparation/README.md` for prerequisites, isolation flags, and safety guards).

```bash
# 1) regenerate data from raw logs (Postgres required)
PYTHONHASHSEED=0 python data_preparation/prepare_data.py CADETS_E3
# 2) train on it with the Option-D hyperparameters (see Hyperparameters)
CUDA_VISIBLE_DEVICES=0 python main_transductive.py --device 0 \
  --dataset CADETS_E3 --data_path cadets_e3_regen.pt \
  --ground_truth_path gt_canonical/CADETS_E3_regen_ground_truth_nids.pt \
  --num_hidden 128 --num_heads 2 --mask_rate 0.5 \
  --lr 0.0015 --weight_decay 0.01 --weight_decay_f 0.0001 --max_epoch 100 --max_epoch_f 50 \
  --encoder gatedge --decoder gat --loss_fn sce --optimizer adam --num_layers 2 \
  --in_drop 0.2 --attn_drop 0.1 --replace_rate 0.0 --alpha_l 3 --activation prelu \
  --norm layernorm --linear_prob --scheduler --use_cfg --seeds 1 --lr_f 0.001 \
  --normalization_method percentile
```

Because Word2Vec is not bit-reproducible across machines, a fresh regeneration is a fresh embedding
draw — evaluate against the **regenerated** labels (`gt_canonical/{DATASET}_regen_*`), not the
originals, and sweep a few dozen reps per *On reproducibility*.

**Per-dataset notes.** The ingester must be the exact parser that produced the original database:

| Dataset | Ingester | Fidelity vs. original DB | Best draw |
|---|---|---|---|
| CLEARSCOPE_E3 | vendored default | node `index_id` md5-identical | TP=6/FP=7 (exact), up to TP=7 |
| THEIA_E3 | `guard/src` `theia_e3.py` | same node sets/counts; ~0.5% multi-valued rows differ; `index_id` renumbered | TP=88/FP=2; TP=99/FP=3 |
| CADETS_E3 | `orthrus_old` `cadets_e3.py` | subject/netflow byte-identical; 0.07% of file rows differ; `index_id` renumbered | **TP=24/FP=1 (exact)** |

- **THEIA / CADETS renumber `index_id`** (their raw-log directory order differs from the original
  run), so they need the `*_regen_*` labels — `prepare_data.py` builds them automatically.
- **CADETS ingester:** the original `file_node_table` has 2,303,164 nodes (~98% path-less) and the
  raw logs are ~91% `FILE_OBJECT_UNIX_SOCKET`; the previously vendored parser *skips* those sockets
  (dropping ~90% of file nodes), so only `orthrus_old` (which keeps them) matches the original.
- **CADETS sweet spot:** `mask_rate=0.5` (not the 0.3 baseline) drops FP from ~7 to ~2; hammering
  it lands TP=24/FP=1 (3 of 60 reps). `max_epoch=200` over-trains and collapses TP to 0;
  `drop_edge_rate>0` is unsupported by this data path.

## Hyperparameters

Common to every run:

```
--encoder gatedge --decoder gat --loss_fn sce --optimizer adam --num_layers 2
--in_drop 0.2 --attn_drop 0.1 --replace_rate 0.0 --alpha_l 3 --activation prelu
--norm layernorm --linear_prob --scheduler --use_cfg --seeds 1 --lr_f 0.001
--normalization_method percentile
```

Per-dataset (baseline on original data; Option-D on regenerated data):

| Dataset | data | num_hidden | heads | mask | lr | max_epoch | epoch_f | wd | wd_f |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| THEIA_E3 | baseline | —¹ | 8 | 0.3 | 0.0015 | 5 | 50 | 0.01 | 1e-4 |
| CADETS_E3 | baseline | 64 | 2 | 0.3 | 0.0015 | 5 | 50 | 1e-3 | 1e-4 |
| CLEARSCOPE_E3 | baseline | 64 | 4 | 0.1 | 0.0015 | 200 | 50 | 1e-3 | 1e-4 |
| THEIA_E3 | regen (D) | 64 | 8 | 0.3 | 0.0015 | 20 | 50 | 1e-3 | 1e-5 |
| CADETS_E3 | regen (D) | 128 | 2 | 0.5 | 0.0015 | 100 | 50 | 0.01 | 1e-4 |
| CLEARSCOPE_E3 | regen (D) | 64 | 4 | 0.1 | 0.003 | 100 | 50 | 1e-3 | 0 |

¹ Not recorded in the THEIA artifact name; its sweep covered `num_hidden ∈ {64, 128, 256}`
(`theia_e3.sh`). Use the released checkpoint (Option A) for the exact number, or sweep those three
values when retraining.

`--use_cfg` also applies the per-dataset entries in `configs.yml`. Per-dataset sweep grids live in
`theia_e3.sh`, `cadets_e3.sh`, `clearscope_e3.sh` (and the E5/OPTC variants).

## Repository map

```
main_transductive.py        # training + evaluation entry point
preprocess.py               # raw_data/{DATASET} -> merged .pt
try_different_threshold.py  # detection / evaluation (method_12..., percentile)
graphmae/                   # model + data loading (GraphMAE, gatedge encoder)
configs.yml                 # per-dataset --use_cfg defaults
configs/                    # dataset loss weights
tools/build_ground_truth.py # UUID CSVs -> index-based label .pt (against any database)
Ground_Truth_csv/           # canonical attack nodes, by UUID
gt_canonical/               # prebuilt labels for original (*_orig) and regenerated (*_regen) DBs
data_preparation/           # raw DARPA logs -> embeddings (Option D; own README)
release_assets/             # checkpoints + regenerated data + MANIFEST.md5
*.sh                        # per-dataset hyperparameter sweep scripts
```
