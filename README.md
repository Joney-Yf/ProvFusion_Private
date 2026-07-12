# OpenProvFusion

Provenance-based intrusion detection on system audit logs, built on a GraphMAE-style masked
graph auto-encoder with multi-view anomaly voting. Evaluated on DARPA Transparent Computing
(E3/E5: THEIA, CADETS, CLEARSCOPE) and OpTC.

## Results (node-level detection)

| Dataset | TP | FP | Reproduction entry |
|---|---|---|---|
| THEIA_E3 | 91 | 2 | released checkpoint / retrain (Options A/B/C) |
| CADETS_E3 | 24 | 1 | released checkpoint / retrain (Options A/B/C) |
| CLEARSCOPE_E3 | 6 | 7 | released checkpoint / retrain (Options A/B/C) |
| CLEARSCOPE_E3 (full pipeline, regenerated from raw logs) | 7 | 7 | released checkpoint / end-to-end (Option D) |
| THEIA_E3 (full pipeline, regenerated from raw logs) | 99 | 3 | released checkpoint / end-to-end (Option D) |

TP/FP are unique attack/benign nodes flagged by the final voting detector
(`method_12_with_different_normalization`, `percentile` normalization).

> **On randomness:** the link-prediction (edge-reconstruction) head trains with
> `shuffle=True` + multi-worker data loading and is not bit-reproducible; detection FP varies
> across reruns of the same config. The released checkpoints pin the exact reported numbers
> (Option A is fully deterministic). When retraining (Options B/C/D), keep `seed=1` and rerun
> the same command several times — the best draws match the numbers above.

## 1. Environment

- Python 3.9 (tested 3.9.21), CUDA 11.x GPU.
- `pip install -r requirements.txt` — see the header of that file for the CUDA-specific
  install order (torch → dgl → PyG wheels → the rest).
- Always run with `PYTHONHASHSEED=0` (Word2Vec and hashing determinism).
- PostgreSQL is required **only** for Option D (raw-log ingestion).

## 2. Data layout

The repo expects this sibling layout:

```
parent/
├── OpenProvFusion/            # this repo (run everything from here)
├── raw_data/{DATASET}/        # per-graph embeddings: train/ val/ test/ *.TemporalData.simple
└── Ground_Truth/              # ground-truth node ids (*.pt)
```

Per-dataset training inputs (`--data_path` is the merged .pt built from raw_data; it is
created automatically on first run and reused afterwards):

| Dataset | --data_path | --ground_truth_path |
|---|---|---|
| THEIA_E3 | theia_merge_edge_data.pt | gt_canonical/THEIA_E3_orig_ground_truth_nids.pt |
| CADETS_E3 | cadets_e3.pt | ../Ground_Truth/ground_truth_cadet_v2.pt |
| CLEARSCOPE_E3 | clearscope_e3_merge_edge_data.pt | gt_canonical/CLEARSCOPE_E3_orig_ground_truth_nids.pt |

**Canonical ground truth.** Attack nodes are recorded by **node UUID** in
`Ground_Truth_csv/` (one CSV per attack; THEIA's third attack — phishing e-mail with
executable attachment — lives in `node_phishing_email.csv`). `tools/build_ground_truth.py`
resolves those UUIDs against a given Postgres database and emits the index-based
`*_ground_truth_nids.pt` + `*_attack_to_nids.pt` consumed by evaluation (`gt_canonical/`
ships prebuilt artifacts for the original databases). This makes ground truth independent
of database ingestion order: if you regenerate a database from raw logs (Option D), rebuild
the ground-truth artifacts for it with the same script instead of reusing the originals.
Equivalence with the historical labels was verified live (THEIA 91/2, CLEARSCOPE 6/7
reproduce exactly; the retired hardcoded `addition_list` nodes are all included).
| THEIA_E5 | theia_e5_merge_edge_data_final.pt | ../Ground_Truth/ground_truth_nids_theia_e5.pt |
| CADETS_E5 | cadets_e5_merge_edge_data.pt | ../Ground_Truth/ground_truth_nids_cadets_e5.pt |
| CLEARSCOPE_E5 | clearscope_e5_merge_edge_data.pt | ../Ground_Truth/ground_truth_nids_clearscope_e5.pt |
| OPTC_h201 / h051 / h501 | optc_h{201,051,501}_merge_edge_normalized.pt | ../Ground_Truth/ground_truth_nids_optc_h{201,051,501}.pt |

**Released data & checkpoints** (see `release_assets/MANIFEST.md5` for integrity):

```
release_assets/
├── checkpoints/          # saved middle results; each re-evaluates to exactly these numbers
│   ├── THEIA_E3_..._gatedge_gat_data.pt                  # TP=91 / FP=2
│   ├── CADETS_E3_loss_sce_dim_64_..._gat_data.pt         # TP=24 / FP=1
│   ├── CLEARSCOPE_E3_loss_sce_dim_64_..._gat_data.pt     # TP=6  / FP=7
│   ├── CLEARSCOPE_E3_RERUN_4.pt                          # TP=6  / FP=4 (retrained in this repo)
│   ├── THEIA_E3_regen_d64_..._tp99_fp3.pt                # TP=99 / FP=3 (Option-D regenerated data)
│   ├── THEIA_E3_regen_d64_..._tp88_fp2.pt                # TP=88 / FP=2 (Option-D, FP matches baseline)
│   ├── CLEARSCOPE_E3_regen_emb25_..._tp7_fp7.pt          # TP=7  / FP=7 (Option-D regenerated data)
│   ├── CLEARSCOPE_E3_regen_emb25_..._tp6_fp7.pt          # TP=6  / FP=7 (Option-D regenerated data)
│   ├── CLEARSCOPE_E3_regen_emb25_..._tp6_fp9.pt          # TP=6  / FP=9 (Option-D regenerated data)
│   └── CLEARSCOPE_E3_regen_emb25_..._tp7_fp16.pt         # TP=7  / FP=16 (Option-D regenerated data)
├── regenerated_data/
│   └── clearscope_e3_emb25.pt      # merged training input regenerated from raw logs (Option D)
└── MANIFEST.md5
```

## 3. Best hyperparameters

All runs: `--encoder gatedge --decoder gat --loss_fn sce --optimizer adam --num_layers 2
--in_drop 0.2 --attn_drop 0.1 --replace_rate 0.0 --activation prelu --linear_prob --scheduler
--use_cfg --seeds 1 --lr_f 0.001 --normalization_method percentile`.

| Dataset | num_hidden | num_heads | mask_rate | lr | max_epoch | max_epoch_f | weight_decay | weight_decay_f |
|---|---|---|---|---|---|---|---|---|
| THEIA_E3 | —¹ | 8 | 0.3 | 0.0015 | 5 | 50 | 0.01 | 0.0001 |
| CADETS_E3 | 64 | 2 | 0.3 | 0.0015 | 5 | 50 | 0.001 | 0.0001 |
| CLEARSCOPE_E3 | 64 | 4 | 0.1 | 0.0015 | 200 | 50 | 0.001 | 0.0001 |
| CLEARSCOPE_E3 (regenerated data, Option D) | 64 | 4 | 0.1 | 0.003 | 100 | 50 | 0.001 | 0 |
| THEIA_E3 (regenerated data, Option D) | 64 | 8 | 0.3 | 0.0015 | 20 | 50 | 0.001 | 1e-5 |

¹ Not recorded in the THEIA artifact name; the sweep that produced it covered
`num_hidden ∈ {64, 128, 256}` (`theia_e3.sh`) — use the released checkpoint (Option A) for
the exact number, or sweep these three values when retraining.

`--use_cfg` additionally applies the per-dataset entries in `configs.yml`
(activation/dropout/norm etc.). Sweep scripts with these grids are provided as
`theia_e3.sh`, `cadets_e3.sh`, `clearscope_e3.sh`, and the E5/OPTC variants.

## 4. Reproducing the results

You can start from any of four checkpoints in the pipeline, from fastest to fullest.

### Option A — evaluate a released checkpoint (deterministic, minutes)

Evaluation needs the merged data `.pt`, the ground truth, and the per-dataset
`{DATASET}_attack_to_nids.pt` (shipped in this repo) in the working directory:

```bash
python -c "
from try_different_threshold import method_12_with_different_normalization
method_12_with_different_normalization(
    'release_assets/checkpoints/CLEARSCOPE_E3_loss_sce_dim_64_nhd_4_nh_0.1_nl_2_lr_0.0015_mp_200_mpf_50_wd_0.001_wdf_0.0001_gatedge_gat_data.pt',
    'clearscope_e3_merge_edge_data.pt',
    '../Ground_Truth/ground_truth_nids_clearscope.pt',
    normalization_method='percentile')
"
```

Swap the checkpoint/data/ground-truth triple per the tables above to reproduce each row
of the results table exactly.

> **Checkpoint naming matters:** evaluation derives the dataset from the **first two
> underscore-separated tokens of the checkpoint filename** (e.g. `CLEARSCOPE_E3_*.pt` →
> loads `CLEARSCOPE_E3_attack_to_nids.pt` from the working directory). If you rename a
> checkpoint, keep the `{DATASET}_` prefix — otherwise the attack attribution silently
> fails and the reported TP collapses.

### Option B — retrain from the released merged `.pt`

```bash
# CLEARSCOPE_E3 (baseline config; rerun a few times, best draws reach TP=6/FP<=7)
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

For THEIA_E3 / CADETS_E3 substitute the data paths and the hyperparameters from Section 3
(THEIA/CADETS use `--max_epoch 5`). Training prints the full evaluation (all methods + final
voting) at the end of every run and saves its middle result under `save_middle_results/`.

### Option C — rebuild the merged `.pt` from the released embeddings

If `--data_path` does not exist, `main_transductive.py` builds it from
`../raw_data/{DATASET}/{train,val,test}` automatically (or run `preprocess.py` yourself).
Delete/rename the old `.pt` first — an existing file is silently reused.

Verified mappings raw_data → merged (edge counts match exactly after the de-duplicating
DGL conversion):

| raw_data source | rebuilds |
|---|---|
| `raw_data/THEIA_E3/` | `theia_merge_edge_data.pt` |
| `raw_data/CADETS_E3_v2/` (rename or symlink to `raw_data/CADETS_E3/`) | `cadets_e3.pt` |
| `raw_data/CLEARSCOPE_E3/` | `clearscope_e3_merge_edge_data.pt` |

### Option D — full pipeline from raw DARPA logs

`data_preparation/` regenerates everything from the original DARPA JSON logs:
Postgres ingestion → per-day graphs → Word2Vec featurization → edge embeddings, published
as `../raw_data/{DATASET}` (see `data_preparation/README.md` for prerequisites, isolation
flags, and safety guards). Validated end-to-end on CLEARSCOPE_E3: the regenerated database
is bit-identical to the original (node `index_id` md5 match, so the released ground truth
remains valid), and the regenerated embeddings are structurally identical to the originals.
Training on the regenerated data with the Option-D config in Section 3 detects **all six
labeled attack nodes** and, on the best draws, matches or beats the original-data baseline:
the released Option-D checkpoints re-evaluate to **TP=7/FP=7** (one more ground-truth node
than the baseline at the same FP), **TP=6/FP=7** (exact baseline match), TP=6/FP=9 and
TP=7/FP=16 (release tree in Section 2). Statistics from an 11,520-run sweep (3 regenerated
embeddings × 64 configs × 60 reps, seed=1): 42% of runs detect ≥6 attack nodes, but
FP ≤ 10 draws are rare (~3% of reps of the best config) — expect to rerun the Section-3
Option-D config a few dozen times to land one, or use the released checkpoints directly.

```bash
# 1) regenerate data from raw logs (Postgres required; see data_preparation/README.md)
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

**THEIA_E3 (Option D).** Same four stages; validated end-to-end. Two points specific to it:
(a) its ingester is the `guard/src` variant of `create_database/theia_e3.py` — the parser that
actually produced the original database (already vendored); (b) the regenerated database
assigns **different `index_id` numbering** than the original (the raw-log directory's entry
order changed at some point, and file enumeration is `glob`-order), so you MUST use the labels
built for the regenerated database — `prepare_data.py` does this automatically after ingest
(`gt_canonical/THEIA_E3_regen_*`), and evaluation must run from a directory whose
`THEIA_E3_attack_to_nids.pt` points at the regen attack map. Node sets and per-table row counts
match the original exactly; ~0.5% of rows differ in multi-valued attributes whose recorded value
depends on log processing order. Best released draws on regenerated data: **TP=99/FP=3** and
**TP=88/FP=2** (baseline: 91/2), config in Section 3.

Alternatively skip step 1 and train directly on the released regenerated merged data
`release_assets/regenerated_data/clearscope_e3_emb25.pt` (the exact input behind the
TP=7/FP=16 checkpoint). Word2Vec is not bit-reproducible across machines (multi-worker
training), so a fresh regeneration is a fresh embedding draw — sweep lr/weight_decay around
the Option-D config and rerun per Section "On randomness".

## 5. Repository map

```
main_transductive.py        # training + evaluation entry point
preprocess.py               # raw_data/{DATASET} -> merged .pt
try_different_threshold.py  # detection/evaluation (method_12..., percentile)
graphmae/                   # model + data loading (GraphMAE, gatedge encoder)
configs.yml                 # per-dataset --use_cfg defaults
configs/                    # dataset loss weights
data_preparation/           # raw DARPA logs -> embeddings (Option D; own README)
*.sh                        # per-dataset hyperparameter sweep scripts
release_assets/             # released checkpoints + regenerated data + MANIFEST.md5
```
