"""
Data-preparation entry point: raw DARPA logs  ->  edge_embeds (a new dataset).

This migrates the front-end of the original `orthrus_for_data_prepration` project
(the PIDSMaker/Orthrus framework) into OpenProvFusion. It runs ONLY the four stages
that produce the per-edge embedded graphs the current training pipeline consumes:

    1. create_db       raw JSON logs        -> Postgres database
    2. graphs          Postgres database    -> per-time-window networkx graphs (nx/)
    3. featurization   Postgres node msgs   -> feature_word2vec.model
    4. embed           graphs + word2vec    -> edge_embeds/{train,val,test}/*.TemporalData.simple

The detection / attack_reconstruction stages of the original project are intentionally
NOT migrated.

The produced `edge_embeds/{train,val,test}` directory is byte-compatible with what
`graphmae/datasets/data_util.py::Custimized` (used by `main_transductive.py` via
`preprocess.py`) expects under `../raw_data/{DATASET}/`. The final "publish" step
exposes it there as a new dataset, so the existing training/eval pipeline can load it
unchanged.

Usage (run from the OpenProvFusion project root):

    # Full pipeline from raw logs, then publish as ../raw_data/CLEARSCOPE_E3
    PYTHONHASHSEED=0 python data_preparation/prepare_data.py CLEARSCOPE_E3

    # Resume from a later stage (DB already built, graphs already built, ...)
    PYTHONHASHSEED=0 python data_preparation/prepare_data.py CLEARSCOPE_E3 --run_from graphs

    # Override any orthrus.yml param from the CLI (same dotted syntax as the original):
    PYTHONHASHSEED=0 python data_preparation/prepare_data.py CLEARSCOPE_E3 \
        --graph_construction.build_graphs.time_window_size=1.0

    # Validate WITHOUT touching the original DB or the original dataset folder (dev):
    #   first create the suffixed DB+schema:
    #     data_preparation/postgres/init-create-databases.sh _Test_for_OpenSource
    PYTHONHASHSEED=0 python data_preparation/prepare_data.py CLEARSCOPE_E3 \
        --out_suffix _Test_for_OpenSource --raw_data_dir ../raw_data_opensource
    # -> ingests into DB clearscope_e3_test_for_opensource (original DB untouched),
    #    publishes ../raw_data_opensource/CLEARSCOPE_E3 (folder name kept = CLEARSCOPE_E3,
    #    original ../raw_data/CLEARSCOPE_E3 untouched). Then train with the SAME name:
    #      python main_transductive.py --dataset CLEARSCOPE_E3 \
    #          --raw_data_dir ../raw_data_opensource --data_path clearscope_e3_opensource.pt ...
    # Drop --out_suffix / --raw_data_dir (defaults) for the clean release run on originals.

NOTE on reproducibility: the original project relies on `PYTHONHASHSEED=0` so that
Gensim's Word2Vec produces deterministic vectors. Always export it (as above) if you
care about reproducing previous embeddings.

Prerequisites for stage 1 (create_db) — server-environment, unchanged from the original:
    - A running Postgres reachable with `DATABASE_DEFAULT_CONFIG` in config.py
      (host=localhost, user=postgres, password=yangfan, port=5432).
    - The target database AND its tables must already exist: run
      `data_preparation/postgres/init-create-databases.sh [out_suffix]` (the full-schema
      variant, not `init-create-empty-databases.sh`) once beforehand. The per-dataset
      scripts here only INSERT rows; they create no databases and no tables. Pass the same
      suffix you give `--out_suffix` to build the isolated DB instead of the original.
    - The database must be EMPTY (stage 1 aborts otherwise): the INSERT scripts have no
      TRUNCATE/upsert logic, and re-ingesting from scratch can assign different index_ids,
      silently invalidating Ground_Truth nid files generated against a previous ingestion.
      To reuse an already-populated database, resume with `--run_from graphs`.
    - Raw JSON logs present at the `raw_dir` baked into DATASET_DEFAULT_CONFIG in config.py.
"""

import argparse
import os
import sys
import random
import shutil
import time

# MIGRATION: make `data_preparation/` the import root so the vendored modules keep their
# original bare imports (`from config import *`, `from graph_construction import ...`),
# exactly as they resolved under the original `src/` directory. No churn to the algorithm
# files is needed.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import psycopg2
import torch

from provnet_utils import init_database_connection, log
from config import get_yml_cfg, get_runtime_required_args

# Stage 1 dispatcher: we import the per-dataset modules directly (rather than copying the
# original top-level `create_database.py` dispatcher) because a `create_database.py` module
# would collide with the `create_database/` package on the import path.
from create_database import (
    cadets_e3,
    cadets_e5,
    clearscope_e3,
    clearscope_e5,
    theia_e3,
    theia_e5,
)
from graph_construction import build_orthrus_graphs
from edge_featurization import (
    build_feature_word2vec,
    embed_edges_feature_word2vec,
)

# Ordered pipeline stages; --run_from selects the first one to execute.
STAGES = ["create_db", "graphs", "featurization", "embed"]

_CREATE_DB_DISPATCH = {
    "CADETS_E3": cadets_e3,
    "CADETS_E5": cadets_e5,
    "CLEARSCOPE_E3": clearscope_e3,
    "CLEARSCOPE_E5": clearscope_e5,
    "THEIA_E3": theia_e3,
    "THEIA_E5": theia_e5,
}


def _assert_db_ready_and_empty(cfg):
    """
    Stage-1 guard. The create_database scripts INSERT into pre-existing tables with no
    TRUNCATE/upsert logic, so re-running them on a populated database crashes on duplicate
    keys. Worse, a from-scratch re-ingestion can assign different index_ids (file
    enumeration order is filesystem-dependent), which silently invalidates Ground_Truth
    nid files generated against the previous ingestion. Fail loudly up front instead.
    """
    if cfg.graph_construction.build_graphs.use_all_files:
        database_name = cfg.dataset.database_all_file
    else:
        database_name = cfg.dataset.database

    try:
        cur, connect = init_database_connection(cfg)
    except psycopg2.OperationalError as e:
        raise RuntimeError(
            f"Cannot connect to Postgres database '{database_name}': {e}\n"
            f"Stage create_db needs a running Postgres with the database and tables "
            f"already created — see data_preparation/postgres/init-create-databases.sh "
            f"(the full-schema variant) and DATABASE_DEFAULT_CONFIG in config.py."
        ) from e

    try:
        non_empty = []
        for table in ("netflow_node_table", "subject_node_table",
                      "file_node_table", "event_table"):
            try:
                cur.execute(f"SELECT EXISTS (SELECT 1 FROM {table});")
            except psycopg2.ProgrammingError as e:
                raise RuntimeError(
                    f"Table '{table}' is missing in database '{database_name}'. The "
                    f"stage-1 scripts only INSERT rows; create the schema first with "
                    f"data_preparation/postgres/init-create-databases.sh (the "
                    f"full-schema variant, not the empty one)."
                ) from e
            if cur.fetchone()[0]:
                non_empty.append(table)
    finally:
        connect.close()

    if non_empty:
        raise RuntimeError(
            f"Database '{database_name}' already contains data "
            f"({', '.join(non_empty)}).\n"
            f"Refusing to re-run create_db: the INSERT scripts would crash on duplicate "
            f"keys, and a from-scratch re-ingestion can reassign index_ids, silently "
            f"invalidating Ground_Truth nid files generated against the existing data.\n"
            f"Either resume with `--run_from graphs` to reuse this database, or TRUNCATE "
            f"all four tables yourself if you really mean to rebuild (and then regenerate "
            f"the ground-truth nids)."
        )


def apply_out_suffix(cfg, out_suffix):
    """
    Isolate the Postgres DATABASE for this run so stage 1 never ingests into the original DB.

    The suffix is appended ONLY to the Postgres database names — lowercased to match the
    `init-create-databases.sh` convention (unquoted CREATE DATABASE folds identifiers to
    lowercase, so a mixed-case connect name would fail to match) — so every read and write
    in the whole pipeline (all of which go through `init_database_connection`, which reads
    only `cfg.dataset.database` / `cfg.dataset.database_all_file`) hits the isolated DB.

    It deliberately does NOT rename the published dataset folder: training keys configs.yml
    best-configs and DATASET_WEIGHTS by `--dataset` NAME, so the folder name must stay
    `cfg.dataset.name` (e.g. CLEARSCOPE_E3). Isolate the published DATA instead with
    `--raw_data_dir` (publish into a separate parent dir, same folder name), keeping the
    dataset name — and thus the matching hyperparameters/weights — unchanged.

    Empty suffix => original DB names are used (the stage-1 emptiness guard still protects
    them). Artifact paths are keyed by `cfg.dataset.name` + the task-config hash, NOT by the
    database name, so changing the DB name here does not perturb any artifact location.
    """
    if not out_suffix:
        return
    cfg.dataset.database = (cfg.dataset.database + out_suffix).lower()
    cfg.dataset.database_all_file = (cfg.dataset.database_all_file + out_suffix).lower()
    log(f"[out_suffix] Isolated DATABASE active (original DB untouched):")
    log(f"[out_suffix]   database     = {cfg.dataset.database}")
    log(f"[out_suffix]   database_all = {cfg.dataset.database_all_file}")


def run_create_database(cfg):
    """Stage 1: ingest raw JSON logs into Postgres (mirrors create_database.py::main)."""
    name = cfg.dataset.name
    if name not in _CREATE_DB_DISPATCH:
        raise ValueError(
            f"No create_database script for dataset '{name}'. "
            f"Available: {sorted(_CREATE_DB_DISPATCH)}"
        )
    _assert_db_ready_and_empty(cfg)
    _CREATE_DB_DISPATCH[name].main(cfg)


def publish_as_dataset(cfg, raw_data_dir, publish_mode, force_publish=False):
    """
    Expose the produced edge_embeds as a dataset under `raw_data_dir/{DATASET}`, so the
    existing loader (`Custimized`) picks it up with no changes.

    The folder name is always `cfg.dataset.name` (unchanged) so training can keep its
    `--dataset` NAME and the matching configs.yml / DATASET_WEIGHTS entries. To publish a
    fresh copy WITHOUT clobbering the original `../raw_data/{DATASET}`, point `--raw_data_dir`
    at a separate parent directory (the folder name stays the same there).

    edge_embeds layout produced by stage 4:
        <artifact>/edge_featurization/<DATASET>/embed_edges/<hash>/edge_embeds/{train,val,test}/
    """
    edge_embeds_dir = cfg.edge_featurization.embed_edges._edge_embeds_dir
    edge_embeds_dir = os.path.abspath(edge_embeds_dir)

    if not os.path.isdir(edge_embeds_dir):
        raise FileNotFoundError(
            f"Expected edge_embeds dir not found: {edge_embeds_dir}\n"
            f"Did stage 'embed' run successfully?"
        )

    target = os.path.join(raw_data_dir, cfg.dataset.name)

    if publish_mode == "none":
        log(f"[publish] Skipped (--publish_mode none). edge_embeds at: {edge_embeds_dir}")
        log(f"[publish] To use it, point the trainer at this dir, e.g. "
            f"--raw_data_dir {os.path.dirname(edge_embeds_dir)} (renamed to {cfg.dataset.name}).")
        return

    os.makedirs(raw_data_dir, exist_ok=True)

    # Clear a previous publication so re-runs (possibly with a new config hash) stay fresh.
    # Only a symlink is known to be ours; a real directory/file at the target may be a
    # pre-existing dataset (Mode A) that must never be deleted silently.
    if os.path.islink(target):
        os.unlink(target)
    elif os.path.exists(target):
        if not force_publish:
            raise RuntimeError(
                f"Refusing to overwrite existing non-symlink dataset at: {target}\n"
                f"It is not something this pipeline published (publications in symlink "
                f"mode are symlinks) — it may be a pre-existing real dataset.\n"
                f"Options: --force_publish to delete and replace it, --raw_data_dir to "
                f"publish elsewhere, or --publish_mode none.\n"
                f"The produced edge_embeds are intact at: {edge_embeds_dir}"
            )
        log(f"[publish] --force_publish: removing existing {target}")
        if os.path.isdir(target):
            shutil.rmtree(target)
        else:
            os.remove(target)

    if publish_mode == "symlink":
        os.symlink(edge_embeds_dir, target)
        log(f"[publish] Symlinked dataset: {target} -> {edge_embeds_dir}")
    elif publish_mode == "copy":
        shutil.copytree(edge_embeds_dir, target)
        log(f"[publish] Copied dataset to: {target}")
    else:
        raise ValueError(f"Unknown publish_mode: {publish_mode}")

    for split in ("train", "val", "test"):
        split_dir = os.path.join(target, split)
        if not os.path.isdir(split_dir):
            log(f"[publish] WARNING: expected split '{split}' missing at {split_dir}")

    log(f"[publish] NOTE: training merges this dataset into a single .pt via "
        f"preprocess.py, which SKIPS regeneration when the output file already exists. "
        f"To train on THIS freshly published data, pass a NEW --data_path (e.g. "
        f"{cfg.dataset.name.lower()}_<tag>.pt) — pointing at an existing merged .pt makes "
        f"training silently keep using the stale data.")


def main(cfg, prep_args):
    # Redirect every DB read/write and the publish folder into an isolated namespace
    # (no-op when --out_suffix is empty). Must run before any stage so stage 1's DB
    # connection already targets the suffixed database.
    apply_out_suffix(cfg, prep_args.out_suffix)

    # Mirror the seeding the original word2vec stage relies on (each vendored stage also
    # seeds itself; this just makes the top-level run deterministic end-to-end).
    if cfg.edge_featurization.embed_nodes.use_seed:
        seed = cfg._seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    start = STAGES.index(prep_args.run_from)
    stop = STAGES.index(prep_args.run_to)
    if start > stop:
        raise ValueError(
            f"--run_from ({prep_args.run_from}) is after --run_to ({prep_args.run_to}) "
            f"in the stage order {STAGES}."
        )

    def _selected(stage):
        return start <= STAGES.index(stage) <= stop

    times = {}
    t0 = time.time()

    t = t0
    if _selected("create_db"):
        log("=" * 60)
        log("[stage 1/4] create_db: raw logs -> Postgres")
        run_create_database(cfg)
        times["create_db"] = round(time.time() - t, 2)

        # Canonical labels: ground truth is keyed by node UUID (Ground_Truth_csv/); the
        # index-based artifacts consumed by evaluation are DERIVED from the freshly
        # ingested database, so label indices always match THIS database's numbering
        # (ingestion order differences can never invalidate them).
        if not prep_args.skip_build_gt:
            repo_root = os.path.dirname(_THIS_DIR)
            builder = os.path.join(repo_root, "tools", "build_ground_truth.py")
            tag = "regen" if prep_args.out_suffix else "orig"
            out_prefix = os.path.join(repo_root, "gt_canonical", f"{cfg.dataset.name}_{tag}")
            import subprocess
            r = subprocess.run(
                [sys.executable, builder, cfg.dataset.name, cfg.dataset.database, out_prefix],
                capture_output=True, text=True, cwd=repo_root,
            )
            for ln in (r.stdout + r.stderr).strip().splitlines():
                log(f"[build_gt] {ln}")
            if r.returncode != 0:
                if "KeyError" in r.stderr:
                    log(f"[build_gt] dataset {cfg.dataset.name} has no canonical CSV spec — skipped")
                else:
                    raise RuntimeError("build_ground_truth failed; see [build_gt] log lines above")
            else:
                log(f"[build_gt] labels for database '{cfg.dataset.database}' -> {out_prefix}_*.pt")

    t = time.time()
    if _selected("graphs"):
        log("=" * 60)
        log("[stage 2/4] graphs: Postgres -> per-time-window nx graphs")
        build_orthrus_graphs.main(cfg)
        times["graphs"] = round(time.time() - t, 2)

    t = time.time()
    if _selected("featurization"):
        log("=" * 60)
        log("[stage 3/4] featurization: node msgs -> feature_word2vec.model")
        build_feature_word2vec.main(cfg)
        times["featurization"] = round(time.time() - t, 2)

    t = time.time()
    if _selected("embed"):
        log("=" * 60)
        log("[stage 4/4] embed: graphs + word2vec -> edge_embeds")
        embed_edges_feature_word2vec.main(cfg)
        times["embed"] = round(time.time() - t, 2)

    # Publish only when this invocation actually produced edge_embeds (i.e. ran through the
    # embed stage). Stopping earlier (e.g. --run_to graphs) skips publish so we never expose
    # a stale/absent edge_embeds dir.
    if _selected("embed"):
        log("=" * 60)
        log("[publish] Exposing edge_embeds as a dataset")
        publish_as_dataset(
            cfg, prep_args.raw_data_dir, prep_args.publish_mode, prep_args.force_publish
        )
    else:
        log("=" * 60)
        log(f"[publish] Skipped: --run_to {prep_args.run_to} stops before the embed stage.")

    times["total"] = round(time.time() - t0, 2)
    log("=" * 60)
    log("Data preparation finished. Time per stage (s):")
    for k, v in times.items():
        log(f"  {k}: {v}")


def parse_args(argv=None):
    # Prep-only flags, kept separate from the original config arg parser. We strip these
    # first, then hand the remainder (incl. the positional `dataset` and any dotted cfg
    # overrides) to the original `get_runtime_required_args`.
    # allow_abbrev=False so a config passthrough arg (e.g. --run_from_training, handled by
    # the original config parser) can never be prefix-matched/swallowed by --run_from.
    prep_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    prep_parser.add_argument(
        "--run_from", choices=STAGES, default="create_db",
        help="First pipeline stage to run (resume support). Default: create_db.",
    )
    prep_parser.add_argument(
        "--run_to", choices=STAGES, default="embed",
        help="Last pipeline stage to run (inclusive). Default: embed (full pipeline). "
             "Use --run_from X --run_to X to run a single stage for step-by-step "
             "verification. Publish runs only when the embed stage is included.",
    )
    prep_parser.add_argument(
        "--raw_data_dir", default="../raw_data",
        help="Where to publish the dataset, matching the trainer's loader "
             "(default ../raw_data, i.e. produces ../raw_data/{DATASET}).",
    )
    prep_parser.add_argument(
        "--publish_mode", choices=["symlink", "copy", "none"], default="symlink",
        help="How to expose edge_embeds as ../raw_data/{DATASET}: "
             "symlink (default, no duplication), copy (self-contained), or none.",
    )
    prep_parser.add_argument(
        "--out_suffix", default="",
        help="DATABASE isolation suffix. Appended (lowercased) to the Postgres database "
             "names so ALL stage reads/writes hit a separate DB and the original is never "
             "ingested into. Does NOT rename the published dataset folder (training keys "
             "configs.yml / DATASET_WEIGHTS by --dataset NAME, so that stays cfg.dataset.name); "
             "isolate the published DATA with --raw_data_dir instead. Default empty = "
             "original DB names. Create the matching DB first with "
             "postgres/init-create-databases.sh {out_suffix}.",
    )
    prep_parser.add_argument(
        "--skip_build_gt", action="store_true",
        help="Skip the automatic post-ingest generation of canonical ground-truth "
             "artifacts (tools/build_ground_truth.py) for the ingested database.",
    )
    prep_parser.add_argument(
        "--force_publish", action="store_true",
        help="Allow the publish step to delete an existing REAL (non-symlink) "
             "../raw_data/{DATASET} before publishing. Without it, only a previous "
             "symlink publication is replaced; a real dataset at the target aborts.",
    )
    prep_args, remaining = prep_parser.parse_known_args(argv)

    # Original config parser (positional `dataset`, --seed, dotted cfg overrides, ...).
    # Mirror the original orthrus.py entry point: reject unknown args loudly. Without
    # this, parse_known_args silently drops them, so a typo'd dotted override (e.g.
    # --graph_constrution.…) would fall back to defaults with no warning.
    cfg_args, unknown_args = get_runtime_required_args(
        return_unknown_args=True, args=remaining
    )
    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown arguments: {unknown_args}")
    return prep_args, cfg_args


if __name__ == "__main__":
    prep_args, cfg_args = parse_args()
    cfg = get_yml_cfg(cfg_args)
    main(cfg, prep_args)
