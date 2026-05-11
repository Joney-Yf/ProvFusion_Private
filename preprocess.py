import argparse
import os
import torch
from graphmae.datasets.data_util import Custimized


def preprocess_dataset(dataset_name, raw_data_dir, output_path):
    """
    Convert raw per-graph .pt files into a single merged .pt ready for main_transductive.py.

    The output is a 6-tuple:
        (graphs, edge_counts, num_features, edge_features, num_classes, reverse_graphs)

    Skips silently if output_path already exists.
    """
    if os.path.exists(output_path):
        print(f"[preprocess] {output_path} already exists — skipping.")
        return

    raw_path = os.path.join(raw_data_dir, dataset_name)
    if not os.path.isdir(raw_path):
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_path}\n"
            f"Expected structure: {raw_path}/train/, {raw_path}/val/, {raw_path}/test/"
        )

    print(f"[preprocess] Processing {dataset_name} from {raw_path} ...")
    graphs, edge_counts, reverse_graphs = Custimized(raw_path)

    num_features = graphs['train'][0].ndata["feat"].shape[1]
    edge_features = graphs['train'][0].edata['label'].shape[1]
    num_classes = 3  # fixed for all provenance datasets

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(
        (graphs, edge_counts, num_features, edge_features, num_classes, reverse_graphs),
        output_path,
    )
    print(f"[preprocess] Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw provenance graph files into a merged .pt for training."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, e.g. THEIA_E5, OPTC_h201")
    parser.add_argument("--raw_data_dir", type=str, default="../raw_data",
                        help="Parent directory containing {DATASET}/train|val|test/ (default: ../raw_data)")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Where to write the merged .pt file (default: current dir)")
    args = parser.parse_args()

    output_path = os.path.join(
        args.output_dir,
        f"{args.dataset.lower()}_merge_edge_normalized.pt",
    )
    preprocess_dataset(args.dataset, args.raw_data_dir, output_path)


if __name__ == "__main__":
    main()
