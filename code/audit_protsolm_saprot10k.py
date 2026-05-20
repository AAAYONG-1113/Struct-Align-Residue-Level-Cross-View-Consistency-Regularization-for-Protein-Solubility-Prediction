import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from accelerate.utils import set_seed
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm


DEFAULT_PROTSOLM_ROOT = "/home/heyong/ProtSolM"
DEFAULT_DATASET_ROOT = "/home/heyong/Protein_ADMET_SaProt/protsolm_saprot10k"
DEFAULT_GNN_MODEL_PATH = "/home/heyong/ProtSolM/model/protssn_k20_h512.pt"
DEFAULT_MODEL_DIR = "/home/heyong/Protein_ADMET_SaProt/results/protsolm_k20_h512"
DEFAULT_MODEL_NAME = "feature_attention1d_k20_h512_saprot10k.pt"
DEFAULT_IDENTITY_PATH = "/home/heyong/Protein_ADMET_SaProt/test_vs_train.m8"


def compute_metrics(labels, probs, threshold=None, threshold_metric="acc"):
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5

    if threshold is None:
        threshold = 0.5
        best_score = None
        best_acc = accuracy_score(labels, probs > threshold)
        try:
            best_mcc = matthews_corrcoef(labels, probs > threshold)
        except Exception:
            best_mcc = 0.0

        for thr in np.linspace(0.05, 0.95, 181):
            preds = probs > thr
            curr_acc = accuracy_score(labels, preds)
            try:
                curr_mcc = matthews_corrcoef(labels, preds)
            except Exception:
                curr_mcc = 0.0

            if threshold_metric == "mcc":
                score = curr_mcc
                tie_break = curr_acc
                best_tie_break = best_acc
            else:
                score = curr_acc
                tie_break = curr_mcc
                best_tie_break = best_mcc

            if best_score is None or score > best_score or (np.isclose(score, best_score) and tie_break > best_tie_break):
                best_score = score
                best_acc = curr_acc
                best_mcc = curr_mcc
                threshold = thr

    preds = probs > threshold
    try:
        mcc = matthews_corrcoef(labels, preds)
    except Exception:
        mcc = 0.0

    return {
        "auc": auc,
        "acc": accuracy_score(labels, preds),
        "mcc": mcc,
        "threshold": float(threshold),
    }


def load_max_identity_map(identity_path):
    max_identity = {}
    if not identity_path or not os.path.exists(identity_path):
        return max_identity

    with open(identity_path, "r", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3 or not row[0].startswith("test_"):
                continue
            try:
                test_idx = int(row[0].replace("test_", "", 1))
                pident = float(row[2])
            except ValueError:
                continue
            prev = max_identity.get(test_idx, 0.0)
            if pident > prev:
                max_identity[test_idx] = pident
    return max_identity


def assign_homology_bucket(max_identity):
    if max_identity < 30.0:
        return "<30%"
    if max_identity < 50.0:
        return "30-50%"
    return ">=50%"


def summarize_homology_buckets(results, threshold, identity_path):
    if results["orig_idx"].size == 0:
        return []

    max_identity = load_max_identity_map(identity_path)
    if not max_identity:
        return []

    bucket_rows = []
    for bucket_name in ["<30%", "30-50%", ">=50%"]:
        selected = []
        identities = []
        for pos, orig_idx in enumerate(results["orig_idx"]):
            identity = max_identity.get(int(orig_idx), 0.0)
            if assign_homology_bucket(identity) == bucket_name:
                selected.append(pos)
                identities.append(identity)

        if not selected:
            continue

        labels = results["labels"][selected]
        probs = results["probs"][selected]
        metrics = compute_metrics(labels, probs, threshold=threshold)
        bucket_rows.append(
            {
                "bucket": bucket_name,
                "n": len(selected),
                "pos_rate": float(np.mean(labels)),
                "mean_max_identity": float(np.mean(identities)),
                "auc": metrics["auc"],
                "acc": metrics["acc"],
                "mcc": metrics["mcc"],
            }
        )
    return bucket_rows


def print_bucket_report(bucket_rows):
    if not bucket_rows:
        return

    print("Homology Bucket Report (threshold fixed on validation split)")
    for row in bucket_rows:
        print(
            f" - {row['bucket']}: n={row['n']} | pos_rate={row['pos_rate']:.3f} | "
            f"mean_max_id={row['mean_max_identity']:.1f} | AUC={row['auc']:.4f} | "
            f"Acc={row['acc']:.4f} | MCC={row['mcc']:.4f}"
        )


def save_prediction_csv(results, threshold, output_path, model_name="ProtSolM", dataset_name="SaProt10k"):
    if not output_path:
        return
    out_df = pd.DataFrame(
        {
            "dataset": dataset_name,
            "model": model_name,
            "split": "test",
            "orig_idx": results["orig_idx"].astype(int),
            "label": results["labels"].astype(int),
            "prob": results["probs"].astype(float),
        }
    )
    out_df["threshold"] = float(threshold)
    out_df["pred"] = (out_df["prob"] > float(threshold)).astype(int)
    out_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit ProtSolM on SaProt 10k with validation-fixed thresholding.")
    parser.add_argument("--protsolm-root", default=DEFAULT_PROTSOLM_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--gnn-model-path", default=DEFAULT_GNN_MODEL_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--identity-path", default=DEFAULT_IDENTITY_PATH)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--gnn", default="egnn")
    parser.add_argument("--gnn-config", default="src/config/egnn.yaml")
    parser.add_argument("--gnn-hidden-dim", type=int, default=512)
    parser.add_argument("--plm", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--plm-hidden-size", type=int, default=1280)
    parser.add_argument("--pooling-method", default="attention1d")
    parser.add_argument("--pooling-dropout", type=float, default=0.1)
    parser.add_argument("--feature-file", default=None)
    parser.add_argument(
        "--feature-name",
        nargs="+",
        default=["aa_composition", "gravy", "ss_composition", "hygrogen_bonds", "exposed_res_fraction", "pLDDT"],
    )
    parser.add_argument("--feature-dim", type=int, default=0)
    parser.add_argument("--feature-embed-dim", type=int, default=None)
    parser.add_argument("--use-plddt-penalty", action="store_true")
    parser.add_argument("--c-alpha-max-neighbors", type=int, default=20)
    parser.add_argument("--batch-token-num", type=int, default=16000)
    parser.add_argument("--max-graph-token-num", type=int, default=3000)
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--problem-type", default="classification")
    parser.add_argument("--save-preds", default=None)
    parser.add_argument("--pred-model-name", default="ProtSolM")
    parser.add_argument("--pred-dataset-name", default="SaProt10k")
    return parser.parse_args()


def apply_saved_config(args):
    config_path = Path(args.model_dir) / "config.json"
    if not config_path.exists():
        return args

    with open(config_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)

    # Rebuild the model with the same hyperparameters used during training.
    for key, value in saved.items():
        if key == "gnn_config":
            continue
        setattr(args, key.replace("-", "_"), value)
    return args


def build_feature_dict(args):
    if not args.feature_file:
        return {}, args

    feature_df = pd.read_csv(args.feature_file)
    feature_dict = {}
    if not isinstance(args.feature_name, list):
        args.feature_name = [args.feature_name]

    feature_groups = {
        "aa_composition": ["1-C", "1-D", "1-E", "1-R", "1-H", "Turn-forming residues fraction"],
        "gravy": ["GRAVY"],
        "ss_composition": ["ss8-G", "ss8-H", "ss8-I", "ss8-B", "ss8-E", "ss8-T", "ss8-S", "ss8-P", "ss8-L", "ss3-H", "ss3-E", "ss3-C"],
        "hygrogen_bonds": ["Hydrogen bonds", "Hydrogen bonds per 100 residues"],
        "exposed_res_fraction": [
            "Exposed residues fraction by 5%", "Exposed residues fraction by 10%", "Exposed residues fraction by 15%",
            "Exposed residues fraction by 20%", "Exposed residues fraction by 25%", "Exposed residues fraction by 30%",
            "Exposed residues fraction by 35%", "Exposed residues fraction by 40%", "Exposed residues fraction by 45%",
            "Exposed residues fraction by 50%", "Exposed residues fraction by 55%", "Exposed residues fraction by 60%",
            "Exposed residues fraction by 65%", "Exposed residues fraction by 70%", "Exposed residues fraction by 75%",
            "Exposed residues fraction by 80%", "Exposed residues fraction by 85%", "Exposed residues fraction by 90%",
            "Exposed residues fraction by 95%", "Exposed residues fraction by 100%",
        ],
        "pLDDT": ["pLDDT"],
    }

    selected_frames = {}
    args.feature_dim = 0
    for name in args.feature_name:
        cols = feature_groups[name]
        selected_frames[name] = feature_df[cols]
        args.feature_dim += len(cols)

    for i in range(len(feature_df)):
        protein_name = feature_df["protein name"][i].split(".")[0]
        feature_dict[protein_name] = []
        for name in args.feature_name:
            feature_dict[protein_name] += list(selected_frames[name].iloc[i])

    return feature_dict, args


def main():
    args = parse_args()
    args = apply_saved_config(args)
    set_seed(args.seed)

    if args.feature_file is None:
        candidate = Path(args.dataset_root) / "SaProt10k_feature.csv"
        if candidate.exists():
            args.feature_file = str(candidate)

    sys.path.insert(0, args.protsolm_root)
    from src.dataset.supervise_dataset import SuperviseDataset
    from src.models import GNN_model, PLM_model, ProtssnClassification
    from src.utils.data_utils import BatchSampler
    from src.utils.dataset_utils import NormalizeProtein

    gnn_cfg_path = Path(args.protsolm_root) / args.gnn_config
    args.gnn_config = yaml.load(open(gnn_cfg_path), Loader=yaml.FullLoader)[args.gnn]
    args.gnn_config["hidden_channels"] = args.gnn_hidden_dim

    feature_dict, args = build_feature_dict(args)

    dataset_root = Path(args.dataset_root)
    datatset_name = dataset_root.name
    if (dataset_root / "esmfold_pdb").exists():
        pdb_dir = dataset_root / "esmfold_pdb"
    elif (dataset_root / "pdb").exists():
        pdb_dir = dataset_root / "pdb"
    else:
        raise ValueError("No pdb or esmfold_pdb directory found in the dataset")

    graph_dir = f"{datatset_name}_k{args.c_alpha_max_neighbors}"
    supervise_dataset = SuperviseDataset(
        root=str(dataset_root),
        raw_dir=str(pdb_dir),
        name=graph_dir,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        pre_transform=NormalizeProtein(filename=str(Path(args.protsolm_root) / f"norm/cath_k{args.c_alpha_max_neighbors}_mean_attr.pt")),
    )
    del supervise_dataset

    splits = {
        "valid": pd.read_csv(dataset_root / "valid.csv"),
        "test": pd.read_csv(dataset_root / "test.csv"),
    }
    label_dict = {}
    orig_idx_dict = {}
    split_names = {}
    split_node_nums = {}
    for split_name, df in splits.items():
        split_names[split_name] = df["name"].tolist()
        split_node_nums[split_name] = [len(seq) for seq in df["aa_seq"].tolist()]
        for row in df.itertuples(index=False):
            label_dict[row.name] = int(row.label)
            orig_idx_dict[row.name] = int(row.orig_idx)

    processed_dir = dataset_root / graph_dir.capitalize() / "processed"

    def process_data(name):
        data = torch.load(processed_dir / f"{name}.pt", weights_only=False)
        data.label = torch.tensor(label_dict[name]).view(1)
        data.orig_idx = torch.tensor(orig_idx_dict[name]).view(1)
        if feature_dict:
            data.feature = torch.tensor(feature_dict[name], dtype=torch.float32).view(1, -1)
        return data

    def collect_fn(batch):
        batch_data = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_data, name) for name in batch]
            for future in as_completed(futures):
                batch_data.append(future.result())
        return batch_data

    loaders = {}
    for split_name in ["valid", "test"]:
        loaders[split_name] = DataLoader(
            dataset=split_names[split_name],
            num_workers=4,
            collate_fn=lambda x: collect_fn(x),
            batch_sampler=BatchSampler(
                node_num=split_node_nums[split_name],
                max_len=args.max_graph_token_num,
                batch_token_num=args.batch_token_num,
                shuffle=False,
            ),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plm_model = PLM_model(args).to(device)
    gnn_model = GNN_model(args).to(device)
    gnn_model.load_state_dict(torch.load(args.gnn_model_path, map_location=device))
    model = ProtssnClassification(args).to(device)
    ckpt = torch.load(Path(args.model_dir) / args.model_name, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def collect_predictions(loader):
        all_probs, all_labels, all_orig_idx = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval"):
                logits = model(plm_model, gnn_model, batch)
                probs = torch.softmax(logits, dim=1)[:, 1]
                labels = torch.cat([data.label for data in batch]).to(probs.device)
                orig_idx = torch.cat([data.orig_idx for data in batch]).to(probs.device)
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_orig_idx.extend(orig_idx.cpu().numpy().tolist())
        return {
            "probs": np.asarray(all_probs, dtype=np.float32),
            "labels": np.asarray(all_labels, dtype=np.float32),
            "orig_idx": np.asarray(all_orig_idx, dtype=np.int64),
        }

    valid_results = collect_predictions(loaders["valid"])
    valid_metrics = compute_metrics(valid_results["labels"], valid_results["probs"])
    val_thr = valid_metrics["threshold"]

    test_results = collect_predictions(loaders["test"])
    test_metrics = compute_metrics(test_results["labels"], test_results["probs"], threshold=val_thr)
    bucket_rows = summarize_homology_buckets(test_results, val_thr, args.identity_path)
    save_prediction_csv(test_results, val_thr, args.save_preds, args.pred_model_name, args.pred_dataset_name)

    print("ProtSolM Audit Results")
    print(json.dumps({"valid": valid_metrics, "test": test_metrics}, indent=2))
    print(f"Validation-fixed threshold: {val_thr:.3f}")
    print_bucket_report(bucket_rows)


if __name__ == "__main__":
    main()
