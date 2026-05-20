import argparse
import csv
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GlobalAttention
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
import numpy as np

import sys
sys.path.append('/home/heyong/Protein_ADMET_SaProt')
from gvp import GVP, GVPConv

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

sys.path.append('/home/heyong/ProBindLM')
from esm.models.esm3 import ESM3

AA_TO_TOKEN = {
    "L": 4, "A": 5, "G": 6, "V": 7, "S": 8,
    "E": 9, "R": 10, "T": 11, "I": 12, "D": 13,
    "P": 14, "K": 15, "Q": 16, "N": 17, "F": 18,
    "Y": 19, "M": 20, "H": 21, "W": 22, "C": 23,
}
MASK_TOKEN_ID = 32
DEFAULT_CSV_PATH = "/home/heyong/Protein_ADMET_SaProt/solubility_10000_SaProt_Ready.csv"
DEFAULT_IDENTITY_PATH = "/home/heyong/Protein_ADMET_SaProt/test_vs_train.m8"
DEFAULT_CHECKPOINT_PATH = "best_esm3_attn_pool_supcon.pth"
DEFAULT_META_PATH = "best_esm3_attn_pool_supcon_meta.json"

SEQUENCE_COLUMN_CANDIDATES = ["protein", "sequence", "seq", "aa_seq", "Sequence", "Protein"]
LABEL_COLUMN_CANDIDATES = ["label", "labels", "target", "solubility", "Label"]
STAGE_COLUMN_CANDIDATES = ["stage", "split", "subset", "partition", "set"]
PDB_COLUMN_CANDIDATES = ["pdb_path", "structure_path", "pdb_file", "pdb"]
STAGE_ALIASES = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "dev": "valid",
    "test": "test",
    "te": "test",
}

def extract_biophysical_features(seq):
    clean_seq = "".join([aa for aa in seq if aa in "ACDEFGHIKLMNPQRSTVWY"])
    if len(clean_seq) == 0:
        return [0.0] * 8
    analyzer = ProteinAnalysis(clean_seq)
    mw = analyzer.molecular_weight() / 100000.0 
    pi = analyzer.isoelectric_point() / 14.0 
    gravy = analyzer.gravy()
    arom = analyzer.aromaticity()
    instab = analyzer.instability_index() / 100.0
    sec_struc = analyzer.secondary_structure_fraction()
    return [mw, pi, gravy, arom, instab, sec_struc[0], sec_struc[1], sec_struc[2]]

def encode_sequence_tokens(sequence: str) -> torch.Tensor:
    token_ids = [0]
    token_ids.extend(AA_TO_TOKEN.get(aa, MASK_TOKEN_ID) for aa in sequence)
    token_ids.append(2)
    return torch.tensor(token_ids, dtype=torch.long)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int):
    def _worker_init_fn(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn


def find_first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def standardize_solubility_dataframe(df):
    df = df.copy()

    seq_col = find_first_existing_column(df, SEQUENCE_COLUMN_CANDIDATES)
    if seq_col is None:
        raise ValueError(f"Could not find a sequence column. Tried: {SEQUENCE_COLUMN_CANDIDATES}")

    label_col = find_first_existing_column(df, LABEL_COLUMN_CANDIDATES)
    if label_col is None:
        raise ValueError(f"Could not find a label column. Tried: {LABEL_COLUMN_CANDIDATES}")

    stage_col = find_first_existing_column(df, STAGE_COLUMN_CANDIDATES)
    if stage_col is None:
        raise ValueError(
            "Could not find a split column. Expected one of "
            f"{STAGE_COLUMN_CANDIDATES}. Use the benchmark preparation script first."
        )

    rename_map = {}
    if seq_col != "protein":
        rename_map[seq_col] = "protein"
    if label_col != "label":
        rename_map[label_col] = "label"
    if stage_col != "stage":
        rename_map[stage_col] = "stage"

    pdb_col = find_first_existing_column(df, PDB_COLUMN_CANDIDATES)
    if pdb_col is not None and pdb_col != "pdb_path":
        rename_map[pdb_col] = "pdb_path"

    if rename_map:
        df = df.rename(columns=rename_map)

    df["protein"] = df["protein"].fillna("").astype(str).str.strip()
    df["stage"] = (
        df["stage"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: STAGE_ALIASES.get(x, x))
    )
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    keep_mask = (
        df["protein"].ne("")
        & df["label"].notna()
        & df["stage"].isin(["train", "valid", "test"])
    )
    df = df.loc[keep_mask].reset_index(drop=True)

    if "orig_idx" not in df.columns:
        df["orig_idx"] = np.arange(len(df))
    else:
        df["orig_idx"] = pd.to_numeric(df["orig_idx"], errors="coerce").fillna(-1).astype(int)

    return df


def build_knn_graph_from_coords(final_coords, k_neighbors):
    dist = torch.cdist(final_coords, final_coords)
    dist.fill_diagonal_(float("inf"))
    k = min(k_neighbors, final_coords.size(0) - 1)
    if k <= 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 1), dtype=torch.float32),
            torch.zeros((0, 1, 3), dtype=torch.float32),
        )

    _, edge_targets = torch.topk(dist, k, dim=1, largest=False)
    edge_sources = torch.arange(final_coords.size(0)).unsqueeze(1).repeat(1, k)
    edge_index = torch.stack([edge_sources.flatten(), edge_targets.flatten()], dim=0)

    src_coords = final_coords[edge_index[0]]
    dst_coords = final_coords[edge_index[1]]
    vec = dst_coords - src_coords
    d = torch.norm(vec, dim=-1, keepdim=True)
    u = vec / (d + 1e-6)
    return edge_index, d, u.unsqueeze(1)


def build_chain_graph(num_nodes, seq_window=2):
    if num_nodes <= 1 or seq_window <= 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 1), dtype=torch.float32),
            torch.zeros((0, 1, 3), dtype=torch.float32),
        )

    edge_pairs = []
    for offset in range(1, min(seq_window + 1, num_nodes)):
        src = torch.arange(0, num_nodes - offset, dtype=torch.long)
        dst = src + offset
        edge_pairs.append(torch.stack([src, dst], dim=0))
        edge_pairs.append(torch.stack([dst, src], dim=0))

    edge_index = torch.cat(edge_pairs, dim=1)
    delta = (edge_index[1] - edge_index[0]).float().unsqueeze(-1)
    edge_s = delta.abs()
    direction = torch.zeros((delta.size(0), 3), dtype=torch.float32)
    direction[:, 0] = torch.sign(delta.squeeze(-1))
    edge_v = direction.unsqueeze(1)
    return edge_index, edge_s, edge_v

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        # Normalize features against feature collapse / explosive values
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -mean_log_prob_pos.mean()
        return loss

class ProteinGraphDataset10kESM3(Dataset):
    def __init__(self, df, max_len=256, k_neighbors=20, seq_window=2):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.k_neighbors = k_neighbors
        self.seq_window = seq_window
        self.use_local_pdb_lookup = all(col in self.df.columns for col in ("foldseek_seq", "saprot_input"))
        self.parser = PDBParser(QUIET=True)

    def __len__(self):
        return len(self.df)

    def extract_coords(self, pdb_path):
        if not pdb_path:
            return torch.zeros((0, 3))
        if not os.path.exists(pdb_path):
            return torch.zeros((0, 3))
        structure = self.parser.get_structure('protein', pdb_path)
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
        if len(coords) == 0:
            return torch.zeros((0, 3))
        return torch.tensor(np.array(coords), dtype=torch.float32)

    def resolve_pdb_path(self, row, orig_idx):
        explicit_path = row.get("pdb_path")
        if isinstance(explicit_path, str) and explicit_path.strip():
            return explicit_path.strip()

        if not self.use_local_pdb_lookup:
            return None

        if 0 <= orig_idx < 3000:
            return f"/home/heyong/Protein_ADMET_SaProt/pdbs_3000/seq_{orig_idx}.pdb"
        if orig_idx >= 3000:
            return f"/home/heyong/Protein_ADMET_SaProt/pdbs_7000/seq_7k_{orig_idx - 3000}.pdb"
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['protein']
        label = float(row['label'])
        orig_idx = int(row['orig_idx'])

        surface_features = extract_biophysical_features(seq)
        pdb_path = self.resolve_pdb_path(row, orig_idx)
        coords = self.extract_coords(pdb_path)

        seq = seq[:self.max_len - 2]
        input_ids = encode_sequence_tokens(seq)

        seq_len = min(len(coords), len(input_ids) - 2)
        final_coords = torch.zeros((len(input_ids), 3), dtype=torch.float32)
        if seq_len > 0:
            final_coords[1:1+seq_len] = coords[:seq_len]

        has_structure = seq_len > 0
        if has_structure:
            edge_index, edge_s, edge_v = build_knn_graph_from_coords(final_coords, self.k_neighbors)
        else:
            edge_index, edge_s, edge_v = build_chain_graph(len(input_ids), self.seq_window)

        data = Data(
            x_ids=input_ids,
            edge_index=edge_index,
            edge_s=edge_s,
            edge_v=edge_v,
            y=torch.tensor([label], dtype=torch.float32),
            orig_idx=torch.tensor([orig_idx], dtype=torch.long),
            coords=final_coords,
            num_nodes=len(input_ids),
            surface_feats=torch.tensor(surface_features, dtype=torch.float32).unsqueeze(0)
        )
        return data

class ESM3GVPAttnPoolSupCon(nn.Module):
    def __init__(self, plm_name="esm3_sm_open_v1", freeze_plm=True):
        super().__init__()
        self.plm = ESM3.from_pretrained(plm_name)
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
                
        plm_dim = 1536
        
        self.node_s_proj = nn.Linear(plm_dim, 128)
        
        self.gvp_conv1 = GVPConv((128, 1), (1, 1), (128, 16))
        self.gvp_conv2 = GVPConv((128, 16), (1, 1), (128, 16))
        
        node_repr_dim = 128 + 16 * 3
        
        gate_nn = nn.Sequential(
            nn.Linear(node_repr_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.attn_pool = GlobalAttention(gate_nn=gate_nn)
        
        fusion_dim = node_repr_dim + 8 
        
        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_ids, edge_index, edge_s, edge_v, batch_index, surface_feats, return_repr=False):
        from torch_geometric.utils import unbatch
        
        device = x_ids.device
        ids_list = unbatch(x_ids, batch_index)
        
        max_l = max([ids.size(0) for ids in ids_list])
        B = len(ids_list)
        
        plm_ids = torch.ones(B, max_l, dtype=torch.long, device=device)
        for i in range(B):
            L = ids_list[i].size(0)
            plm_ids[i, :L] = ids_list[i]
            
        with torch.autocast("cuda", dtype=torch.bfloat16):
            plm_out = self.plm(sequence_tokens=plm_ids).embeddings
        
        plm_nodes = []
        for i in range(B):
            L = ids_list[i].size(0)
            plm_nodes.append(plm_out[i, :L, :].float())
        node_feats = torch.cat(plm_nodes, dim=0) 
        
        node_s = F.relu(self.node_s_proj(node_feats))
        node_v = torch.zeros((node_s.size(0), 1, 3), device=device) 
        
        out_s, out_v = self.gvp_conv1((node_s, node_v), edge_index, (edge_s, edge_v))
        out_s, out_v = self.gvp_conv2((out_s, out_v), edge_index, (edge_s, edge_v))
        
        out_v_flat = out_v.reshape(out_v.size(0), -1)
        node_repr = torch.cat([out_s, out_v_flat], dim=1)
        
        graph_repr = self.attn_pool(node_repr, batch_index)
        fused_repr = torch.cat([graph_repr, surface_feats], dim=1)
        
        preds = self.out_proj(fused_repr)
        
        if return_repr:
            return preds.squeeze(-1), fused_repr
        return preds.squeeze(-1)

def collect_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_orig_idx = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            batch = batch.to(device)
            out = model(batch.x_ids, batch.edge_index, batch.edge_s, batch.edge_v, batch.batch, batch.surface_feats)
            probs = torch.sigmoid(out)
            y_true = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y_true.cpu().numpy().tolist())
            if hasattr(batch, "orig_idx"):
                all_orig_idx.extend(batch.orig_idx.view(-1).cpu().numpy().tolist())

    return {
        "probs": np.asarray(all_probs, dtype=np.float32),
        "labels": np.asarray(all_labels, dtype=np.float32),
        "orig_idx": np.asarray(all_orig_idx, dtype=np.int64),
    }

def save_prediction_csv(results, threshold, output_path, model_name="GVP+SupCon", dataset_name="SaProt10k"):
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
        bucket_rows.append({
            "bucket": bucket_name,
            "n": len(selected),
            "pos_rate": float(np.mean(labels)),
            "mean_max_identity": float(np.mean(identities)),
            "auc": metrics["auc"],
            "acc": metrics["acc"],
            "mcc": metrics["mcc"],
        })
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

def save_meta(meta_path, meta):
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

def load_meta(meta_path):
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def build_loaders(df, max_len, k_neighbors, batch_size, num_workers, seq_window=2, seed=None):
    df = standardize_solubility_dataframe(df)
    train_df = df[df["stage"] == "train"]
    val_df = df[df["stage"] == "valid"]
    test_df = df[df["stage"] == "test"]

    train_ds = ProteinGraphDataset10kESM3(train_df, max_len=max_len, k_neighbors=k_neighbors, seq_window=seq_window)
    val_ds = ProteinGraphDataset10kESM3(val_df, max_len=max_len, k_neighbors=k_neighbors, seq_window=seq_window)
    test_ds = ProteinGraphDataset10kESM3(test_df, max_len=max_len, k_neighbors=k_neighbors, seq_window=seq_window)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    if seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)
        loader_kwargs["worker_init_fn"] = make_worker_init_fn(seed)
        loader_kwargs["generator"] = train_generator

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader

def parse_args():
    parser = argparse.ArgumentParser(description="ESM3 + GVP + attention pooling + SupCon for solubility.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--identity-path", default=DEFAULT_IDENTITY_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--meta-path", default=DEFAULT_META_PATH)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--k-neighbors", type=int, default=20)
    parser.add_argument("--seq-window", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-preds", default=None)
    parser.add_argument("--pred-model-name", default="GVP+SupCon")
    parser.add_argument("--pred-dataset-name", default="SaProt10k")
    return parser.parse_args()

def main():
    args = parse_args()
    set_global_seed(args.seed)
    print("Loading 10k dataset... (ESM3 + ATTN POOLING + GVP + SUPCON)")
    df = pd.read_csv(args.csv_path)
    df['orig_idx'] = np.arange(len(df))

    train_loader, val_loader, test_loader = build_loaders(
        df,
        max_len=args.max_len,
        k_neighbors=args.k_neighbors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_window=args.seq_window,
        seed=args.seed,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESM3GVPAttnPoolSupCon().to(device)

    if args.eval_only:
        print(f"Eval-only mode: loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        meta = load_meta(args.meta_path)
        if "best_val_threshold" in meta:
            val_thr = float(meta["best_val_threshold"])
            print(f"Using saved validation threshold: {val_thr:.3f}")
        else:
            print("No saved validation threshold found, recomputing on validation split...")
            val_results = collect_predictions(model, val_loader, device)
            val_thr = compute_metrics(val_results["labels"], val_results["probs"])["threshold"]

        test_results = collect_predictions(model, test_loader, device)
        test_metrics = compute_metrics(test_results["labels"], test_results["probs"], threshold=val_thr)
        bucket_rows = summarize_homology_buckets(test_results, val_thr, args.identity_path)
        save_prediction_csv(test_results, val_thr, args.save_preds, args.pred_model_name, args.pred_dataset_name)

        print("🌟 FINAL ESM3 ATTN-POOLING + SUPCON TEST RESULTS 🌟")
        print(f" - AUC Benchmark: {test_metrics['auc']:.4f}")
        print(f" - Calibrated Acc (Val Thr={val_thr:.3f}): {test_metrics['acc']:.4f}")
        print(f" - Calibrated MCC: {test_metrics['mcc']:.4f}")
        print_bucket_report(bucket_rows)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_supcon = SupConLoss(temperature=0.1)

    alpha = args.alpha
    best_auc = 0
    best_val_thr = 0.5
    best_epoch = 0
    epochs = args.epochs
    patience = 5
    patience_counter = 0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_bce = 0
        total_supcon = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out, reprs = model(batch.x_ids, batch.edge_index, batch.edge_s, batch.edge_v, batch.batch, batch.surface_feats, return_repr=True)
            y_true = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y
            
            loss_bce = criterion_bce(out, y_true)
            loss_supcon = criterion_supcon(reprs, y_true)
            
            loss = loss_bce + alpha * loss_supcon
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_supcon += loss_supcon.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f} (BCE: {total_bce/len(train_loader):.4f}, SupCon: {total_supcon/len(train_loader):.4f})")
        
        val_results = collect_predictions(model, val_loader, device)
        val_metrics = compute_metrics(val_results["labels"], val_results["probs"])
        print(
            f"Epoch {epoch+1} Val Acc(Thr={val_metrics['threshold']:.3f}): {val_metrics['acc']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | Val MCC: {val_metrics['mcc']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_val_thr = val_metrics["threshold"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.checkpoint_path)
            save_meta(
                args.meta_path,
                {
                    "best_epoch": best_epoch,
                    "best_val_auc": best_auc,
                    "best_val_threshold": best_val_thr,
                    "alpha": alpha,
                },
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
            
    print("Training finished. Evaluating on Test Split...")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    meta = load_meta(args.meta_path)
    val_thr = float(meta.get("best_val_threshold", best_val_thr))
    test_results = collect_predictions(model, test_loader, device)
    test_metrics = compute_metrics(test_results["labels"], test_results["probs"], threshold=val_thr)
    bucket_rows = summarize_homology_buckets(test_results, val_thr, args.identity_path)
    save_prediction_csv(test_results, val_thr, args.save_preds, args.pred_model_name, args.pred_dataset_name)

    print("🌟 FINAL ESM3 ATTN-POOLING + SUPCON TEST RESULTS 🌟")
    print(f" - Best epoch by Val AUC: {int(meta.get('best_epoch', best_epoch))}")
    print(f" - AUC Benchmark: {test_metrics['auc']:.4f}")
    print(f" - Calibrated Acc (Val Thr={val_thr:.3f}): {test_metrics['acc']:.4f}")
    print(f" - Calibrated MCC: {test_metrics['mcc']:.4f}")
    print_bucket_report(bucket_rows)

if __name__ == "__main__":
    main()
