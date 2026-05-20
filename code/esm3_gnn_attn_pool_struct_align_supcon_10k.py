import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention
from tqdm import tqdm

from esm3_gnn_attn_pool_supcon_10k import (
    DEFAULT_CSV_PATH,
    DEFAULT_IDENTITY_PATH,
    SupConLoss,
    build_loaders,
    compute_metrics,
    load_meta,
    print_bucket_report,
    save_meta,
    set_global_seed,
    summarize_homology_buckets,
)
from gvp import GVPConv

import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

sys.path.append('/home/heyong/ProBindLM')
from esm.models.esm3 import ESM3


DEFAULT_CHECKPOINT_PATH = "/home/heyong/best_esm3_attn_pool_struct_align_supcon.pth"
DEFAULT_META_PATH = "/home/heyong/best_esm3_attn_pool_struct_align_supcon_meta.json"


class ResidueAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.1, max_residues=2048):
        super().__init__()
        self.temperature = temperature
        self.max_residues = max_residues

    def forward(self, seq_repr, struct_repr, mask):
        if mask.sum() < 2:
            return seq_repr.new_tensor(0.0)

        seq_repr = seq_repr[mask]
        struct_repr = struct_repr[mask]
        if seq_repr.size(0) > self.max_residues:
            idx = torch.randperm(seq_repr.size(0), device=seq_repr.device)[: self.max_residues]
            seq_repr = seq_repr[idx]
            struct_repr = struct_repr[idx]

        seq_repr = F.normalize(seq_repr, dim=1)
        struct_repr = F.normalize(struct_repr, dim=1)
        logits = seq_repr @ struct_repr.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class ESM3GVPAttnPoolStructAlignSupCon(nn.Module):
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

        self.seq_align_proj = nn.Linear(128, 128)
        self.struct_align_proj = nn.Linear(128, 128)

        node_repr_dim = 128 + 16 * 3
        self.attn_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(node_repr_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        )
        self.out_proj = nn.Sequential(
            nn.Linear(node_repr_dim + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x_ids, edge_index, edge_s, edge_v, batch_index, surface_feats, coords=None, return_aux=False):
        from torch_geometric.utils import unbatch

        device = x_ids.device
        ids_list = unbatch(x_ids, batch_index)
        batch_size = len(ids_list)
        max_len = max(ids.size(0) for ids in ids_list)

        plm_ids = torch.ones(batch_size, max_len, dtype=torch.long, device=device)
        for i, ids in enumerate(ids_list):
            plm_ids[i, : ids.size(0)] = ids

        with torch.autocast("cuda", dtype=torch.bfloat16):
            plm_out = self.plm(sequence_tokens=plm_ids).embeddings

        node_feats = torch.cat(
            [plm_out[i, : ids.size(0), :].float() for i, ids in enumerate(ids_list)],
            dim=0,
        )
        node_s0 = F.relu(self.node_s_proj(node_feats))
        node_v0 = torch.zeros((node_s0.size(0), 1, 3), device=device)

        out_s, out_v = self.gvp_conv1((node_s0, node_v0), edge_index, (edge_s, edge_v))
        out_s, out_v = self.gvp_conv2((out_s, out_v), edge_index, (edge_s, edge_v))

        node_repr = torch.cat([out_s, out_v.reshape(out_v.size(0), -1)], dim=1)
        graph_repr = self.attn_pool(node_repr, batch_index)
        fused_repr = torch.cat([graph_repr, surface_feats], dim=1)
        preds = self.out_proj(fused_repr).squeeze(-1)

        if not return_aux:
            return preds

        if coords is None:
            coord_mask = torch.ones_like(x_ids, dtype=torch.bool)
        else:
            coord_mask = coords.abs().sum(dim=-1) > 0
        residue_mask = coord_mask & (x_ids != 0) & (x_ids != 2)

        return preds, fused_repr, {
            "seq_repr": self.seq_align_proj(node_s0),
            "struct_repr": self.struct_align_proj(out_s),
            "residue_mask": residue_mask,
        }


def collect_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_orig_idx = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            batch = batch.to(device)
            out = model(
                batch.x_ids, batch.edge_index, batch.edge_s, batch.edge_v, batch.batch, batch.surface_feats, coords=batch.coords
            )
            probs = torch.sigmoid(out)
            y_true = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y_true.cpu().numpy().tolist())
            all_orig_idx.extend(batch.orig_idx.view(-1).cpu().numpy().tolist())
    return {
        "probs": np.asarray(all_probs, dtype=np.float32),
        "labels": np.asarray(all_labels, dtype=np.float32),
        "orig_idx": np.asarray(all_orig_idx, dtype=np.int64),
    }


def save_prediction_csv(results, threshold, output_path, model_name="Struct-Align v1", dataset_name="SaProt10k"):
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
    parser = argparse.ArgumentParser(description="ESM3 attn-pool + residue-level structure alignment.")
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
    parser.add_argument("--graph-alpha", type=float, default=0.1)
    parser.add_argument("--align-beta", type=float, default=0.2)
    parser.add_argument("--align-temp", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-preds", default=None)
    parser.add_argument("--pred-model-name", default="Struct-Align v1")
    parser.add_argument("--pred-dataset-name", default="SaProt10k")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    print("Loading 10k dataset... (ESM3 + ATTN POOLING + RESIDUE-ALIGN + SMALL SUPCON)")
    df = pd.read_csv(args.csv_path)
    df["orig_idx"] = np.arange(len(df))

    train_loader, val_loader, test_loader = build_loaders(
        df,
        max_len=args.max_len,
        k_neighbors=args.k_neighbors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESM3GVPAttnPoolStructAlignSupCon().to(device)

    if args.eval_only:
        print(f"Eval-only mode: loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        meta = load_meta(args.meta_path)
        val_thr = float(meta.get("best_val_threshold", 0.5))
        if "best_val_threshold" not in meta:
            val_results = collect_predictions(model, val_loader, device)
            val_thr = compute_metrics(val_results["labels"], val_results["probs"])["threshold"]
        test_results = collect_predictions(model, test_loader, device)
        test_metrics = compute_metrics(test_results["labels"], test_results["probs"], threshold=val_thr)
        bucket_rows = summarize_homology_buckets(test_results, val_thr, args.identity_path)
        save_prediction_csv(test_results, val_thr, args.save_preds, args.pred_model_name, args.pred_dataset_name)
        print("🌟 FINAL ESM3 ATTN-POOLING + STRUCT-ALIGN TEST RESULTS 🌟")
        print(f" - AUC Benchmark: {test_metrics['auc']:.4f}")
        print(f" - Calibrated Acc (Val Thr={val_thr:.3f}): {test_metrics['acc']:.4f}")
        print(f" - Calibrated MCC: {test_metrics['mcc']:.4f}")
        print_bucket_report(bucket_rows)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_supcon = SupConLoss(temperature=0.1)
    criterion_align = ResidueAlignmentLoss(temperature=args.align_temp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0
    best_epoch = 0
    best_val_thr = 0.5
    patience = 5
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = total_bce = total_graph = total_align = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out, graph_repr, aux = model(
                batch.x_ids,
                batch.edge_index,
                batch.edge_s,
                batch.edge_v,
                batch.batch,
                batch.surface_feats,
                coords=batch.coords,
                return_aux=True,
            )
            y_true = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y

            loss_bce = criterion_bce(out, y_true)
            loss_graph = criterion_supcon(graph_repr, y_true)
            loss_align = criterion_align(aux["seq_repr"], aux["struct_repr"], aux["residue_mask"])
            loss = loss_bce + args.graph_alpha * loss_graph + args.align_beta * loss_align
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_graph += loss_graph.item()
            total_align += loss_align.item()

        scheduler.step()
        print(
            f"Epoch {epoch + 1} Train Loss: {total_loss / len(train_loader):.4f} "
            f"(BCE: {total_bce / len(train_loader):.4f}, GraphSupCon: {total_graph / len(train_loader):.4f}, "
            f"Align: {total_align / len(train_loader):.4f})"
        )

        val_results = collect_predictions(model, val_loader, device)
        val_metrics = compute_metrics(val_results["labels"], val_results["probs"])
        print(
            f"Epoch {epoch + 1} Val Acc(Thr={val_metrics['threshold']:.3f}): {val_metrics['acc']:.4f} | "
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
                    "graph_alpha": args.graph_alpha,
                    "align_beta": args.align_beta,
                    "seed": args.seed,
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

    print("🌟 FINAL ESM3 ATTN-POOLING + STRUCT-ALIGN TEST RESULTS 🌟")
    print(f" - Best epoch by Val AUC: {int(meta.get('best_epoch', best_epoch))}")
    print(f" - AUC Benchmark: {test_metrics['auc']:.4f}")
    print(f" - Calibrated Acc (Val Thr={val_thr:.3f}): {test_metrics['acc']:.4f}")
    print(f" - Calibrated MCC: {test_metrics['mcc']:.4f}")
    print_bucket_report(bucket_rows)


if __name__ == "__main__":
    main()
