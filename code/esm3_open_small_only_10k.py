import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from esm3_gnn_attn_pool_supcon_10k import (
    DEFAULT_CSV_PATH,
    DEFAULT_IDENTITY_PATH,
    compute_metrics,
    encode_sequence_tokens,
    load_meta,
    print_bucket_report,
    save_meta,
    set_global_seed,
    standardize_solubility_dataframe,
    summarize_homology_buckets,
)

sys.path.append("/home/heyong/ProBindLM")
from esm.models.esm3 import ESM3


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")


DEFAULT_MODEL_NAME = "esm3_sm_open_v1"
DEFAULT_CHECKPOINT_PATH = "/home/heyong/best_esm3_open_small_only_10k.pth"
DEFAULT_META_PATH = "/home/heyong/best_esm3_open_small_only_10k_meta.json"

PAD_TOKEN_ID = 1
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 2


class SolubilitySequenceDataset(Dataset):
    def __init__(self, df, max_len):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = str(row["protein"])[: self.max_len]
        return {
            "sequence_tokens": encode_sequence_tokens(sequence),
            "label": float(row["label"]),
            "orig_idx": int(row["orig_idx"]),
        }


class SequenceBatchCollator:
    def __call__(self, batch):
        lengths = [item["sequence_tokens"].size(0) for item in batch]
        max_len = max(lengths)

        tokens = torch.full((len(batch), max_len), PAD_TOKEN_ID, dtype=torch.long)
        for i, item in enumerate(batch):
            seq_tokens = item["sequence_tokens"]
            tokens[i, : seq_tokens.size(0)] = seq_tokens

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        orig_idx = torch.tensor([item["orig_idx"] for item in batch], dtype=torch.long)
        return {
            "sequence_tokens": tokens,
            "labels": labels,
            "orig_idx": orig_idx,
        }


class ESM3OpenSmallOnlyClassifier(nn.Module):
    def __init__(self, plm_name, freeze_backbone=True, dropout=0.2):
        super().__init__()
        self.plm = ESM3.from_pretrained(plm_name)
        if freeze_backbone:
            for param in self.plm.parameters():
                param.requires_grad = False

        hidden_dim = 1536
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence_tokens):
        use_autocast = sequence_tokens.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            embeddings = self.plm(sequence_tokens=sequence_tokens).embeddings

        token_mask = (
            sequence_tokens.ne(PAD_TOKEN_ID)
            & sequence_tokens.ne(BOS_TOKEN_ID)
            & sequence_tokens.ne(EOS_TOKEN_ID)
        )
        denom = token_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (embeddings.float() * token_mask.unsqueeze(-1)).sum(dim=1) / denom
        logits = self.head(pooled).squeeze(-1)
        return logits


def build_loaders(csv_path, max_len, batch_size, num_workers):
    df = pd.read_csv(csv_path)
    df = standardize_solubility_dataframe(df)

    train_df = df[df["stage"] == "train"].reset_index(drop=True)
    val_df = df[df["stage"] == "valid"].reset_index(drop=True)
    test_df = df[df["stage"] == "test"].reset_index(drop=True)

    collator = SequenceBatchCollator()
    train_loader = DataLoader(
        SolubilitySequenceDataset(train_df, max_len=max_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        SolubilitySequenceDataset(val_df, max_len=max_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        SolubilitySequenceDataset(test_df, max_len=max_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    return train_loader, val_loader, test_loader


def evaluate_model(model, loader, device, criterion=None):
    model.eval()
    all_probs = []
    all_labels = []
    all_orig_idx = []
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            labels = batch["labels"].to(device)
            orig_idx = batch["orig_idx"]
            sequence_tokens = batch["sequence_tokens"].to(device)

            logits = model(sequence_tokens=sequence_tokens)
            if criterion is not None:
                total_loss += criterion(logits, labels).item()
                total_batches += 1

            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_orig_idx.extend(orig_idx.numpy().tolist())

    return {
        "probs": np.asarray(all_probs, dtype=np.float32),
        "labels": np.asarray(all_labels, dtype=np.float32),
        "orig_idx": np.asarray(all_orig_idx, dtype=np.int64),
        "loss": total_loss / max(1, total_batches),
    }


def save_prediction_csv(results, threshold, output_path, model_name="ESM3-only", dataset_name="SaProt10k"):
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
    parser = argparse.ArgumentParser(description="ESM3-open-small-only baseline on local SaProt 10k.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--identity-path", default=DEFAULT_IDENTITY_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--meta-path", default=DEFAULT_META_PATH)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-preds", default=None)
    parser.add_argument("--pred-model-name", default="ESM3-only")
    parser.add_argument("--pred-dataset-name", default="SaProt10k")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    freeze_backbone = not args.unfreeze_backbone
    print(f"Loading dataset... (ESM3-open-small-only baseline, model={args.model_name})")

    train_loader, val_loader, test_loader = build_loaders(
        csv_path=args.csv_path,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESM3OpenSmallOnlyClassifier(
        plm_name=args.model_name,
        freeze_backbone=freeze_backbone,
        dropout=args.dropout,
    ).to(device)

    if args.eval_only:
        print(f"Eval-only mode: loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        meta = load_meta(args.meta_path)
        val_thr = float(meta.get("best_val_threshold", 0.5))
        if "best_val_threshold" not in meta:
            val_results = evaluate_model(model, val_loader, device)
            val_thr = compute_metrics(val_results["labels"], val_results["probs"])["threshold"]
        test_results = evaluate_model(model, test_loader, device)
        test_metrics = compute_metrics(test_results["labels"], test_results["probs"], threshold=val_thr)
        bucket_rows = summarize_homology_buckets(test_results, val_thr, args.identity_path)
        save_prediction_csv(test_results, val_thr, args.save_preds, args.pred_model_name, args.pred_dataset_name)
        print("FINAL ESM3-ONLY TEST RESULTS")
        print(f" - AUC Benchmark: {test_metrics['auc']:.4f}")
        print(f" - Calibrated Acc (Val Thr={val_thr:.3f}): {test_metrics['acc']:.4f}")
        print(f" - Calibrated MCC: {test_metrics['mcc']:.4f}")
        print_bucket_report(bucket_rows)
        return

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.plm.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": args.lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_epoch = 0
    best_val_thr = 0.5
    patience = 4
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}"), start=1):
            labels = batch["labels"].to(device)
            sequence_tokens = batch["sequence_tokens"].to(device)

            logits = model(sequence_tokens=sequence_tokens)
            loss = criterion(logits, labels)
            (loss / args.grad_accum_steps).backward()

            if step_idx % args.grad_accum_steps == 0 or step_idx == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1} Train Loss: {total_loss / len(train_loader):.4f}")

        val_results = evaluate_model(model, val_loader, device, criterion=criterion)
        val_metrics = compute_metrics(val_results["labels"], val_results["probs"])
        print(
            f"Epoch {epoch + 1} Val Loss: {val_results['loss']:.4f} | "
            f"Val Acc(Thr={val_metrics['threshold']:.3f}): {val_metrics['acc']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | Val MCC: {val_metrics['mcc']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_val_thr = val_metrics["threshold"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.checkpoint_path)
            save_meta(
                args.meta_path,
                {
                    "best_epoch": best_epoch,
                    "best_val_auc": best_val_auc,
                    "best_val_threshold": best_val_thr,
                    "model_name": args.model_name,
                    "freeze_backbone": freeze_backbone,
                    "max_len": args.max_len,
                    "dropout": args.dropout,
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
    test_results = evaluate_model(model, test_loader, device)
    test_metrics = compute_metrics(test_results["labels"], test_results["probs"], threshold=val_thr)
    bucket_rows = summarize_homology_buckets(test_results, val_thr, args.identity_path)
    save_prediction_csv(test_results, val_thr, args.save_preds, args.pred_model_name, args.pred_dataset_name)

    print("FINAL ESM3-ONLY TEST RESULTS")
    print(f" - Best epoch by Val AUC: {int(meta.get('best_epoch', best_epoch))}")
    print(f" - AUC Benchmark: {test_metrics['auc']:.4f}")
    print(f" - Calibrated Acc (Val Thr={val_thr:.3f}): {test_metrics['acc']:.4f}")
    print(f" - Calibrated MCC: {test_metrics['mcc']:.4f}")
    print_bucket_report(bucket_rows)


if __name__ == "__main__":
    main()
