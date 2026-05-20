import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer

from esm3_gnn_attn_pool_supcon_10k import (
    DEFAULT_CSV_PATH,
    DEFAULT_IDENTITY_PATH,
    compute_metrics,
    load_meta,
    print_bucket_report,
    save_meta,
    standardize_solubility_dataframe,
    summarize_homology_buckets,
)


DEFAULT_MODEL_NAME = "facebook/esm1b_t33_650M_UR50S"
DEFAULT_CHECKPOINT_PATH = "/home/heyong/best_netsolp_official_esm1b_10k.pth"
DEFAULT_META_PATH = "/home/heyong/best_netsolp_official_esm1b_10k_meta.json"


class SolubilitySequenceDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "sequence": str(row["protein"]),
            "label": float(row["label"]),
            "orig_idx": int(row["orig_idx"]),
        }


class SequenceBatchCollator:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        seqs = [item["sequence"][: self.max_len] for item in batch]
        toks = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len + 2,
        )
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        orig_idx = torch.tensor([item["orig_idx"] for item in batch], dtype=torch.long)
        toks["labels"] = labels
        toks["orig_idx"] = orig_idx
        return toks


class NetSolPOfficialESM1b(nn.Module):
    def __init__(self, model_name, freeze_backbone=True, dropout=0.2):
        super().__init__()
        self.backbone = EsmModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        special_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        pad_token_id = getattr(self.backbone.config, "pad_token_id", None)
        bos_token_id = getattr(self.backbone.config, "bos_token_id", None)
        eos_token_id = getattr(self.backbone.config, "eos_token_id", None)
        cls_token_id = getattr(self.backbone.config, "cls_token_id", None)

        if pad_token_id is not None:
            special_mask |= input_ids.eq(pad_token_id)
        if bos_token_id is not None:
            special_mask |= input_ids.eq(bos_token_id)
        if cls_token_id is not None:
            special_mask |= input_ids.eq(cls_token_id)
        if eos_token_id is not None:
            special_mask |= input_ids.eq(eos_token_id)

        token_mask = attention_mask.bool() & (~special_mask)
        denom = token_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (hidden * token_mask.unsqueeze(-1)).sum(dim=1) / denom
        logits = self.head(pooled).squeeze(-1)
        return logits


def build_loaders(csv_path, model_name, max_len, batch_size, num_workers):
    df = pd.read_csv(csv_path)
    df = standardize_solubility_dataframe(df)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    collator = SequenceBatchCollator(tokenizer=tokenizer, max_len=max_len)

    train_df = df[df["stage"] == "train"].reset_index(drop=True)
    val_df = df[df["stage"] == "valid"].reset_index(drop=True)
    test_df = df[df["stage"] == "test"].reset_index(drop=True)

    train_loader = DataLoader(
        SolubilitySequenceDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        SolubilitySequenceDataset(val_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        SolubilitySequenceDataset(test_df),
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
            labels = batch.pop("labels")
            orig_idx = batch.pop("orig_idx")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
            if criterion is not None:
                total_loss += criterion(logits, labels.to(device)).item()
                total_batches += 1
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_orig_idx.extend(orig_idx.numpy().tolist())
    return {
        "probs": np.asarray(all_probs, dtype=np.float32),
        "labels": np.asarray(all_labels, dtype=np.float32),
        "orig_idx": np.asarray(all_orig_idx, dtype=np.int64),
        "loss": total_loss / max(1, total_batches),
    }


def save_prediction_csv(results, threshold, output_path, model_name="Official-like ESM1b", dataset_name="SaProt10k"):
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
    parser = argparse.ArgumentParser(description="Official NetSolP-like ESM1b baseline on local 10k dataset.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--identity-path", default=DEFAULT_IDENTITY_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--meta-path", default=DEFAULT_META_PATH)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=510)
    parser.add_argument("--lr-head", type=float, default=2e-5)
    parser.add_argument("--lr-backbone", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--save-preds", default=None)
    parser.add_argument("--pred-model-name", default="Official-like ESM1b")
    parser.add_argument("--pred-dataset-name", default="SaProt10k")
    return parser.parse_args()


def main():
    args = parse_args()
    freeze_backbone = not args.unfreeze_backbone
    print(f"Loading dataset... (Official NetSolP ESM1b baseline, model={args.model_name})")

    train_loader, val_loader, test_loader = build_loaders(
        csv_path=args.csv_path,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetSolPOfficialESM1b(model_name=args.model_name, freeze_backbone=freeze_backbone).to(device)

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
        print("🌟 FINAL OFFICIAL NETSOLP ESM1B BASELINE TEST RESULTS 🌟")
        print(f" - AUC Benchmark: {test_metrics['auc']:.4f}")
        print(f" - Calibrated Acc (Val Thr={val_thr:.3f}): {test_metrics['acc']:.4f}")
        print(f" - Calibrated MCC: {test_metrics['mcc']:.4f}")
        print_bucket_report(bucket_rows)
        return

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": args.lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_thr = 0.5
    patience = 4
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        # Re-run loop with proper optimizer stepping semantics.
        optimizer.zero_grad()
        total_loss = 0.0
        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}"), start=1):
            labels = batch.pop("labels").to(device)
            batch.pop("orig_idx")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
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

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            best_val_thr = val_metrics["threshold"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.checkpoint_path)
            save_meta(
                args.meta_path,
                {
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_threshold": best_val_thr,
                    "model_name": args.model_name,
                    "freeze_backbone": freeze_backbone,
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

    print("🌟 FINAL OFFICIAL NETSOLP ESM1B BASELINE TEST RESULTS 🌟")
    print(f" - Best epoch by Val Loss: {int(meta.get('best_epoch', best_epoch))}")
    print(f" - AUC Benchmark: {test_metrics['auc']:.4f}")
    print(f" - Calibrated Acc (Val Thr={val_thr:.3f}): {test_metrics['acc']:.4f}")
    print(f" - Calibrated MCC: {test_metrics['mcc']:.4f}")
    print_bucket_report(bucket_rows)


if __name__ == "__main__":
    main()
