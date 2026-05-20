#!/usr/bin/env python3
"""Bootstrap confidence intervals and paired significance from prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score


METRICS = ["auc", "acc", "mcc"]


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:
        pass

    columns = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        rows.append(values)

    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    table.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(table)


def load_max_identity_map(identity_path: str | None) -> dict[int, float]:
    max_identity: dict[int, float] = {}
    if not identity_path or not Path(identity_path).exists():
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
            max_identity[test_idx] = max(max_identity.get(test_idx, 0.0), pident)
    return max_identity


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(int)
    probs = probs.astype(float)
    preds = probs > threshold
    out: dict[str, float] = {}
    try:
        out["auc"] = float(roc_auc_score(labels, probs))
    except Exception:
        out["auc"] = float("nan")
    out["acc"] = float(accuracy_score(labels, preds))
    try:
        out["mcc"] = float(matthews_corrcoef(labels, preds))
    except Exception:
        out["mcc"] = 0.0
    return out


def bootstrap_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    n = labels.size
    samples = {metric: [] for metric in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.unique(labels[idx]).size < 2:
            continue
        vals = compute_metrics(labels[idx], probs[idx], threshold)
        for metric in METRICS:
            if not math.isnan(vals[metric]):
                samples[metric].append(vals[metric])
    ci = {}
    for metric in METRICS:
        arr = np.asarray(samples[metric], dtype=float)
        if arr.size == 0:
            ci[metric] = (float("nan"), float("nan"))
        else:
            ci[metric] = tuple(np.percentile(arr, [2.5, 97.5]).tolist())
    return ci


def paired_bootstrap(
    labels: np.ndarray,
    ref_probs: np.ndarray,
    ref_thr: float,
    base_probs: np.ndarray,
    base_thr: float,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    n = labels.size
    deltas = {metric: [] for metric in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.unique(labels[idx]).size < 2:
            continue
        ref = compute_metrics(labels[idx], ref_probs[idx], ref_thr)
        base = compute_metrics(labels[idx], base_probs[idx], base_thr)
        for metric in METRICS:
            if not math.isnan(ref[metric]) and not math.isnan(base[metric]):
                deltas[metric].append(ref[metric] - base[metric])

    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        arr = np.asarray(deltas[metric], dtype=float)
        if arr.size == 0:
            out[metric] = {"delta_ci_low": float("nan"), "delta_ci_high": float("nan"), "p_two_sided": float("nan")}
            continue
        p_left = float(np.mean(arr <= 0.0))
        p_right = float(np.mean(arr >= 0.0))
        out[metric] = {
            "delta_ci_low": float(np.percentile(arr, 2.5)),
            "delta_ci_high": float(np.percentile(arr, 97.5)),
            "p_two_sided": min(1.0, 2.0 * min(p_left, p_right)),
        }
    return out


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(pred_dir.glob("*.csv")):
        df = pd.read_csv(path)
        required = {"model", "orig_idx", "label", "prob", "threshold"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frames.append(df)
    if not frames:
        raise ValueError(f"No prediction CSV files found in {pred_dir}")
    return pd.concat(frames, ignore_index=True)


def subset_frame(df: pd.DataFrame, subset: str, identity_map: dict[int, float]) -> pd.DataFrame:
    if subset == "all":
        return df.copy()
    if subset == "lt30":
        if not identity_map:
            raise ValueError("lt30 subset requested but identity map is empty")
        keep = df["orig_idx"].map(lambda x: identity_map.get(int(x), 0.0) < 30.0)
        return df.loc[keep].copy()
    raise ValueError(f"Unknown subset: {subset}")


def build_outputs(args: argparse.Namespace) -> None:
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_predictions(pred_dir)
    identity_map = load_max_identity_map(args.identity_path)
    rng = np.random.default_rng(args.seed)

    all_ci_rows = []
    all_pair_rows = []

    for subset in args.subsets:
        df = subset_frame(df_all, subset, identity_map)
        labels_by_model = {}
        probs_by_model = {}
        thresholds = {}
        model_order = []

        for model, g in df.groupby("model", sort=False):
            g = g.sort_values("orig_idx")
            labels_by_model[model] = g["label"].to_numpy(dtype=int)
            probs_by_model[model] = g["prob"].to_numpy(dtype=float)
            thresholds[model] = float(g["threshold"].iloc[0])
            model_order.append(model)

        if args.reference_model not in probs_by_model:
            raise ValueError(f"Reference model {args.reference_model!r} not found for subset {subset}")

        ref_idx = df[df["model"] == args.reference_model].sort_values("orig_idx")["orig_idx"].to_numpy(dtype=int)
        ref_labels = labels_by_model[args.reference_model]

        for model in model_order:
            g_idx = df[df["model"] == model].sort_values("orig_idx")["orig_idx"].to_numpy(dtype=int)
            if not np.array_equal(g_idx, ref_idx):
                raise ValueError(f"Model {model} is not aligned with reference model on subset {subset}")
            if not np.array_equal(labels_by_model[model], ref_labels):
                raise ValueError(f"Labels differ between {model} and reference on subset {subset}")

            point = compute_metrics(ref_labels, probs_by_model[model], thresholds[model])
            ci = bootstrap_metrics(ref_labels, probs_by_model[model], thresholds[model], args.n_boot, rng)
            row = {
                "subset": subset,
                "model": model,
                "n": int(ref_labels.size),
                "threshold": thresholds[model],
            }
            for metric in METRICS:
                row[metric] = point[metric]
                row[f"{metric}_ci_low"] = ci[metric][0]
                row[f"{metric}_ci_high"] = ci[metric][1]
            all_ci_rows.append(row)

        for model in model_order:
            if model == args.reference_model:
                continue
            paired = paired_bootstrap(
                ref_labels,
                probs_by_model[args.reference_model],
                thresholds[args.reference_model],
                probs_by_model[model],
                thresholds[model],
                args.n_boot,
                rng,
            )
            point_ref = compute_metrics(ref_labels, probs_by_model[args.reference_model], thresholds[args.reference_model])
            point_base = compute_metrics(ref_labels, probs_by_model[model], thresholds[model])
            pair_row = {
                "subset": subset,
                "reference_model": args.reference_model,
                "baseline_model": model,
                "n": int(ref_labels.size),
            }
            for metric in METRICS:
                pair_row[f"{metric}_delta"] = point_ref[metric] - point_base[metric]
                pair_row[f"{metric}_delta_ci_low"] = paired[metric]["delta_ci_low"]
                pair_row[f"{metric}_delta_ci_high"] = paired[metric]["delta_ci_high"]
                pair_row[f"{metric}_p_two_sided"] = paired[metric]["p_two_sided"]
            all_pair_rows.append(pair_row)

    ci_df = pd.DataFrame(all_ci_rows)
    pair_df = pd.DataFrame(all_pair_rows)
    ci_path = out_dir / "bootstrap_metric_ci.csv"
    pair_path = out_dir / "paired_bootstrap_significance.csv"
    ci_df.to_csv(ci_path, index=False)
    pair_df.to_csv(pair_path, index=False)

    md_lines = ["# Bootstrap CI and paired bootstrap significance", ""]
    md_lines.append(f"- Prediction directory: `{pred_dir}`")
    md_lines.append(f"- Reference model: `{args.reference_model}`")
    md_lines.append(f"- Bootstrap resamples: `{args.n_boot}`")
    md_lines.append("")
    md_lines.append("## Metric confidence intervals")
    md_lines.append("")
    md_lines.append(df_to_markdown(ci_df, floatfmt=".4f"))
    md_lines.append("")
    md_lines.append("## Paired bootstrap significance")
    md_lines.append("")
    md_lines.append(df_to_markdown(pair_df, floatfmt=".4f"))
    md_lines.append("")
    (out_dir / "bootstrap_significance_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    manifest = {
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "reference_model": args.reference_model,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "subsets": args.subsets,
        "ci_csv": str(ci_path),
        "paired_csv": str(pair_path),
    }
    (out_dir / "bootstrap_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {ci_path}")
    print(f"Wrote {pair_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-model", default="Struct-Align v1")
    parser.add_argument("--identity-path", default="/home/heyong/Protein_ADMET_SaProt/test_vs_train.m8")
    parser.add_argument("--subsets", nargs="+", default=["all", "lt30"], choices=["all", "lt30"])
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260430)
    return parser.parse_args()


if __name__ == "__main__":
    build_outputs(parse_args())
