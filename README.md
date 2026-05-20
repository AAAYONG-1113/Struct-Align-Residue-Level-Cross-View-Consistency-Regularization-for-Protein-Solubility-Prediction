# Struct-Align: Protein Solubility Prediction

This repository contains code, result tables, figures, and reproducibility materials for the manuscript:

**Struct-Align: Residue-Level Cross-View Consistency Regularization for Protein Solubility Prediction**

Struct-Align is a multimodal protein solubility predictor that combines frozen ESM3-open-small residue embeddings, a GVP structure graph encoder, supervised contrastive learning, physicochemical features, and a residue-level sequence-structure alignment objective used during training.

## Main Results

The primary benchmark is **SaProt10k**, a 10,000-protein subset derived from the local SaProt-based solubility dataset used in this study.

| Dataset | Model | AUC | Accuracy | MCC |
|---|---|---:|---:|---:|
| SaProt10k | ESM3-only | 0.7205 | 0.6154 | 0.2485 |
| SaProt10k | ProtSolM | 0.8291 | 0.7260 | 0.4606 |
| SaProt10k | Struct-Align w/o Align Loss | 0.8237 | 0.7115 | 0.4360 |
| SaProt10k | Surface FiLM | 0.8175 | 0.7356 | 0.4801 |
| SaProt10k | Struct-Align | 0.8565 | 0.7740 | 0.5544 |
| NetSolP official | ProtSolM | 0.7653 | 0.7211 | 0.3643 |
| NetSolP official | Struct-Align w/o Align Loss | 0.7830 | 0.7256 | 0.3961 |
| NetSolP official | Struct-Align | 0.7677 | 0.7226 | 0.3680 |

The NetSolP result should be interpreted as an external protocol-transfer evaluation. Struct-Align is strongest on the SaProt10k main split, while the no-alignment graph-contrastive variant performs best under the NetSolP official protocol.

## Repository Layout

```text
.
├── code/                                          # Core model, baseline, and utility code
├── data/
│   ├── saprot10k/                                 # SaProt10k CSV and homology file
│   ├── netsolp/                                   # Cleaned NetSolP external-protocol CSV
│   └── splits/                                    # Random CV5 and homology split metadata
├── diagnostics/                                   # Lightweight diagnostic result tables
├── figures/                                      # Final paper figures
├── manifests/                                    # Mapping from reported results to source artifacts
├── manuscript/                                   # BIBM manuscript source and compiled PDF
└── tables/                                       # Machine-readable paper result tables
```

Large intermediate folders such as `pdbs_3000/`, `pdbs_7000/`, `protsolm_saprot10k/`, `protsolm_netsolp_official/`, and full ESMFold structure directories are not recommended for normal Git upload. Use Git LFS, a GitHub Release, Zenodo, OSF, or institutional storage for these artifacts.

Figure-generation scripts and manuscript-facing reports are not included in this GitHub submission package. They remain in the local working archive and can be provided separately if needed.

## Environment

The experiments were run with Python 3, PyTorch, PyTorch Geometric, ESM3, Biopython, scikit-learn, pandas, NumPy, SciPy, and matplotlib.

A lightweight dependency snapshot is provided in:

```text
requirements.txt
```

The ESM3 scripts expect a local ESM3/ProBindLM installation and currently include local path assumptions such as:

```text
/home/heyong/ProBindLM
/home/heyong/Protein_ADMET_SaProt
```

Before running on another machine, update these paths or set equivalent local symlinks.

## Data

### Required for SaProt10k

```text
data/saprot10k/solubility_10000_SaProt_Ready.csv
data/saprot10k/test_vs_train.m8
pdbs_3000/
pdbs_7000/
```

The CSV contains protein sequences, binary solubility labels, split labels, and structure-compatible fields. The PDB folders provide the structure inputs used by the graph branch. The `test_vs_train.m8` file is used for the low-homology bucket analysis.

### Required for NetSolP External Evaluation

```text
data/netsolp/netsolp_ready_esmfold_struct_clean.csv
external_benchmarks/
```

NetSolP structural inputs were generated with ESMFold and are treated as predicted-structure external-protocol inputs.

### Optional Robustness Splits

```text
data/splits/cv5_random/
data/splits/cv5_homology/
```

These folders store random five-fold and homology-aware split metadata used for robustness checks.

## Training and Evaluation

Run commands assume the current working directory is the repository root.

### Struct-Align on SaProt10k

```bash
python code/esm3_gnn_attn_pool_struct_align_supcon_10k.py \
  --csv-path data/saprot10k/solubility_10000_SaProt_Ready.csv \
  --identity-path data/saprot10k/test_vs_train.m8 \
  --epochs 20 \
  --batch-size 8 \
  --align-beta 0.2 \
  --graph-alpha 0.1 \
  --checkpoint-path best_esm3_attn_pool_struct_align_supcon.pth \
  --meta-path best_esm3_attn_pool_struct_align_supcon_meta.json \
  --save-preds predictions_struct_align_saprot10k.csv
```

### Architecture-Matched No-Alignment Baseline

```bash
python code/esm3_gnn_attn_pool_supcon_10k.py \
  --csv-path data/saprot10k/solubility_10000_SaProt_Ready.csv \
  --identity-path data/saprot10k/test_vs_train.m8 \
  --epochs 20 \
  --batch-size 8 \
  --checkpoint-path best_esm3_attn_pool_supcon.pth \
  --meta-path best_esm3_attn_pool_supcon_meta.json \
  --save-preds predictions_no_align_saprot10k.csv
```

### ESM3-Only Baseline

```bash
python code/esm3_open_small_only_10k.py \
  --csv-path data/saprot10k/solubility_10000_SaProt_Ready.csv \
  --identity-path data/saprot10k/test_vs_train.m8 \
  --save-preds predictions_esm3_only_saprot10k.csv
```

### Diagnostic Outputs

Alignment-weight, regularization-probe, and structure-perturbation summaries are included as result files:

```text
diagnostics/alignment_weight_sweep/alignment_weight_sweep_summary.csv
diagnostics/regularization_probe/regularization_probe_saprot10k.csv
diagnostics/structure_noise/
```

## Result Tables and Figures

Paper-facing result tables:

```text
tables/paper_total_results_table.csv
tables/saprot10k_benchmark_summary.csv
tables/netsolp_official_summary.csv
tables/robustness_summary.csv
```

BIBM manuscript figures:

```text
figures/fig1_workflow.png
figures/fig2_saprot10k_results.pdf
figures/fig3_low_homology.pdf
figures/fig4_external_robustness.pdf
figures/fig5_ablation_analysis.pdf
figures/fig6_alignment_weight_sweep.pdf
figures/fig7_paired_bootstrap.pdf
figures/fig8_netsolp_error_analysis.pdf
```

## Manuscript

The current BIBM manuscript files are:

```text
manuscript/bibm_ieee_draft.tex
manuscript/bibm_ieee_draft.pdf
```

Compile with:

```bash
cd manuscript
tectonic bibm_ieee_draft.tex
```

## Citation

If this repository is useful, please cite the associated manuscript after it is available.

## License

Add the final license before public release. For academic code release, MIT or Apache-2.0 are common choices; confirm compatibility with ESM3, ProtSolM, and any downloaded benchmark licenses before publishing all artifacts.
