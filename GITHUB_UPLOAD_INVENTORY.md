# GitHub Upload Inventory for Manuscript Submission

This file lists the materials that should be uploaded or archived for a journal/conference submission. The goal is to make the paper reproducible without pushing unnecessary multi-GB intermediate folders into normal Git history.

## 1. Minimum Upload Set

These files should be uploaded to GitHub.

### 1.1 Core Model Code

```text
gvp.py
esm3_gnn_attn_pool_struct_align_supcon_10k.py
esm3_gnn_attn_pool_supcon_10k.py
esm3_open_small_only_10k.py
audit_protsolm_saprot10k.py
generate_protsolm_features.py
prepare_protsolm_saprot10k.py
netsolp_official_esm1b_10k.py
bootstrap_significance_from_predictions.py
```

Purpose:

- Train and evaluate Struct-Align.
- Run the architecture-matched no-alignment baseline.
- Run sequence-only and ProtSolM-style baselines.
- Generate prediction files and bootstrap tests.

### 1.2 BIBM Submission Package

```text
manuscript/bibm_ieee_draft.tex
manuscript/bibm_ieee_draft.pdf
manuscript/figure_plan.md
figures/
```

Keep the final figures used in the paper. Figure-generation scripts are not included in this GitHub submission package.

### 1.3 Machine-Readable Result Tables

```text
tables/paper_total_results_table.csv
tables/saprot10k_benchmark_summary.csv
tables/netsolp_official_summary.csv
tables/robustness_summary.csv
diagnostics/alignment_weight_sweep/alignment_weight_sweep_summary.csv
diagnostics/regularization_probe/regularization_probe_saprot10k.csv
diagnostics/structure_noise/*/structure_noise_sensitivity_saprot10k.csv
```

Purpose:

- Preserve the exact numbers used in the manuscript.
- Allow reviewers to regenerate figures without rerunning all models.
- Document the alignment-weight and structure-perturbation diagnostics.

### 1.4 Reports and Manifests

```text
manifests/source_artifacts_manifest.csv
manifests/run_dirs_manifest.csv
requirements.txt
README.md
GITHUB_UPLOAD_INVENTORY.md
```

Purpose:

- Show where each result came from.
- Provide reviewer-readable provenance.

## 2. Data Artifacts

### 2.1 Small or Medium Files Suitable for Git

These can usually be uploaded directly, depending on repository size limits and license constraints.

```text
data/saprot10k/solubility_10000_SaProt_Ready.csv
data/saprot10k/test_vs_train.m8
data/splits/cv5_random/
data/splits/cv5_homology/
data/netsolp/netsolp_ready_esmfold_struct_clean.csv
```

Before public release, confirm that the source licenses permit redistribution.

### 2.2 Large Files Better Uploaded Separately

Do not push these through normal Git unless using Git LFS.

```text
pdbs_3000/
pdbs_7000/
SaProt_Solubility_3D/
external_benchmarks/          # full structure archive
protsolm_saprot10k/
protsolm_netsolp_official/
tmp/
```

Recommended handling:

- Put full structure archives in Zenodo, OSF, Figshare, institutional storage, or a GitHub Release.
- Add a download link and checksum to `README.md`.
- Keep only small metadata and result summaries in Git.

## 3. Model Checkpoints

Upload checkpoints only if needed for reviewer inference.

Recommended checkpoint artifacts:

```text
best_esm3_attn_pool_struct_align_supcon.pth
best_esm3_attn_pool_struct_align_supcon_meta.json
best_esm3_attn_pool_supcon.pth
best_esm3_attn_pool_supcon_meta.json
```

Suggested handling:

- Use Git LFS or a GitHub Release.
- Include metadata JSON files in normal Git.
- Provide SHA256 checksums for checkpoint files.

## 4. Results-to-Script Mapping

| Manuscript Component | Data/Output | Reproduction Entry Point |
|---|---|---|
| SaProt10k main table | `tables/paper_total_results_table.csv` | core code in `code/` plus exported result tables |
| ESM3-only baseline | SaProt10k metrics | `code/esm3_open_small_only_10k.py` |
| No-alignment baseline | SaProt10k metrics | `code/esm3_gnn_attn_pool_supcon_10k.py` |
| Struct-Align | SaProt10k metrics | `code/esm3_gnn_attn_pool_struct_align_supcon_10k.py` |
| ProtSolM reproduction | ProtSolM-compatible features and metrics | `code/generate_protsolm_features.py`, `code/prepare_protsolm_saprot10k.py`, `code/audit_protsolm_saprot10k.py` |
| Low-homology analysis | `data/saprot10k/test_vs_train.m8` | core evaluation code |
| NetSolP external evaluation | `data/netsolp/netsolp_ready_esmfold_struct_clean.csv` | core evaluation code |
| Alignment-weight sweep | `diagnostics/alignment_weight_sweep/alignment_weight_sweep_summary.csv` | result table only in this package |
| Structure perturbation | `diagnostics/structure_noise/*/structure_noise_sensitivity_saprot10k.csv` | result table only in this package |
| Figures | `figures/*.pdf`, `figures/*.png` | final figure files only |
| Manuscript PDF | `bibm_ieee_draft.pdf` | `tectonic bibm_ieee_draft.tex` |

## 5. Files to Exclude From Normal Git

Recommended `.gitignore` patterns:

```text
__pycache__/
*.pyc
*.log
*.pid
*.aux
*.out
*.toc
*.bbl
*.blg
*.pth
*.pt
*.tar.gz
tmp/
protsolm_saprot10k/
protsolm_netsolp_official/
pdbs_3000/
pdbs_7000/
SaProt_Solubility_3D/
external_benchmarks/**/*.pdb
external_benchmarks/**/*.pt
```

If a file excluded above is required for review, upload it separately and link it from `README.md`.

## 6. Pre-Submission Checklist

- [ ] Confirm the final manuscript PDF compiles from `bibm_ieee_draft.tex`.
- [ ] Confirm all tables in the manuscript match `paper_total_results_table.csv`.
- [ ] Confirm all figures used in the manuscript are present in `bibm_submission_20260511/figures/`.
- [ ] Confirm data redistribution permissions for SaProt10k and NetSolP-derived files.
- [ ] Upload large structures/checkpoints to an external archive or Git LFS.
- [ ] Add final external download URLs and checksums to `README.md`.
- [ ] Add a license file.
- [ ] Remove or archive draft AI-generated figure attempts if they are not part of the final paper.
- [ ] Verify local absolute paths in core code are documented or patched before public reuse.
