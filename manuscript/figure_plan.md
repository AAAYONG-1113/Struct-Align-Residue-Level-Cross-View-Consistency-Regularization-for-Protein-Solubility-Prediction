# Figure Plan for BIBM Draft

Goal: redraw all figures in a clean IEEE/BIBM style. Use consistent fonts, restrained colors, high contrast, and simple vector-friendly layouts. Generate PDF for LaTeX and PNG for inspection.

## Global Style

- Canvas: white background.
- Font: DejaVu Sans or Arial-like sans-serif for generated figures; IEEE paper will handle body text.
- Color strategy:
  - Sequence baselines: gray.
  - External multimodal baseline: muted purple.
  - In-house graph baseline: blue/green.
  - Proposed Struct-Align: dark red.
  - Robustness/limitations: neutral blue/gray/red.
- Avoid decorative gradients, shadows, and dense text.
- Label each panel with A, B, C when a figure has multiple panels.
- Use 300-450 dpi PNG for review and vector PDF for manuscript.

## Fig. 1 Workflow

Purpose: explain the model and evaluation protocol in one clean diagram.

Recommended panels:

- A. Inputs: sequence, structure graph, surface/physicochemical features.
- B. Encoders: frozen ESM3-open-small, GVP graph branch.
- C. Alignment: residue-level symmetric sequence-structure alignment.
- D. Prediction and evaluation: attention pooling, classifier, validation-fixed threshold, SaProt10k/NetSolP evaluation.

Key text to include inside figure:

- ESM3 residue embeddings
- GVP residue graph
- Residue alignment loss
- SupCon + BCE
- AUC / Accuracy / MCC

Avoid:

- Too many implementation hyperparameters.
- Large paragraph text inside boxes.

## Fig. 2 SaProt10k Main Results

Purpose: show the primary benchmark result.

Recommended layout:

- Three small horizontal bar panels: AUC, Accuracy, MCC.
- Models sorted in a clear order:
  - ESM3-only
  - ProtSolM
  - GVP+SupCon
  - Surface FiLM
  - Struct-Align
- Optionally include other ablations in a smaller secondary figure or appendix.

Main values:

| Model | AUC | Accuracy | MCC |
|---|---:|---:|---:|
| ESM3-only | 0.7205 | 0.6154 | 0.2485 |
| ProtSolM | 0.8291 | 0.7260 | 0.4606 |
| GVP+SupCon | 0.8237 | 0.7115 | 0.4360 |
| Surface FiLM | 0.8175 | 0.7356 | 0.4801 |
| Struct-Align | 0.8565 | 0.7740 | 0.5544 |

Caption message:

Struct-Align gives the strongest single-split SaProt10k result and improves most clearly in MCC.

## Fig. 3 Low-Homology and Ablation

Purpose: support the claim that the model is not only exploiting close homologs and that alignment contributes to the gain.

Recommended layout:

- Panel A: less-than-30% identity AUC/Accuracy/MCC for ESM3-only, ProtSolM, GVP+SupCon, Struct-Align.
- Panel B: ablation ladder line plot or grouped bars from ESM3-only to SaProt-GVP to GVP+SupCon to Struct-Align.

Low-homology values:

| Model | <30% AUC | <30% Accuracy | <30% MCC |
|---|---:|---:|---:|
| ESM3-only | 0.7471 | 0.6099 | 0.2441 |
| ProtSolM | 0.7963 | 0.7092 | 0.4261 |
| GVP+SupCon | 0.8048 | 0.6950 | 0.4100 |
| Struct-Align | 0.8326 | 0.7589 | 0.5390 |

Ablation values:

| Model | AUC | Accuracy | MCC |
|---|---:|---:|---:|
| ESM3-only | 0.7205 | 0.6154 | 0.2485 |
| SaProt-GVP | 0.8316 | 0.6875 | 0.4001 |
| GVP+SupCon | 0.8237 | 0.7115 | 0.4360 |
| Struct-Align | 0.8565 | 0.7740 | 0.5544 |

Caption message:

The less-than-30% subset and the architecture-matched ablation both support the value of explicit residue-level alignment.

## Fig. 4 External NetSolP and Robustness

Purpose: show external validation honestly and avoid overclaiming.

Recommended layout:

- Panel A: NetSolP official protocol bar plot for AUC/Accuracy/MCC.
- Panel B: robustness summary for main split, same-split 3 seeds, random CV5.

NetSolP values:

| Model | AUC | Accuracy | MCC |
|---|---:|---:|---:|
| Official-like ESM1b | 0.6305 | 0.5817 | 0.2358 |
| ESM3-only | 0.7265 | 0.6931 | 0.2845 |
| ProtSolM | 0.7653 | 0.7211 | 0.3643 |
| GVP+SupCon | 0.7830 | 0.7256 | 0.3961 |
| Struct-Align + ESMFold | 0.7677 | 0.7226 | 0.3680 |

Robustness values:

| Setting | AUC | Accuracy | MCC |
|---|---:|---:|---:|
| Main single split | 0.8565 | 0.7740 | 0.5544 |
| Same split, 3 seeds mean | 0.8227 | 0.7147 | 0.4430 |
| Random CV5 mean | 0.8107 | 0.7275 | 0.4379 |

Caption message:

Struct-Align remains competitive externally, but GVP+SupCon is strongest on NetSolP; robustness estimates are lower than the headline split.

## Drawing Order

1. Fig. 1 Workflow.
2. Fig. 2 SaProt10k main result.
3. Fig. 3 Low-homology and ablation.
4. Fig. 4 External NetSolP and robustness.

Review process:

- Generate one figure.
- Inspect PNG/PDF.
- Revise layout and text.
- Move to the next figure only after approval.
