# Group Project 2-Page Summary Report (Draft)

This draft is designed to map directly to the required project-report format and then be trimmed/formatted into a 2-page PDF.

- **Project title:** Re-implementation Study of GAT vs GATv2
- **Primary paper:** Brody, Alon, and Yahav, *How Attentive are Graph Attention Networks?* (ICLR 2022)
- **Team:** Allen Vinoy, Aryan Patel, Anushka Vijay, Ram Vaidya

---

## 1. Introduction

Graph Attention Networks (GATs) are widely used for node- and graph-level prediction, but the GATv2 paper argues that standard GAT has a key expressive limitation: attention ranking is effectively query-independent (static). The paper proposes GATv2, which changes the ordering of linear transformation and attention scoring so that ranking can depend on the query node (dynamic attention).

Our project reproduces this core mechanistic claim and tests whether the claimed advantage translates to downstream tasks. We focus on both:

- **Controlled synthetic tasks** where attention behavior is easy to inspect (dictionary lookup, structural noise), and
- **Real benchmarks** where practical performance and optimization dynamics matter (Cora, VarMisuse, QM9, OGB).

The main contribution of our project is a unified re-implementation with iterative experiments in notebooks and shared model code in `src/models.py`, including custom sparse edge-aware GAT/GATv2 variants for molecular and OGB settings.

---

## 2. Chosen Result

### Target result from the paper

We chose to reproduce the paper's central static-vs-dynamic attention phenomenon (Figure 1 style behavior) and then evaluate whether that mechanistic difference produces practical gains in broader datasets.

- **Paper references used:** Figure 1 and Eq. (6)/(7) comparison of GAT vs GATv2 attention scoring.
- **Why this was chosen:** It is the core claim of the paper. If this is not reproduced, later benchmark comparisons are less meaningful.

### Why this matters for our goals

Our project goal was not only to "match one metric" but to test an explanation pipeline:

1. Confirm the attention-mechanism claim in a controlled setting.
2. Check if gains hold under noise and higher-density graphs.
3. Evaluate practical sensitivity to hyperparameters, depth, residuals, normalization, and edge features.

This gives a more complete replication story than only reporting one final table row.

![Figure 1a: Dictionary lookup attention (GAT)](results/notebooks__01_dictionary_lookup__cell007__out00.png)
*Figure 1a. Static-style attention visualization from our dictionary lookup run.*

![Figure 1b: Dictionary lookup attention (GATv2)](results/notebooks__01_dictionary_lookup__cell007__out01.png)
*Figure 1b. Dynamic attention visualization from our dictionary lookup run.*

---

## 3. Methodology

### 3.1 Re-implementation setup

We used a notebook-first workflow (`notebooks/01` to `06`) with shared reusable code in `src/models.py`.

- Dense baseline models for synthetic and Cora settings.
- Sparse custom QM9/OGB models supporting `edge_index` and optional `edge_attr`.
- Separate model builders for task-specific configurations (`gat_qm9`, `gatv2_qm9`, `gat_ogb`, `gatv2_ogb`).

### 3.2 Datasets and tasks

- **Synthetic dictionary lookup** (`notebooks/01_dictionary_lookup.ipynb`): tests static vs dynamic attention directly.
- **Structural noise synthetic test** (`notebooks/02_structural_noise.ipynb`): increasing topological noise.
- **Cora baseline** (`notebooks/05_gat_cora_baseline.ipynb`): node classification sanity check.
- **VarMisuse** (`notebooks/04_varmisuse.ipynb`): code-graph variable-slot prediction (accuracy).
- **QM9** (`notebooks/06_chemical_qm9.ipynb`): multi-target molecular regression.
- **OGB high-density tasks** (`notebooks/03_high_density_ogb.ipynb`): `ogbg-molhiv` / `ogbg-ppa`.

### 3.3 Metrics

- **Classification tasks:** accuracy or ROC-AUC (dataset standard metric).
- **QM9 regression:** mean absolute error (MAE), including per-target MAE.
- **Training diagnostics:** validation/test curves and early-stopping checkpoints.

### 3.4 Key modifications from original paper setup

We intentionally made practical adaptations to keep experiments feasible and interpretable:

- Added **target normalization** for QM9 multi-target regression.
- Added **validation-based early stopping** for QM9 and OGB runs.
- Used a **2k QM9 subset** in some runs for compute feasibility.
- Added **separate GATv2 hyperparameter tuning** (lower LR, larger hidden size/depth, residual + LayerNorm options in OGB variant).
- Implemented custom sparse GAT/GATv2 layers in `src/models.py` to reduce dependency on black-box operators and keep architecture control explicit.

### 3.5 Implementation obstacles and engineering choices

- PyTorch 2.6+ changed `torch.load` defaults (`weights_only=True`), which broke OGB/PyG object loading until safe globals were explicitly added.
- OGB dataset feature dtypes required explicit casting to float to avoid linear-layer dtype mismatch.
- Dynamic-attention models in dense-style implementations had higher memory/time cost; sparse formulations and adjusted configs were necessary for stable runs.

---

## 4. Results and Analysis

### 4.1 Core mechanistic reproduction (dictionary lookup)

In `notebooks/01_dictionary_lookup.ipynb`, we observe behavior consistent with the paper's main claim:

- **GAT final accuracy:** ~10.00%
- **GATv2 training reached:** 100% (reported at epoch 900 in run output)

Interpretation: static attention struggles to adapt ranking by query in this setup, while dynamic attention can represent the required mapping.

| Model | Metric | Value |
|---|---|---:|
| GAT | Final accuracy | 10.00% |
| GATv2 | Reached 100% train accuracy | Epoch 900 |

### 4.2 Structural noise behavior

In `notebooks/02_structural_noise.ipynb`, both models degrade as random graph noise increases, but GATv2 becomes more robust at higher noise:

- Noise 20%: GAT 0.61 vs GATv2 0.68
- Noise 30%: GAT 0.50 vs GATv2 0.63
- Noise 50%: GAT 0.40 vs GATv2 0.56

Interpretation: dynamic attention appears more resilient when graph neighborhoods include irrelevant edges.

| Noise level | GAT accuracy | GATv2 accuracy |
|---:|---:|---:|
| 20% | 0.61 | 0.68 |
| 30% | 0.50 | 0.63 |
| 50% | 0.40 | 0.56 |

![Figure 2: Structural noise trend](results/notebooks__02_structural_noise__cell011__out01.png)
*Figure 2. Accuracy under increasing topological noise.*

### 4.3 QM9 (multi-target regression)

From `notebooks/06_chemical_qm9.ipynb` output:

- **Best mean Val/Test MAE**
  - GAT: 1.4542 / 1.5243
  - GATv2: 1.3885 / 1.4827
- **Per-target test MAE**
  - mu: GAT 0.9544 vs GATv2 0.9297
  - alpha: GAT 3.3101 vs GATv2 3.2381
  - HOMO: GAT 0.3083 vs GATv2 0.2802

Interpretation: in this run, tuned GATv2 improves over GAT on all selected targets, but margins are moderate and depend on subset size and training budget.

| QM9 metric | GAT | GATv2 |
|---|---:|---:|
| Best mean val MAE | 1.4542 | 1.3885 |
| Best mean test MAE | 1.5243 | 1.4827 |
| Test MAE (mu) | 0.9544 | 0.9297 |
| Test MAE (alpha) | 3.3101 | 3.2381 |
| Test MAE (HOMO) | 0.3083 | 0.2802 |

![Figure 3: QM9 per-target learning curves](results/notebooks__06_chemical_qm9__cell013__out00.png)
*Figure 3. Per-target validation/test MAE trajectories for GAT vs GATv2.*

### 4.4 OGB (`ogbg-molhiv` run shown)

From `notebooks/03_high_density_ogb.ipynb` output:

- GAT best val ROC-AUC: 0.7052, test@best-val: 0.6962
- GATv2 best val ROC-AUC: 0.6705, test@best-val: 0.6890

Interpretation: this specific run does not show a clear GATv2 win. This is still informative and consistent with our broader finding that dynamic attention advantage is task- and optimization-sensitive rather than universal in any single run.

| OGB molhiv metric | GAT | GATv2 |
|---|---:|---:|
| Best val ROC-AUC | 0.7052 | 0.6705 |
| Test ROC-AUC @ best val | 0.6962 | 0.6890 |

![Figure 4: OGB training curves](results/notebooks__03_high_density_ogb__cell006__out00.png)
*Figure 4. Validation/test metric curves and loss trajectories for OGB run.*

### 4.5 Additional benchmark signal (VarMisuse / Cora)

- **VarMisuse:** GAT test acc 0.5972, GATv2 test acc 0.6071 (small but consistent gain).
- **Cora baseline:** one run reports test accuracy around 0.79 (used primarily as sanity baseline and protocol check).

### 4.6 Discrepancies vs paper and why they are reasonable

Main reasons we may not exactly match paper-level aggregate numbers:

- smaller subsets / fewer runs in some experiments,
- single-seed or low-seed comparisons in exploratory phases,
- implementation details omitted/implicit in paper,
- different hardware/runtime limits requiring adjusted architecture depth and batch settings.

Given the project objective, our evidence is still strong: we reproduced the mechanism and explored practical boundary conditions with multiple iterative interventions.

---

## 5. Reflections

### Lessons learned

1. **Mechanistic reproduction is easier than benchmark domination.**  
   We could clearly reproduce static-vs-dynamic attention behavior on synthetic tasks, but real benchmarks required substantial tuning and did not always favor GATv2 by default.

2. **Optimization details are first-order effects.**  
   Learning rate, depth, normalization, residual connections, early stopping, and target normalization materially changed outcomes.

3. **Engineering robustness matters in reproducibility.**  
   Library-version issues (e.g., PyTorch load safety changes), dtype mismatches, and memory constraints can dominate project time if not handled explicitly.

4. **Evaluation should be multi-view, not one-number.**  
   Per-target curves (QM9), val-vs-test trajectories, and synthetic diagnostics gave better insight than only final scalar metrics.

### Future directions

- run larger multi-seed sweeps and report confidence intervals systematically,
- evaluate more QM9 targets jointly with calibrated normalization schemes,
- expand OGB experiments with stronger regularization/search protocols,
- add ablations on edge features, residual design, and layer normalization placement.

---

## 6. References

1. Brody, S., Alon, U., & Yahav, E. (2022). *How Attentive are Graph Attention Networks?* ICLR. [https://arxiv.org/abs/2105.14491](https://arxiv.org/abs/2105.14491)  
2. Open Graph Benchmark (OGB): [https://ogb.stanford.edu](https://ogb.stanford.edu)  
3. PyTorch Geometric documentation: [https://pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)

---

## Optional extra visuals (if space remains in PDF)

- `results/notebooks__04_varmisuse__cell026__out00.png` (VarMisuse training curves)
- `results/notebooks__05_gat_cora_baseline__cell013__out01.png` (Cora comparison figure)
- `results/notebooks__05_gat_cora_baseline__cell015__out01.png` (additional Cora summary)

---

