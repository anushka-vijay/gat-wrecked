# GAT vs GATv2: Replication Study

**Paper:** [How Attentive are Graph Attention Networks? (ICLR 2022)](https://arxiv.org/abs/2105.14491)  
**Team Members:** Allen Vinoy (av642), Aryan Patel (ap2568), Anushka Vijay (asv48), Ram Vaidya (rmv42)

## 1. Introduction

This repo is our project re-implementation of Brody et al. (2022), testing when GATv2’s dynamic attention helps over standard GAT across synthetic and real graph tasks.  
The paper’s core contribution is that standard GAT attention ranking is query-independent (static), while GATv2 changes operation order to support more expressive query-dependent attention.

## 2. Chosen Result

We primarily target the paper’s static-vs-dynamic claim shown in **Figure 1** (dictionary-style attention behavior) and extend comparison to additional benchmarks (QM9, OGB graph property tasks).  
Paper reference points used in this repo: **Eq. (6)/(7)** for GAT vs GATv2 scoring and benchmark tables for QM9/OGB-style evaluation.

## 3. GitHub Contents

```text
gat-wrecked/
├── data/
├── notebooks/
│   ├── dataset/
│   ├── 01_dictionary_lookup.ipynb
│   ├── 02_structural_noise.ipynb
│   ├── 03_high_density_ogb.ipynb
│   ├── 04_varmisuse.ipynb
│   ├── 05_gat_cora_baseline.ipynb
│   └── 06_chemical_qm9.ipynb
├── paper/
│   └── 2105.14491v3.pdf
├── poster/
├── report/
│   └── gat_wrecked.pdf
├── results/
│   └── notebooks__*.png
├── scripts/
├── src/
│   ├── __init__.py
│   ├── data_loaders.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 4. Re-implementation Details

We use notebook-driven experiments with shared model code in `src/models.py`: dense GAT/GATv2, sparse QM9 variants, and sparse OGB graph-level variants (`OGB_GAT_Model`, `OGB_GATv2_Model`).  
Datasets include synthetic dictionary/noise setups, Cora, QM9 multi-target regression, OGB graph classification (`ogbg-molhiv`), and VarMisuse; metrics follow dataset conventions (e.g., ROC-AUC, Accuracy, MAE).

**Experiment map (current repo state):**

- `01_dictionary_lookup.ipynb`: synthetic dictionary lookup unit-test for static vs dynamic attention (paper Figure 1 style behavior).
- `02_structural_noise.ipynb`: synthetic structural-noise robustness sweeps on generated graph topologies.
- `03_high_density_ogb.ipynb`: OGB graph classification (`ogbg-molhiv`) with val/test metric tracking.
- `04_varmisuse.ipynb`: VarMisuse program-analysis task on graph-structured code data.
- `05_gat_cora_baseline.ipynb`: Cora baseline comparison for sanity/teaching reference.
- `06_chemical_qm9.ipynb`: QM9 multi-target regression with target normalization, early stopping, and per-target curves.

## 5. Reproduction Steps

1. Create environment and install required libs (PyTorch, PyG, OGB, NumPy, Matplotlib).  
2. Open desired notebook in `notebooks/` and run top-to-bottom (each notebook has its own config block).  
3. For OGB notebook (`03_high_density_ogb.ipynb`), set `DATASET_NAME` to `ogbg-molhiv`; for QM9 (`06_chemical_qm9.ipynb`), set target IDs and model configs.

Example setup (local):

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install torch-geometric ogb numpy matplotlib
```

Compute guidance: CPU runs are possible for small settings, but GPU is strongly recommended for OGB/QM9 full runs.

## 6. Results / Insights

Expected outcome: synthetic dictionary experiments reproduce static-vs-dynamic attention behavior, while real-task gains are dataset- and tuning-dependent (GATv2 is not guaranteed to win on every single run).  
Current notebooks include per-target and per-epoch visual diagnostics (e.g., QM9 labeled target curves; OGB val/test metric curves) to inspect where gains come from.

## 7. Conclusion

Our re-implementation supports the paper’s main mechanistic claim in controlled settings and provides practical benchmark extensions with custom model variants.  
A key lesson is that dynamic attention benefits are sensitive to task structure, model capacity, and optimization choices.

## 8. References

1. Brody, S., Alon, U., & Yahav, E. (2022). *How Attentive are Graph Attention Networks?* ICLR. [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)  
2. OGB benchmark docs and datasets: [https://ogb.stanford.edu](https://ogb.stanford.edu)  
3. PyTorch Geometric docs: [https://pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)

## 9. Acknowledgements

This project was developed as coursework replication/re-implementation effort and reflects iterative peer collaboration across notebooks and shared model code.  
We acknowledge the original authors for open-sourcing the core ideas and benchmark framing that made this reproduction possible.
