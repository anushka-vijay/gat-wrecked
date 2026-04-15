***

# GAT vs GATv2: Replication Study
**Paper:** [How Attentive are Graph Attention Networks? (ICLR 2022)](https://arxiv.org/abs/2105.14491)  
**Team Members:** Allen Vinoy (av642), Aryan Patel (ap2568), Anushka Vijay (asv48), Ram Vaidya (rmv42)

## Overview
This repository contains the code for our replication study of the paper *How Attentive are Graph Attention Networks?* The original authors demonstrate that standard Graph Attention Networks (GATs) suffer from **"static attention,"** where the ranking of a node's attention scores toward its neighbors is unconditioned on the query node itself. 

By altering the order of operations in the attention mechanism (applying the non-linearity before the weight matrix), the proposed **GATv2** achieves **"dynamic attention."** 

Our goal is to empirically validate this theoretical claim and demonstrate GATv2's superiority across synthetic, noisy, high-density, and highly inductive real-world datasets.

## The Experiments
We focus on four key experiments designed to explicitly stress-test the expressiveness of dynamic attention:

1. **DictionaryLookup (The Theoretical Proof):** A synthetic benchmark acting as a unit test for dynamic attention. We demonstrate that standard GAT completely fails this query-dependent task, while GATv2 succeeds.
2. **Structural Noise Robustness (`ogbn-arxiv`):** We inject fake edges (up to 100% of the original edge count) to prove GATv2 can dynamically "attend away" from irrelevant, noisy connections better than static baselines.
3. **High-Density Graph Benchmark (`ogbn-proteins`):** Testing on a dense graph (average node degree of 597) where the ability to dynamically filter hundreds of less relevant neighbors is critical to performance.
4. **Program Analysis (VARMISUSE):** Predicting misused variables in Abstract Syntax Tree (AST) graphs. This acts as the ultimate inductive stress test, requiring aggressive semantic conditioning.

## Repository Structure

```text
gatv2-replication/
├── README.md
├── requirements.txt
├── notebooks/                  # Colab notebooks (Execution only)
│   ├── 01_dictionary_lookup.ipynb   
│   ├── 02_structural_noise.ipynb    
│   ├── 03_high_density_ogb.ipynb    
│   └── 04_varmisuse.ipynb           
├── src/                        # Shared logic (.py files)
│   ├── __init__.py
│   ├── models.py               # GAT and GATv2 PyTorch modules
│   ├── data_loaders.py         # PyG NeighborLoaders and custom AST parsing
│   ├── train.py                # Standardized training loops
│   └── utils.py                # Noise injection and W&B logging
└── scripts/                    # Automation
    └── download_varmisuse.sh   # Bash script for fetching JSONlines
```

## Team Collaboration Workflow
We utilize a distributed execution model. **Do not clone this repository into a mounted Google Drive.** ### 1. Data Storage (Shared Drive)
All datasets (OGB, VARMISUSE) and saved `.pth` model weights are stored in our shared Google Drive folder: `GATv2_Data/datasets`, `GATv2_Data/model_weights`.

### 2. Execution (Google Colab)
(Need to figure out branches and stuff).

At the start of your working session in Colab, run the following setup to mount the data and pull the freshest code into temporary storage:

```python
# 1. Mount the shared data
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone the code into Colab's fast local storage
!git clone https://github.com/your-org/gatv2-replication.git
%cd gatv2-replication
!pip install -r requirements.txt

# 3. Enable auto-reloading for local module edits
%load_ext autoreload
%autoreload 2
```

### 3. Experiment Tracking (Optional)
All runs are logged to our team's Weights & Biases (W&B) dashboard. Ensure you authenticate your W&B account before initializing the training loop:
```python
import wandb
wandb.login()
```

***