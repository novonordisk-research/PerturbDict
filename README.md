# PerturbDict

A lightweight data class for storing transcriptomics data aggregated by perturbation.

## Overview

PerturbDict provides a simple way to aggregate single-cell transcriptomics data by perturbation, storing average gene expression vectors in a lightweight dictionary format. This approach:

- Aggregates data per perturbation to reduce memory footprint
- Avoids heavy dependencies like scanpy for downstream analysis
- Creates a simple dict structure that can be easily fed into LLMs or other tools
- Includes differential expression analysis capabilities
- Built-in support for k-fold splits and data loaders

## Installation

```bash
git clone https://github.com/novonordisk-research/PerturbDict
cd PerturbDict
uv sync --all-extras  # includes anndata support for loading from AnnData objects
```

## Getting Started

```python
import numpy as np
import anndata as ad
from perturbdict import PerturbDict

# Create toy AnnData with 3 perturbations, 3 cells each, 10 genes
X = np.random.randn(9, 10)
obs = {'perturbation': ['NT', 'NT', 'NT', 'TP53', 'TP53', 'TP53', 'BRCA1', 'BRCA1', 'BRCA1']}
var = {'gene_names': [f'Gene{i}' for i in range(10)]}
adata = ad.AnnData(X=X, obs=obs, var=var)

# Load and aggregate by perturbation
pert_dict = PerturbDict.from_adata(adata, perturbation_col='perturbation', control_name='NT')

# Access data
tp53_expr = pert_dict.get_expression('TP53')
top_genes = pert_dict.get_top_de_genes('TP53', k=5, return_type='names')
all_data = pert_dict.get_observed_dict()  # {pert_name: expr_vector}

# Save/load
pert_dict.save('data.pkl')
pert_dict = PerturbDict.from_cache('data.pkl')
```

## Train/Test Splits

```python
# Simple split
train_data, test_data = pert_dict.get_split_data(k=5, fold=0, seed=42)

# PyTorch DataLoader
from torch.utils.data import DataLoader
train_iter, test_iter = pert_dict.get_split_iterators(k=5, fold=0)
train_loader = DataLoader(train_iter, batch_size=32, shuffle=True)

# Evaluate on top-k DE genes
mask = pert_dict.get_de_mask('TP53', k=20)
mse = ((pred[mask] - observed[mask]) ** 2).mean()
```
