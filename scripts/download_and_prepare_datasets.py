#!/usr/bin/env python3
"""
Download and prepare Replogle and Nadig perturbation datasets.

This script downloads 4 perturbation datasets:
- Replogle K562 essential genes
- Replogle RPE1 essential genes
- Nadig HepG2
- Nadig Jurkat

It then preprocesses them and creates a MultiContextPerturbDict object.

Usage:
    python download_and_prepare_datasets.py
"""

import os
import requests
from zipfile import ZipFile
import scanpy as sc
from perturbdict import MultiContextPerturbDict
from perturbdict.utils import filter_perturbations_by_de_count


# ============================================================================
# Configuration - Edit these as needed
# ============================================================================

# Directory to store downloaded and processed datasets
DATA_DIR = './data'

# Filtering parameters
MIN_DE_GENES = 20
PADJ_THRESHOLD = 0.05

# Skip steps if already completed
SKIP_DOWNLOAD = False
SKIP_PREPROCESS = False

# Dataset URLs and configuration
DATASETS = {
    'replogle_k562_essential': {
        'type': 'zip',
        'url': 'https://dataverse.harvard.edu/api/access/datafile/7458695',
        'raw_filename': 'perturb_processed.h5ad',
    },
    'replogle_rpe1_essential': {
        'type': 'zip',
        'url': 'https://dataverse.harvard.edu/api/access/datafile/7458694',
        'raw_filename': 'perturb_processed.h5ad',
    },
    'nadig_hepg2': {
        'type': 'h5ad',
        'url': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fhepg2%5Fraw%5Fsinglecell%5F01%2Eh5ad',
        'raw_filename': 'raw.h5ad',
        'cell_type': 'hepg2',
    },
    'nadig_jurkat': {
        'type': 'h5ad',
        'url': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fjurkat%5Fraw%5Fsinglecell%5F01%2Eh5ad',
        'raw_filename': 'raw.h5ad',
        'cell_type': 'jurkat',
    },
}


# ============================================================================
# Download Functions
# ============================================================================

def download_file(url, output_path):
    """Download file from URL."""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to {output_path}")


def download_zip_dataset(dataset_name, url, data_dir):
    """Download and extract zip dataset."""
    dataset_path = os.path.join(data_dir, dataset_name)
    
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f"{dataset_name}.zip")
    
    download_file(url, zip_path)
    
    print(f"Extracting {zip_path}...")
    with ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall(path=data_dir)
    
    os.remove(zip_path)
    print(f"Extracted to {dataset_path}")
    return dataset_path


def download_h5ad_dataset(dataset_name, url, data_dir, filename='raw.h5ad'):
    """Download h5ad file directly into a dataset folder."""
    dataset_path = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    
    h5ad_path = os.path.join(dataset_path, filename)
    
    if os.path.exists(h5ad_path):
        print(f"Dataset already exists at {h5ad_path}")
        return h5ad_path
    
    download_file(url, h5ad_path)
    return h5ad_path


# ============================================================================
# Preprocessing Functions
# ============================================================================

def postprocess_replogle(adata):
    """
    Postprocess Replogle datasets (K562, RPE1).
    
    - Select 1000 highly variable genes (note: adata.X is already log-normalized)
    - Clean up obs columns to match standard format
    
    Args:
        adata: AnnData object
        
    Returns:
        Processed AnnData object
    """
    # Choose 1000 highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor='seurat', subset=True)
    
    # Clean up obs columns
    adata.obs['perturbation'] = adata.obs['condition'].str.replace('+ctrl', '')
    adata.obs = adata.obs[['cell_type', 'perturbation']]
    
    return adata


def postprocess_nadig(adata, cell_type):
    """
    Postprocess Nadig datasets (HepG2, Jurkat).
    
    - Select 1000 highly variable genes
    - Normalize and log-transform
    - Standardize metadata format
    
    Args:
        adata: AnnData object
        cell_type: Cell type name (e.g., 'hepg2', 'jurkat')
        
    Returns:
        Processed AnnData object
    """
    # Choose 1000 highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor='seurat_v3', subset=True)
    
    # Normalize and log1p transform
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    # Add cell_type column
    adata.obs['cell_type'] = cell_type
    
    # Rename 'gene' column to 'perturbation'
    adata.obs['perturbation'] = adata.obs['gene']
    
    # Replace perturbation label "non-targeting" with "ctrl"
    adata.obs['perturbation'] = adata.obs['perturbation'].replace('non-targeting', 'ctrl')
    
    # Keep cell_type and perturbation columns to match replogle format
    adata.obs = adata.obs[['cell_type', 'perturbation']]
    
    return adata


# ============================================================================
# Main Pipeline
# ============================================================================

def download_all_datasets(data_dir):
    """Download all datasets."""
    print("=" * 80)
    print("STEP 1: Downloading datasets")
    print("=" * 80)
    
    paths = {}
    
    for dataset_name, config in DATASETS.items():
        print(f"\n--- {dataset_name} ---")
        if config['type'] == 'zip':
            paths[dataset_name] = download_zip_dataset(
                dataset_name,
                config['url'],
                data_dir
            )
        elif config['type'] == 'h5ad':
            paths[dataset_name] = download_h5ad_dataset(
                dataset_name,
                config['url'],
                data_dir,
                config['raw_filename']
            )
    
    return paths


def preprocess_all_datasets(data_dir):
    """Preprocess all downloaded datasets."""
    print("\n" + "=" * 80)
    print("STEP 2: Preprocessing datasets")
    print("=" * 80)
    
    # Process Replogle K562
    print("\n--- Processing replogle_k562_essential ---")
    replogle_k562 = sc.read_h5ad(
        os.path.join(data_dir, 'replogle_k562_essential', 'perturb_processed.h5ad')
    )
    replogle_k562 = postprocess_replogle(replogle_k562)
    output_path = os.path.join(data_dir, 'replogle_k562_essential', 'final.h5ad')
    replogle_k562.write(output_path)
    print(f"Saved to {output_path}")
    
    # Process Replogle RPE1
    print("\n--- Processing replogle_rpe1_essential ---")
    replogle_rpe1 = sc.read_h5ad(
        os.path.join(data_dir, 'replogle_rpe1_essential', 'perturb_processed.h5ad')
    )
    replogle_rpe1 = postprocess_replogle(replogle_rpe1)
    output_path = os.path.join(data_dir, 'replogle_rpe1_essential', 'final.h5ad')
    replogle_rpe1.write(output_path)
    print(f"Saved to {output_path}")
    
    # Process Nadig HepG2
    print("\n--- Processing nadig_hepg2 ---")
    nadig_hepg2 = sc.read_h5ad(
        os.path.join(data_dir, 'nadig_hepg2', 'raw.h5ad')
    )
    nadig_hepg2 = postprocess_nadig(nadig_hepg2, cell_type='hepg2')
    output_path = os.path.join(data_dir, 'nadig_hepg2', 'final.h5ad')
    nadig_hepg2.write(output_path)
    print(f"Saved to {output_path}")
    
    # Process Nadig Jurkat
    print("\n--- Processing nadig_jurkat ---")
    nadig_jurkat = sc.read_h5ad(
        os.path.join(data_dir, 'nadig_jurkat', 'raw.h5ad')
    )
    nadig_jurkat = postprocess_nadig(nadig_jurkat, cell_type='jurkat')
    output_path = os.path.join(data_dir, 'nadig_jurkat', 'final.h5ad')
    nadig_jurkat.write(output_path)
    print(f"Saved to {output_path}")


def create_perturbdict(data_dir, min_de_genes=20, padj_threshold=0.05):
    """Load preprocessed datasets and create MultiContextPerturbDict."""
    print("\n" + "=" * 80)
    print("STEP 3: Creating MultiContextPerturbDict")
    print("=" * 80)
    
    # Load each dataset
    print("\nLoading preprocessed datasets...")
    adata_dict = {
        'k562': sc.read_h5ad(
            os.path.join(data_dir, 'replogle_k562_essential', 'final.h5ad')
        ),
        'rpe1': sc.read_h5ad(
            os.path.join(data_dir, 'replogle_rpe1_essential', 'final.h5ad')
        ),
        'hepg2': sc.read_h5ad(
            os.path.join(data_dir, 'nadig_hepg2', 'final.h5ad')
        ),
        'jurkat': sc.read_h5ad(
            os.path.join(data_dir, 'nadig_jurkat', 'final.h5ad')
        ),
    }
    
    # Filter perturbations by DE count
    print(f"\nFiltering perturbations (min_de_genes={min_de_genes}, "
          f"padj_threshold={padj_threshold})...")
    for dataset_key in adata_dict:
        print(f"  - Filtering {dataset_key}...")
        adata = adata_dict[dataset_key]
        adata_filtered = filter_perturbations_by_de_count(
            adata,
            min_de_genes=min_de_genes,
            padj_threshold=padj_threshold,
            control_name="ctrl"
        )
        adata_dict[dataset_key] = adata_filtered
    
    # Create MultiContextPerturbDict
    print("\nCreating MultiContextPerturbDict...")
    pert_dict = MultiContextPerturbDict.from_adata_dict(
        adata_dict,
        control_name="ctrl"
    )
    
    # Display statistics
    print("\n--- Perturbation Availability ---")
    availability_df = pert_dict.get_perturbation_availability()
    print(availability_df)
    
    print("\n--- Overlap Statistics ---")
    overlap_stats = pert_dict.get_overlap_stats()
    print(overlap_stats)
    
    # Save
    output_path = os.path.join(data_dir, 'perturbdict_replogle_nadig.pkl')
    pert_dict.save(output_path)
    print(f"\nSaved MultiContextPerturbDict to {output_path}")
    
    return pert_dict


def main():
    """Main execution function."""
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Using data directory: {DATA_DIR}")
    
    # Run pipeline
    if not SKIP_DOWNLOAD:
        download_all_datasets(DATA_DIR)
    
    if not SKIP_PREPROCESS:
        preprocess_all_datasets(DATA_DIR)
    
    create_perturbdict(
        DATA_DIR,
        min_de_genes=MIN_DE_GENES,
        padj_threshold=PADJ_THRESHOLD
    )
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
