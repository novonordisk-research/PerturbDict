from typing import Dict, List, Union, Optional, Callable
import numpy as np
import pandas as pd
from scipy import sparse


def _get_data_dict(pert_dict: Union[Dict, 'PerturbDict']) -> Dict:
    """Extract the underlying dictionary from either a PerturbDict object or raw dict."""
    if hasattr(pert_dict, '_data'):
        return pert_dict._data
    return pert_dict


def validate_perturbation_dict(pert_dict: Union[Dict, 'PerturbDict'], perturbation: str, require_de: bool = True) -> Dict:
    """Validate and return perturbation data. Raises ValueError if not found or missing DE info."""
    data_dict = _get_data_dict(pert_dict)
    if perturbation not in data_dict['perturbations']:
        raise ValueError(f"Perturbation '{perturbation}' not found in dictionary")

    pert_data = data_dict['perturbations'][perturbation]

    if require_de and 'de_ranked' not in pert_data:
        raise ValueError(f"No DE information available for perturbation '{perturbation}'")

    return pert_data


def get_gene_names_from_indices(pert_dict: Union[Dict, 'PerturbDict'], indices: List[int]) -> List[str]:
    """Convert gene indices to gene names."""
    data_dict = _get_data_dict(pert_dict)
    gene_names = data_dict['gene_names']
    return [gene_names[i] for i in indices]


def get_expression_values(pert_data: Dict, indices: List[int]) -> np.ndarray:
    """Get expression values for specified gene indices."""
    exprs = pert_data['exprs']
    if isinstance(exprs, np.ndarray):
        return exprs[indices]
    else:
        return np.array(exprs)[indices]


def format_de_results(
    pert_dict: Union[Dict, 'PerturbDict'],
    pert_data: Dict,
    indices: List[int],
    pvals: List[float],
    effect_sizes: List[float],
    padj: List[float],
    return_type: str = 'indices'
) -> Union[List[int], List[str], Dict]:
    """Format DE results based on return_type: 'indices', 'names', 'stats', or 'all'."""
    if return_type == 'indices':
        return indices
    elif return_type == 'names':
        return get_gene_names_from_indices(pert_dict, indices)
    elif return_type == 'stats':
        return {
            'indices': indices,
            'names': get_gene_names_from_indices(pert_dict, indices),
            'pvalues': pvals,
            'effect_sizes': effect_sizes,
            'padj': padj
        }
    elif return_type == 'all':
        return {
            'indices': indices,
            'names': get_gene_names_from_indices(pert_dict, indices),
            'pvalues': pvals,
            'effect_sizes': effect_sizes,
            'padj': padj,
            'expressions': get_expression_values(pert_data, indices)
        }
    else:
        raise ValueError(f"return_type must be one of: ['indices', 'names', 'stats', 'all']")


def create_de_ranked_structure(de_results: pd.DataFrame, gene_names: List[str]) -> Dict:
    """Convert DE results DataFrame to ranked dict structure."""
    return {
        "gene_indices": [gene_names.index(gene) for gene in de_results['gene'].tolist()],
        "pvalues": de_results['pval'].tolist(),
        "effect_sizes": de_results['effect_size'].tolist(),
        "padj": de_results['padj'].tolist()
    }


def apply_thresholds(de_ranked: Dict, pval_threshold: float = 0.05,
                    padj_threshold: Optional[float] = None,
                    effect_size_threshold: Optional[float] = None) -> np.ndarray:
    """Apply statistical thresholds and return boolean mask for significant genes."""
    pvals = np.array(de_ranked['pvalues'])
    padj = np.array(de_ranked['padj'])
    effect_sizes = np.array(de_ranked['effect_sizes'])

    # P-value or adjusted p-value threshold
    if padj_threshold is not None:
        pval_mask = padj <= padj_threshold
    else:
        pval_mask = pvals <= pval_threshold

    # Effect size threshold
    if effect_size_threshold is not None:
        effect_mask = np.abs(effect_sizes) >= effect_size_threshold
        final_mask = pval_mask & effect_mask
    else:
        final_mask = pval_mask

    return final_mask


def summarize_de_genes_from_adata(
    adata,
    perturbation_col: str = "perturbation",
    control_name: str = "ctrl",
    pval_threshold: float = 0.05,
    padj_threshold: Optional[float] = None,
    effect_size_threshold: Optional[float] = None,
    de_method: str = "ttest",
    n_jobs: Optional[int] = None,
    return_dataframe: bool = True
) -> Union[Dict, pd.DataFrame]:
    """
    Create a summary of DE gene counts directly from an AnnData object.
    
    Performs differential expression analysis on-the-fly for all perturbations
    vs control and summarizes the number of significant genes.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with perturbation information in .obs
    perturbation_col : str, default="perturbation"
        Column name in adata.obs containing perturbation labels
    control_name : str, default="NT"
        Name of the control condition in perturbation_col
    pval_threshold : float, default=0.05
        P-value threshold for significance
    padj_threshold : float, optional
        Adjusted p-value threshold (overrides pval_threshold if provided)
    effect_size_threshold : float, optional
        Minimum absolute effect size threshold
    de_method : str, default="ttest"
        Method for differential expression ("ttest" or "mannwhitneyu")
    n_jobs : int, optional
        Number of parallel jobs (only for mannwhitneyu)
    return_dataframe : bool, default=True
        If True, return results as DataFrame; if False, return as dict
        
    Returns
    -------
    Union[Dict, pd.DataFrame]
        Summary with columns/keys:
        - perturbation: name of the perturbation
        - n_cells_pert: number of cells in perturbation group
        - n_cells_control: number of cells in control group
        - n_de_genes: number of significant DE genes
        - n_upregulated: number of upregulated genes
        - n_downregulated: number of downregulated genes
        
    Examples
    --------
    >>> from perturbdict.utils import summarize_de_genes_from_adata
    >>> summary = summarize_de_genes_from_adata(adata, padj_threshold=0.05)
    >>> print(summary.sort_values('n_de_genes', ascending=False))
    """
    from .diff_exp_analysis import find_de_genes_by_perturbation
    
    # Run DE analysis
    _, de_results_dict = find_de_genes_by_perturbation(
        adata,
        perturbation_col=perturbation_col,
        control_name=control_name,
        n_top_genes=[],  # We don't need top genes, just full results
        de_method=de_method,
        n_jobs=n_jobs,
        show_progress=False
    )
    
    # Count cells
    pert_counts = adata.obs[perturbation_col].value_counts()
    n_control = pert_counts.get(control_name, 0)
    
    # Summarize each perturbation
    summary = []
    for pert_name, results_df in de_results_dict.items():
        # Apply thresholds
        if padj_threshold is not None:
            sig_mask = results_df['padj'] <= padj_threshold
        else:
            sig_mask = results_df['pval'] <= pval_threshold
        
        if effect_size_threshold is not None:
            sig_mask = sig_mask & (np.abs(results_df['effect_size']) >= effect_size_threshold)
        
        sig_genes = results_df[sig_mask]
        n_de_genes = len(sig_genes)
        n_upregulated = (sig_genes['effect_size'] > 0).sum()
        n_downregulated = (sig_genes['effect_size'] < 0).sum()
        
        summary.append({
            'perturbation': pert_name,
            'n_cells_pert': int(pert_counts.get(pert_name, 0)),
            'n_cells_control': int(n_control),
            'n_de_genes': int(n_de_genes),
            'n_upregulated': int(n_upregulated),
            'n_downregulated': int(n_downregulated)
        })
    
    if return_dataframe:
        df = pd.DataFrame(summary)
        # Sort by number of DE genes descending
        return df.sort_values('n_de_genes', ascending=False).reset_index(drop=True)
    else:
        return {item['perturbation']: {k: v for k, v in item.items() if k != 'perturbation'} 
                for item in summary}


def filter_perturbations_by_de_count(
    adata,
    min_de_genes: int,
    perturbation_col: str = "perturbation",
    control_name: str = "ctrl",
    padj_threshold: Optional[float] = 0.05,
    pval_threshold: Optional[float] = None,
    effect_size_threshold: Optional[float] = None,
    de_method: str = "ttest",
    n_jobs: Optional[int] = None,
    keep_control: bool = True,
    return_adata: bool = True
) -> Union[List[str], 'AnnData']:
    """
    Filter perturbations based on number of DE genes.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with perturbation information
    min_de_genes : int
        Minimum number of DE genes required to keep a perturbation
    perturbation_col : str, default="perturbation"
        Column name in adata.obs containing perturbation labels
    control_name : str, default="NT"
        Name of the control condition
    padj_threshold : float, optional, default=0.05
        Adjusted p-value threshold for significance
    pval_threshold : float, optional
        P-value threshold (used if padj_threshold is None)
    effect_size_threshold : float, optional
        Minimum absolute effect size threshold
    de_method : str, default="ttest"
        Method for differential expression
    n_jobs : int, optional
        Number of parallel jobs (only for mannwhitneyu)
    keep_control : bool, default=True
        Whether to include control cells in filtered result
    return_adata : bool, default=False
        If True, return filtered AnnData object; if False, return list of perturbation names
        
    Returns
    -------
    Union[List[str], AnnData]
        Either list of perturbation names meeting criteria, or filtered AnnData object.
    """
    # Get summary
    summary_df = summarize_de_genes_from_adata(
        adata,
        perturbation_col=perturbation_col,
        control_name=control_name,
        pval_threshold=pval_threshold if padj_threshold is None else 0.05,
        padj_threshold=padj_threshold,
        effect_size_threshold=effect_size_threshold,
        de_method=de_method,
        n_jobs=n_jobs,
        return_dataframe=True
    )
    
    # Filter perturbations by minimum DE gene count
    passing_perts = summary_df[summary_df['n_de_genes'] >= min_de_genes]['perturbation'].tolist()
    
    if return_adata:
        # Filter AnnData object
        if keep_control:
            keep_mask = adata.obs[perturbation_col].isin(passing_perts + [control_name])
        else:
            keep_mask = adata.obs[perturbation_col].isin(passing_perts)
        
        return adata[keep_mask].copy()
    else:
        # Just return list of perturbations
        return passing_perts