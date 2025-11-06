from typing import Dict, List, Union, Optional, Callable
import numpy as np
import pandas as pd


def validate_perturbation_dict(pert_dict: Dict, perturbation: str, require_de: bool = True) -> Dict:
    """Validate and return perturbation data. Raises ValueError if not found or missing DE info."""
    if perturbation not in pert_dict['perturbations']:
        raise ValueError(f"Perturbation '{perturbation}' not found in dictionary")

    pert_data = pert_dict['perturbations'][perturbation]

    if require_de and 'de_ranked' not in pert_data:
        raise ValueError(f"No DE information available for perturbation '{perturbation}'")

    return pert_data


def get_gene_names_from_indices(pert_dict: Dict, indices: List[int]) -> List[str]:
    """Convert gene indices to gene names."""
    gene_names = pert_dict['gene_names']
    return [gene_names[i] for i in indices]


def get_expression_values(pert_data: Dict, indices: List[int]) -> np.ndarray:
    """Get expression values for specified gene indices."""
    exprs = pert_data['exprs']
    if isinstance(exprs, np.ndarray):
        return exprs[indices]
    else:
        return np.array(exprs)[indices]


def format_de_results(
    pert_dict: Dict,
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