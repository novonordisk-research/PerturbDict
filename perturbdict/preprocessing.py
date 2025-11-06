from typing import Dict, List, Tuple, Union
import numpy as np
from scipy import sparse


def aggregate_by_perturbation_simple(
    adata,
    perturbation_col: str = "perturbation",
    exclude_controls: bool = True,
    control_name: str = "NT"
) -> Tuple[Dict[str, np.ndarray], List[str], Union[np.ndarray, None]]:
    """Calculate mean expression per perturbation. Returns (pert_dict, pert_names, ctrl_mean)."""
    # Get perturbations to process
    all_perturbations = adata.obs[perturbation_col].unique()
    perturbations = all_perturbations[all_perturbations != control_name] if exclude_controls else all_perturbations

    # Calculate mean expression for control cells
    ctrl_mask = adata.obs[perturbation_col] == control_name
    if ctrl_mask.any():
        if sparse.issparse(adata.X):
            ctrl_mean = adata[ctrl_mask].X.mean(axis=0).A1
        else:
            ctrl_mean = adata[ctrl_mask].X.mean(axis=0)
    else:
        ctrl_mean = None

    # Calculate mean expression for each perturbation
    perturbation_dict = {}
    for perturbation in perturbations:
        pert_mask = adata.obs[perturbation_col] == perturbation

        # Calculate mean, handling sparse matrices
        if sparse.issparse(adata.X):
            pert_mean = adata[pert_mask].X.mean(axis=0).A1
        else:
            pert_mean = adata[pert_mask].X.mean(axis=0)

        perturbation_dict[perturbation] = pert_mean

    return perturbation_dict, list(perturbations), ctrl_mean