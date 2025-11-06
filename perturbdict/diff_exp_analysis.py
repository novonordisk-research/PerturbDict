from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from scipy import stats, sparse
from statsmodels.stats.multitest import multipletests
import warnings

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x


def _extract_expression_data(adata, mask):
    """Extract expression data as dense array."""
    if sparse.issparse(adata.X):
        return adata[mask].X.toarray()
    else:
        return adata[mask].X


def _perform_ttest(pert_X, control_X):
    """Perform Welch's t-test between perturbation and control groups."""
    n_pert, n_control = pert_X.shape[0], control_X.shape[0]

    # Calculate means and variances
    pert_mean = np.mean(pert_X, axis=0)
    control_mean = np.mean(control_X, axis=0)
    pert_var = np.var(pert_X, axis=0, ddof=1)
    control_var = np.var(control_X, axis=0, ddof=1)

    # Calculate t-statistic and p-values
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = (pert_mean - control_mean) / np.sqrt(pert_var/n_pert + control_var/n_control)

        # Degrees of freedom (Welch-Satterthwaite equation)
        df_num = (pert_var/n_pert + control_var/n_control)**2
        df_denom = (pert_var/n_pert)**2/(n_pert-1) + (control_var/n_control)**2/(n_control-1)
        df = np.where(df_denom > 0, df_num / df_denom, 1.0)

        # Calculate p-values for valid statistics
        mask_valid = ~np.isnan(t_stat) & ~np.isinf(t_stat) & (df > 0) & (df_denom > 0)
        pvals = np.ones(len(t_stat))
        pvals[mask_valid] = 2 * (1 - stats.t.cdf(np.abs(t_stat[mask_valid]), df[mask_valid]))

    return pvals, t_stat, pert_mean, control_mean


def _perform_mannwhitneyu(pert_X, control_X, n_jobs: Optional[int] = None, show_progress: bool = False):
    """Perform Mann-Whitney U test between perturbation and control groups."""
    n_genes = pert_X.shape[1]

    # Try parallel if joblib available and n_jobs specified
    if n_jobs is not None and n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            def test_gene(i):
                try:
                    stat, pval = stats.mannwhitneyu(pert_X[:, i], control_X[:, i], alternative='two-sided')
                    return stat, pval
                except:
                    return 0.0, 1.0

            results = Parallel(n_jobs=n_jobs)(delayed(test_gene)(i) for i in range(n_genes))
            u_stats, pvals = zip(*results)
            pvals = np.array(pvals)
            u_stats = np.array(u_stats)
        except ImportError:
            # Fallback to serial
            pvals, u_stats = _mannwhitneyu_serial(pert_X, control_X, n_genes, show_progress)
    else:
        # Serial execution
        pvals, u_stats = _mannwhitneyu_serial(pert_X, control_X, n_genes, show_progress)

    pert_mean = np.mean(pert_X, axis=0)
    control_mean = np.mean(control_X, axis=0)

    return pvals, u_stats, pert_mean, control_mean


def _mannwhitneyu_serial(pert_X, control_X, n_genes, show_progress: bool = False):
    """Serial Mann-Whitney U test implementation."""
    pvals = np.ones(n_genes)
    u_stats = np.zeros(n_genes)

    gene_iter = tqdm(range(n_genes), desc="Mann-Whitney U", disable=not show_progress) if HAS_TQDM else range(n_genes)
    for i in gene_iter:
        try:
            stat, pval = stats.mannwhitneyu(pert_X[:, i], control_X[:, i], alternative='two-sided')
            pvals[i] = pval
            u_stats[i] = stat
        except:
            pass  # Keep p-value at 1.0

    return pvals, u_stats


def find_de_genes_by_perturbation(
    adata,
    perturbation_col: str = "perturbation",
    control_name: str = "NT",
    n_top_genes: Union[int, List[int]] = [20],
    de_method: str = "ttest",
    n_jobs: Optional[int] = None,
    show_progress: bool = True
) -> Tuple[Dict[str, Dict[int, List[int]]], Dict[str, pd.DataFrame]]:
    """Find DE genes for each perturbation vs control. Returns (de_indices, de_results)."""
    # Setup
    if isinstance(n_top_genes, int):
        n_top_genes = [n_top_genes]

    all_perturbations = adata.obs[perturbation_col].unique()
    perturbations = all_perturbations[all_perturbations != control_name]
    gene_names = adata.var_names.tolist()

    # Early return if no control
    if control_name not in all_perturbations:
        return {}, {}

    # Extract control data once
    control_mask = adata.obs[perturbation_col] == control_name
    control_X = _extract_expression_data(adata, control_mask)

    de_indices_dict = {}
    de_results_dict = {}

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                               message='.*Precision loss occurred in moment calculation.*')

        perturbation_iter = tqdm(perturbations, desc="DE analysis", disable=not show_progress) if HAS_TQDM else perturbations
        for perturbation in perturbation_iter:
            # Extract perturbation data
            pert_mask = adata.obs[perturbation_col] == perturbation
            pert_X = _extract_expression_data(adata, pert_mask)

            # Perform statistical test
            if de_method == 'ttest':
                pvals, stats_values, pert_mean, control_mean = _perform_ttest(pert_X, control_X)
            elif de_method == 'mannwhitneyu':
                pvals, stats_values, pert_mean, control_mean = _perform_mannwhitneyu(pert_X, control_X, n_jobs, show_progress)
            else:
                raise ValueError(f"Unknown DE method: {de_method}")

            # Multiple testing correction
            padj = multipletests(pvals, method='fdr_bh')[1]

            # Create and sort results DataFrame
            results_df = pd.DataFrame({
                'gene': gene_names,
                'pval': pvals,
                'padj': padj,
                'effect_size': pert_mean - control_mean,
                'perturbation_mean': pert_mean,
                'control_mean': control_mean,
                'statistic': stats_values
            }).sort_values('pval')

            # Store results and generate indices for requested thresholds
            de_results_dict[perturbation] = results_df
            de_indices_dict[perturbation] = {}

            for n in n_top_genes:
                n_to_get = min(n, len(gene_names))
                top_genes = results_df['gene'].iloc[:n_to_get].tolist()
                de_indices_dict[perturbation][n] = [gene_names.index(gene) for gene in top_genes]

    return de_indices_dict, de_results_dict