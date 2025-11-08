from typing import Dict, List, Tuple, Union, Optional, Set
import pandas as pd
import numpy as np
import pickle
import json
import warnings
from pathlib import Path
from .preprocessing import aggregate_by_perturbation_simple
from .diff_exp_analysis import find_de_genes_by_perturbation
from .utils import (
    validate_perturbation_dict,
    format_de_results,
    create_de_ranked_structure,
    apply_thresholds
)
from .splits import get_train_test_split, PerturbDataIterator

# Optional AnnData import
try:
    import anndata
    HAS_ANNDATA = True
except ImportError:
    anndata = None
    HAS_ANNDATA = False


class PerturbDict:
    """
    A lightweight data class for storing transcriptomics data aggregated by perturbation.

    This class provides a clean interface for working with perturbation data, including
    mean expression values and ranked differential expression information.

    Attributes:
    -----------
    gene_names : List[str]
        List of all gene names in the dataset
    perturbations : Set[str]
        Set of perturbation names in the dataset
    de_results : Dict[str, pd.DataFrame]
        Detailed DE statistics for each perturbation (if DE analysis was performed)
    ctrl_mean : Optional[np.ndarray]
        Mean expression values for control cells (if control cells are present)
    """

    def __init__(self):
        """Initialize empty PerturbDict. Use from_adata(), from_cache(), or from_dict() to load data."""
        self._initialize_empty()

    def _initialize_empty(self):
        """Initialize object in empty state."""
        self.gene_names = []
        self.perturbations = set()
        self.de_results = {}
        self.ctrl_mean = None
        self._data = {
            "gene_names": [],
            "perturbations": {}
        }
        self._is_loaded = False

    def load_from_adata(
        self,
        adata,
        perturbation_col: str = 'perturbation',
        exclude_controls: bool = True,
        control_name: str = 'NT',
        find_de_genes: bool = True,
        de_method: str = 'ttest'
    ):
        """Load and aggregate data from AnnData object. Returns self for chaining."""
        if not HAS_ANNDATA:
            raise ImportError(
                "AnnData is required for load_from_adata(). "
                "Install with: pip install perturbdict[scanpy] or pip install anndata"
            )

        # Store parameters
        self._perturbation_col = perturbation_col
        self._exclude_controls = exclude_controls
        self._control_name = control_name
        self._find_de_genes = find_de_genes
        self._de_method = de_method

        # Build the data structure
        self._build_from_adata(adata)
        self._is_loaded = True

        return self

    def _build_from_adata(self, adata):
        """Build the perturbation dictionary from AnnData object."""
        # Get gene names
        self.gene_names = adata.var_names.tolist()

        # Get mean expression for each perturbation, along with ctrl mean
        pert_expressions, perturbations, self.ctrl_mean = aggregate_by_perturbation_simple(
            adata,
            perturbation_col=self._perturbation_col,
            exclude_controls=self._exclude_controls,
            control_name=self._control_name
        )

        # Find DE genes if requested
        self.de_results = {}
        if self._find_de_genes:
            _, self.de_results = find_de_genes_by_perturbation(
                adata,
                perturbation_col=self._perturbation_col,
                control_name=self._control_name,
                n_top_genes=[len(self.gene_names)],  # Get all genes ranked
                de_method=self._de_method
            )

            # Warn if no DE results were generated (likely no control cells found)
            if not self.de_results:
                warnings.warn(
                    f"No differential expression results generated. "
                    f"This likely means no control cells with name '{self._control_name}' "
                    f"were found in column '{self._perturbation_col}'. "
                    f"Available values: {adata.obs[self._perturbation_col].unique().tolist()[:10]}",
                    UserWarning
                )

        # Create the internal data structure
        self._data = {
            "gene_names": self.gene_names,
            "perturbations": {}
        }

        # Store perturbations as a set for easy access
        self.perturbations = set(perturbations)

        # Build perturbation data
        for pert in perturbations:
            self._data["perturbations"][pert] = {
                "exprs": pert_expressions[pert]
            }

            # Add ranked DE information if available
            if self._find_de_genes and pert in self.de_results:
                de_df = self.de_results[pert]
                self._data["perturbations"][pert]["de_ranked"] = create_de_ranked_structure(de_df, self.gene_names)

    @classmethod
    def from_adata(
        cls,
        adata,
        perturbation_col: str = 'perturbation',
        exclude_controls: bool = True,
        control_name: str = 'NT',
        find_de_genes: bool = True,
        de_method: str = 'ttest'
    ):
        """Create PerturbDict from AnnData object."""
        instance = cls()
        instance.load_from_adata(
            adata,
            perturbation_col=perturbation_col,
            exclude_controls=exclude_controls,
            control_name=control_name,
            find_de_genes=find_de_genes,
            de_method=de_method
        )
        return instance

    @classmethod
    def from_dict(cls, data_dict: Dict, de_results: Optional[Dict[str, pd.DataFrame]] = None,
                  ctrl_mean: Optional[np.ndarray] = None):
        """Create PerturbDict from dictionary structure."""
        instance = cls.__new__(cls)
        instance._initialize_empty()

        instance._data = data_dict
        instance.gene_names = data_dict["gene_names"]
        instance.perturbations = set(data_dict["perturbations"].keys())
        instance.de_results = de_results or {}
        instance.ctrl_mean = ctrl_mean
        instance._is_loaded = True

        return instance

    @classmethod
    def from_cache(cls, filepath: Union[str, Path]):
        """Load PerturbDict from saved file (.pkl or .json)."""
        instance = cls()
        instance.load(filepath)
        return instance

    def get_expression(self, perturbation: str) -> np.ndarray:
        """Get mean expression vector for a perturbation."""
        self._check_loaded()

        if perturbation not in self.perturbations:
            raise ValueError(f"Perturbation '{perturbation}' not found")

        return self._data["perturbations"][perturbation]["exprs"]

    def get_ctrl_mean(self) -> np.ndarray:
        """Get mean expression vector for control cells."""
        self._check_loaded()

        if self.ctrl_mean is None:
            raise ValueError(
                "No control mean available. This likely means no control cells were found in the data."
            )

        return self.ctrl_mean

    def get_top_de_genes(
        self,
        perturbation: str,
        k: int = 20,
        return_type: str = 'indices'
    ) -> Union[List[int], List[str], Dict]:
        """Get top k DE genes. return_type: 'indices', 'names', 'stats', or 'all'."""
        self._check_loaded()

        # Validate inputs and get perturbation data
        pert_data = validate_perturbation_dict(self._data, perturbation, require_de=True)
        de_ranked = pert_data['de_ranked']

        # Limit k to available genes
        n_genes_available = len(de_ranked['gene_indices'])
        k = min(k, n_genes_available)

        # Get top k data
        top_indices = de_ranked['gene_indices'][:k]
        top_pvals = de_ranked['pvalues'][:k]
        top_effects = de_ranked['effect_sizes'][:k]
        top_padj = de_ranked['padj'][:k]

        # Format and return results
        return format_de_results(
            self._data, pert_data, top_indices, top_pvals, top_effects, top_padj, return_type
        )

    def get_de_mask(self, perturbation: str, k: int = 20) -> np.ndarray:
        """Get boolean mask for top-k DE genes (useful for subsetting predictions)."""
        indices = self.get_top_de_genes(perturbation, k=k, return_type='indices')
        mask = np.zeros(len(self.gene_names), dtype=bool)
        mask[indices] = True
        return mask

    def get_de_genes_by_threshold(
        self,
        perturbation: str,
        pval_threshold: float = 0.05,
        padj_threshold: Optional[float] = None,
        effect_size_threshold: Optional[float] = None,
        return_type: str = 'indices'
    ) -> Union[List[int], List[str], Dict]:
        """Get DE genes meeting statistical thresholds (pval, padj, effect size)."""
        # Validate inputs and get perturbation data
        pert_data = validate_perturbation_dict(self._data, perturbation, require_de=True)
        de_ranked = pert_data['de_ranked']

        # Apply thresholds to get significant genes
        final_mask = apply_thresholds(
            de_ranked, pval_threshold, padj_threshold, effect_size_threshold
        )

        # Get data for significant genes
        significant_positions = np.where(final_mask)[0]
        significant_indices = [de_ranked['gene_indices'][i] for i in significant_positions]

        pvals = np.array(de_ranked['pvalues'])
        effect_sizes = np.array(de_ranked['effect_sizes'])
        padj = np.array(de_ranked['padj'])

        sig_pvals = pvals[final_mask].tolist()
        sig_effects = effect_sizes[final_mask].tolist()
        sig_padj = padj[final_mask].tolist()

        # Format and return results
        return format_de_results(
            self._data, pert_data, significant_indices, sig_pvals, sig_effects, sig_padj, return_type
        )

    def save(self, filepath: Union[str, Path], format: str = 'pickle'):
        """Save to file (format: 'pickle' or 'json'). Returns self for chaining."""
        if not self._is_loaded:
            raise ValueError("Cannot save empty PerturbDict. Load data first with .load_from_adata() or .load()")

        filepath = Path(filepath)
        save_data = {
            'data': self._data,
            'gene_names': self.gene_names,
            'perturbations': list(self.perturbations),
            'ctrl_mean': self.ctrl_mean,
            'metadata': {'version': '0.1.0', 'format': format}
        }

        if format == 'pickle':
            save_data['de_results'] = self.de_results
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)

        elif format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_data = self._data.copy()
            for pert in json_data['perturbations']:
                if 'exprs' in json_data['perturbations'][pert]:
                    json_data['perturbations'][pert]['exprs'] = json_data['perturbations'][pert]['exprs'].tolist()

            save_data['data'] = json_data
            # Convert ctrl_mean to list
            if self.ctrl_mean is not None:
                save_data['ctrl_mean'] = self.ctrl_mean.tolist()
            # Note: de_results DataFrames not included in JSON format
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'")

        return self

    def load(self, filepath: Union[str, Path]):
        """Load from saved file (.pkl or .json). Returns self for chaining."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Determine format from file extension
        format_type = 'json' if filepath.suffix == '.json' else 'pickle'

        try:
            if format_type == 'pickle':
                with open(filepath, 'rb') as f:
                    save_data = pickle.load(f)
                self.de_results = save_data.get('de_results', {})

            else:  # json
                with open(filepath, 'r') as f:
                    save_data = json.load(f)

                # Convert lists back to numpy arrays
                for pert in save_data['data']['perturbations']:
                    if 'exprs' in save_data['data']['perturbations'][pert]:
                        save_data['data']['perturbations'][pert]['exprs'] = np.array(
                            save_data['data']['perturbations'][pert]['exprs']
                        )
                self.de_results = {}  # Not saved in JSON format

            # Set common attributes
            self._data = save_data['data']
            self.gene_names = save_data['gene_names']
            self.perturbations = set(save_data['perturbations'])

            # Load ctrl_mean
            self.ctrl_mean = save_data.get('ctrl_mean')

            # Convert to numpy array if loaded from JSON
            if format_type == 'json' and self.ctrl_mean is not None:
                self.ctrl_mean = np.array(self.ctrl_mean)

            self._is_loaded = True

        except Exception as e:
            raise ValueError(f"Failed to load file {filepath}: {e}")

        return self

    def to_dict(self) -> Tuple[Dict, Dict[str, pd.DataFrame]]:
        """Export the data structure and DE results."""
        self._check_loaded()
        return self._data.copy(), self.de_results.copy()

    def get_observed_dict(self) -> Dict[str, np.ndarray]:
        """Get expression data as dict: {perturbation_name: expression_vector}."""
        self._check_loaded()
        return {pert: self._data["perturbations"][pert]["exprs"]
                for pert in self.perturbations}

    def get_split(self, k: int = 5, fold: int = 0, seed: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """Get k-fold train/test split. Returns (train_perts, test_perts)."""
        self._check_loaded()
        # Sort perturbations to ensure consistent ordering across runs
        sorted_perts = sorted(list(self.perturbations))
        train, test = get_train_test_split(sorted_perts, k=k, fold=fold, seed=seed)
        return train.tolist(), test.tolist()

    def get_split_data(self, k: int = 5, fold: int = 0, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Get k-fold train/test data split. Returns (train_data, test_data) dicts."""
        train_perts, test_perts = self.get_split(k=k, fold=fold, seed=seed)
        observed = self.get_observed_dict()
        train_data = {p: observed[p] for p in train_perts}
        test_data = {p: observed[p] for p in test_perts}
        return train_data, test_data

    def get_split_iterators(self, k: int = 5, fold: int = 0, seed: Optional[int] = None):
        """Get k-fold train/test iterators. Compatible with PyTorch DataLoader."""
        self._check_loaded()
        train_perts, test_perts = self.get_split(k=k, fold=fold, seed=seed)
        train_iter = PerturbDataIterator(self, train_perts)
        test_iter = PerturbDataIterator(self, test_perts)
        return train_iter, test_iter

    def find_knn(self, query: Union[str, List[str]], k: int = 5,
                 reference_perts: Optional[List[str]] = None,
                 metric: str = "euclidean") -> Dict[str, List[str]]:
        """Find k-nearest neighbor perturbations by expression similarity."""
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise ImportError("Requires scikit-learn: pip install scikit-learn")

        self._check_loaded()
        if isinstance(query, str):
            query = [query]

        all_perts = reference_perts if reference_perts is not None else list(self.perturbations)
        profiles = np.array([self.get_expression(p) for p in all_perts])

        nn = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
        nn.fit(profiles)

        knn_dict = {}
        for q in query:
            if q not in self.perturbations:
                raise ValueError(f"Query '{q}' not found")

            _, indices = nn.kneighbors(self.get_expression(q).reshape(1, -1))
            neighbors = [all_perts[i] for i in indices[0] if all_perts[i] != q][:k]
            knn_dict[q] = neighbors

        return knn_dict

    def _check_loaded(self):
        """Check if data is loaded, raise error if not."""
        if not self._is_loaded:
            raise ValueError(
                "No data loaded. Use from_adata(), from_cache(), or load_from_adata() first."
            )

    def __repr__(self) -> str:
        """String representation of PerturbDict."""
        if not self._is_loaded:
            return "PerturbDict(empty - no data loaded)"

        n_genes = len(self.gene_names)
        n_perts = len(self.perturbations)
        de_status = "with DE analysis" if self.de_results else "without DE analysis"

        return f"PerturbDict({n_perts} perturbations, {n_genes} genes, {de_status})"
