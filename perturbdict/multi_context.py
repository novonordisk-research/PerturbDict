from typing import Dict, List, Tuple, Union, Optional, Set
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from .core import PerturbDict
from .splits import get_train_test_split

# Optional AnnData import
try:
    import anndata
    HAS_ANNDATA = True
except ImportError:
    anndata = None
    HAS_ANNDATA = False


class MultiContextPerturbDict:
    """
    A container class for managing perturbation data across multiple contexts (e.g., cell types).
    
    Each context is represented by a PerturbDict instance. Perturbation names may overlap
    across contexts but are not required to be identical.
    
    Attributes:
    -----------
    contexts : Dict[str, PerturbDict]
        Dictionary mapping context names to their PerturbDict instances
    context_names : List[str]
        List of all context names
    """
    
    def __init__(self, contexts: Optional[Dict[str, PerturbDict]] = None):
        """
        Initialize MultiContextPerturbDict.
        
        Parameters:
        -----------
        contexts : Dict[str, PerturbDict], optional
            Dictionary of context_name -> PerturbDict mappings
        """
        if contexts is None:
            contexts = {}
        
        self.contexts = contexts
        self.context_names = list(contexts.keys())
        self._is_loaded = len(contexts) > 0
        
        # Build cached mappings for efficient lookup
        self._build_perturbation_mappings()
    
    def _build_perturbation_mappings(self):
        """Build cached mappings between perturbations and contexts."""
        self._perturbation_to_contexts = {}  # pert -> set of contexts
        self._context_to_perturbations = {}  # context -> set of perts
        
        for context_name, pert_dict in self.contexts.items():
            self._context_to_perturbations[context_name] = pert_dict.perturbations.copy()
            
            for pert in pert_dict.perturbations:
                if pert not in self._perturbation_to_contexts:
                    self._perturbation_to_contexts[pert] = set()
                self._perturbation_to_contexts[pert].add(context_name)
    
    @classmethod
    def from_adata_dict(
        cls,
        adatas: Dict[str, 'anndata.AnnData'],
        perturbation_col: str = 'perturbation',
        exclude_controls: bool = True,
        control_name: str = 'ctrl',
        find_de_genes: bool = True,
        de_method: str = 'ttest'
    ):
        """
        Create MultiContextPerturbDict from multiple AnnData objects.
        
        Parameters:
        -----------
        adatas : Dict[str, AnnData]
            Dictionary mapping context names to AnnData objects
        perturbation_col : str
            Column name in adata.obs containing perturbation labels
        exclude_controls : bool
            Whether to exclude control cells from perturbation dictionary
        control_name : str
            Name of control condition in perturbation_col
        find_de_genes : bool
            Whether to perform differential expression analysis
        de_method : str
            Method for DE analysis ('ttest' or 'wilcoxon')
        
        Returns:
        --------
        MultiContextPerturbDict
        """
        if not HAS_ANNDATA:
            raise ImportError(
                "AnnData is required for from_adata_dict(). "
                "Install with: pip install perturbdict[scanpy] or pip install anndata"
            )
        
        contexts = {}
        for context_name, adata in adatas.items():
            contexts[context_name] = PerturbDict.from_adata(
                adata,
                perturbation_col=perturbation_col,
                exclude_controls=exclude_controls,
                control_name=control_name,
                find_de_genes=find_de_genes,
                de_method=de_method
            )
        
        return cls(contexts)
    
    @classmethod
    def from_perturb_dict_collection(cls, datasets: Dict[str, PerturbDict]):
        """
        Create MultiContextPerturbDict from existing PerturbDict instances.
        
        Parameters:
        -----------
        datasets : Dict[str, PerturbDict]
            Dictionary mapping context names to PerturbDict instances
        
        Returns:
        --------
        MultiContextPerturbDict
        """
        return cls(datasets)
    
    @classmethod
    def from_cache(cls, filepath: Union[str, Path]):
        """Load MultiContextPerturbDict from saved file (.pkl)."""
        instance = cls()
        instance.load(filepath)
        return instance
    
    def get_expression(self, perturbation: str, context: str) -> np.ndarray:
        """
        Get mean expression vector for a perturbation in a specific context.
        
        Parameters:
        -----------
        perturbation : str
            Perturbation name
        context : str
            Context name
        
        Returns:
        --------
        np.ndarray
            Mean expression vector
        """
        self._check_loaded()
        
        if context not in self.contexts:
            raise ValueError(f"Context '{context}' not found. Available: {self.context_names}")
        
        return self.contexts[context].get_expression(perturbation)
    
    def get_ctrl_mean(self, context: str) -> np.ndarray:
        """
        Get control mean expression for a specific context.
        
        Parameters:
        -----------
        context : str
            Context name
        
        Returns:
        --------
        np.ndarray
            Control mean expression vector
        """
        self._check_loaded()
        
        if context not in self.contexts:
            raise ValueError(f"Context '{context}' not found. Available: {self.context_names}")
        
        return self.contexts[context].get_ctrl_mean()
    
    def get_all_perturbations(self) -> Set[str]:
        """
        Get union of all perturbations across all contexts.
        
        Returns:
        --------
        Set[str]
            Set of all perturbation names
        """
        self._check_loaded()
        return set(self._perturbation_to_contexts.keys())
    
    def get_shared_perturbations(self, min_contexts: int = 2) -> Set[str]:
        """
        Get perturbations present in at least min_contexts contexts.
        
        Parameters:
        -----------
        min_contexts : int
            Minimum number of contexts a perturbation must be present in
        
        Returns:
        --------
        Set[str]
            Set of perturbations meeting the criteria
        """
        self._check_loaded()
        
        if min_contexts > len(self.contexts):
            raise ValueError(
                f"min_contexts ({min_contexts}) cannot exceed number of contexts ({len(self.contexts)})"
            )
        
        return {
            pert for pert, contexts in self._perturbation_to_contexts.items()
            if len(contexts) >= min_contexts
        }
    
    def get_context_specific_perturbations(self, context: str) -> Set[str]:
        """
        Get perturbations that only exist in the specified context.
        
        Parameters:
        -----------
        context : str
            Context name
        
        Returns:
        --------
        Set[str]
            Set of perturbations unique to this context
        """
        self._check_loaded()
        
        if context not in self.contexts:
            raise ValueError(f"Context '{context}' not found. Available: {self.context_names}")
        
        context_perts = self._context_to_perturbations[context]
        return {
            pert for pert in context_perts
            if len(self._perturbation_to_contexts[pert]) == 1
        }
    
    def get_perturbation_availability(self) -> pd.DataFrame:
        """
        Get a DataFrame showing which perturbations exist in which contexts.
        
        Returns:
        --------
        pd.DataFrame
            Boolean DataFrame with perturbations as rows and contexts as columns
        """
        self._check_loaded()
        
        all_perts = sorted(self.get_all_perturbations())
        
        data = {}
        for context in self.context_names:
            data[context] = [pert in self._context_to_perturbations[context] for pert in all_perts]
        
        df = pd.DataFrame(data, index=all_perts)
        df.index.name = 'perturbation'
        
        return df
    
    def get_split_availability(
        self,
        train_dict: Dict[str, List[str]],
        test_dict: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Get a DataFrame showing train/test assignment for each (context, perturbation) pair.
        
        Parameters:
        -----------
        train_dict : Dict[str, List[str]]
            Dictionary from get_split() containing training perturbations per context
        test_dict : Dict[str, List[str]]
            Dictionary from get_split() containing test perturbations per context
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with perturbations as rows and contexts as columns.
            Values are 'train', 'test', or NaN (if perturbation not present in that context)
        """
        self._check_loaded()
        
        all_perts = sorted(self.get_all_perturbations())
        
        data = {}
        for context in self.context_names:
            train_set = set(train_dict.get(context, []))
            test_set = set(test_dict.get(context, []))
            
            column = []
            for pert in all_perts:
                if pert in train_set:
                    column.append('train')
                elif pert in test_set:
                    column.append('test')
                elif pert in self._context_to_perturbations[context]:
                    # Present in context but not in split (shouldn't normally happen)
                    column.append('unknown')
                else:
                    column.append(np.nan)
            
            data[context] = column
        
        df = pd.DataFrame(data, index=all_perts)
        df.index.name = 'perturbation'
        
        return df
    
    def get_overlap_stats(self) -> Dict:
        """
        Get summary statistics about perturbation overlap across contexts.
        
        Returns:
        --------
        Dict
            Dictionary containing overlap statistics
        """
        self._check_loaded()
        
        all_perts = self.get_all_perturbations()
        n_contexts = len(self.contexts)
        
        # Count perturbations by number of contexts they appear in
        coverage_counts = {}
        for i in range(1, n_contexts + 1):
            coverage_counts[f"in_{i}_context{'s' if i > 1 else ''}"] = len(
                [p for p in all_perts if len(self._perturbation_to_contexts[p]) == i]
            )
        
        # Per-context stats
        context_stats = {}
        for context in self.context_names:
            context_perts = self._context_to_perturbations[context]
            context_stats[context] = {
                'n_perturbations': len(context_perts),
                'n_unique': len(self.get_context_specific_perturbations(context)),
                'n_genes': len(self.contexts[context].gene_names)
            }
        
        return {
            'n_contexts': n_contexts,
            'total_unique_perturbations': len(all_perts),
            'coverage_distribution': coverage_counts,
            'per_context_stats': context_stats
        }
    
    def get_split(
        self,
        train_ratio: float = 0.8,
        seed: Optional[int] = None
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Get random train/test split of (context, perturbation) pairs.
        
        Splits at the (context, perturbation) pair level. If a perturbation appears
        in multiple contexts, some instances may be in train and some in test.
        
        Parameters:
        -----------
        train_ratio : float
            Proportion of (context, perturbation) pairs for training (default: 0.8)
        seed : Optional[int]
            Random seed for reproducibility
            
        Returns:
        --------
        train_dict : Dict[str, List[str]]
            {context_name: [train_perturbations_in_this_context]}
        test_dict : Dict[str, List[str]]
            {context_name: [test_perturbations_in_this_context]}
        """
        self._check_loaded()
        
        # Get all (context, perturbation) pairs
        all_pairs = []
        for context in self.context_names:
            for pert in self._context_to_perturbations[context]:
                all_pairs.append((context, pert))
        
        # Shuffle and split pairs
        all_pairs = sorted(all_pairs)  # Sort for reproducibility
        rng = np.random.RandomState(seed)
        rng.shuffle(all_pairs)
        
        n_train = int(len(all_pairs) * train_ratio)
        train_pairs = all_pairs[:n_train]
        test_pairs = all_pairs[n_train:]
        
        # Organize by context
        train_dict = {context: [] for context in self.context_names}
        test_dict = {context: [] for context in self.context_names}
        
        for context, pert in train_pairs:
            train_dict[context].append(pert)
        
        for context, pert in test_pairs:
            test_dict[context].append(pert)
        
        # Sort for consistency
        for context in self.context_names:
            train_dict[context] = sorted(train_dict[context])
            test_dict[context] = sorted(test_dict[context])
        
        return train_dict, test_dict
    
    def get_split_iterators(
        self,
        train_ratio: float = 0.8,
        seed: Optional[int] = None
    ):
        """
        Get train/test iterators for multi-context data.
        
        Parameters:
        -----------
        train_ratio : float
            Proportion of (context, perturbation) pairs for training
        seed : int, optional
            Random seed
        
        Returns:
        --------
        Tuple[MultiContextPerturbDataIterator, MultiContextPerturbDataIterator]
            (train_iterator, test_iterator)
        """
        # Import here to avoid circular dependency
        from .loaders import MultiContextPerturbDataIterator
        
        self._check_loaded()
        
        train_dict, test_dict = self.get_split(train_ratio=train_ratio, seed=seed)
        
        train_iter = MultiContextPerturbDataIterator(self, train_dict)
        test_iter = MultiContextPerturbDataIterator(self, test_dict)
        
        return train_iter, test_iter
    
    def save(self, filepath: Union[str, Path]):
        """
        Save MultiContextPerturbDict to file (pickle format).
        
        Parameters:
        -----------
        filepath : str or Path
            Path to save file (will use .pkl extension)
        
        Returns:
        --------
        self
            Returns self for method chaining
        """
        if not self._is_loaded:
            raise ValueError("Cannot save empty MultiContextPerturbDict")
        
        filepath = Path(filepath)
        
        save_data = {
            'contexts': self.contexts,
            'context_names': self.context_names,
            'metadata': {
                'version': '0.1.0',
                'type': 'MultiContextPerturbDict'
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        return self
    
    def load(self, filepath: Union[str, Path]):
        """
        Load MultiContextPerturbDict from saved file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to saved file
        
        Returns:
        --------
        self
            Returns self for method chaining
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.contexts = save_data['contexts']
            self.context_names = save_data['context_names']
            self._is_loaded = True
            
            # Rebuild mappings
            self._build_perturbation_mappings()
            
        except Exception as e:
            raise ValueError(f"Failed to load file {filepath}: {e}")
        
        return self
    
    def _check_loaded(self):
        """Check if data is loaded, raise error if not."""
        if not self._is_loaded:
            raise ValueError(
                "No data loaded. Use from_adata_dict(), from_perturb_dict_collection(), "
                "or from_cache() first."
            )
    
    def __repr__(self) -> str:
        """String representation of MultiContextPerturbDict."""
        if not self._is_loaded:
            return "MultiContextPerturbDict(empty - no data loaded)"
        
        n_contexts = len(self.contexts)
        n_total_perts = len(self.get_all_perturbations())
        
        return (
            f"MultiContextPerturbDict({n_contexts} contexts, "
            f"{n_total_perts} unique perturbations)"
        )
    
    def __getitem__(self, context: str) -> PerturbDict:
        """Access a specific context's PerturbDict by name."""
        self._check_loaded()
        
        if context not in self.contexts:
            raise KeyError(f"Context '{context}' not found. Available: {self.context_names}")
        
        return self.contexts[context]
