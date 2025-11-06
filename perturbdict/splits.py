from typing import List, Tuple
import numpy as np
from sklearn.model_selection import KFold


def partition_perturbations_into_k_folds(perturbations: List[str], k: int = 5, seed: int = None):
    """Partition perturbations into k folds. Returns (test_folds, train_folds)."""
    perturbations = np.array(perturbations)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    test_perturbations = []
    train_perturbations = []

    for train_index, test_index in kf.split(perturbations):
        test_perturbations.append(perturbations[test_index])
        train_perturbations.append(perturbations[train_index])

    return test_perturbations, train_perturbations


def get_train_test_split(perturbations: List[str], k: int = 5, fold: int = 0, seed: int = None):
    """Get train/test split for specific fold. Returns (train_perts, test_perts)."""
    test_perts, train_perts = partition_perturbations_into_k_folds(perturbations, k=k, seed=seed)
    return train_perts[fold], test_perts[fold]


class PerturbDataIterator:
    """Iterator over perturbation data. Compatible with torch.utils.data.DataLoader."""

    def __init__(self, pert_dict, perturbations: List[str]):
        """Create iterator for given perturbations."""
        self.pert_dict = pert_dict
        self.perturbations = list(perturbations)
        self._index = 0

    def __len__(self):
        return len(self.perturbations)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.perturbations):
            raise StopIteration
        pert_name = self.perturbations[self._index]
        expression = self.pert_dict.get_expression(pert_name)
        self._index += 1
        return pert_name, expression

    def __getitem__(self, idx):
        """For PyTorch DataLoader compatibility."""
        pert_name = self.perturbations[idx]
        expression = self.pert_dict.get_expression(pert_name)
        return pert_name, expression
