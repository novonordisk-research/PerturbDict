from typing import List, Tuple, Dict
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
