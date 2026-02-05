from typing import List, Dict


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


class MultiContextPerturbDataIterator:
    """
    Iterator over multi-context perturbation data.
    
    Yields (context, perturbation, expression) tuples.
    Compatible with torch.utils.data.DataLoader.
    """
    
    def __init__(self, multi_pert_dict, context_perturbations: Dict[str, List[str]]):
        """
        Create iterator for multi-context data.
        
        Parameters:
        -----------
        multi_pert_dict : MultiContextPerturbDict
            The multi-context perturbation dictionary
        context_perturbations : Dict[str, List[str]]
            Dictionary mapping context names to lists of perturbations
        """
        self.multi_pert_dict = multi_pert_dict
        self.context_perturbations = context_perturbations
        
        # Flatten to list of (context, perturbation) tuples
        self._items = []
        for context, perts in context_perturbations.items():
            for pert in perts:
                self._items.append((context, pert))
        
        self._index = 0
    
    def __len__(self):
        return len(self._items)
    
    def __iter__(self):
        self._index = 0
        return self
    
    def __next__(self):
        if self._index >= len(self._items):
            raise StopIteration
        
        context, pert_name = self._items[self._index]
        expression = self.multi_pert_dict.get_expression(pert_name, context)
        self._index += 1
        
        return context, pert_name, expression
    
    def __getitem__(self, idx):
        """For PyTorch DataLoader compatibility."""
        context, pert_name = self._items[idx]
        expression = self.multi_pert_dict.get_expression(pert_name, context)
        return context, pert_name, expression
