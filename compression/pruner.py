```python
import torch
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.2):
    """
    Prunes the smallest magnitude weights from all Linear layers.
    
    Args:
        model (nn.Module): The model to prune.
        amount (float): Fraction of weights to prune per layer.
    """
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model
```
