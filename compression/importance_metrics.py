```python
import torch

def compute_weight_magnitudes(model):
    """
    Computes the L1 norm of weights for all Linear layers.

    Args:
        model (nn.Module): The model to analyze.

    Returns:
        dict: Mapping of layer names to L1 norms.
    """
    importance = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data.abs()
            importance[name] = weight.mean().item()
    return importance
```
