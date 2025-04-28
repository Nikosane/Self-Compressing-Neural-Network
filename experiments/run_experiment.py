```python
import torch

from data.load_data import get_dataloaders
from models.base_model import BaseModel
from trainer.train import train
from utils.logger import Logger

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_dataloaders(dataset_name="MNIST", batch_size=64)

    model = BaseModel()
    logger = Logger()

    model = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=20,
        device=device,
        prune_every=5,
        prune_amount=0.1
    )

    params = count_parameters(model)
    accuracy = logger.logs[-1]["accuracy"] if logger.logs else 0

    logger.log(epoch="Final", loss=0, accuracy=accuracy, params_count=params)
    logger.save()

if __name__ == "__main__":
    main()
```
