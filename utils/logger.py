```python
import os
import json

class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logs = []

    def log(self, epoch, loss, accuracy, params_count):
        entry = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "parameters": params_count
        }
        self.logs.append(entry)

    def save(self, filename="training_log.json"):
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=4)

    def print_last(self):
        if self.logs:
            last = self.logs[-1]
            print(f"[Epoch {last['epoch']}] Loss: {last['loss']:.4f} | Accuracy: {last['accuracy']:.2f}% | Params: {last['parameters']}")
```
