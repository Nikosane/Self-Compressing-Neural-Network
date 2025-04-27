```python
import torch
import torch.nn as nn
import torch.optim as optim

from compression.pruner import prune_model

def train(model, train_loader, test_loader, epochs=20, device="cpu", prune_every=5, prune_amount=0.1):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        test_accuracy = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")

        # Prune after certain number of epochs
        if (epoch + 1) % prune_every == 0:
            model = prune_model(model, amount=prune_amount)
            print(f"Pruned model at epoch {epoch+1}")

    return model

def evaluate(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
```
