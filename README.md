# Self-Compressing Neural Networks

> Networks that optimize their own architectures during training to become smaller, smarter, and faster — without human intervention.
> Inspired by Neural Darwinism: survival of the fittest neurons.

## 🚀 Project Overview
This project builds a neural network that self-compresses during training by:
- Identifying and pruning unnecessary neurons or weights.
- Retraining after pruning to recover performance.
- Evolving to become more efficient over time.

The network learns to optimize itself without manual tuning.
---

## ⚙️ How It Works
1. Train an initial large network.
2. Periodically prune low-importance neurons/weights.
3. Fine-tune the network after pruning.
4. Repeat the cycle until reaching a compact and performant model.

## 🛠 Requirements
- Python 3.8+
- PyTorch
- Numpy
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🧪 Running an Experiment
```bash
python experiments/run_experiment.py
```

## 📊 What We Track
- Model size (neurons/parameters) over time
- Accuracy vs. compression
- Speed and memory improvements
- Visualization of pruning dynamics

## 🌟 Future Extensions
- Dynamic neuron regrowth
- Meta-learned pruning strategies
- Evolutionary competitions between sub-networks
