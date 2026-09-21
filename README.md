# 🔢 MNIST Digit Classifier — PyTorch

A simple feedforward neural network built in PyTorch to classify handwritten digits from the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Built as a learning project to understand the core building blocks of deep learning: layers, activation functions, loss functions, optimizers, and the training loop.

## What it does

The model looks at a 28×28 grayscale image of a handwritten digit and predicts which digit (0–9) it is. It's trained on 60,000 labeled digit images and evaluated on 10,000 unseen ones.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~97% |
| Test Loss | *fill in after running* |
| Training Epochs | 20 |

*(Update these numbers with your own run's output.)*

## Model Architecture

A simple fully-connected ("dense") network — no convolutions, just linear layers:

```
Input (28×28 image)
   ↓ Flatten → 784 values
Linear(784 → 128) → ReLU → Dropout(0.3)
   ↓
Linear(128 → 64) → ReLU
   ↓
Linear(64 → 10)  →  10 digit scores
```

- **Flatten** — turns each 28×28 image into a flat list of 784 pixel values.
- **Linear layers** — learn weighted combinations of inputs, gradually compressing 784 → 128 → 64 → 10.
- **ReLU** — introduces non-linearity so the network can learn more than just straight-line patterns.
- **Dropout(0.3)** — randomly disables 30% of neurons during training to reduce overfitting.
- **Output layer (10 units)** — one raw score per digit class (0–9).

## Training Setup

- **Loss function:** `CrossEntropyLoss` — standard choice for multi-class classification.
- **Optimizer:** `Adam` with learning rate `0.001`.
- **Batch size:** 64
- **Epochs:** 20

Each training step follows the standard PyTorch pattern:

```python
optimizer.zero_grad()   # clear old gradients
pred = model(data)      # forward pass
loss = criterion(pred, targets)
loss.backward()         # compute gradients
optimizer.step()        # update weights
```

## Project Structure

```
.
├── pytorch-version.py     # Main script: data loading, training, evaluation, visualization
├── data/                  # MNIST dataset (auto-downloaded on first run)
├── predictions.png        # Sample of 10 test predictions (generated after running)
└── misclassified.png      # Sample of 10 misclassified digits (generated after running)
```

## Getting Started

### Requirements

```bash
pip install torch torchvision matplotlib
```

### Run it

```bash
python pytorch-version.py
```

On first run, the MNIST dataset will be automatically downloaded into a local `./data` folder. Training progress (loss per epoch) prints to the console, followed by the final test accuracy. Two image files — `predictions.png` and `misclassified.png` — are saved showing sample predictions and errors.

## What This Project Taught Me

This was built while learning PyTorch fundamentals — some of the core concepts explored here:

- Why input data needs to be normalized before training
- How `DataLoader` batches and shuffles data for efficient training
- The difference between a forward pass and backpropagation
- Why gradients must be manually zeroed each step (`optimizer.zero_grad()`)
- How `softmax` converts raw model outputs into interpretable probabilities
- Why `model.eval()` and `torch.no_grad()` matter during evaluation

## Possible Next Steps

- [ ] Swap the fully-connected layers for a CNN (`Conv2d`) and compare accuracy
- [ ] Try different optimizers (SGD vs. Adam) and compare training curves
- [ ] Experiment with hidden layer sizes and depth
- [ ] Add a validation split to monitor overfitting during training
- [ ] Try the same architecture on a different dataset (e.g., Fashion-MNIST)

## License

This project is for educational purposes. Feel free to fork and experiment.
