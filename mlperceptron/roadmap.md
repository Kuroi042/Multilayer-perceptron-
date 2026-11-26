# Multilayer Perceptron (from scratch)

Predict whether a breast cancer is **malignant (M)** or **benign (B)** using a multilayer perceptron implemented with NumPy only. No neural network libraries.

- **Language:** Python 3
- **Allowed libraries:** `numpy`, `pandas` (CSV loading), `matplotlib` (curves)
- **Forbidden:** TensorFlow, PyTorch, Keras, scikit-learn (avoid it entirely, even for the split)
- **Deadline:** 7 days, mandatory part + bonus

> The bonus is only graded if the mandatory part is **perfect**. Lock the mandatory part by Day 5.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install numpy pandas matplotlib
pip freeze > requirements.txt
```

## Project layout

```
mlp/
├── split.py            # program 1: split dataset into train / validation
├── train.py            # program 2: train, plot curves, save model
├── predict.py          # program 3: load model, predict, print BCE + accuracy
├── mlp/
│   ├── activations.py  # sigmoid, tanh, relu, softmax (+ derivatives)
│   ├── layers.py       # DenseLayer, weight initializers
│   ├── losses.py       # categorical + binary cross-entropy
│   ├── optimizers.py   # SGD (mandatory), Momentum/Nesterov/RMSprop/Adam (bonus)
│   ├── model.py        # Network: forward, backward
│   ├── metrics.py      # accuracy, precision, recall, F1, confusion matrix
│   └── utils.py        # normalization, one-hot, save/load
├── plot_compare.py     # bonus: overlay several histories on one graph
├── requirements.txt
└── README.md
```

## Target usage

```bash
python split.py --dataset data.csv --seed 42
python train.py --dataset data_training.csv --valid data_validation.csv \
    --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy \
    --batch_size 8 --learning_rate 0.0314
python predict.py --dataset data_validation.csv --model saved_model.npy
```

Expected training output:

```
x_train shape : (455, 30)
x_valid shape : (114, 30)
epoch 01/84 - loss: 0.6882 - val_loss: 0.6788
...
> saving model './saved_model.npy' to disk...
```

---

## Roadmap

### Day 1: Data understanding and preprocessing
- [ ] Load the CSV (32 columns, **no header**): column 0 = ID, column 1 = diagnosis, columns 2-31 = features
- [ ] Drop ID, encode `M -> 1`, `B -> 0`
- [ ] Explore: class balance, histograms, correlations (keep a notebook to show the corrector)
- [ ] Standardize features: `(x - mean) / std`, computed on the **training set only**
- [ ] Write `split.py`: seeded shuffle, ~80/20 split, stratified if possible, save two CSVs
- [ ] Save mean/std so `predict.py` can reuse them

### Day 2: Feedforward
- [ ] `DenseLayer`: `W` of shape `(n_in, n_out)`, `b` of shape `(1, n_out)`
- [ ] Initializers: He uniform (default), Xavier, random normal
- [ ] Activations and derivatives: sigmoid, tanh, ReLU
- [ ] Softmax with stability trick (`z - max(z)`) on the 2-neuron output layer
- [ ] One-hot labels: `Y = np.eye(2)[labels]`
- [ ] Forward pass caches `X`, `Z`, `A` per layer
- [ ] Sanity test: each output row sums to 1

### Day 3: Loss and backpropagation (the core)
- [ ] Categorical cross-entropy and binary cross-entropy, with `np.clip(p, 1e-15, 1)`
- [ ] Output layer (softmax + CE): `dZ = A_out - Y`
- [ ] Hidden layers: `dZ = (dZ_next @ W_next.T) * activation'(Z)`
- [ ] Gradients: `dW = A_prev.T @ dZ / m`, `db = sum(dZ) / m`
- [ ] **Gradient check**: analytic vs numeric `(L(w+e) - L(w-e)) / 2e`, must match to ~6 digits
- [ ] Gradient descent: `W -= lr * dW`

### Day 4: Training loop, saving, predicting
- [ ] Mini-batch loop, reshuffle every epoch
- [ ] Per epoch: loss and accuracy on train and validation, printed in the subject's format
- [ ] Print `x_train shape` and `x_valid shape`
- [ ] Two learning-curve plots at the end (loss, accuracy; train vs validation)
- [ ] Save model: topology, activations, weights, biases, **normalization stats**
- [ ] `predict.py`: load model, normalize with saved stats, print BCE and accuracy
- [ ] CLI args: `--layer`, `--epochs`, `--loss`, `--batch_size`, `--learning_rate`
- [ ] Default: at least **2 hidden layers**
- [ ] Checkpoint: validation accuracy ~95%+ (lower means a bug)

### Day 5: Mandatory polish and defense prep
- [ ] Edge cases: missing file, bad args, empty layer list, invalid learning rate
- [ ] Clean, structured, commented code
- [ ] Re-read the subject line by line and tick every requirement
- [ ] Rehearse explanations out loud (see below)

### Day 6: Bonus (only if the mandatory part is perfect)
In order of value per effort:
- [ ] **Metrics history**: save per-epoch metrics to JSON
- [ ] **Early stopping**: patience counter, restore the **best** weights
- [ ] **Multiple metrics**: precision, recall, F1, confusion matrix
- [ ] **Multiple curves on one graph**: overlay several history files (same seed and split for fair comparison)
- [ ] **Better optimizers**: Momentum, Nesterov, RMSprop, **Adam** (`--optimizer adam`, lr around 0.001)
- [ ] Optional extras: L2 regularization, dropout, learning-rate decay

### Day 7: Buffer and final review
- [ ] Test from a **fresh clone** in a clean environment
- [ ] Run the full pipeline: split -> train -> predict
- [ ] Final README with usage examples and a results graph
- [ ] Rehearse the defense once more
- [ ] Keep this day as slack

---

## Defense prep: concepts to explain without notes

| Concept | One-line explanation |
|---|---|
| **Feedforward** | Data flows input to output; each neuron computes a weighted sum plus bias, then applies an activation. |
| **Backpropagation** | The chain rule applied from the loss back through the layers to find each weight's contribution to the error. |
| **Gradient descent** | Move each weight opposite its gradient, scaled by the learning rate, to reduce the loss. |
| **Softmax** | Turns raw outputs into a probability distribution that sums to 1. |
| **Cross-entropy** | Penalizes confident wrong predictions heavily; pairs with softmax to give `dZ = A - Y`. |
| **Normalization** | Keeps features on the same scale so training is stable and fast. |
| **Validation set** | Measures performance on unseen data and reveals overfitting. |
| **Overfitting** | Training loss keeps falling while validation loss rises; visible on the curves. |

## Common pitfalls

| Problem | Likely fix |
|---|---|
| Loss is `nan` | Clip probabilities in the log, stabilize softmax, lower the learning rate |
| Stuck near 60% accuracy / loss near 0.69 | Missing normalization, or learning rate too small |
| Validation much worse than training | Overfitting: early stopping or regularization |
| Shape errors in backprop | Print shapes; keep the `(batch, features)` convention |
| `predict.py` results differ from training | Normalization stats not saved or not reused |
| Adam looks broken | Use lr around 0.001, not 0.0314; separate `m` and `v` buffers per parameter; apply bias correction |
| Early stopping saves a worse model | Restore the best weights before saving |

## Submission rules

- Only the work inside the Git repository is evaluated
- No Makefile needed (Python is not compiled)
- Double-check folder and file names before pushing
