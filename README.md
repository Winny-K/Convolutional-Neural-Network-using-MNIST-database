# Convolutional Neural Network — MNIST Digit Recognition

A PyTorch implementation of a Convolutional Neural Network (CNN) trained on the MNIST dataset to classify handwritten digits (0–9).

---

## 🧠 Model Architecture

The model consists of two convolutional layers followed by three fully connected layers:

```
Input (1x28x28)
  → Conv2d(1, 6, kernel=3) + ReLU + MaxPool2d
  → Conv2d(6, 16, kernel=3) + ReLU + MaxPool2d
  → Flatten
  → Linear(400, 120) + ReLU
  → Linear(120, 84)  + ReLU
  → Linear(84, 10)
  → LogSoftmax
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Epochs | 5 |
| Batch Size | 10 |
| Optimizer | Adam (lr=0.0009) |
| Loss Function | CrossEntropyLoss |
| Test Accuracy | ~98% |

Training and testing loss/accuracy are plotted per epoch for visual evaluation.

---

## 📁 Project Structure

```
├── CNN_MNIST.ipynb       # Main notebook
├── CNN_data/             # Auto-downloaded MNIST dataset
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision scikit-learn matplotlib numpy pandas
```

### Run

Open and run the Jupyter notebook:

```bash
jupyter notebook CNN_MNIST.ipynb
```

The MNIST dataset will be automatically downloaded on first run.

---

## 📈 Sample Outputs

- **Loss per Epoch** — training vs testing loss curve
- **Accuracy per Epoch** — training vs testing accuracy curve
- **Single Image Prediction** — passes a sample test image through the model and returns the predicted digit

---

## 🛠️ Built With

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/)
- [Scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)

---
