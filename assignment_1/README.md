# Feedforward Neural Networks for Tabular Data

**Course:** AI600 — Deep Learning, Spring 2026  
**Institution:** Lahore University of Management Sciences (LUMS)  
**Author:** Abdul Hafeez

---

## Overview
This assignment implements and analyzes a **two-hidden-layer feedforward neural network** architecture applied to tabular data. The objective is to build an MLP from scratch using only **NumPy** to classify NYC Airbnb listing prices into four categories: *Budget*, *Moderate*, *Premium*, and *Luxury*.

## Dataset

The dataset consists of **NYC Airbnb listings** with various features categorical and numeric such as neighbourhood, room type, minimum nights, and number of reviews.

| File | Description |
|------|-------------|
| `data/train.csv` | Training set |
| `data/test.csv` | Test set |

## Architecture

The MLP consists of:

| Layer | Neurons | Activation |
|-------|---------|------------|
| Input | 17 (# features) | — |
| Hidden 1 | 64 | ReLU |
| Hidden 2 | 32 | ReLU |
| Output | 4 (classes) | Softmax |


## Project Structure

```
assignment_1/
├── data/
│   ├── train.csv
│   └── test.csv
├── docs/
│   └── AI600 - Assignment 1.pdf      # Assignment prompt
├── notebooks/
│   └── mlp.ipynb                     # EDA, training, and analysis notebook
├── outputs/
│   ├── bivariate_categorical.png
│   ├── bivariate_numeric.png
│   ├── correlation_matrix.png
│   ├── feature_importance_aggarwal.png
│   ├── gradient_magnitude_comparison.png
│   ├── mlp_activation_comparison.png
│   ├── numeric_distributions.png
│   ├── numeric_qq_plots.png
│   ├── skewness_comparison.png
│   └── target_distribution.png
├── src/
│   └── mlp.py                        # NumPy MLP implementation
└── README.md
```

## How to Run

1. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn jupyter scikit-learn scipy
   ```

2. **Run the notebook:**
   ```bash
   jupyter notebook notebooks/mlp.ipynb
   ```

   The notebook covers the full pipeline — EDA, preprocessing, model training, gradient analysis, and evaluation.

3. **Use the MLP module directly:**
   ```python
   from src.mlp import TwoLayerMLP

   model = TwoLayerMLP(M0=n_features, M1=64, M2=32, M3=4, activation='relu')
   history = model.train(X_train, y_train, X_val, y_val, iterations=200)
   ```
