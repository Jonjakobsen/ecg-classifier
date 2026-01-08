# ECG Abnormality Classification

End-to-end machine learning system for binary ECG abnormality classification, combining classical machine learning and deep learning approaches. The project is implemented as a production-ready Python package with a command-line interface (CLI) and Docker support, enabling reproducible inference on single ECG files.

---

## Project Overview

The goal of this project is to classify ECG recordings as **NORMAL** or **NOT NORMAL**. Two complementary model types are supported:

- **Logistic Regression** using handcrafted time–frequency features
- **GRU (Gated Recurrent Unit)** neural network operating on raw or minimally processed ECG signals

The project emphasizes not only model performance, but also **clean software architecture**, **reproducibility**, and **deployment-readiness**.

---

## Dataset

The models are trained on ECG data derived from **PTB-XL**, a large, publicly available clinical ECG dataset containing 12-lead recordings with expert annotations.

For this project:

- The task is simplified to **binary classification**:
  - `0 = NORMAL`
  - `1 = NOT NORMAL`
- Original multi-label diagnostic annotations are mapped to this binary target

The dataset is slighty imbalanced with approximately 57% of the samples labeled as "NOT NORMAL".

---

## Demo Data

Demo ECG files are provided as .csv and in wfdb-format. These "signals" are **synthetic and randomly generated** and are **not real patient data**.

These demo files exist solely to:

- Validate the end-to-end inference pipeline
- Test the CLI and Docker setup

They should **not** be interpreted as physiologically realistic ECGs and are **not suitable for model validation or performance claims**.

---

## Key Features

- End-to-end ML pipeline: data loading → preprocessing → training → evaluation → inference
- Multiple model backends (logistic regression and GRU)
- Model artifacts bundled with the package
- Command-line interface for inference
- Docker image for reproducible, environment-independent execution
- Clear separation between training code and inference package

---

## Repository Structure

```
ecg-classifier/
├── src/ecg_classifier/       # Python package
│   ├── cli.py                # CLI entry point
│   ├── inference.py          # Inference logic
│   ├── models/               # Model definitions (LogReg, GRU)
│   ├── io/                   # ECG loading (CSV, WFDB)
│   ├── artifacts/            # Trained model files (.joblib, .pt)
│   └── demo/                 # Synthetic demo ECG files in csv. and wfdb-format
├── scripts/                  # Training scripts and demo data generation
├── tests/                    # PowerShell-based smoke test verifying CLI functionality.
├── data/                     # Path to training data (data not included)
├── plots/                    # Model performance, training etc.
├── Dockerfile                # Docker image optimized for inference
├── pyproject.toml            # Packaging and dependencies
└── README.md
```

---

## Installation

### Local (virtual environment)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the package in editable mode for development and testing.

---

## Usage

## CLI Usage

The package exposes a command-line interface for running ECG classification either on user-provided data or on bundled demo data. The same interface is available both locally and when running inside Docker.

---

### Demo (recommended first run)

Run inference on synthetic ECG signals bundled with the package.  
This is the fastest way to verify that the full system works end-to-end.

#### Local execution

```bash
ecg-classifier demo
```

Optional arguments:

- `--format`: `wfdb` or `csv` (default: `wfdb`)
- `--model`: `logreg` or `gru` (default: `logreg`)

Example:

```bash
ecg-classifier demo --format csv --model gru
```

The demo uses **synthetic, randomly generated ECG signals** and exists solely to validate the CLI, model loading, and inference pipeline.

#### Docker execution

First, build the Docker image:

```bash
docker build -t ecg-classifier .
```

Run the demo inside Docker:

```bash
docker run --rm ecg-classifier demo
```

Example with explicit options:

```bash
docker run --rm ecg-classifier demo --format csv --model gru
```

Because the demo data is bundled inside the package, **no volume mounting is required** when running the demo in Docker.

---

### Inference on user-provided data

Run inference on ECG files supplied by the user.

#### Local execution

```bash
ecg-classifier run   --input path/to/ecgfile   --format csv   --model logreg
```

Arguments:

- `--input`: Path to an ECG file or directory
- `--format`: `wfdb` or `csv` (default: `wfdb`)
- `--model`: `logreg` or `gru` (default: `logreg`)

#### Docker execution

When running inference on user-provided data in Docker, the input data must be mounted into the container.

Example:

```bash
docker run --rm   -v /path/to/local/data:/data   ecg-classifier run   --input /data/test_ecg_12lead.csv   --format csv   --model logreg
```

In this example:

- `/path/to/local/data` is a directory on the host machine containing ECG files
- `/data` is the corresponding directory inside the container
- The `--input` path must refer to the container path, not the host path

---

### Output format

```text
ECG classification result
-------------------------
Label      : NORMAL
Confidence : 0.87
```

The confidence corresponds to the predicted class probability returned by the selected model.

---

## Note on Out-of-Distribution (OOD) Behavior

When running the demo using the logistic regression model, it can be observed that the model may confidently classify certain **synthetic or non-ECG-like signals** as **NORMAL**, even when the input clearly does not resemble a physiological ECG waveform.

This behavior is **expected** and represents a classical example of the **out-of-distribution (OOD) problem** in machine learning.

The logistic regression model is a *discriminative classifier* trained exclusively on in-distribution ECG data (i.e. real ECG recordings labeled as normal or abnormal). As a result, it is only capable of answering the question:

> “Does this input look more like NORMAL or NOT NORMAL ECG data, given the training distribution?”

It is **not designed to determine whether an input signal is an ECG at all**. When presented with arbitrary signals (e.g. noise or unrealistic synthetic data), the model is forced to extrapolate and will often assign a high-confidence prediction to one of the known classes.

This highlights an important system-level consideration for production machine learning systems:  
**robust input validation and OOD handling must be addressed explicitly and cannot be delegated to the classifier alone.**

### Potential mitigation strategies

Possible approaches to mitigate this issue include:

- **Signal sanity checks prior to inference**, such as amplitude range checks, RMS thresholds, dominant frequency analysis, or flatline detection.
- **Explicit OOD detection**, for example using a reconstruction-based model (e.g. an autoencoder) trained to model the ECG signal manifold and reject inputs with high reconstruction error.
- **Pipeline-level rejection**, where inputs failing validation are labeled as `INVALID` rather than being forced into a clinical classification.

In this project, the observed behavior is intentionally documented to illustrate a well-known limitation of discriminative models and to emphasize the importance of system design considerations beyond model accuracy.



---

## Model training

Model training is performed via standalone scripts and is intentionally kept **outside the package**.

Example:

```bash
pip install -e .
python -m scripts.train_logreg
python -m scripts.train_gru
```

Training outputs artifacts (e.g. `.joblib`, `.pt`) into `ecg_classifier/artifacts/` for inference.


---

## Model performance for logistic regression

The logistic regression model is trained on the training split and evaluated on a separate validation set for model comparison. Final performance is reported on a held-out test set using the same metrics as the GRU model to ensure fair comparison.

The performance on the test-set was:

**Test accuracy: 0.732**



## Training curves for GRU

The figure below shows the training and validation loss per epoch for the GRU-based ECG classifier.

![GRU loss curves](plots/gru_loss_curves.png)

- The initial loss plateau around 0.68 corresponds to a validation accuracy of approximately 57%, which matches the proportion of the “not normal” class in the dataset. At this stage, the model effectively predicts “not normal” for all samples, equivalent to a majority-class baseline.

- After approximately five epochs, the training loss begins to decrease steadily, indicating that the model starts to learn meaningful temporal patterns from the ECG signals.

- The validation loss follows a similar downward trend, suggesting that the learned representations generalize beyond the training data.

- From around epoch 24, the validation loss begins to plateau while the training loss continues to decrease, indicating the onset of overfitting.


The validation loss is used for **early stopping and model selection**.  
The saved model corresponds to the epoch with the lowest validation loss, rather than the final training epoch.

Loss curves are primarily used as a **diagnostic tool** to assess convergence behavior and overfitting, while final model performance is reported separately on a held-out test set. The performance on the test-set was:

**Test accuracy: 0.835**




## Design Principles

- Production-oriented ML system design
- Clear separation of concerns (training vs inference)
- Deterministic, reproducible inference
- Minimal, explicit dependencies


---


## Skills Demonstrated

System design for ML applications, data preprocessing and feature engineering, classical ML and deep learning, ML tooling, packaging, containerization in Docker, and production-oriented deployment practices.

---

## Future Work

This project is intentionally scoped with a primary focus on software engineering aspects of machine learning systems, including modular design, reproducibility, packaging, and deployment. Several natural extensions exist, particularly on the modeling side, which were deliberately deprioritized in favor of system-level concerns.

Potential future work includes:

- Systematic hyperparameter tuning
Both the logistic regression and GRU models could benefit from structured hyperparameter search (e.g. regularization strength, GRU hidden size, learning rate, batch size). This could be implemented via grid search or Bayesian optimization, with the existing train/validation split already supporting such workflows.

- Explicit regularization strategies
While some regularization is implicitly present (e.g. class weighting in logistic regression, early stopping in GRU training), further techniques such as L1/L2 penalties, dropout, or weight decay could be explored to improve generalization and model robustness.

- Cross-validation for classical models
For the logistic regression model in particular, k-fold cross-validation could be added to obtain more stable performance estimates and to decouple model selection more cleanly from the held-out test set.

- More advanced evaluation
While this project reports a limited set of core metrics for clarity, additional evaluation tools such as precision–recall curves or ROC AUC could be added in future work to further analyze model behavior, especially in the presence of class imbalance.


Overall, these extensions are technically straightforward within the current architecture. The existing codebase is designed to accommodate such improvements without major refactoring, underscoring the project’s emphasis on maintainable and extensible ML system design rather than exhaustive model optimization.

---


