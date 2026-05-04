# EEG-Based Sleep Stage Classification

**EE559 Course Project — Spring 2026**  
Authors: Shana Roshanghiyas, Pooyan Pahlavan

Automatic sleep stage classification from single-channel EEG signals using five classifiers built entirely from scratch in NumPy. The full pipeline covers data acquisition, signal filtering, feature engineering, class-imbalance handling, model training, cross-validation, PCA analysis, and detailed visualisation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Features](#features)
- [Classifiers](#classifiers)
- [Results & Outputs](#results--outputs)
- [Dependencies](#dependencies)
- [Notes](#notes)

---

## Overview

Sleep staging is the process of labelling each 30-second window of an overnight EEG recording as one of five clinical stages:

| Label | Stage | Description |
|-------|-------|-------------|
| W | Wake | Eyes open or closed, awake |
| N1 | NREM Stage 1 | Light sleep, transition |
| N2 | NREM Stage 2 | Sleep spindles, K-complexes |
| N3 | NREM Stage 3 | Slow-wave / deep sleep |
| REM | REM Sleep | Rapid eye movement, dreaming |

This project trains and compares five classifiers on a 20-dimensional hand-crafted feature set extracted from the Fpz-Cz EEG channel. All machine-learning components — scalers, SMOTE, classifiers, PCA, cross-validation — are implemented from scratch using only NumPy.

---

## Dataset

**Sleep-EDF Database Expanded** — publicly available on [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/).

- 8 subjects (Cassette recordings, age 25–34)
- Each subject: one full-night PSG recording + hypnogram annotation
- Sampling rate: 100 Hz
- Channel used: `EEG Fpz-Cz`
- Epoch length: 30 seconds → ~15,000 labelled epochs total

**No manual download required.** The notebook fetches the data automatically on first run via `mne.datasets.sleep_physionet` and caches it locally. Subsequent runs skip the download.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install mne scipy numpy pandas matplotlib ipywidgets
```

Or run the first cell of the notebook — it installs everything for you.

### 3. Open the notebook

```bash
jupyter notebook Sleep_Final_GitHub.ipynb
```

Then **run cells in order from top to bottom**. The first cell downloads the data (~200 MB) and all subsequent cells build on variables in memory.

> **GitHub Codespaces / JupyterHub:** open the `.ipynb` file directly in the file browser and run all cells. No extra setup needed.

---

## Project Structure

```
.
├── Sleep_Final_GitHub.ipynb   # Main notebook (run this)
├── README.md
└── outputs/                   # Auto-created; all figures saved here
    ├── fig1_train_confusion.png
    ├── fig2_test_confusion.png
    ├── fig3_train_vs_test.png
    ├── fig4_per_class_accuracy.png
    ├── fig5_training_loss.png
    ├── fig6_generalisation_gap.png
    ├── fig7_scree_plot.png
    ├── fig8_cumulative_variance.png
    ├── fig9_pca_scatter.png
    ├── fig10_pca_vs_accuracy.png
    ├── fig11_cv_results.png
    ├── fig12_cv_per_fold.png
    ├── figA_raw_eeg.png
    ├── figB_hypnogram.png
    ├── figC_psd.png
    ├── figD_class_dist.png
    ├── figE_pipeline.png
    ├── feature_importance.png
    ├── feature_profiles.png
    └── feature_distributions.png
```

---

## Pipeline

The notebook is divided into seven cells that must be run in order:

| Cell | Name | What it does |
|------|------|--------------|
| 0 | **Setup & Data Download** | Installs packages, downloads Sleep-EDF from PhysioNet via MNE |
| 1 | **Main Pipeline** | Feature extraction, train/test split, SMOTE, trains all 5 classifiers |
| 2 | **PCA + Cross-Validation** | PCA from scratch, 4-fold subject-wise CV |
| 3 | **Plotting Module** | Confusion matrices, accuracy/F1 bar charts, loss curves, generalisation gap |
| 4 | **Extra Figures** | Raw EEG epochs, hypnogram, PSD per stage, class distribution, pipeline diagram |
| 5 | **PCA & CV Plots** | Scree plot, cumulative variance, 2D scatter, accuracy vs. components, CV bars |
| 6 | **Feature Importance & Error Analysis** | Permutation importance, class-conditional heatmap, error breakdown |

---

## Features

Each 30-second epoch is described by **20 features** (no library models — all computed with NumPy/SciPy):

**Time-domain (10)**

| Feature | Description |
|---------|-------------|
| Mean | Average amplitude |
| Std Dev | Signal variability |
| Variance | Squared variability |
| RMS | Root mean square energy |
| Peak-to-Peak | Amplitude range |
| Skewness | Asymmetry of distribution |
| Kurtosis | Tailedness (excess) |
| Zero-Crossing Rate | Frequency of sign changes |
| Hjorth Mobility | Ratio of std of derivative to std of signal |
| Hjorth Complexity | Rate of change of the signal's slope |

**Frequency-domain (10)**

| Feature | Description |
|---------|-------------|
| δ ratio | Delta band power (0.5–4 Hz) / total |
| θ ratio | Theta band power (4–8 Hz) / total |
| α ratio | Alpha band power (8–12 Hz) / total |
| β ratio | Beta band power (12–30 Hz) / total |
| δ/θ | Delta-to-theta ratio |
| θ/α | Theta-to-alpha ratio |
| α/β | Alpha-to-beta ratio |
| Spectral Entropy | Shannon entropy of band power distribution |
| Dominant Frequency | Frequency with highest PSD (Welch) |
| Log Total Power | Log of summed band power |

---

## Classifiers

All five classifiers are implemented **from scratch** using only NumPy:

| Classifier | Key design choices |
|------------|--------------------|
| **k-NN (k=7)** | Euclidean distance, distance-weighted voting |
| **Softmax Regression** | Mini-batch SGD, L2 regularisation, learning-rate decay |
| **MLP (2 hidden layers)** | ReLU activations, He initialisation, SGD + momentum, weighted cross-entropy |
| **Linear SVM (OVR)** | One-vs-rest, sub-gradient SGD, hinge loss |
| **Gaussian Naive Bayes** | Class-conditional Gaussian, Laplace-smoothed priors |

**Class imbalance** is handled with a minimal SMOTE implementation: minority classes are oversampled by interpolating between random same-class neighbours until every class matches the majority count. SMOTE is applied only to the training fold to prevent data leakage.

**Train/test split** is done subject-wise (75/25) so no epochs from the same subject appear in both sets.

---

## Results & Outputs

Running all cells produces the following figures in `outputs/`:

**Model evaluation**
- `fig1` / `fig2` — Normalised confusion matrices for train and test sets
- `fig3` — Grouped bar chart: accuracy and macro-F1, train vs. test
- `fig4` — Per-class accuracy by sleep stage, train vs. test
- `fig5` — Training loss curves (Softmax and MLP)
- `fig6` — Generalisation gap (train error vs. test error)

**PCA analysis**
- `fig7` — Scree plot (variance per component)
- `fig8` — Cumulative explained variance with 90%/95% markers
- `fig9` — 2D PCA scatter coloured by sleep stage
- `fig10` — Classification accuracy and F1 as a function of number of PCA components
- `fig11` / `fig12` — 4-fold cross-validation bar charts and per-fold line plots

**Signal visualisation**
- `figA` — Representative 30-second EEG epoch for each sleep stage
- `figB` — Full-night hypnogram (colour-coded timeline)
- `figC` — Power spectral density per stage (Welch method)
- `figD` — Class distribution bar chart
- `figE` — End-to-end pipeline diagram

**Feature analysis**
- `feature_importance.png` — Permutation feature importance (Softmax, test set)
- `feature_profiles.png` — Class-conditional feature heatmap (z-scored)
- `feature_distributions.png` — Box plots of top 6 features by sleep stage

---

## Dependencies

| Package | Version tested | Purpose |
|---------|---------------|---------|
| `mne` | ≥ 1.0 | EDF file I/O, data download, epoching, filtering |
| `numpy` | ≥ 1.23 | All numerical computation |
| `scipy` | ≥ 1.9 | Welch PSD (`scipy.signal.welch`) |
| `matplotlib` | ≥ 3.6 | All figures |
| `pandas` | ≥ 1.5 | Minor tabular operations |
| `ipywidgets` | ≥ 7.0 | Notebook interactivity |

Install all at once:

```bash
pip install mne scipy numpy pandas matplotlib ipywidgets
```

---

## Notes

- **Run order matters.** Each cell depends on variables defined in earlier cells. Always run from Cell 0 downward; do not skip cells.
- **First run takes time.** MNE will download ~200 MB from PhysioNet. A progress bar is shown. After that, data is cached and instantly available.
- **Random seed is fixed** (`np.random.seed(42)`) for reproducibility.
- **All outputs** are saved to `outputs/` relative to the notebook. The folder is created automatically.
- **No GPU required.** All computation is CPU-based NumPy. A full run (all cells) takes approximately 20–40 minutes depending on hardware, dominated by the k-NN and SVM training on the balanced dataset.
