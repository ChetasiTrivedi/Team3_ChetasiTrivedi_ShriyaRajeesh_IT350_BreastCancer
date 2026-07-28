# Hybrid Deep Learning and Machine Learning Framework for Breast Cancer Detection Using Infrared Thermography

A hybrid AI framework for breast cancer detection from infrared thermographic images, combining deep learning, handcrafted feature extraction, and ensemble machine learning.

This project explores how hybrid architectures can improve medical image classification on small, low-contrast thermographic datasets, where traditional CNNs alone often struggle.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Models Implemented](#models-implemented)
- [Performance Comparison](#performance-comparison)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Data Augmentation](#data-augmentation)
- [Evaluation Metrics](#evaluation-metrics)
- [Technologies Used](#technologies-used)
- [Key Takeaways](#key-takeaways)

---

## Project Overview

Breast cancer is one of the leading causes of death among women worldwide. Early detection significantly improves treatment success and survival rates.

Traditional screening methods such as mammography:

- are expensive,
- expose patients to radiation,
- and may not perform well for early-stage detection.

This project uses **Infrared Thermography** as a:

- non-invasive,
- radiation-free,
- cost-effective alternative.

The proposed system combines:

- Deep learning feature extraction
- Handcrafted texture features (HOG)
- Ensemble learning
- Attention mechanisms (CBAM)
- Knowledge distillation
- Few-shot learning

to improve detection accuracy and generalization on limited medical imaging data.

---

## Features

- Hybrid Deep Learning + Machine Learning pipeline
- CNN-based feature extraction
- HOG (Histogram of Oriented Gradients) feature fusion
- PCA-based dimensionality reduction
- Attention mechanism using CBAM (Convolutional Block Attention Module)
- Multiple ensemble learning strategies (stacking, snapshot)
- Few-shot learning implementation (Prototypical Networks)
- Knowledge distillation framework (teacher–student)
- Shared 5-fold cross-validation evaluation across all experiments
- Unified metric reporting: Accuracy, F1-score, Recall, Precision, Sensitivity, Specificity, AUROC
- Side-by-side performance comparison across all models

---

## Models Implemented

### 1. Baseline CNN (ResNet50)

Initial deep learning model trained directly on raw thermographic images, with no handcrafted features or ensembling.

- **Accuracy:** ~66%
- Suffered from severe overfitting on the small dataset.

### 2. CNN + HOG Hybrid Model

Combines deep CNN features with HOG handcrafted texture features, then reduces dimensionality with PCA before classification.

- **Accuracy:** ~92.45%

### 3. CBAM + LightGBM

Uses CBAM to focus the network on the most diagnostically relevant image regions, then feeds the resulting features into a LightGBM classifier.

- **Accuracy:** ~94.34%
- **ROC-AUC:** 0.974

### 4. SVM Classification

Tested across multiple kernels — Linear, RBF, Polynomial, and Sigmoid.

- **Best kernel:** RBF
- **Accuracy:** ~81.13%

### 5. Knowledge Distillation

A teacher–student setup where a large model transfers its learned representations to a lightweight student model.

- **Teacher:** EfficientNet + CBAM
- **Student:** LightGBM
- **Accuracy:** ~90.57%
- **ROC-AUC:** 0.988

### 6. Few-Shot Learning Ensemble

Designed for low-data regimes using Prototypical Networks on top of a pretrained ResNet feature extractor.

- **Accuracy:** ~86.79%
- **Cancer recall:** ~95% (high sensitivity to positive cases)

### 7. Snapshot Ensemble

Trains a single network but captures multiple "snapshots" during training (via cosine annealing) and ensembles them at inference time.

- **Accuracy:** ~90.57%

### 8. Stacking Ensemble (Best Model)

Combines predictions from multiple base learners using a meta-model.

- **Base models:** LightGBM, Random Forest, SVM, Logistic Regression
- **Meta-model:** combines base-learner outputs into a final prediction

**Best Result:**
- **Accuracy:** 96.23%
- **ROC-AUC:** 0.989
- **F1-score:** 0.95

---



---

## Dataset

- **Dataset used:** DMR-IR Thermal Breast Dataset (Breast Cancer Detection using Thermography)
- **Source:** Kaggle Public Dataset
- **Image type:** Infrared thermal images
- **Classes:**
  - Normal (Healthy)
  - Abnormal / Sick (Cancerous)

---

## Preprocessing

- Image resizing → 224 × 224
- Normalization
- Noise handling
- Feature extraction (CNN + HOG)
- PCA dimensionality reduction

---

## Data Augmentation

Applied transformations:

- Rotation
- Horizontal flipping
- Zoom transformations
- Brightness variation

**Observation:** Standard augmentation slightly *reduced* performance, since these transformations disrupted the thermal integrity of the images (thermal images encode temperature information spatially, so geometric/brightness distortions can introduce misleading signal).


---

## Evaluation Metrics

All experiments are scored with a shared, unified evaluation utility (5-fold reporting) covering:

- Accuracy (%)
- F1-score
- Recall
- Precision
- Sensitivity
- Specificity
- AUROC

This ensures every model — from the baseline CNN to the final stacking ensemble — is compared on a consistent, like-for-like basis.

---

## Technologies Used

- Python
- TensorFlow / Keras
- PyTorch
- OpenCV
- Scikit-learn
- LightGBM
- NumPy
- Pandas
- Matplotlib

---

## Key Takeaways

- Deep learning alone (baseline CNN) struggles on small, low-contrast thermographic datasets — hybrid approaches are essential.
- Combining handcrafted features (HOG) with learned CNN features consistently boosts accuracy over either approach alone.
- Attention mechanisms (CBAM) meaningfully improve both accuracy and ROC-AUC by focusing on diagnostically relevant regions.
- Ensemble stacking of diverse base learners (LightGBM, Random Forest, SVM, Logistic Regression) yields the best overall performance.
- Standard image augmentation can *hurt* performance on thermal data — domain-specific augmentation strategies are needed.
- Few-shot learning is a promising direction given the limited availability of labeled medical thermography data.
