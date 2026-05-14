# Hybrid Deep Learning and Machine Learning Framework for Breast Cancer Detection Using Infrared Thermography

A hybrid AI-based framework for breast cancer detection using infrared thermographic images by combining deep learning, handcrafted feature extraction, and ensemble machine learning techniques.

This project explores how hybrid architectures can improve medical image classification performance on small and low-contrast thermographic datasets, where traditional CNNs often struggle.

---

# Project Overview

Breast cancer is one of the leading causes of death among women worldwide. Early detection significantly improves treatment success and survival rates.

Traditional screening methods such as mammography:

* are expensive,
* expose patients to radiation,
* and may not perform well for early-stage detection.

This project uses Infrared Thermography as a:

* non-invasive,
* radiation-free,
* cost-effective alternative.

The proposed system combines:

* Deep learning feature extraction
* Handcrafted texture features (HOG)
* Ensemble learning
* Attention mechanisms
* Knowledge distillation
* Few-shot learning

to improve detection accuracy and generalization.

---

# Features

* Hybrid Deep Learning + Machine Learning pipeline
* CNN-based feature extraction
* HOG (Histogram of Oriented Gradients) feature fusion
* PCA-based dimensionality reduction
* Attention mechanism using CBAM
* Ensemble learning approaches
* Few-shot learning implementation
* Knowledge distillation framework
* Snapshot ensemble training
* Cross-validation evaluation
* Performance comparison across models

---

# Models Implemented

## 1. Baseline CNN (ResNet50)

Initial deep learning model trained directly on thermographic images.

Result:

* Accuracy: ~60%

---

## 2. CNN + HOG Hybrid Model

Combined:

* Deep CNN features
* HOG handcrafted texture features
* PCA for dimensionality reduction

Result:

* Accuracy: ~92.45%

---

## 3. CBAM + LightGBM

Used:

* CBAM (Convolutional Block Attention Module)
* LightGBM classifier

Result:

* Accuracy: ~92.45%
* ROC-AUC: 0.975

---

## 4. SVM Classification

Tested multiple kernels:

* Linear
* RBF
* Polynomial
* Sigmoid

Best performing kernel:

* Sigmoid

Result:

* Accuracy: ~92.4%

---

## 5. Knowledge Distillation

Teacher-Student architecture:

* Teacher: EfficientNet + CBAM
* Student: LightGBM

Result:

* Accuracy: ~90.56%
* ROC-AUC: 0.986

---

## 6. Few-Shot Learning Ensemble

Implemented:

* Prototypical Networks
* Pretrained ResNet feature extractor

Result:

* Accuracy: ~90.5%
* High cancer recall (~95%)

---

## 7. Snapshot Ensemble

Used:

* Cosine annealing
* Multiple model snapshots during training

Result:

* Accuracy: ~88.6%

---

## 8. Stacking Ensemble (Best Model)

Base Models:

* LightGBM
* Random Forest
* SVM
* Logistic Regression

Meta-model combines predictions from all base learners.

Best Result:

* Accuracy: 96.23%
* ROC-AUC: 0.986
* F1-score: 0.96

---

# Dataset

## Dataset Used

DMR-IR Thermal Breast Dataset

Source:

* Kaggle Public Dataset

## Image Type

* Infrared Thermal Images

## Classes

* Normal (Healthy)
* Abnormal (Cancerous)

---

# Preprocessing

* Image resizing → 224 × 224
* Normalization
* Noise handling
* Feature extraction
* PCA dimensionality reduction

---

# Data Augmentation

Applied:

* Rotation
* Horizontal flipping
* Zoom transformations
* Brightness variation

Observation:

Standard augmentation slightly reduced performance because thermal integrity was affected.

| Model                | Accuracy |
| -------------------- | -------- |
| Without Augmentation | 90.56%   |
| With Augmentation    | 88.68%   |

---

# Performance Comparison

| Model                  | Accuracy | ROC-AUC | Key Insight                |
| ---------------------- | -------- | ------- | -------------------------- |
| Baseline CNN           | 60.0%    | 0.6875  | Severe overfitting         |
| XGBoost                | 90.56%   | 0.9288  | Strong ML baseline         |
| CNN + HOG              | 92.45%   | —       | Strong feature extraction  |
| CBAM + LightGBM        | 92.45%   | 0.975   | Excellent cancer recall    |
| Few-Shot Learning      | ~90.5%   | —       | Works well on limited data |
| Snapshot Ensemble      | ~88.6%   | —       | Stable but lower diversity |
| Knowledge Distillation | ~90.56%  | 0.986   | Better generalization      |
| Stacking Ensemble      | 96.23%   | 0.986   | Best overall model         |

---

# Technologies Used

* Python
* TensorFlow / Keras
* PyTorch
* OpenCV
* Scikit-learn
* LightGBM
* NumPy
* Pandas
* Matplotlib
