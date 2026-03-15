# BUSI Ultrasound Classification with Imbalance Handling Techniques

This project focuses on the classification of breast ultrasound images using deep learning. The goal is to identify three categories of breast ultrasound images:

- **Benign**
- **Malignant**
- **Normal**

The project uses the **BUSI (Breast Ultrasound Images) Dataset** and implements a **MobileNetV2-based transfer learning model**. In addition to the baseline model, several techniques are explored to address **class imbalance**, including **class weighting, data augmentation, and SMOTE**.

---

## Project Objective

Breast cancer is one of the most common cancers among women worldwide. Early detection plays a critical role in improving treatment outcomes. Ultrasound imaging is widely used for breast cancer screening because it is safe, affordable, and effective.

The main objectives of this project are:

- Build a deep learning model for breast ultrasound image classification
- Use transfer learning with MobileNetV2
- Investigate the impact of class imbalance on model performance
- Compare different techniques for handling class imbalance
- Evaluate the effectiveness of each technique

---

## Dataset

The project uses the **Breast Ultrasound Images Dataset (BUSI)**.

Each sample contains:
- Ultrasound image
- Segmentation mask

For this project, **only the ultrasound images are used**. Mask images (`_mask.png`) are removed to avoid information leakage.

### Dataset Classes

| Class | Description |
|------|-------------|
| Benign | Non-cancerous tumors |
| Malignant | Cancerous tumors |
| Normal | Healthy breast tissue |

### Dataset Distribution

| Class | Number of Images |
|------|------------------|
| Benign | 437 |
| Malignant | 210 |
| Normal | 133 |
| **Total** | **780** |

The dataset is **imbalanced**, which motivates the use of imbalance handling techniques.

---

## Methodology

The project pipeline consists of the following steps:

1. Dataset preprocessing
2. Removing segmentation mask images
3. Train-test split
4. Image preprocessing and normalization
5. Transfer learning using MobileNetV2
6. Model training using different imbalance handling techniques
7. Performance comparison

---

## Experiments

Four different approaches were implemented and compared.

### 1. Baseline Model
The MobileNetV2 model is trained on the dataset **without applying any imbalance handling techniques**.

This experiment serves as a reference to observe how class imbalance affects classification performance.

---

### 2. Class Weight Balancing
Class weights are computed based on the class distribution and applied during model training.

This approach increases the penalty for misclassifying minority classes, encouraging the model to learn better representations for underrepresented categories.

---

### 3. Data Augmentation
Data augmentation is used to increase the diversity of the training data. The following transformations are applied:

- Small rotations
- Zooming
- Horizontal flipping

These transformations help improve generalization and reduce overfitting.

---

### 4. SMOTE Oversampling
SMOTE (Synthetic Minority Oversampling Technique) is applied to **feature representations extracted from MobileNetV2**.

SMOTE generates synthetic samples for minority classes, creating a balanced feature space that improves classifier learning.

---

## Model Architecture

The classification model uses **MobileNetV2** with transfer learning.

Main components:

- Pretrained **MobileNetV2 backbone**
- **Global Average Pooling**
- Dense layer with ReLU activation
- Softmax output layer for 3-class classification

---

## Results

The performance of different techniques was compared using classification accuracy.

| Method | Accuracy |
|------|----------|
| Baseline Model | 76.3% |
| Class Weight Balancing | 80.8% |
| Data Augmentation | 80.1% |
| SMOTE Oversampling | 97.4% |

### Key Observations

- The **baseline model** performs reasonably well due to transfer learning.
- **Class weights** improve performance by addressing dataset imbalance.
- **Data augmentation** increases dataset diversity and improves generalization.
- **SMOTE-based balancing** achieves the highest accuracy by generating synthetic minority samples in feature space.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Imbalanced-learn (SMOTE)
- NumPy
- Matplotlib
- Pandas

---

