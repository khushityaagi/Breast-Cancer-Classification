# Breast Cancer Classification using Deep Learning

## Project Summary

Early detection of breast cancer significantly improves survival rates. This project focuses on applying deep learning techniques to automatically classify breast tumor images into **benign** and **malignant** categories.

The main objective was to design, train, and compare multiple state-of-the-art deep learning architectures to evaluate their effectiveness in medical image classification and identify the most reliable model for accurate diagnosis support.

---

## Datasets Used

To ensure diverse learning, the models were trained and evaluated using three well-known medical imaging datasets:

### BreakHis Dataset
- **Type:** Histopathology images  
- **Classes:** Benign (0) | Malignant (1)  
- **Usage:** Used with ResNet-50 and Vision Transformer to analyze tissue-level cancer patterns.

---

### DDSM Dataset
- **Type:** Mammogram images  
- **Classes:** Benign (0) | Malignant (1)  
- **Usage:** Used for training CNN-based architectures including Custom VGG and EfficientNet.

---

### INbreast Dataset
- **Type:** Mammography images  
- **Classes:** Benign (0) | Malignant (1)  
- **Usage:** Evaluated using both CNN and transformer-based approaches for comparative analysis.

> Due to large file sizes, datasets are not included in this repository. Download links are provided separately.

---

## Data Preparation & Processing

Key preprocessing steps performed:

- Image resizing and normalization  
- Data augmentation to improve generalization  
- Handling class imbalance using class weights  
- Training–validation split for evaluation  
- Early stopping to prevent overfitting  

---

## Deep Learning Models Implemented

### ResNet-50 (Transfer Learning)

A pretrained convolutional neural network fine-tuned for binary classification.

**Observation:** High training accuracy but noticeable overfitting during validation.

---

### EfficientNetB3

A lightweight and efficient architecture optimized for feature extraction.

**Observation:** Strong training performance with moderate generalization ability.

---

### Vision Transformer (ViT) — Best Performing Model

An attention-based architecture that captures global image relationships using patch embeddings.

**Performance Achieved:**
- **Accuracy:** 88.51%
- **F1 Score:** 0.87

---

### Custom VGG-Based CNN

A tailored CNN architecture with dropout, batch normalization, and regularization techniques.

**Observation:** Stable and consistent performance across both training and validation datasets.

---

## Results & Model Performance

### Vision Transformer Results

<p align="center">
  <img src="Results/VIT_result.png" width="300">
  <br>
  <img src="Results/Vision_Transformer_matrix.png" width="280">
</p>

---

### ResNet-50 Results

<p align="center">
  <img src="Results/Resnet.png" width="300">
  <br>
  <img src="Results/Resnet_confusion_matrix.png" width="280">
</p>

---

### Custom VGG Results

<p align="center">
  <img src="Results/VGG_RESULT.png" width="300">
  <br>
  <img src="Results/VGG_matrix.png" width="280">
</p>

---

### EfficientNet Results

<p align="center">
  <img src="Results/EfficientNet.png" width="300">
</p>

---

## Performance Comparison

| Model | Key Insight |
|------|-------------|
| ResNet-50 | Overfitting observed |
| EfficientNetB3 | High training accuracy |
| Vision Transformer | Best accuracy & generalization |
| Custom VGG | Stable and reliable performance |

---

## Technologies & Tools

Python • TensorFlow • Keras • OpenCV • NumPy • Pandas • Matplotlib • Scikit-learn

---

## Key Takeaways

- Deep learning models can effectively classify breast cancer from medical images.  
- Transformer-based architectures showed superior ability in capturing complex visual patterns.  
- Proper preprocessing and augmentation significantly improved performance.  
- Comparative analysis helped identify the most reliable architecture.

---

## Author

**Khushi Tyagi**  
B.Tech — Data Science
