# Breast Cancer Classification Using Deep Learning

## Overview

This project focuses on the automated classification of breast cancer into benign and malignant categories using deep learning techniques applied to medical imaging data. The objective is to evaluate and compare multiple neural network architectures for accurate tumor classification.

---

## Datasets

The study utilizes three publicly available medical imaging datasets:

### BreakHis Dataset
- Type: Histopathology images  
- Classes: Benign (0), Malignant (1)  
- Application: Used for training and evaluation with ResNet-50 and Vision Transformer models.

---

### DDSM Dataset
- Type: Mammogram images  
- Classes: Benign (0), Malignant (1)  
- Application: Used with Custom VGG, EfficientNet, and Vision Transformer architectures.

---

### INbreast Dataset
- Type: Mammography images  
- Classes: Benign (0), Malignant (1)  
- Application: Evaluated using CNN and transformer-based models.

*Note: Datasets are not included due to large size. Download links are provided separately.*

---

## Data Processing

- Image resizing and normalization  
- Data augmentation techniques  
- Class imbalance handling  
- Training–validation splitting  
- Early stopping and checkpointing  

---

## Models Implemented

### ResNet-50 (Transfer Learning)
- Pretrained on ImageNet  
- Fine-tuned top layers  
- Result: High training accuracy with validation overfitting.

---

### EfficientNetB3
- Transfer learning approach  
- Optimized using Adam optimizer  
- Result: Strong training performance with reduced generalization on unseen data.

---

### Vision Transformer (ViT)
- Patch-based attention architecture  
- Captures global image relationships effectively  

**Performance:**
- Accuracy: **88.51%**  
- F1 Score: **0.87**

This model achieved the best overall results.

---

### Custom VGG-Based CNN
- Three convolution layers with regularization techniques  
- Includes dropout and batch normalization  
- Result: Stable performance across training and validation sets.

---

## Results

| Model | Key Observation |
|------|----------------|
| ResNet-50 | Overfitting observed |
| EfficientNetB3 | High training accuracy |
| Vision Transformer | Best overall performance |
| Custom VGG | Stable validation accuracy |

---

## Technologies Used

Python, TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn

---

## Conclusion

The project demonstrates the effectiveness of deep learning methods for medical image classification. Among the evaluated architectures, the Vision Transformer provided the highest accuracy and showed superior capability in capturing complex image patterns. Proper preprocessing and augmentation significantly improved model performance.

---

## Author

Khushi Tyagi  
B.Tech – Data Science
