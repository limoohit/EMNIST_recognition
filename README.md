# EMNIST Handwritten Character Recognition & Secret Message Decoding

## 📝Project Overview | 项目概述
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten letters from the EMNIST dataset. Beyond simple classification, it includes a specialized script to decode hidden messages embedded in pixel data, demonstrating the practical application of deep learning in pattern recognition.


---

## 🚀 Key Features | 项目亮点
* **High Performance**: Achieved a validation accuracy of **81.28%** after 14 epochs of training.
* **Automated Decoding**: Successfully decoded a hidden message from `hidden_message_images.pkl` using the trained model.


---

## 📂 File Structure | 文件结构
* `emnist_classifier.py`: Main script containing model architecture, training loop, and decoding logic.
* `helper_utils.py`: Utility functions for data visualization and preprocessing.
* `hidden_message_images.pkl`: The encrypted pixel data for the secret message.
* `.gitignore`: Configured to exclude large datasets (e.g., `EMNIST_data/`) to keep the repository clean.

---
## 📊 Dataset: EMNIST (Extended MNIST) | 数据集说明

This project utilizes the **EMNIST (Extended MNIST)** dataset, specifically targeting handwritten alphabets. 
* **Data Acquisition**: The dataset is automatically downloaded and loaded via `torchvision.datasets.EMNIST`.
* **Exclusion Note**: To adhere to best engineering practices, the raw `EMNIST_data/` directory is excluded via `.gitignore` to maintain a lightweight repository footprint.
  
---
## 🛠️ Environment & Setup | 环境配置
* **Operating System**: Windows 10/11
* **Python Version**: `3.11+`
* **Deep Learning Framework**: **PyTorch `2.7.0+cu118`** (CUDA 11.8 enabled)
* **Computer Vision**: **Torchvision `0.22.0+cu118`**
* **Key Dependencies**: `numpy`, `matplotlib` (for CCN-style data visualization)
