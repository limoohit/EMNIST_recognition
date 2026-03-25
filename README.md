# EMNIST Handwritten Character Recognition & Secret Message Decoding

## 📝Project Overview | 项目概述
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten letters from the EMNIST dataset. Beyond simple classification, it includes a specialized script to decode hidden messages embedded in pixel data, demonstrating the practical application of deep learning in pattern recognition.

本项目使用 PyTorch 框架构建了一个卷积神经网络 (CNN)，用于对 EMNIST 数据集中的手写字母进行分类。除了基础的分类任务，本项目还实现了一个特殊的脚本，用于解码嵌入在像素数据中的神秘信息，展示了深度学习在模式识别中的实际应用。

---

## 🚀 Key Features | 项目亮点
* **High Performance**: Achieved a validation accuracy of **81.28%** after 14 epochs of training.
* **Automated Decoding**: Successfully decoded a hidden message from `hidden_message_images.pkl` using the trained model.


---

## 📂 File Structure | 文件结构
* `emnist_classifier.py`: Main script containing model architecture, training loop, and decoding logic.
    （主程序：包含模型架构、训练循环及解码逻辑）
* `helper_utils.py`: Utility functions for data visualization and preprocessing.
    （工具类：用于数据可视化和预处理）
* `hidden_message_images.pkl`: The encrypted pixel data for the secret message.
    （数据源：神秘信息的原始像素数据）
* `.gitignore`: Configured to exclude large datasets (e.g., `EMNIST_data/`) to keep the repository clean.
    （配置：排除大型数据集，保持仓库整洁专业）

---

## 🛠️ Environment & Setup | 环境配置
* **Operating System**: Windows 10/11
* **Python Version**: `3.11+`
* **Deep Learning Framework**: **PyTorch `2.7.0+cu118`** (CUDA 11.8 enabled)
* **Computer Vision**: **Torchvision `0.22.0+cu118`**
* **Key Dependencies**: `numpy`, `matplotlib` (for CCN-style data visualization)
