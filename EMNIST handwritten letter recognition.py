import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 导入 Coursera 提供的辅助工具（需确保 helper_utils.py 在同一目录下）
try:
    import helper_utils
except ImportError:
    print("Warning: 'helper_utils.py' not found. Decoding features will be disabled.")

# --- 全局配置 (Hyperparameters & Config) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_CLASSES = 26  # EMNIST Letters: a-z
LEARNING_RATE = 0.001
NUM_EPOCHS = 14   # 恢复为你原本设定的 14 轮，以保证 89%+ 的测试集准确率

def get_dataloaders(data_path='./EMNIST_data', batch_size=BATCH_SIZE):
    """
    加载 EMNIST Letters 数据集，应用归一化处理，并返回 DataLoaders。
    """
    # EMNIST Letters 数据集的预计算均值和标准差
    mean = (0.1736,)
    std = (0.3317,)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    download = not os.path.exists(data_path)
    if download:
        print("Downloading EMNIST dataset...")

    train_dataset = datasets.EMNIST(root=data_path, split='letters', train=True, download=download, transform=transform)
    test_dataset = datasets.EMNIST(root=data_path, split='letters', train=False, download=download, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def initialize_model(num_classes=NUM_CLASSES):
    """
    初始化 MLP 模型架构、损失函数和优化器 (Adam)
    架构: Flatten -> Linear(256) -> ReLU -> Linear(128) -> ReLU -> Linear(num_classes)
    """
    torch.manual_seed(42) # 固定随机种子以保证结果可复现

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

    loss_function = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, loss_function, optimizer

def train_epoch(model, loss_function, optimizer, train_loader, device):
    """
    训练模型一个 Epoch，返回平均损失和准确率
    """
    model.to(device)
    model.train()
    
    running_loss = 0.0
    num_correct_predictions = 0
    total_predictions = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # EMNIST Letters 的标签是 1-26，需减 1 调整为 0-25 以匹配 CrossEntropyLoss
        targets = targets - 1  

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        # 统计 Loss 
        running_loss += loss.item()

        # 统计正确率
        predicted_indices = outputs.argmax(dim=1)
        num_correct_predictions += predicted_indices.eq(targets).sum().item()
        total_predictions += targets.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = (num_correct_predictions / total_predictions) * 100
    return avg_loss, accuracy

def evaluate(model, test_loader, device):
    """
    在测试集上评估模型，返回准确率
    """
    model.eval()
    num_correct_predictions = 0
    total_predictions = 0 

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets - 1

            outputs = model(inputs)
            predicted_indices = outputs.argmax(dim=1)
            num_correct_predictions += predicted_indices.eq(targets).sum().item()
            total_predictions += targets.size(0)

    accuracy = (num_correct_predictions / total_predictions) * 100
    return accuracy

def decode_secret_message(model, device):
    """
    加载神秘信件图像，使用训练好的模型进行字母预测并拼接成句子。
    展示模型在实际手写场景中的泛化能力。
    """
    print("\n--- Decoding Secret Message ---")
    try:
        message_imgs = helper_utils.load_hidden_message_images()
    except Exception as e:
        print(f"Failed to load hidden message images: {e}")
        return

    model.eval()
    with torch.no_grad():
        for sentence_imgs in message_imgs:
            decoded_sentence = []
            for word_imgs in sentence_imgs:
                decoded_chars = []
                for char_img in word_imgs:
                    char_img = char_img.unsqueeze(0).to(device)
                    output = model(char_img)
                    predicted_label = output.argmax(dim=1).item()
                    
                    # 将 0-25 的预测索引映射回 'a'-'z'
                    lowercase_char = chr(ord("a") + predicted_label)
                    decoded_chars.append(lowercase_char)
                    
                decoded_word = "".join(decoded_chars)
                decoded_sentence.append(decoded_word)
            
            print(" ".join(decoded_sentence))
    print("-------------------------------\n")

def main():
    print(f"Starting EMNIST Letter Recognition Pipeline on: {DEVICE}")

    # 1. 加载与预处理数据
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # 2. 初始化模型架构
    model, loss_function, optimizer = initialize_model(num_classes=NUM_CLASSES)

    # 3. 训练与验证循环
    print(f"Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, loss_function, optimizer, train_loader, DEVICE)
        test_acc = evaluate(model, test_loader, DEVICE)

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

    # 4. 保存模型权重 (专业做法)
    save_path = 'trained_emnist_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel training complete. Weights saved to '{save_path}'")

    # 5. 模型应用测试：解码神秘信件！
    if 'helper_utils' in globals():
        decode_secret_message(model, DEVICE)

if __name__ == "__main__":
    main()