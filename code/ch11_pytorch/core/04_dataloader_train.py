# ---
# chapter: 11
# topic: 深度学习与PyTorch
# section: 11.2.4 DataLoader 与训练流程
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: torch, scikit-learn, numpy
# run: python 04_dataloader_train.py
# expected_runtime: 30-90s (10 epochs on 5000 samples, CPU)
# expected_output: 10 行 epoch 日志, 最终 train/test loss 与 accuracy
# ---
# See: ../tutorial/11_深度学习与PyTorch.md#11.2.4-dataloader-与训练流程
#
# Interview hooks:
#  1. 训练循环中, optimizer.zero_grad() 为什么必须显式调用? PyTorch 不自动清零的原因?
#  2. model.train() 与 model.eval() 影响哪些层的行为 (BN/Dropout)?
#  3. @torch.no_grad() 装饰器相对 with no_grad 的优势? 何时仍需 torch.inference_mode?
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ========== 本地定义的简单模型 (与 03_nn_module_mlp.py 共享相同结构) ==========
class MLPClassifier(nn.Module):
    """简洁 MLP 分类器 — 避免跨文件 import."""
    def __init__(self, input_dim, hidden_dim=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ========== 自定义数据集 ==========
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========== 标准训练循环模板 ==========
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # 1. 清零梯度
        optimizer.zero_grad()

        # 2. 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 3. 反向传播
        loss.backward()

        # 4. 参数更新
        optimizer.step()

        # 统计
        total_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        total_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / total, correct / total

# ========== 完整训练流程 ==========
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # 数据
    X, y = make_classification(n_samples=5000, n_features=20, n_classes=3,
                               n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 模型、损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim=20, hidden_dim=128, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()  # 内含 Softmax
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, "
              f"Test Acc={test_acc:.4f}")
    print("OK")
