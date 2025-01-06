import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from math import ceil
from torch.utils.data import DataLoader, TensorDataset

# 定义神经网络
class PickupPointPredictor(nn.Module):
    def __init__(self):
        super(PickupPointPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),  # 输入为4个值
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # 输出为2个值 (row, col)
        )

    def forward(self, x):
        return self.fc(x)

# 自定义损失函数 (欧几里得距离)
def euclidean_loss(predicted, target):
    return torch.sqrt(((predicted - target) ** 2).sum(dim=1)).mean()

# 数据准备
def prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    df['taxi_location_rowcol'] = df['taxi_location_rowcol'].apply(eval)
    df['user_location_rowcol'] = df['user_location_rowcol'].apply(eval)
    df['pick_up_location_rowcol'] = df['pick_up_location_rowcol'].apply(eval)

    # 准备输入 (taxi_row, taxi_col, user_row, user_col)
    X = df.apply(lambda row: row['taxi_location_rowcol'] + row['user_location_rowcol'], axis=1)
    y = df['pick_up_location_rowcol']

    # 分割数据集 (75% 训练集, 25% 验证集)
    X_train, X_val, y_train, y_val = train_test_split(
        X.tolist(), y.tolist(), test_size=0.25, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    return X_train, X_val, y_train, y_val

# 模型训练
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                outputs = model(X_val_batch)
                val_loss += criterion(outputs, y_val_batch).item()

                predicted = torch.round(outputs)
                total += y_val_batch.size(0)
                correct += ((torch.abs(predicted - y_val_batch) < 0.5).all(dim=1)).sum().item()

        val_accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")

# 主函数
if __name__ == "__main__":
    csv_file = "train_data_rowcol.csv"
    X_train, X_val, y_train, y_val = prepare_data(csv_file)
    print("read file")

    # 构造数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("load_data")

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PickupPointPredictor().to(device)
    criterion = euclidean_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("define_model")

    # 训练模型
    train_model(model, criterion, optimizer, train_loader, val_loader, epochs=50, device=device)
    print("model_trained！")
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, "model.pth")
    print("Model saved to model.pth")