#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN水体分类模型训练 - 使用AlphaEarth特征
基于geeCNN_Water_Classification.ipynb，适配3x3窗口和新数据格式
根据项目规则，使用简单直接的代码
"""

import torch
import random
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 设置随机种子确保可重现性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检查CUDA可用性并设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 设置默认数据类型
torch.set_default_dtype(torch.float32)

# 设置matplotlib字体为Arial
plt.rcParams['font.family'] = 'Arial'

print("开始训练CNN水体分类模型...")
print("使用AlphaEarth特征和5x5窗口")

# 读取数据
print("\n1. 读取数据...")
data = pd.read_csv("data/water_CNN_with_AlphaEarth_all_merged.csv")
print(f"数据形状: {data.shape}")
print(f"数据列: {list(data.columns)}")

# 检查标签分布
landcover_counts = data['landcover'].value_counts()
print(f"\n原始标签分布:")
for label, count in landcover_counts.items():
    percentage = (count / len(data)) * 100
    print(f"  类别 {label}: {count} 个样本 ({percentage:.1f}%)")

# 将标签重新映射：0->0 (非水体), 11->1 (水体)
print("\n重新映射标签: 0->0 (非水体), 11->1 (水体)")
data['landcover_binary'] = data['landcover'].map({0: 0, 11: 1})

# 检查重新映射后的标签分布
binary_counts = data['landcover_binary'].value_counts()
print(f"\n重新映射后标签分布:")
for label, count in binary_counts.items():
    label_name = "非水体" if label == 0 else "水体"
    percentage = (count / len(data)) * 100
    print(f"  类别 {label} ({label_name}): {count} 个样本 ({percentage:.1f}%)")

# 数据划分：70%训练，30%验证
print("\n2. 数据划分...")
train_data = data[data['random'] <= 0.7]
val_data = data[data['random'] > 0.7]
print(f"训练集大小: {len(train_data)}")
print(f"验证集大小: {len(val_data)}")

# 特征列：包含AlphaEarth和Sentinel-2波段
feature_columns = ['A31', 'A36', 'A46', 'A47', 'A63', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
print(f"\n使用特征: {feature_columns}")
print(f"特征数量: {len(feature_columns)}")

# 处理训练数据
print("\n3. 处理训练数据...")
train_features_list = []
for col in feature_columns:
    col_data = np.array(train_data[col].apply(ast.literal_eval).values.tolist())
    train_features_list.append(col_data)

# 堆叠特征 (样本数, 特征数, 3, 3) - 注意这里改为3x3
train_features = np.stack(train_features_list, axis=1)
print(f"训练特征形状: {train_features.shape}")

# 数据是5x5窗口，保持原始尺寸
print(f"使用5x5窗口，训练特征形状: {train_features.shape}")

train_features = torch.tensor(train_features, dtype=torch.float32)
train_labels = torch.tensor(train_data['landcover_binary'].values, dtype=torch.long)

# 处理验证数据
print("\n4. 处理验证数据...")
val_features_list = []
for col in feature_columns:
    col_data = np.array(val_data[col].apply(ast.literal_eval).values.tolist())
    val_features_list.append(col_data)

val_features = np.stack(val_features_list, axis=1)
print(f"验证特征形状: {val_features.shape}")

# 验证数据也是5x5窗口
print(f"验证特征形状: {val_features.shape}")

val_features = torch.tensor(val_features, dtype=torch.float32)
val_labels = torch.tensor(val_data['landcover_binary'].values, dtype=torch.long)

# 创建数据加载器
print("\n5. 创建数据加载器...")
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"批次大小: {batch_size}")
print(f"训练批次数: {len(train_loader)}")
print(f"验证批次数: {len(val_loader)}")

# 定义CNN模型 - 适配5x5输入和15个特征通道，参考原始geeCNN架构
print("\n6. 定义CNN模型...")
class WaterCNN(nn.Module):
    def __init__(self, num_features=15, num_classes=2):
        super(WaterCNN, self).__init__()

        # 参考原始geeCNN架构，适配5x5窗口
        self.conv1 = nn.Conv2d(in_channels=num_features, out_channels=16, kernel_size=(3, 3), padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0)
        self.conv3 = nn.Conv2d(in_channels=32 + num_features, out_channels=64, kernel_size=(1, 1))  # 连接中心1x1
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=(1, 1))

    def forward(self, x):
        # x shape: (batch, 15, 5, 5)
        x1 = F.relu(self.conv1(x))  # (batch, 16, 3, 3) - 5x5 -> 3x3
        x2 = F.relu(self.conv2(x1))  # (batch, 32, 1, 1) - 3x3 -> 1x1

        # 取原始输入的中心1x1区域
        center_1x1 = x[:, :, 2:3, 2:3]  # (batch, 15, 1, 1) - 从5x5中取中心

        # 连接特征
        x3_input = torch.cat((x2, center_1x1), dim=1)  # (batch, 32+15=47, 1, 1)
        x3 = F.relu(self.conv3(x3_input))  # (batch, 64, 1, 1)
        x4 = F.relu(self.conv4(x3))  # (batch, 128, 1, 1)
        x5 = self.conv5(x4)  # (batch, 2, 1, 1)

        # 展平为分类输出
        return x5.view(x5.size(0), -1)  # (batch, 2)

# 创建模型并移到GPU
model = WaterCNN(num_features=len(feature_columns), num_classes=2)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# 确保模型完全移动到设备
model = model.to(device)
print(f"模型已移动到设备: {device}")

# 检查模型参数设备
for name, param in model.named_parameters():
    if param.device != device:
        print(f"警告: 参数 {name} 在设备 {param.device}，应该在 {device}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print(f"损失函数: {criterion}")
print(f"优化器: {optimizer}")

# 训练模型
print("\n7. 开始训练...")
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_acc = 0.0
best_model_state = None

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
    
    # 计算平均损失和准确率
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
    
    # 更新学习率
    scheduler.step()
    
    # 打印进度
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")

# 加载最佳模型进行最终评估
model.load_state_dict(best_model_state)

# 最终评估
print("\n8. 最终评估...")
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

# 计算评估指标
final_accuracy = accuracy_score(all_targets, all_predictions)
print(f"最终验证准确率: {final_accuracy:.4f}")

print("\n分类报告:")
print(classification_report(all_targets, all_predictions, 
                          target_names=['非水体', '水体']))

# 保存模型
model_path = "model/water_cnn_alpha_model.pth"
import os
os.makedirs("model", exist_ok=True)
torch.save(best_model_state, model_path)
print(f"\n模型已保存到: {model_path}")

# 绘制训练曲线
print("\n9. 绘制训练曲线...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(train_losses, label='Training Loss', color='blue')
ax1.plot(val_losses, label='Validation Loss', color='red')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# 准确率曲线
ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('model/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制混淆矩阵
print("\n10. 绘制混淆矩阵...")
cm = confusion_matrix(all_targets, all_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Water', 'Water'],
            yticklabels=['Non-Water', 'Water'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('model/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 CNN模型训练完成！")
print(f"最终验证准确率: {final_accuracy:.4f}")
print(f"模型文件: {model_path}")
print("训练曲线: model/training_curves.png")
print("混淆矩阵: model/confusion_matrix.png")
