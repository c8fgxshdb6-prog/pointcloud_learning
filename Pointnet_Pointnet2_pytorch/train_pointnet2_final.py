import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import time
import numpy as np

# 添加路径
sys.path.append('models')
sys.path.append('data_utils')

from pointnet2_cls_ssg import get_model
from ModelNetH5Loader import ModelNetH5Dataset

def train_pointnet2():
    # 训练配置
    config = {
        'data_path': '../data/modelnet40_ply_hdf5_2048',
        'batch_size': 16,
        'num_points': 1024,
        'epochs': 10,
        'learning_rate': 0.001,
        'log_dir': 'logs_pointnet2',
        'model': 'pointnet2_cls_ssg'
    }
    
    print("🚀 Starting PointNet++ Training")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 检查数据路径
    if not os.path.exists(config['data_path']):
        print(f"❌ Error: Data path does not exist: {config['data_path']}")
        return
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  CUDA capability: {torch.cuda.get_device_capability()}")
    
    # 加载数据
    print("📊 Loading datasets...")
    train_dataset = ModelNetH5Dataset(
        root=config['data_path'], 
        num_points=config['num_points'], 
        split='train'
    )
    test_dataset = ModelNetH5Dataset(
        root=config['data_path'], 
        num_points=config['num_points'], 
        split='test'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"✅ Data loaded:")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Number of classes: 40")
    
    # 创建模型
    print("🔄 Creating PointNet++ model...")
    model = get_model(num_class=40, normal_channel=False)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 创建日志目录
    os.makedirs(config['log_dir'], exist_ok=True)
    
    print("🎯 Starting training...")
    best_accuracy = 0.0
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (points, labels) in enumerate(train_loader):
            points, labels = points.to(device), labels.to(device)
            points = points.transpose(2, 1)  # (B, N, 3) -> (B, 3, N)
            
            optimizer.zero_grad()
            pred, _ = model(points)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = pred.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # 学习率调度
        scheduler.step()
        
        train_accuracy = 100. * correct / total
        epoch_time = time.time() - start_time
        
        print(f'📈 Epoch {epoch+1}/{config["epochs"]}:')
        print(f'   Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'   Train Accuracy: {train_accuracy:.2f}%')
        print(f'   Time: {epoch_time:.2f}s')
        
        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for points, labels in test_loader:
                points, labels = points.to(device), labels.to(device)
                points = points.transpose(2, 1)
                pred, _ = model(points)
                _, predicted = pred.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_accuracy = 100. * test_correct / test_total
        print(f'   Test Accuracy: {test_accuracy:.2f}%')
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy
            }, os.path.join(config['log_dir'], 'best_model.pth'))
            print(f'   💾 New best model saved! Accuracy: {best_accuracy:.2f}%')
        
        print('-' * 50)
    
    print(f'🎉 Training completed!')
    print(f'   Best test accuracy: {best_accuracy:.2f}%')
    print(f'   Model saved in: {config["log_dir"]}')

if __name__ == "__main__":
    train_pointnet2()