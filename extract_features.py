# extract_features.py
import sys
import os
sys.path.append('./Pointnet_Pointnet2_pytorch')
sys.path.append('./Pointnet_Pointnet2_pytorch/models')

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob

class ModelNet40PLY(Dataset):
    def __init__(self, root, num_points=1024, use_normals=True):
        self.root = root
        self.num_points = num_points
        self.use_normals = use_normals
        self.classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.points = []
        self.labels = []
        for cls in self.classes:
            cls_path = os.path.join(self.root, cls)
            files = glob.glob(os.path.join(cls_path, '*.txt'))
            for f in files:
                self.points.append(f)
                self.labels.append(self.class_to_idx[cls])
        print(f"Loaded {len(self.points)} samples from {root}")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        filepath = self.points[idx]
        label = self.labels[idx]
        # 指定分隔符为逗号
        data = np.loadtxt(filepath, dtype=np.float32, delimiter=',')
        if data.shape[0] < self.num_points:
            choice = np.random.choice(data.shape[0], self.num_points, replace=True)
        else:
            choice = np.random.choice(data.shape[0], self.num_points, replace=False)
        data = data[choice, :]
        if not self.use_normals:
            data = data[:, :3]
        return data, label

def extract_features():
    # ========== 配置路径 ==========
    data_root = 'D:/Users/yxf/Desktop/pointcloud_learning/data/modelnet40_normal_resampled/test'
    checkpoint_path = 'D:/Users/yxf/Desktop/pointcloud_learning/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth'
    normal_channel = True   # 模型需要6维输入（xyz + 法向量）
    BATCH_SIZE = 16
    NUM_POINTS = 1024
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集
    dataset = ModelNet40PLY(root=data_root, num_points=NUM_POINTS, use_normals=normal_channel)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 加载模型
    from models.pointnet2_cls_msg import get_model
    model = get_model(num_class=40, normal_channel=normal_channel).to(DEVICE)
    if os.path.exists(checkpoint_path):
        # 添加 weights_only=False 以消除警告（可选）
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 预训练模型加载成功")
    else:
        print("❌ 模型文件不存在，请检查路径：", checkpoint_path)
        return
    model.eval()

    # 提取特征
    feats_sa1, feats_sa2, feats_sa3 = [], [], []
    with torch.no_grad():
        for i, (points, _) in enumerate(dataloader):
            points = points.transpose(2, 1).to(DEVICE)  # (B, 6, N)
            _, l1, l2, l3 = model(points)
            feats_sa1.append(l1.cpu().numpy())
            feats_sa2.append(l2.cpu().numpy())
            feats_sa3.append(l3.cpu().numpy())
            if (i+1) % 10 == 0:
                print(f"已处理 {i+1} / {len(dataloader)} 个batch")

    feats_sa1 = np.concatenate(feats_sa1, axis=0)
    feats_sa2 = np.concatenate(feats_sa2, axis=0)
    feats_sa3 = np.concatenate(feats_sa3, axis=0)
    os.makedirs('results', exist_ok=True)
    np.save('results/clean_features_sa1.npy', feats_sa1)
    np.save('results/clean_features_sa2.npy', feats_sa2)
    np.save('results/clean_features_sa3.npy', feats_sa3)
    print("特征提取完成！")
    print(f"SA1 形状: {feats_sa1.shape}")
    print(f"SA2 形状: {feats_sa2.shape}")
    print(f"SA3 形状: {feats_sa3.shape}")

if __name__ == '__main__':
    extract_features()