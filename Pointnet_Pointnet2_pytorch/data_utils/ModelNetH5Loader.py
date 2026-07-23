import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

class ModelNetH5Dataset(Dataset):
    def __init__(self, root, num_points=1024, split='train', use_normals=False, process_data=False):
        self.root = root
        self.npoints = num_points
        self.split = split
        self.process_data = process_data
        
        print(f"Looking for data in: {root}")
        
        # 根据split确定文件
        if self.split == 'train':
            self.files = [os.path.join(root, f'ply_data_train{i}.h5') for i in range(5)]
        else:  # test
            self.files = [os.path.join(root, 'ply_data_test0.h5'),
                         os.path.join(root, 'ply_data_test1.h5')]
        
        # 检查文件是否存在
        existing_files = []
        for file_path in self.files:
            if os.path.exists(file_path):
                existing_files.append(file_path)
                print(f"Found: {file_path}")
            else:
                print(f"Missing: {file_path}")
        
        if not existing_files:
            raise FileNotFoundError(f"No HDF5 files found in {root}")
        
        # 加载数据
        self.data = []
        self.labels = []
        for file_path in existing_files:
            print(f"Loading {file_path}...")
            with h5py.File(file_path, 'r') as f:
                file_data = f['data'][:]
                file_labels = f['label'][:]
                self.data.append(file_data)
                self.labels.append(file_labels)
                print(f"Loaded {file_data.shape} from {file_path}")
        
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0).squeeze()
        
        print(f'Successfully loaded ModelNetH5 {split} set: {self.data.shape}')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        point_set = self.data[index].astype(np.float32)
        cls = self.labels[index].astype(np.int64)
        
        # 随机采样到固定点数
        if point_set.shape[0] > self.npoints:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        else:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        
        point_set = point_set[choice, :]
        
        # 数据归一化
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist
        
        return point_set, cls

if __name__ == '__main__':
    import torch
    
    # 测试不同的路径
    test_paths = [
        '../../data/modelnet40_ply_hdf5_2048',
        '../data/modelnet40_ply_hdf5_2048',
        'data/modelnet40_processed'
    ]
    
    for path in test_paths:
        print(f"\nTesting path: {path}")
        if os.path.exists(path):
            print(f"Path exists: {path}")
            try:
                data = ModelNetH5Dataset(root=path, split='train')
                DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
                for point, label in DataLoader:
                    print(f"Data shape: {point.shape}")
                    print(f"Label shape: {label.shape}")
                    break
                print("SUCCESS!")
                break
            except Exception as e:
                print(f"Failed: {e}")
        else:
            print(f"Path does not exist: {path}")