import os
import h5py

# 测试不同的路径
test_paths = [
    '../../data/modelnet40_ply_hdf5_2048',
    '../data/modelnet40_ply_hdf5_2048', 
    'D:/Users/yxf/Desktop/pointcloud_learning/data/modelnet40_ply_hdf5_2048'
]

for path in test_paths:
    print(f"\nTesting: {path}")
    if os.path.exists(path):
        print(f"✓ Path exists")
        # 检查HDF5文件
        h5_files = [f for f in os.listdir(path) if f.endswith('.h5')]
        print(f"HDF5 files: {h5_files[:3]}...")  # 只显示前3个
        
        if h5_files:
            # 尝试读取第一个文件
            try:
                first_file = os.path.join(path, h5_files[0])
                with h5py.File(first_file, 'r') as f:
                    print(f"Data shape: {f['data'][:].shape}")
                    print(f"Labels shape: {f['label'][:].shape}")
                print("✓ SUCCESS - Can read HDF5 files")
                correct_path = path
                break
            except Exception as e:
                print(f"✗ Error reading HDF5: {e}")
    else:
        print(f"✗ Path does not exist")

print(f"\nRecommended path: {correct_path}")