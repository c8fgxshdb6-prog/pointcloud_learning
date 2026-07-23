import h5py
import numpy as np
import os

def convert_h5_to_pointnet2_format():
    """将HDF5格式转换为PointNet++期望的格式"""
    input_path = "../data/modelnet40_ply_hdf5_2048"
    output_path = "data/modelnet40_processed"
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
    
    print("🔧 开始转换数据格式...")
    
    # 处理训练数据
    train_files = []
    for i in range(5):
        train_file = os.path.join(input_path, f"ply_data_train{i}.h5")
        if os.path.exists(train_file):
            train_files.append(train_file)
            print(f"找到训练文件: {train_file}")
    
    # 处理测试数据  
    test_files = [
        os.path.join(input_path, "ply_data_test0.h5"),
        os.path.join(input_path, "ply_data_test1.h5")
    ]
    
    # 转换训练数据
    train_data = []
    train_labels = []
    for file in train_files:
        with h5py.File(file, 'r') as f:
            data = f['data'][:]
            labels = f['label'][:]
            train_data.append(data)
            train_labels.append(labels)
            print(f"加载训练文件 {file}: {data.shape}")
    
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0).squeeze()
    
    # 转换测试数据
    test_data = []
    test_labels = []
    for file in test_files:
        if os.path.exists(file):
            with h5py.File(file, 'r') as f:
                data = f['data'][:]
                labels = f['label'][:]
                test_data.append(data)
                test_labels.append(labels)
                print(f"加载测试文件 {file}: {data.shape}")
    
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0).squeeze()
    
    print(f"📊 数据统计:")
    print(f"  训练数据: {train_data.shape}")
    print(f"  训练标签: {train_labels.shape}")
    print(f"  测试数据: {test_data.shape}") 
    print(f"  测试标签: {test_labels.shape}")
    
    # 保存为numpy格式
    np.save(os.path.join(output_path, "train_data.npy"), train_data)
    np.save(os.path.join(output_path, "train_labels.npy"), train_labels)
    np.save(os.path.join(output_path, "test_data.npy"), test_data) 
    np.save(os.path.join(output_path, "test_labels.npy"), test_labels)
    
    print("✅ 数据格式转换完成！")
    print(f"输出目录: {output_path}")

if __name__ == "__main__":
    convert_h5_to_pointnet2_format()