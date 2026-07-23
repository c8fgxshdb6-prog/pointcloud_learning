import h5py
import os
import numpy as np

def verify_modelnet40():
    dataset_path = "../data/modelnet40_ply_hdf5_2048"
    
    print("🔍 验证 ModelNet40 数据集...")
    
    # 检查目录是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集目录不存在: {dataset_path}")
        return False
    
    print(f"✅ 找到数据集目录: {dataset_path}")
    
    # 列出所有文件
    files = os.listdir(dataset_path)
    h5_files = [f for f in files if f.endswith('.h5')]
    
    print("📁 HDF5 文件列表:")
    for file in sorted(h5_files):
        file_path = os.path.join(dataset_path, file)
        size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"  {file} ({size:.1f} MB)")
    
    # 检查关键文件
    required_files = ["ply_data_train0.h5", "ply_data_test0.h5"]
    found_files = [f for f in required_files if f in files]
    
    print(f"✅ 找到 {len(found_files)}/{len(required_files)} 个关键文件")
    
    # 尝试读取数据
    try:
        # 读取训练数据
        train_file = os.path.join(dataset_path, "ply_data_train0.h5")
        with h5py.File(train_file, 'r') as f:
            data = f['data'][:]
            labels = f['label'][:]
            print(f"✅ 成功读取训练数据")
            print(f"  点云数据形状: {data.shape}")
            print(f"  标签数据形状: {labels.shape}")
            print(f"  数据类型: {data.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return False

if __name__ == "__main__":
    success = verify_modelnet40()
    if success:
        print("\n🎉 ModelNet40 数据集验证成功！")
    else:
        print("\n⚠️ 数据集验证失败")