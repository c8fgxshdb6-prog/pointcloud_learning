#!/usr/bin/env python3
"""
项目结构验证测试
确认VSCode配置和项目结构是否正确
"""

import os
import sys
import torch
import open3d as o3d
import numpy as np

def check_project_structure():
    """检查项目文件夹结构"""
    print("🔍 检查项目结构...")
    
    folders = ['.vscode', 'data', 'notebooks', 'scripts', 'outputs']
    missing_folders = []
    
    for folder in folders:
        if os.path.exists(folder):
            print(f"   ✅ {folder}/")
        else:
            print(f"   ❌ {folder}/ (缺失)")
            missing_folders.append(folder)
    
    return len(missing_folders) == 0

def check_environment():
    """检查Python环境"""
    print("\n🔍 检查Python环境...")
    
    print(f"   Python路径: {sys.executable}")
    print(f"   Python版本: {sys.version.split()[0]}")
    
    # 检查关键库
    libraries = {
        'torch': torch.__version__,
        'open3d': o3d.__version__,
        'numpy': np.__version__
    }
    
    for lib, version in libraries.items():
        print(f"   ✅ {lib}: {version}")
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"   ✅ CUDA可用: {cuda_available}")
    if cuda_available:
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")

def main():
    print("=" * 50)
    print("VSCode项目配置验证")
    print("=" * 50)
    
    # 检查项目结构
    structure_ok = check_project_structure()
    
    # 检查环境
    check_environment()
    
    print("\n" + "=" * 50)
    if structure_ok:
        print("🎉 所有配置检查通过！")
        print("🚀 你可以开始点云学习了！")
    else:
        print("⚠️  项目结构不完整，请检查文件夹创建")
    print("=" * 50)

if __name__ == "__main__":
    main()