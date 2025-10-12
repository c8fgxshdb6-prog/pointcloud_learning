#!/usr/bin/env python3
"""
最终项目配置验证
确认所有组件都能正常工作
"""

import os
import sys
import subprocess

def check_project_structure():
    """检查项目结构完整性"""
    print("🔍 检查项目结构完整性...")
    
    required_items = {
        'folders': ['.vscode', 'data', 'notebooks', 'scripts', 'outputs'],
        'files': [
            '.vscode/settings.json',
            '.vscode/launch.json',
            'scripts/pointcloud_visualizer.py',
            'scripts/pointcloud_processor.py',
            'notebooks/01_pointcloud_basics.ipynb',
            'README.md'
        ]
    }
    
    all_good = True
    
    for folder in required_items['folders']:
        if os.path.exists(folder):
            print(f"   ✅ 文件夹: {folder}/")
        else:
            print(f"   ❌ 缺失文件夹: {folder}/")
            all_good = False
    
    for file in required_items['files']:
        if os.path.exists(file):
            print(f"   ✅ 文件: {file}")
        else:
            print(f"   ❌ 缺失文件: {file}")
            all_good = False
    
    return all_good

def test_imports():
    """测试所有模块导入"""
    print("\n🔍 测试模块导入...")
    
    try:
        import torch
        import open3d as o3d
        import numpy as np
        import matplotlib.pyplot as plt
        
        # 测试自定义模块
        sys.path.append('./scripts')
        from pointcloud_visualizer import PointCloudVisualizer
        from pointcloud_processor import PointCloudProcessor
        
        print("   ✅ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        return False

def test_functionality():
    """测试核心功能"""
    print("\n🔍 测试核心功能...")
    
    try:
        # 测试点云创建
        visualizer = PointCloudVisualizer()
        pcd = visualizer.create_sample_pointcloud(100)
        
        # 测试点云处理
        processor = PointCloudProcessor()
        down_pcd = processor.voxel_downsample(pcd, 0.1)
        
        print("   ✅ 核心功能测试通过")
        return True
    except Exception as e:
        print(f"   ❌ 功能测试失败: {e}")
        return False

def main():
    print("=" * 60)
    print("最终项目配置验证")
    print("=" * 60)
    
    structure_ok = check_project_structure()
    imports_ok = test_imports()
    functionality_ok = test_functionality()
    
    print("\n" + "=" * 60)
    if all([structure_ok, imports_ok, functionality_ok]):
        print("🎉 所有配置验证通过！")
        print("🚀 你的点云学习环境已经完全准备就绪！")
        print("\n下一步建议:")
        print("1. 运行: python scripts/pointcloud_visualizer.py")
        print("2. 运行: python scripts/pointcloud_processor.py") 
        print("3. 探索 notebooks/01_pointcloud_basics.ipynb")
    else:
        print("⚠️ 部分配置存在问题，请检查上述错误")
    print("=" * 60)

if __name__ == "__main__":
    main()