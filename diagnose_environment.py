#!/usr/bin/env python3
"""
诊断当前Python环境
"""

import sys
import os
import subprocess

def diagnose_environment():
    print("🔍 环境诊断报告")
    print("=" * 50)
    
    # 当前Python信息
    python_exe = sys.executable
    python_version = sys.version
    print(f"当前Python: {python_exe}")
    print(f"Python版本: {python_version.split()[0]}")
    
    # 检查是否在pointcloud环境中
    if "pointcloud" in python_exe:
        print("✅ 环境状态: 正在使用 pointcloud 环境")
    else:
        print("❌ 环境状态: 未使用 pointcloud 环境")
        print("   当前使用的是基础环境")
    
    # 检查关键库
    print("\n📚 库检查:")
    libraries = ['torch', 'open3d', 'numpy', 'matplotlib']
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', '未知版本')
            print(f"   ✅ {lib}: {version}")
        except ImportError:
            print(f"   ❌ {lib}: 未安装")
    
    # 建议
    print("\n💡 建议:")
    if "pointcloud" not in python_exe:
        print("   1. 运行: conda activate pointcloud")
        print("   2. 或者使用完整路径:")
        print(f'      D:\\Users\\yxf\\anaconda3\\envs\\pointcloud\\python.exe {sys.argv[0]}')
    
    print("=" * 50)

if __name__ == "__main__":
    diagnose_environment()