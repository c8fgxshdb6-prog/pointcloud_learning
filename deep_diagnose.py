#!/usr/bin/env python3
"""
深入诊断 matplotlib 问题
"""

import sys
import os
import subprocess

def deep_diagnose():
    print("🔍 深入诊断 matplotlib 问题")
    print("=" * 50)
    
    # 检查 Python 路径
    print("Python 解释器:", sys.executable)
    print("Python 版本:", sys.version)
    
    # 检查 site-packages 路径
    print("\nSite-packages 路径:")
    for path in sys.path:
        if 'site-packages' in path:
            print(" ", path)
    
    # 检查 matplotlib 是否真的存在
    print("\n检查 matplotlib 文件:")
    try:
        import matplotlib
        print("  ✅ matplotlib 模块可以导入")
        print("  ✅ matplotlib 位置:", matplotlib.__file__)
        print("  ✅ matplotlib 版本:", matplotlib.__version__)
    except ImportError as e:
        print("  ❌ matplotlib 导入失败:", e)
        
        # 尝试手动查找 matplotlib
        for path in sys.path:
            matplotlib_path = os.path.join(path, 'matplotlib')
            if os.path.exists(matplotlib_path):
                print(f"  ✅ 找到 matplotlib 目录: {matplotlib_path}")
                break
        else:
            print("  ❌ 在所有路径中都没有找到 matplotlib")
    
    # 检查其他关键库
    print("\n其他库检查:")
    for lib in ['numpy', 'torch', 'open3d']:
        try:
            module = __import__(lib)
            print(f"  ✅ {lib}: {getattr(module, '__version__', '未知版本')}")
        except ImportError as e:
            print(f"  ❌ {lib}: {e}")

if __name__ == "__main__":
    deep_diagnose()