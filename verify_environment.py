#!/usr/bin/env python3
"""
验证当前使用的Python环境
"""

import sys
import os

def main():
    print("🔍 环境验证")
    print("=" * 50)
    
    python_exe = sys.executable
    print(f"当前Python: {python_exe}")
    
    # 检查是否在pointcloud环境中
    if "pointcloud" in python_exe:
        print("✅ 正确: 正在使用 pointcloud 环境")
        print(f"✅ Python版本: {sys.version.split()[0]}")
        
        # 检查关键库
        libraries = ['torch', 'open3d', 'numpy', 'matplotlib']
        all_ok = True
        
        for lib in libraries:
            try:
                module = __import__(lib)
                version = getattr(module, '__version__', '未知')
                print(f"✅ {lib}: {version}")
            except ImportError as e:
                print(f"❌ {lib}: 导入失败 - {e}")
                all_ok = False
        
        if all_ok:
            print("\n🎉 环境验证通过！所有库都可正常导入")
        else:
            print("\n⚠️ 环境验证失败，部分库无法导入")
            
    else:
        print("❌ 错误: 未使用 pointcloud 环境")
        print("请运行: conda activate pointcloud")
        print("或者使用完整路径: D:\\Users\\yxf\\anaconda3\\envs\\pointcloud\\python.exe")
    
    print("=" * 50)

if __name__ == "__main__":
    main()