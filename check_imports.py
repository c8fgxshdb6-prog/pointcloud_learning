#!/usr/bin/env python3
"""
检查所有导入是否正常工作
"""

import sys
import os

def check_imports():
    print("🔍 检查导入路径...")
    
    # 添加项目根目录到路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    scripts_path = os.path.join(project_root, "scripts")
    
    print(f"项目根目录: {project_root}")
    print(f"脚本路径: {scripts_path}")
    print(f"当前Python路径:")
    for path in sys.path:
        print(f"  - {path}")
    
    # 检查scripts目录是否存在
    if os.path.exists(scripts_path):
        print("✅ scripts目录存在")
        
        # 检查必要的文件
        required_files = [
            "pointcloud_visualizer.py",
            "pointcloud_processor.py", 
            "__init__.py"
        ]
        
        for file in required_files:
            file_path = os.path.join(scripts_path, file)
            if os.path.exists(file_path):
                print(f"✅ {file} 存在")
            else:
                print(f"❌ {file} 缺失")
    
    # 尝试导入
    print("\n🔍 尝试导入模块...")
    try:
        sys.path.insert(0, project_root)
        from scripts.pointcloud_visualizer import PointCloudVisualizer
        print("✅ PointCloudVisualizer 导入成功")
        
        from scripts.pointcloud_processor import PointCloudProcessor
        print("✅ PointCloudProcessor 导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

if __name__ == "__main__":
    success = check_imports()
    if success:
        print("\n🎉 所有导入检查通过！")
    else:
        print("\n⚠️ 存在导入问题，请检查上述错误")