#!/usr/bin/env python3
"""
Stanford 数据验证脚本
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 首先导入 open3d
import open3d as o3d
import numpy as np

from scripts.pointcloud_reader import PointCloudReader

def test_stanford_data():
    """测试 Stanford 数据读取"""
    reader = PointCloudReader()
    
    print("🔍 Stanford 数据验证")
    print("=" * 50)
    
    # 检查 Bunny 数据
    bunny_path = "data/bunny/bunny.ply"
    if os.path.exists(bunny_path):
        print("🐰 读取 Stanford Bunny...")
        bunny_pcd = reader.read_pointcloud(bunny_path)
        
        if bunny_pcd:
            print("✅ Stanford Bunny 读取成功!")
        else:
            print("❌ Stanford Bunny 读取失败!")
    else:
        print("⚠️ Stanford Bunny 文件不存在，请先运行下载脚本")
    
    print("\n" + "-" * 30)
    
    # 检查 Dragon 数据  
    dragon_path = "data/dragon/dragon.ply"
    if os.path.exists(dragon_path):
        print("🐉 读取 Stanford Dragon...")
        dragon_pcd = reader.read_pointcloud(dragon_path)
        
        if dragon_pcd:
            print("✅ Stanford Dragon 读取成功!")
        else:
            print("❌ Stanford Dragon 读取失败!")
    else:
        print("⚠️ Stanford Dragon 文件不存在，请先运行下载脚本")

def visualize_stanford_models():
    """可视化 Stanford 模型"""
    reader = PointCloudReader()
    
    pointclouds = []
    labels = []
    
    # 读取 Bunny
    bunny_path = "data/bunny/bunny.ply"
    if os.path.exists(bunny_path):
        bunny_pcd = reader.read_pointcloud(bunny_path, verbose=False)
        if bunny_pcd:
            bunny_pcd.paint_uniform_color([1, 0, 0])  # 红色
            pointclouds.append(bunny_pcd)
            labels.append("Bunny (红色)")
    
    # 读取 Dragon
    dragon_path = "data/dragon/dragon.ply" 
    if os.path.exists(dragon_path):
        dragon_pcd = reader.read_pointcloud(dragon_path, verbose=False)
        if dragon_pcd:
            dragon_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
            pointclouds.append(dragon_pcd)
            labels.append("Dragon (蓝色)")
    
    if pointclouds:
        print(f"🎨 准备可视化 {len(pointclouds)} 个模型...")
        print("颜色说明:")
        for label in labels:
            print(f"  {label}")
        
        print("\n🖱️ 操作提示: 鼠标左键旋转, 滚轮缩放, 右键平移")
        o3d.visualization.draw_geometries(
            pointclouds,
            window_name="Stanford 3D 模型可视化",
            width=1024,
            height=768
        )
    else:
        print("❌ 没有找到可用的 Stanford 模型文件")

if __name__ == "__main__":
    test_stanford_data()
    
    print("\n" + "=" * 50)
    print("是否进行可视化? (y/n)")
    
    try:
        user_input = input().strip().lower()
        if user_input in ['y', 'yes']:
            visualize_stanford_models()
    except:
        print("跳过可视化")