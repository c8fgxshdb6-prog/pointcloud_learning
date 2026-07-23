#!/usr/bin/env python3
"""
统计滤波专门测试文件
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import open3d as o3d
import numpy as np

def main():
    print("=== 统计滤波去噪演示 ===")
    
    # 创建带噪声的点云
    points = np.random.rand(1000, 3)  # 主要点云
    noise_points = np.random.rand(100, 3) * 2 + 0.5  # 远离主点云的噪声
    all_points = np.vstack([points, noise_points])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    
    print(f"原始点数: {len(pcd.points)}")
    
    # 应用统计滤波
    cleaned_pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20, 
        std_ratio=2.0
    )
    
    print(f"去噪后点数: {len(cleaned_pcd.points)}")
    print(f"移除噪声点: {len(pcd.points) - len(cleaned_pcd.points)}")
    
    # 可视化原始点云
    print("显示原始点云（包含噪声）...")
    o3d.visualization.draw_geometries([pcd], 
                                    window_name="原始点云 - 按ESC关闭")
    
    # 可视化去噪后的点云
    print("显示去噪后的点云...")
    o3d.visualization.draw_geometries([cleaned_pcd], 
                                    window_name="去噪后的点云 - 按ESC关闭")

if __name__ == "__main__":
    main()