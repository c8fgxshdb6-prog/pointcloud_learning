#!/usr/bin/env python3
"""
离群点分析工具
"""
#由于体现ind作用的示范文件
import open3d as o3d
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_pointcloud_with_outliers():
    """创建带明显离群点的测试数据"""
    # 主要点云 - 密集区域
    main_points = np.random.rand(800, 3) * 2 - 1  # [-1, 1]范围内
    
    # 离群点 - 远离主要区域
    outlier_points = np.random.rand(50, 3) * 10 + 5  # [5, 15]范围内
    
    # 合并
    all_points = np.vstack([main_points, outlier_points])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    
    return pcd

def detailed_analysis():
    """详细的离群点分析"""
    pcd = create_pointcloud_with_outliers()
    
    print("原始点云信息:")
    print(f"  总点数: {len(pcd.points)}")
    
    # 应用统计滤波
    cleaned_pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20, 
        std_ratio=2.0
    )
    
    print("\n去噪结果:")
    print(f"  保留点数: {len(cleaned_pcd.points)}")
    print(f"  移除点数: {len(pcd.points) - len(cleaned_pcd.points)}")
    print(f"  保留点索引数量: {len(ind)}")
    
    # 分析被移除的点
    all_indices = set(range(len(pcd.points)))
    kept_indices = set(ind)
    removed_indices = all_indices - kept_indices
    
    print(f"  被移除点索引: {sorted(removed_indices)[:10]}...")  # 显示前10个
    
    # 可视化
    visualize_removal_effect(pcd, cleaned_pcd, ind)
    
    return ind

def visualize_removal_effect(original_pcd, cleaned_pcd, kept_indices):
    """可视化移除效果"""
    # 创建颜色编码
    original_colors = np.zeros((len(original_pcd.points), 3))
    
    # 保留的点标记为蓝色
    original_colors[list(kept_indices)] = [0, 0, 1]  # 蓝色
    
    # 被移除的点标记为红色
    removed_indices = set(range(len(original_pcd.points))) - set(kept_indices)
    original_colors[list(removed_indices)] = [1, 0, 0]  # 红色
    
    original_pcd.colors = o3d.utility.Vector3dVector(original_colors)
    
    print("\n可视化说明:")
    print("  🔵 蓝色: 被保留的点")
    print("  🔴 红色: 被移除的离群点")
    
    o3d.visualization.draw_geometries([original_pcd], 
                                     window_name="离群点分析 - 蓝色:保留 红色:移除")

if __name__ == "__main__":
    detailed_analysis()