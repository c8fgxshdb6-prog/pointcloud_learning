#!/usr/bin/env python3
"""
点云可视化工具集
提供各种点云可视化功能的工具函数
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

class PointCloudVisualizer:
    """点云可视化工具类"""
    
    def __init__(self):
        self.color_map = {
            'red': [1, 0, 0],
            'green': [0, 1, 0],
            'blue': [0, 0, 1],
            'yellow': [1, 1, 0],
            'cyan': [0, 1, 1],
            'magenta': [1, 0, 1],
            'white': [1, 1, 1],
            'black': [0, 0, 0]
        }
    
    def create_sample_pointcloud(self, num_points: int = 1000, 
                               shape: str = 'random') -> o3d.geometry.PointCloud:
        """创建示例点云
        
        Args:
            num_points: 点数
            shape: 点云形状 ('random', 'sphere', 'cube', 'plane')
        
        Returns:
            点云对象
        """
        points = np.random.rand(num_points, 3)
        
        if shape == 'sphere':
            # 创建球体点云
            phi = np.random.uniform(0, 2*np.pi, num_points)
            theta = np.random.uniform(0, np.pi, num_points)
            r = 1.0
            
            points[:, 0] = r * np.sin(theta) * np.cos(phi)
            points[:, 1] = r * np.sin(theta) * np.sin(phi)
            points[:, 2] = r * np.cos(theta)
            
        elif shape == 'cube':
            # 创建立方体点云
            points = np.random.uniform(-1, 1, (num_points, 3))
            
        elif shape == 'plane':
            # 创建平面点云
            points = np.random.uniform(-1, 1, (num_points, 3))
            points[:, 2] = 0  # 所有点z=0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 添加颜色
        self._add_color_gradient(pcd)
        
        return pcd
    
    def _add_color_gradient(self, pcd: o3d.geometry.PointCloud):
        """添加颜色渐变"""
        points = np.asarray(pcd.points)
        colors = np.zeros_like(points)
        
        # 基于z坐标添加颜色渐变
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        if z_max - z_min > 1e-6:
            z_normalized = (points[:, 2] - z_min) / (z_max - z_min)
            colors[:, 0] = z_normalized  # R
            colors[:, 1] = 1 - z_normalized  # G
            colors[:, 2] = 0.5  # B
        else:
            colors = np.random.rand(*points.shape)
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    def visualize_single(self, pcd: o3d.geometry.PointCloud, 
                        title: str = "点云可视化"):
        """可视化单个点云
        
        Args:
            pcd: 点云对象
            title: 窗口标题
        """
        print(f"可视化点云: {title}")
        print(f"点数: {len(pcd.points)}")
        print(f"边界框: {pcd.get_axis_aligned_bounding_box()}")
        
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=title,
            width=1024,
            height=768,
            left=50,
            top=50
        )
    
    def visualize_comparison(self, original_pcd: o3d.geometry.PointCloud,
                           processed_pcd: o3d.geometry.PointCloud,
                           title: str = "处理前后对比"):
        """可视化处理前后的点云对比"""
        # 设置不同颜色以便区分
        original_colors = np.asarray(original_pcd.colors)
        processed_colors = np.ones_like(original_colors) * [1, 0, 0]  # 红色
        
        original_pcd.colors = o3d.utility.Vector3dVector(original_colors)
        processed_pcd.colors = o3d.utility.Vector3dVector(processed_colors)
        
        print("原始点云点数:", len(original_pcd.points))
        print("处理后点数:", len(processed_pcd.points))
        print("减少比例:", f"{(1 - len(processed_pcd.points)/len(original_pcd.points))*100:.1f}%")
        
        o3d.visualization.draw_geometries(
            [original_pcd, processed_pcd],
            window_name=title,
            width=1024,
            height=768
        )
    
    def create_visualization_demo(self):
        """创建可视化演示"""
        print("=" * 50)
        print("点云可视化演示")
        print("=" * 50)
        
        # 1. 随机点云
        random_pcd = self.create_sample_pointcloud(1500, 'random')
        self.visualize_single(random_pcd, "随机点云")
        
        # 2. 球体点云
        sphere_pcd = self.create_sample_pointcloud(1000, 'sphere')
        self.visualize_single(sphere_pcd, "球体点云")
        
        # 3. 立方体点云
        cube_pcd = self.create_sample_pointcloud(800, 'cube')
        self.visualize_single(cube_pcd, "立方体点云")
        
        # 4. 下采样对比演示
        original = self.create_sample_pointcloud(2000, 'random')
        downsampled = original.voxel_down_sample(0.1)
        self.visualize_comparison(original, downsampled, "下采样对比")

def main():
    """主函数：运行可视化演示"""
    visualizer = PointCloudVisualizer()
    visualizer.create_visualization_demo()

if __name__ == "__main__":
    main()