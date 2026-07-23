#!/usr/bin/env python3
"""
点云处理工具集
提供点云预处理和分析功能
"""

import open3d as o3d
import numpy as np
from typing import Tuple
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在导入应该能正常工作
try:
    from scripts.pointcloud_visualizer import PointCloudVisualizer
    print("✅ 成功导入 PointCloudVisualizer")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    # 备用方案：动态导入
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pointcloud_visualizer", 
        os.path.join(project_root, "scripts", "pointcloud_visualizer.py")
    )
    visualizer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visualizer_module)
    PointCloudVisualizer = visualizer_module.PointCloudVisualizer
    print("✅ 使用动态导入成功")

class PointCloudProcessor:
    """点云处理工具类"""
    
    def __init__(self):
        self.stats = {}
    
    def voxel_downsample(self, pcd: o3d.geometry.PointCloud, 
                        voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
        """体素下采样
        
        Args:
            pcd: 输入点云
            voxel_size: 体素尺寸
        
        Returns:
            下采样后的点云
        """
        down_pcd = pcd.voxel_down_sample(voxel_size)
        
        # 记录统计信息
        self.stats['voxel_downsample'] = {
            'original_points': len(pcd.points),
            'downsampled_points': len(down_pcd.points),
            'reduction_ratio': f"{(1 - len(down_pcd.points)/len(pcd.points))*100:.1f}%",
            'voxel_size': voxel_size
        }
        
        return down_pcd
    
    def remove_statistical_outlier(self, pcd: o3d.geometry.PointCloud,
                                 nb_neighbors: int = 20, 
                                 std_ratio: float = 2.0) -> Tuple[o3d.geometry.PointCloud, list]:
        """统计滤波去噪
        
        Args:
            pcd: 输入点云
            nb_neighbors: 邻居点数
            std_ratio: 标准差比率
        
        Returns:
            去噪后的点云和噪声点索引
        """
        cleaned_pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, 
            std_ratio=std_ratio
        )
        
        self.stats['statistical_outlier'] = {
            'original_points': len(pcd.points),
            'cleaned_points': len(cleaned_pcd.points),
            'noise_points': len(pcd.points) - len(cleaned_pcd.points),
            'noise_ratio': f"{((len(pcd.points) - len(cleaned_pcd.points))/len(pcd.points))*100:.1f}%",
            'nb_neighbors': nb_neighbors,
            'std_ratio': std_ratio
        }
        
        return cleaned_pcd, ind
    
    def remove_radius_outlier(self, pcd: o3d.geometry.PointCloud,
                            nb_points: int = 16, 
                            radius: float = 0.05) -> Tuple[o3d.geometry.PointCloud, list]:
        """半径滤波去噪
        
        Args:
            pcd: 输入点云
            nb_points: 最小邻居点数
            radius: 搜索半径
        
        Returns:
            去噪后的点云和噪声点索引
        """
        cleaned_pcd, ind = pcd.remove_radius_outlier(
            nb_points=nb_points, 
            radius=radius
        )
        
        self.stats['radius_outlier'] = {
            'original_points': len(pcd.points),
            'cleaned_points': len(cleaned_pcd.points),
            'noise_points': len(pcd.points) - len(cleaned_pcd.points),
            'noise_ratio': f"{((len(pcd.points) - len(cleaned_pcd.points))/len(pcd.points))*100:.1f}%",
            'nb_points': nb_points,
            'radius': radius
        }
        
        return cleaned_pcd, ind
    
    def estimate_normals(self, pcd: o3d.geometry.PointCloud,
                        search_param: o3d.geometry.KDTreeSearchParamHybrid = None) -> o3d.geometry.PointCloud:
        """估计点云法向量
        
        Args:
            pcd: 输入点云
            search_param: 搜索参数
        
        Returns:
            带有法向量的点云
        """
        if search_param is None:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        
        pcd.estimate_normals(search_param=search_param)
        
        # 定向法向量（使其朝向一致）
        pcd.orient_normals_to_align_with_direction()
        
        self.stats['normals'] = {
            'has_normals': pcd.has_normals(),
            'search_radius': search_param.radius,
            'max_neighbors': search_param.max_nn
        }
        
        return pcd
    
    def get_statistics(self) -> dict:
        """获取处理统计信息"""
        return self.stats
    
    def print_statistics(self):
        """打印统计信息"""
        print("=" * 50)
        print("点云处理统计信息")
        print("=" * 50)
        
        for operation, stats in self.stats.items():
            print(f"\n{operation.replace('_', ' ').title()}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

def demo_processing_pipeline():
    """演示点云处理流程"""
    # 注意：PointCloudVisualizer 已经在文件开头导入了
    # 直接使用即可，不需要再次导入
    
    # 创建可视化器
    visualizer = PointCloudVisualizer()
    
    # 创建处理器
    processor = PointCloudProcessor()
    
    print("开始点云处理流程演示...")
    
    # 1. 创建示例点云
    original_pcd = visualizer.create_sample_pointcloud(2000, 'random')
    print(f"原始点云点数: {len(original_pcd.points)}")
    
    # 2. 下采样
    downsampled_pcd = processor.voxel_downsample(original_pcd, voxel_size=0.1)
    
    # 3. 去噪
    cleaned_pcd, _ = processor.remove_statistical_outlier(downsampled_pcd)
    
    # 4. 估计法向量
    final_pcd = processor.estimate_normals(cleaned_pcd)
    
    # 显示统计信息
    processor.print_statistics()
    
    # 可视化对比
    visualizer.visualize_comparison(original_pcd, final_pcd, "完整处理流程对比")

if __name__ == "__main__":
    demo_processing_pipeline()