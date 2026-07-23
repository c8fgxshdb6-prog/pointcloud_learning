#!/usr/bin/env python3
"""
点云读取工具类
提供各种点云文件的读取和验证功能
"""

import open3d as o3d
import numpy as np
import os
from typing import Optional, Tuple, List

class PointCloudReader:
    """点云读取工具类"""
    
    def __init__(self):
        self.supported_formats = ['.ply', '.pcd', '.xyz', '.xyzn', '.pts']
    
    def read_pointcloud(self, file_path: str, verbose: bool = True) -> Optional[o3d.geometry.PointCloud]:
        """
        读取点云文件
        
        Args:
            file_path: 点云文件路径
            verbose: 是否显示详细信息
            
        Returns:
            点云对象，如果读取失败返回None
        """
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return None
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            print(f"❌ 不支持的文件格式: {file_ext}")
            print(f"✅ 支持的格式: {', '.join(self.supported_formats)}")
            return None
        
        try:
            if verbose:
                print(f"📖 正在读取点云文件: {file_path}")
            
            # 读取点云
            pcd = o3d.io.read_point_cloud(file_path)
            
            if verbose:
                self._print_pointcloud_info(pcd, file_path)
            
            return pcd
            
        except Exception as e:
            print(f"❌ 读取点云失败: {e}")
            return None
    
    def _print_pointcloud_info(self, pcd: o3d.geometry.PointCloud, file_path: str):
        """打印点云信息"""
        print("=" * 50)
        print(f"📊 点云信息 - {os.path.basename(file_path)}")
        print("=" * 50)
        print(f"📏 点数: {len(pcd.points):,}")
        
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            bbox = pcd.get_axis_aligned_bounding_box()
            
            print(f"📐 边界框:")
            print(f"   最小: {bbox.min_bound}")
            print(f"   最大: {bbox.max_bound}")
            print(f"   尺寸: {bbox.max_bound - bbox.min_bound}")
            
            # 统计信息
            print(f"📈 坐标统计:")
            print(f"   X范围: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
            print(f"   Y范围: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
            print(f"   Z范围: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        print(f"🎨 颜色信息: {pcd.has_colors()}")
        print(f"📐 法向量: {pcd.has_normals()}")
        print("=" * 50)
    
    def read_multiple_files(self, file_paths: List[str]) -> List[o3d.geometry.PointCloud]:
        """
        批量读取多个点云文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            点云对象列表
        """
        pointclouds = []
        successful_reads = 0
        
        for file_path in file_paths:
            pcd = self.read_pointcloud(file_path, verbose=True)
            if pcd is not None:
                pointclouds.append(pcd)
                successful_reads += 1
            print()  # 空行分隔
        
        print(f"✅ 成功读取 {successful_reads}/{len(file_paths)} 个文件")
        return pointclouds
    
    def read_from_folder(self, folder_path: str, 
                        extensions: List[str] = None) -> List[o3d.geometry.PointCloud]:
        """
        从文件夹读取所有点云文件
        
        Args:
            folder_path: 文件夹路径
            extensions: 要读取的文件扩展名列表
            
        Returns:
            点云对象列表
        """
        if extensions is None:
            extensions = self.supported_formats
        
        if not os.path.exists(folder_path):
            print(f"❌ 文件夹不存在: {folder_path}")
            return []
        
        pointclouds = []
        file_count = 0
        
        print(f"🔍 在文件夹中搜索点云文件: {folder_path}")
        
        for filename in os.listdir(folder_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in extensions:
                file_path = os.path.join(folder_path, filename)
                pcd = self.read_pointcloud(file_path, verbose=True)
                if pcd is not None:
                    pointclouds.append(pcd)
                    file_count += 1
                print()  # 空行分隔
        
        print(f"✅ 从文件夹读取了 {file_count} 个点云文件")
        return pointclouds

# 使用示例和测试函数
def demo_reading_operations():
    """演示点云读取操作"""
    reader = PointCloudReader()
    
    print("🎯 Open3D 点云读取演示")
    print("=" * 60)
    
    # 1. 使用Open3D内置示例数据
    print("\n1. 📦 使用Open3D内置示例数据")
    try:
        # 尝试获取示例点云数据
        demo_data = o3d.data.DemoICPPointClouds()
        source_path = demo_data.paths[0]
        target_path = demo_data.paths[1]
        
        print("正在下载示例数据...")
        source_pcd = reader.read_pointcloud(source_path)
        target_pcd = reader.read_pointcloud(target_path)
        
        if source_pcd and target_pcd:
            print("✅ 示例数据读取成功!")
        else:
            print("❌ 示例数据读取失败，使用随机生成的数据")
            raise Exception("示例数据不可用")
            
    except Exception as e:
        print(f"⚠️ 示例数据不可用: {e}")
        print("🔄 使用随机生成的点云进行演示...")
        
        # 创建随机点云作为演示
        points = np.random.rand(1000, 3)
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(points)
        
        points2 = np.random.rand(800, 3) + 1.0  # 偏移一些
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(points2)
    
    # 2. 可视化读取的点云
    print("\n2. 👀 可视化点云")
    if source_pcd and target_pcd:
        # 给点云添加不同颜色以便区分
        source_pcd.paint_uniform_color([1, 0, 0])  # 红色
        target_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
        
        print("正在打开可视化窗口...")
        print("操作提示:")
        print("  🟥 红色点云: 源点云")
        print("  🟦 蓝色点云: 目标点云") 
        print("  🖱️ 鼠标左键旋转, 滚轮缩放, 右键平移")
        print("  ⎋ 按ESC关闭窗口")
        
        o3d.visualization.draw_geometries(
            [source_pcd, target_pcd],
            window_name="点云读取演示 - 红色:源点云 蓝色:目标点云",
            width=1024,
            height=768
        )
    
    # 3. 演示批量读取（模拟）
    print("\n3. 📚 批量读取演示")
    test_files = [
        "example1.ply",
        "example2.pcd", 
        "example3.xyz"
    ]
    
    print("模拟批量读取以下文件:")
    for file in test_files:
        print(f"  📄 {file}")
    
    print("\n💡 实际使用时，请将真实文件路径传入 read_multiple_files() 方法")
    
    return reader

def create_sample_pointcloud_files():
    """创建示例点云文件用于测试"""
    print("🛠️ 创建示例点云文件...")
    
    # 创建数据文件夹
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. 创建简单立方体点云
    cube_points = np.random.rand(500, 3) * 2 - 1  # [-1, 1]范围内的点
    cube_pcd = o3d.geometry.PointCloud()
    cube_pcd.points = o3d.utility.Vector3dVector(cube_points)
    cube_pcd.paint_uniform_color([1, 0, 0])  # 红色
    
    # 2. 创建球体点云
    phi = np.random.uniform(0, 2*np.pi, 300)
    theta = np.random.uniform(0, np.pi, 300)
    r = 1.0
    sphere_points = np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi), 
        r * np.cos(theta)
    ]).T
    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd.points = o3d.utility.Vector3dVector(sphere_points)
    sphere_pcd.paint_uniform_color([0, 1, 0])  # 绿色
    
    # 保存文件
    cube_file = os.path.join(data_dir, "sample_cube.ply")
    sphere_file = os.path.join(data_dir, "sample_sphere.pcd")
    
    o3d.io.write_point_cloud(cube_file, cube_pcd)
    o3d.io.write_point_cloud(sphere_file, sphere_pcd)
    
    print(f"✅ 创建示例文件:")
    print(f"   📄 {cube_file}")
    print(f"   📄 {sphere_file}")
    
    return [cube_file, sphere_file]

if __name__ == "__main__":
    # 创建示例文件
    sample_files = create_sample_pointcloud_files()
    
    # 演示读取操作
    reader = demo_reading_operations()
    
    # 读取刚刚创建的示例文件
    print("\n4. 🧪 测试实际文件读取")
    pointclouds = reader.read_multiple_files(sample_files)
    
    if pointclouds:
        print(f"✅ 成功读取 {len(pointclouds)} 个点云文件")
        print("🎉 点云读取功能测试完成!")
    else:
        print("❌ 文件读取测试失败")