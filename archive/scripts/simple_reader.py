#!/usr/bin/env python3
"""
简单点云读取示例 - 增强版
支持 Stanford 3D 数据读取
"""

import open3d as o3d
import numpy as np
import os
import sys

# 添加脚本路径，以便导入自定义模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from pointcloud_reader import PointCloudReader
    from pointcloud_visualizer import PointCloudVisualizer
    HAS_CUSTOM_MODULES = True
except ImportError:
    HAS_CUSTOM_MODULES = False
    print("⚠️ 无法导入自定义模块，使用基础功能")

def simple_read_demo():
    """简单读取演示 - 增强版"""
    print("🚀 增强版点云读取演示")
    print("=" * 50)
    
    # 可用的 Stanford 数据文件
    stanford_files = {
        "1": {"name": "Stanford Bunny", "path": "data/bunny/bunny.ply"},
        "2": {"name": "Stanford Dragon", "path": "data/dragon/dragon.ply"},
        "3": {"name": "随机生成点云", "path": "random"},
        "4": {"name": "自定义文件", "path": "custom"}
    }
    
    print("📚 可用的点云数据:")
    for key, file_info in stanford_files.items():
        if file_info["path"] != "random" and file_info["path"] != "custom":
            exists = os.path.exists(file_info["path"])
            status = "✅" if exists else "❌"
            print(f"  {key}. {file_info['name']} {status}")
        else:
            print(f"  {key}. {file_info['name']}")
    
    print("\n💡 提示: 如果 Stanford 数据不存在，请先运行下载脚本")
    
    # 选择要读取的文件
    while True:
        try:
            choice = input("\n请选择要读取的点云 (输入数字 1-4): ").strip()
            if choice in stanford_files:
                selected_file = stanford_files[choice]
                break
            else:
                print("❌ 无效选择，请输入 1-4")
        except KeyboardInterrupt:
            print("\n👋 用户取消操作")
            return None
    
    # 根据选择读取点云
    if selected_file["path"] == "random":
        print("🎲 创建随机点云...")
        pcd = create_random_pointcloud()
    elif selected_file["path"] == "custom":
        custom_path = input("请输入点云文件路径: ").strip()
        pcd = read_custom_file(custom_path)
    else:
        pcd = read_stanford_file(selected_file["path"], selected_file["name"])
    
    return pcd

def read_stanford_file(file_path, model_name):
    """读取 Stanford 数据文件"""
    print(f"📖 正在读取 {model_name}...")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        print("💡 请先运行下载脚本: python download_stanford_data.py")
        return create_random_pointcloud()  # 回退到随机点云
    
    try:
        # 使用自定义读取器（如果可用）或直接使用 Open3D
        if HAS_CUSTOM_MODULES:
            reader = PointCloudReader()
            pcd = reader.read_pointcloud(file_path, verbose=True)
        else:
            pcd = o3d.io.read_point_cloud(file_path)
            print_basic_info(pcd, model_name)
        
        if pcd and len(pcd.points) > 0:
            print(f"✅ {model_name} 读取成功!")
            return pcd
        else:
            print(f"❌ {model_name} 读取失败或文件为空")
            return create_random_pointcloud()
            
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return create_random_pointcloud()

def read_custom_file(file_path):
    """读取自定义文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return create_random_pointcloud()
    
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) > 0:
            print_basic_info(pcd, "自定义文件")
            print("✅ 自定义文件读取成功!")
            return pcd
        else:
            print("❌ 文件为空或读取失败")
            return create_random_pointcloud()
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return create_random_pointcloud()

def create_random_pointcloud():
    """创建随机点云作为演示"""
    print("🔄 创建随机点云进行演示...")
    
    # 创建更真实的随机点云（球体形状）
    num_points = 2000
    points = np.random.randn(num_points, 3)  # 正态分布
    points = points / np.linalg.norm(points, axis=1, keepdims=True) * 2  # 球体
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 添加渐变色
    colors = np.zeros((num_points, 3))
    z_values = points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    if z_max - z_min > 0:
        normalized_z = (z_values - z_min) / (z_max - z_min)
        colors[:, 0] = normalized_z  # 红色通道
        colors[:, 2] = 1 - normalized_z  # 蓝色通道
    else:
        colors = np.random.rand(num_points, 3)
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print("📊 随机点云信息:")
    print(f"  点数: {len(pcd.points)}")
    print(f"  颜色: {pcd.has_colors()}")
    
    return pcd

def print_basic_info(pcd, name):
    """打印点云基本信息"""
    print(f"📊 {name} 基本信息:")
    print(f"  点数: {len(pcd.points):,}")
    
    if len(pcd.points) > 0:
        points = np.asarray(pcd.points)
        bbox = pcd.get_axis_aligned_bounding_box()
        print(f"  边界框: {bbox.min_bound} -> {bbox.max_bound}")
        print(f"  尺寸: {bbox.max_bound - bbox.min_bound}")
    
    print(f"  颜色: {pcd.has_colors()}")
    print(f"  法向量: {pcd.has_normals()}")

def visualize_with_options(pcd, name):
    """带选项的可视化"""
    if pcd is None:
        print("❌ 没有点云数据可可视化")
        return
    
    print(f"\n🎨 准备可视化: {name}")
    print("=" * 40)
    
    # 可视化选项
    print("可视化选项:")
    print("  1. 🔍 基本可视化")
    print("  2. 🎨 添加颜色效果")
    print("  3. 📐 显示坐标轴")
    print("  4. ❌ 跳过可视化")
    
    while True:
        try:
            viz_choice = input("请选择可视化选项 (1-4): ").strip()
            if viz_choice in ["1", "2", "3", "4"]:
                break
            else:
                print("❌ 无效选择，请输入 1-4")
        except KeyboardInterrupt:
            print("\n👋 用户取消操作")
            return
    
    if viz_choice == "4":
        print("⏭️ 跳过可视化")
        return
    
    # 准备可视化
    geometries = [pcd]
    window_name = f"{name} - 点云可视化"
    
    if viz_choice == "2" and not pcd.has_colors():
        # 添加颜色
        points = np.asarray(pcd.points)
        if len(points) > 0:
            # 基于Z坐标的渐变色
            z_values = points[:, 2]
            z_min, z_max = z_values.min(), z_values.max()
            
            colors = np.zeros_like(points)
            if z_max - z_min > 0:
                normalized_z = (z_values - z_min) / (z_max - z_min)
                colors[:, 0] = normalized_z  # 红色
                colors[:, 1] = 1 - normalized_z  # 绿色
                colors[:, 2] = 0.5  # 蓝色分量固定
            else:
                colors = np.ones_like(points) * 0.7  # 灰色
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            print("✅ 已添加颜色效果")
    
    if viz_choice == "3":
        # 添加坐标轴
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        geometries.append(coordinate_frame)
        window_name += " + 坐标轴"
        print("✅ 已添加坐标轴")
    
    # 可视化
    print("\n🖱️ 可视化操作指南:")
    print("  鼠标左键: 旋转视角")
    print("  鼠标滚轮: 缩放")
    print("  鼠标右键: 平移")
    print("  F键: 切换全屏")
    print("  ESC键: 关闭窗口")
    print("  H键: 显示帮助")
    
    print(f"\n正在打开可视化窗口: {window_name}...")
    
    try:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1024,
            height=768,
            left=50,  # 窗口位置
            top=50
        )
        print("✅ 可视化完成")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

def process_pointcloud(pcd, name):
    """点云处理选项"""
    if pcd is None:
        return pcd
    
    print(f"\n🔧 {name} 处理选项")
    print("=" * 40)
    print("  1. 📉 下采样")
    print("  2. 🧹 去噪")
    print("  3. 📐 估计法向量")
    print("  4. ⏭️ 跳过处理")
    
    while True:
        try:
            process_choice = input("请选择处理选项 (1-4): ").strip()
            if process_choice in ["1", "2", "3", "4"]:
                break
            else:
                print("❌ 无效选择，请输入 1-4")
        except KeyboardInterrupt:
            print("\n👋 用户取消操作")
            return pcd
    
    if process_choice == "4":
        return pcd
    
    try:
        if process_choice == "1":
            # 下采样
            voxel_size = float(input("请输入体素尺寸 (推荐 0.01): ") or "0.01")
            down_pcd = pcd.voxel_down_sample(voxel_size)
            print(f"✅ 下采样完成: {len(pcd.points)} -> {len(down_pcd.points)} 点")
            return down_pcd
            
        elif process_choice == "2":
            # 去噪
            nb_neighbors = int(input("请输入邻居数量 (推荐 20): ") or "20")
            std_ratio = float(input("请输入标准差倍数 (推荐 2.0): ") or "2.0")
            
            cleaned_pcd, ind = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, 
                std_ratio=std_ratio
            )
            removed_count = len(pcd.points) - len(cleaned_pcd.points)
            print(f"✅ 去噪完成: 移除了 {removed_count} 个噪声点")
            return cleaned_pcd
            
        elif process_choice == "3":
            # 估计法向量
            pcd.estimate_normals()
            pcd.orient_normals_to_align_with_direction()
            print("✅ 法向量估计完成")
            return pcd
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return pcd
    
    return pcd

def save_pointcloud(pcd, name):
    """保存点云选项"""
    if pcd is None:
        return
    
    print(f"\n💾 {name} 保存选项")
    print("=" * 40)
    save_choice = input("是否保存点云? (y/n): ").strip().lower()
    
    if save_choice in ['y', 'yes']:
        filename = input("请输入保存文件名 (例如: my_pointcloud.ply): ").strip()
        if not filename:
            filename = "processed_pointcloud.ply"
        
        try:
            # 确保文件扩展名
            if not filename.lower().endswith(('.ply', '.pcd', '.xyz')):
                filename += '.ply'
            
            success = o3d.io.write_point_cloud(filename, pcd)
            if success:
                print(f"✅ 点云已保存: {filename}")
            else:
                print(f"❌ 保存失败: {filename}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

def main():
    """主函数"""
    print("🚀 增强版点云读取工具")
    print("=" * 60)
    
    # 读取点云
    pcd = simple_read_demo()
    
    if pcd is None:
        return
    
    # 获取点云名称用于显示
    pcd_name = "点云数据"
    if hasattr(pcd, 'name'):
        pcd_name = pcd.name
    
    # 处理点云
    processed_pcd = process_pointcloud(pcd, pcd_name)
    
    # 可视化
    visualize_with_options(processed_pcd, pcd_name)
    
    # 保存
    save_pointcloud(processed_pcd, pcd_name)
    
    print("\n" + "=" * 60)
    print("🎉 点云处理流程完成!")
    print("💡 提示: 可以再次运行此脚本处理其他点云")

if __name__ == "__main__":
    main()