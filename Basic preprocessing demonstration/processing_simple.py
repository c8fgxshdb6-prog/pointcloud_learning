import open3d as o3d
import numpy as np
import copy

class PointCloudComparator:
    """点云对比分析器 - 完整的类定义"""
    
    def __init__(self):
        self.comparison_stats = {}
    
    def visualize_comparison_advanced(self, original, processed, title="点云对比", mode="side_by_side"):
        """
        改进的多视图对比可视化
        
        参数:
        - original: 原始点云
        - processed: 处理后的点云  
        - title: 窗口标题
        - mode: 显示模式 ("side_by_side", "overlay", "before_after")
        """
        
        print(f"\n=== {title} ===")
        
        # 创建副本，避免修改原始数据
        original_copy = copy.deepcopy(original)
        processed_copy = copy.deepcopy(processed)
        
        # 计算统计信息
        stats = self._calculate_comparison_stats(original_copy, processed_copy)
        self._print_statistics(stats)
        
        if mode == "side_by_side":
            self._visualize_side_by_side(original_copy, processed_copy, title, stats)
        elif mode == "overlay":
            self._visualize_overlay(original_copy, processed_copy, title, stats)
        elif mode == "before_after":
            self._visualize_before_after(original_copy, processed_copy, title, stats)
    
    def _calculate_comparison_stats(self, original, processed):
        """计算对比统计信息"""
        stats = {
            'original_points': len(original.points),
            'processed_points': len(processed.points),
            'points_removed': len(original.points) - len(processed.points),
            'reduction_ratio': f"{(1 - len(processed.points)/len(original.points))*100:.1f}%",
            'original_bounds': original.get_axis_aligned_bounding_box(),
            'processed_bounds': processed.get_axis_aligned_bounding_box()
        }
        return stats
    
    def _print_statistics(self, stats):
        """打印统计信息"""
        print("📊 对比统计信息:")
        print(f"  原始点数: {stats['original_points']}")
        print(f"  处理后点数: {stats['processed_points']}")
        print(f"  移除点数: {stats['points_removed']}")
        print(f"  减少比例: {stats['reduction_ratio']}")
    
    def _visualize_side_by_side(self, original, processed, title, stats):
        """并排显示模式"""
        # 移动处理后的点云，避免重叠
        processed_moved = processed.translate([3, 0, 0])  # 向右移动3个单位
        
        # 设置不同的颜色
        original.paint_uniform_color([0, 0.5, 1])  # 蓝色
        processed_moved.paint_uniform_color([1, 0.3, 0.3])  # 红色
        
        # 添加坐标轴和标签
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        
        o3d.visualization.draw_geometries(
            [original, processed_moved, coordinate_frame],
            window_name=f"{title} - 并排对比",
            width=1200,
            height=600
        )
    
    def _visualize_overlay(self, original, processed, title, stats):
        """叠加显示模式"""
        # 设置不同颜色
        original.paint_uniform_color([0, 0.5, 1])  # 蓝色
        processed.paint_uniform_color([1, 0.3, 0.3])  # 红色
        
        o3d.visualization.draw_geometries(
            [original, processed],
            window_name=f"{title} - 叠加对比"
        )
    
    def _visualize_before_after(self, original, processed, title, stats):
        """前后对比显示模式"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{title} - 前后对比", width=1000, height=600)
        
        # 添加几何体
        original.paint_uniform_color([0, 0.5, 1])  # 蓝色
        vis.add_geometry(original)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.point_size = 3.0
        render_option.background_color = [0.1, 0.1, 0.1]
        
        print("显示原始点云... (按N键切换到处理后)")
        
        current_view = 'original'
        vis.update_geometry(original)
        vis.poll_events()
        vis.update_renderer()
        
        # 简单的键盘交互
        def key_callback(vis, action, mods):
            nonlocal current_view
            if action == 1:  # 按键按下
                if current_view == 'original':
                    # 切换到处理后
                    vis.clear_geometries()
                    processed.paint_uniform_color([1, 0.3, 0.3])
                    vis.add_geometry(processed)
                    current_view = 'processed'
                    print("切换到处理后点云 (按N键返回原始)")
                else:
                    # 切换回原始
                    vis.clear_geometries()
                    original.paint_uniform_color([0, 0.5, 1])
                    vis.add_geometry(original)
                    current_view = 'original'
                    print("切换回原始点云 (按N键切换到处理后)")
                return True
            return False
        
        vis.register_key_callback(ord('N'), key_callback)
        vis.run()
        vis.destroy_window()

class PointCloudProcessor:
    """点云处理与对比分析器"""
    
    def __init__(self):
        self.results = {}
    
    def compare_processing_results_comprehensive(self, data_source="synthetic"):
        """
        完整的点云处理效果对比分析
        
        参数:
        - data_source: 数据源类型 ("synthetic", "file")
        """
        
        print("=== 点云处理算法效果对比分析 ===")
        
        # 1. 获取点云数据
        original_pcd = self._load_pointcloud_data(data_source)
        if original_pcd is None:
            print("❌ 无法加载点云数据")
            return
        
        print(f"✅ 成功加载点云，包含 {len(original_pcd.points)} 个点")
        
        # 2. 应用多种处理方法
        processing_results = self._apply_all_processing_methods(original_pcd)
        
        # 3. 系统化对比分析
        self._comprehensive_comparison_analysis(original_pcd, processing_results)
        
        # 4. 参数敏感性分析
        self._parameter_sensitivity_analysis(original_pcd)
    
    def _load_pointcloud_data(self, data_source):
        """加载点云数据"""
        if data_source == "synthetic":
            return self._create_synthetic_pointcloud()
        elif data_source == "file":
            # 这里可以扩展为从文件加载
            try:
                # 示例：使用Open3D自带的示例数据
                bunny = o3d.data.BunnyMesh()
                pcd = o3d.io.read_point_cloud(bunny.path)
                return pcd
            except:
                print("⚠️ 无法加载示例文件，使用合成数据替代")
                return self._create_synthetic_pointcloud()
        else:
            return self._create_synthetic_pointcloud()
    
    def _create_synthetic_pointcloud(self):
        """创建合成点云数据（带噪声）"""
        print("创建合成点云数据...")
        
        # 创建基础点云（平面+球体）
        points = []
        
        # 平面点云
        for i in range(30):
            for j in range(30):
                x = i * 0.1 - 1.5
                y = j * 0.1 - 1.5
                z = 0
                points.append([x, y, z])
        
        # 球体点云
        for i in range(500):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = 1.0
            x = r * np.sin(phi) * np.cos(theta) + 1.0
            y = r * np.sin(phi) * np.sin(theta) + 1.0
            z = r * np.cos(phi)
            points.append([x, y, z])
        
        # 添加噪声点
        noise_points = np.random.uniform(-2, 2, (100, 3))
        points.extend(noise_points)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色
        
        print(f"合成点云创建完成: {len(pcd.points)} 个点")
        return pcd
    
    def _apply_all_processing_methods(self, original_pcd):
        """应用所有处理方法"""
        results = {}
        
        print("\n应用点云处理方法...")
        
        # 1. 下采样处理（不同参数）
        print("1. 下采样处理")
        results['downsample_0.02'] = original_pcd.voxel_down_sample(0.02)
        results['downsample_0.05'] = original_pcd.voxel_down_sample(0.05)
        results['downsample_0.1'] = original_pcd.voxel_down_sample(0.1)
        
        # 2. 去噪处理（不同参数）
        print("2. 去噪处理")
        results['denoise_10_1.0'], _ = original_pcd.remove_statistical_outlier(
            nb_neighbors=10, std_ratio=1.0)
        results['denoise_20_2.0'], _ = original_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        results['denoise_30_3.0'], _ = original_pcd.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=3.0)
        
        # 3. 组合处理（先下采样再去噪）
        print("3. 组合处理")
        downsampled = original_pcd.voxel_down_sample(0.05)
        results['combined'], _ = downsampled.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        
        return results
    
    def _comprehensive_comparison_analysis(self, original_pcd, processing_results):
        """综合对比分析"""
        print("\n" + "="*50)
        print("综合对比分析")
        print("="*50)
        
        # 创建对比器
        comparator = PointCloudComparator()
        
        # 对比下采样效果
        print("\n📊 下采样效果对比:")
        comparator.visualize_comparison_advanced(
            original_pcd, processing_results['downsample_0.05'], 
            "下采样对比 (体素尺寸 0.05)", "side_by_side"
        )
        
        # 对比去噪效果
        print("\n📊 去噪效果对比:")
        comparator.visualize_comparison_advanced(
            original_pcd, processing_results['denoise_20_2.0'],
            "去噪对比 (邻居20, 标准差2.0)", "side_by_side"
        )
        
        # 对比组合处理效果
        print("\n📊 组合处理效果对比:")
        comparator.visualize_comparison_advanced(
            original_pcd, processing_results['combined'],
            "组合处理效果", "side_by_side"
        )
    
    def _parameter_sensitivity_analysis(self, original_pcd):
        """参数敏感性分析"""
        print("\n" + "="*50)
        print("参数敏感性分析")
        print("="*50)
        
        comparator = PointCloudComparator()
        
        # 不同下采样参数对比
        print("\n🔬 下采样参数敏感性:")
        voxel_sizes = [0.02, 0.05, 0.1]
        for size in voxel_sizes:
            downsampled = original_pcd.voxel_down_sample(size)
            print(f"体素尺寸 {size}: {len(downsampled.points)} 个点")
            
            # 可视化其中一个示例
            if size == 0.05:
                comparator.visualize_comparison_advanced(
                    original_pcd, downsampled,
                    f"下采样参数对比 (体素尺寸 {size})", "side_by_side"
                )
        
        # 不同去噪参数对比
        print("\n🔬 去噪参数敏感性:")
        denoise_params = [(10, 1.0), (20, 2.0), (30, 3.0)]
        for nb_neighbors, std_ratio in denoise_params:
            denoised, indices = original_pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            print(f"邻居数 {nb_neighbors}, 标准差比率 {std_ratio}: "
                  f"{len(denoised.points)} 个点 (移除 {len(original_pcd.points) - len(denoised.points)})")
            
            # 可视化其中一个示例
            if nb_neighbors == 20 and std_ratio == 2.0:
                comparator.visualize_comparison_advanced(
                    original_pcd, denoised,
                    f"去噪参数对比 (邻居{nb_neighbors}, 标准差{std_ratio})", "side_by_side"
                )

# 简化的入门版本（适合初学者）
def simple_comparison_demo():
    """简化版的点云处理对比演示"""
    print("=== 简化版点云处理对比演示 ===")
    
    # 创建简单点云
    points = np.random.rand(800, 3)
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(points)
    original_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    
    # 应用处理
    downsampled = original_pcd.voxel_down_sample(0.1)
    denoised, _ = original_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 使用对比器
    comparator = PointCloudComparator()
    
    print("1. 下采样效果对比")
    comparator.visualize_comparison_advanced(original_pcd, downsampled, "下采样效果")
    
    print("2. 去噪效果对比")
    comparator.visualize_comparison_advanced(original_pcd, denoised, "去噪效果")

# 运行演示
if __name__ == "__main__":1

print("选择运行模式:")
print("1. 简化版演示 (推荐初学者)")
print("2. 完整版分析")
    
choice = input("请输入选择 (1 或 2): ").strip()
    
if choice == "1":
        simple_comparison_demo()
else:
        processor = PointCloudProcessor()
        processor.compare_processing_results_comprehensive(data_source="synthetic")