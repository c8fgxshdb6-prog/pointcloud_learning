import open3d as o3d
import numpy as np

def demonstrate_normals():
    print("=== 法向量估计演示 ===")
    
    # 创建示例点云（一个倾斜的平面）
    points = []
    for i in range(20):
        for j in range(20):
            x = i * 0.1
            y = j * 0.1
            z = 0.5 * x + 0.3 * y  # 创建倾斜平面
            points.append([x, y, z])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    
    print(f"创建了点云，包含 {len(pcd.points)} 个点")
    
    # 估计法向量
    print("估计法向量...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30)
    )
    
    # 统一法向量方向
    pcd.orient_normals_to_align_with_direction()
    
    # 检查法向量
    normals = np.asarray(pcd.normals)
    print(f"法向量形状: {normals.shape}")
    print(f"前5个点的法向量:")
    for i in range(5):
        print(f"  点{i}: {normals[i]}")
    
    # 可视化（带法向量显示）
    print("打开可视化窗口...")
    print("红色线段表示法向量方向")
    o3d.visualization.draw_geometries([pcd],
                                      window_name="点云法向量可视化",
                                      point_show_normal=True)
    
    # 分析法向量的一致性
    normal_lengths = np.linalg.norm(normals, axis=1)
    print(f"法向量长度统计: 平均={np.mean(normal_lengths):.3f}, 标准差={np.std(normal_lengths):.3f}")

if __name__ == "__main__":
    demonstrate_normals()