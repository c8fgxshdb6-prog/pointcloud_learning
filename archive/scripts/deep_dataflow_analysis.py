# scripts/deep_dataflow_analysis.py
"""
PointNet++ 数据流深度分析
重点：理解每个SA层的内部机制
"""
import torch
import torch.nn as nn
import sys
import os
from datetime import datetime

# 导入PointNet++相关模块
sys.path.append('Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals')
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class DetailedDataflowAnalyzer:
    """详细数据流分析器"""
    
    def __init__(self):
        # 存储各阶段的详细数据
        self.dataflow_records = []
        self.layer_details = {}
        
    def analyze_sa_layer(self, layer_name, sa_module, xyz, points=None):
        """
        深度分析一个SA层的数据流
        
        关键点：
        1. 采样：如何选择关键点
        2. 分组：如何构建局部邻域
        3. PointNet：如何提取局部特征
        """
        print(f"\n{'='*60}")
        print(f"深度分析 {layer_name} 层")
        print(f"{'='*60}")
        
        # 记录输入
        input_info = {
            'layer': layer_name,
            'stage': 'input',
            'xyz_shape': list(xyz.shape) if xyz is not None else None,
            'points_shape': list(points.shape) if points is not None else None,
        }
        self.dataflow_records.append(input_info)
        
        print(f"输入:")
        print(f"  xyz形状: {xyz.shape}")  # [B, 3, N]
        print(f"  点数 N: {xyz.shape[-1]}")
        
        if points is not None:
            print(f"  特征形状: {points.shape}")  # [B, C, N]
            print(f"  特征维度 C: {points.shape[1]}")
        
        # 模拟SA层的前向传播（手动分解）
        # 注意：这里我们假设SA层内部按标准流程工作
        
        # SA层的关键参数分析
        if isinstance(sa_module, PointNetSetAbstractionMsg):
            print(f"\n{layer_name} 是多尺度分组 (PointNetSetAbstractionMsg)")
            # 分析多尺度分组
            self._analyze_multi_scale(layer_name, sa_module, xyz, points)
        else:
            print(f"\n{layer_name} 是单尺度分组 (PointNetSetAbstraction)")
        
        # 执行实际的前向传播
        with torch.no_grad():
            new_xyz, new_points = sa_module(xyz, points)
        
        # 记录输出
        output_info = {
            'layer': layer_name,
            'stage': 'output',
            'new_xyz_shape': list(new_xyz.shape),
            'new_points_shape': list(new_points.shape),
            'point_reduction': xyz.shape[-1] / new_xyz.shape[-1] if new_xyz.shape[-1] > 0 else 0,
            'feature_expansion': new_points.shape[1] / (points.shape[1] if points is not None else 3),
        }
        self.dataflow_records.append(output_info)
        
        print(f"\n输出:")
        print(f"  新xyz形状: {new_xyz.shape}")
        print(f"  新特征形状: {new_points.shape}")
        print(f"  点数压缩率: {output_info['point_reduction']:.1f}x")
        print(f"  特征维度扩展: {output_info['feature_expansion']:.1f}x")
        
        return new_xyz, new_points
    
    def _analyze_multi_scale(self, layer_name, sa_module, xyz, points):
        """分析多尺度分组的具体参数"""
        # 这些参数通常存储在SA层的属性中
        print(f"多尺度参数:")
        
        # 尝试访问SA层的属性（具体属性名可能因实现而异）
        try:
            # 采样点数
            if hasattr(sa_module, 'npoint'):
                print(f"  采样点数: {sa_module.npoint}")
            
            # 半径列表
            if hasattr(sa_module, 'radius_list'):
                print(f"  分组半径: {sa_module.radius_list}")
            
            # 每组点数
            if hasattr(sa_module, 'nsample_list'):
                print(f"  每组点数: {sa_module.nsample_list}")
            
            # MLP配置
            if hasattr(sa_module, 'mlp_convs'):
                print(f"  MLP层数: {len(sa_module.mlp_convs)}")
                for i, conv in enumerate(sa_module.mlp_convs):
                    if hasattr(conv, 'out_channels'):
                        print(f"    第{i+1}层输出通道: {conv.out_channels}")
        except:
            print("  无法直接访问多尺度参数")
        
        print(f"\n多尺度分组过程:")
        print("  1. 对每个尺度:")
        print("     a) 以每个关键点为中心，在指定半径内分组")
        print("     b) 每组包含固定数量的最近邻点")
        print("     c) 对每组点应用共享的PointNet (MLP)")
        print("  2. 将所有尺度的特征拼接起来")
    
    def analyze_feature_evolution(self, model, test_input):
        """分析整个模型的详细特征演化"""
        print("\n" + "="*60)
        print("PointNet++ 完整数据流分析")
        print("="*60)
        
        B, C, N = test_input.shape
        print(f"测试输入: 批次={B}, 通道={C}, 点数={N}")
        
        # 手动执行每一层，深度分析
        xyz = test_input
        
        # SA1层
        l1_xyz, l1_points = self.analyze_sa_layer(
            "SA1", model.sa1, xyz, None
        )
        
        # SA2层
        l2_xyz, l2_points = self.analyze_sa_layer(
            "SA2", model.sa2, l1_xyz, l1_points
        )
        
        # SA3层
        l3_xyz, l3_points = self.analyze_sa_layer(
            "SA3", model.sa3, l2_xyz, l2_points
        )
        
        # 全局特征分析
        print(f"\n{'='*60}")
        print("全局特征分析")
        print(f"{'='*60}")
        
        B = l3_points.shape[0]
        global_feature = l3_points.view(B, -1)
        
        print(f"SA3输出形状: {l3_points.shape}")
        print(f"全局特征形状 (展平后): {global_feature.shape}")
        print(f"特征维度: 1024维")
        
        # 全连接层分析
        print(f"\n{'='*60}")
        print("全连接层分析")
        print(f"{'='*60}")
        
        # 手动执行全连接层
        x = global_feature
        
        for name, module in model.named_children():
            if name.startswith('fc') or name.startswith('bn') or name.startswith('drop'):
                input_shape = x.shape
                with torch.no_grad():
                    x = module(x)
                output_shape = x.shape
                
                print(f"{name}:")
                print(f"  输入形状: {input_shape}")
                print(f"  输出形状: {output_shape}")
                
                if hasattr(module, 'weight'):
                    print(f"  权重形状: {module.weight.shape}")
                
                if name.startswith('drop'):
                    print(f"  Dropout率: {module.p}")
        
        return self.dataflow_records
    
    def generate_comprehensive_report(self, records):
        """生成综合数据流报告"""
        print("\n" + "="*70)
        print("POINTNET++ 数据流综合报告")
        print("="*70)
        
        # 提取关键信息
        sa_layers = [r for r in records if r['layer'].startswith('SA') and r['stage'] == 'output']
        
        print("\n一、层次化特征提取过程")
        print("-"*70)
        
        for i, layer in enumerate(sa_layers):
            layer_name = layer['layer']
            print(f"\n{layer_name}层:")
            print(f"  输入点数: {records[i*2]['xyz_shape'][-1]}")
            print(f"  输出点数: {layer['new_xyz_shape'][-1]}")
            print(f"  输出特征维度: {layer['new_points_shape'][1]}")
            print(f"  点数压缩率: {layer['point_reduction']:.1f}x")
            print(f"  特征维度扩展: {layer['feature_expansion']:.1f}x")
            
            # 语义含义解释
            if layer_name == 'SA1':
                print(f"  语义级别: 局部几何模式")
                print(f"  包含信息: 点法向量、局部曲率、边缘特征")
                print(f"  适用任务: 精细重建、点云补全")
                
            elif layer_name == 'SA2':
                print(f"  语义级别: 部件级结构")
                print(f"  包含信息: 简单部件(桌腿、椅背)、基本形状")
                print(f"  适用任务: 部件分割、物体识别")
                
            elif layer_name == 'SA3':
                print(f"  语义级别: 全局语义")
                print(f"  包含信息: 物体类别、整体结构")
                print(f"  适用任务: 分类、检索")
        
        print("\n二、数据流演化总结")
        print("-"*70)
        
        # 创建演化表格
        headers = ["层", "点数", "特征维", "总元素", "压缩率", "语义级别"]
        data = []
        
        total_elements_original = 0
        for i, layer in enumerate(sa_layers):
            input_record = records[i*2]  # 对应的输入记录
            
            if i == 0:
                input_points = input_record['xyz_shape'][-1]
                input_features = 3  # 坐标维度
            else:
                input_points = input_record['xyz_shape'][-1]
                input_features = input_record['points_shape'][1] if input_record['points_shape'] else 3
            
            output_points = layer['new_xyz_shape'][-1]
            output_features = layer['new_points_shape'][1]
            
            input_elements = input_points * input_features
            output_elements = output_points * output_features
            
            # 第一层的输入元素
            if i == 0:
                total_elements_original = input_elements
            
            compression_rate = output_elements / input_elements if input_elements > 0 else 0
            
            # 语义级别
            semantic_levels = ["局部几何", "部件结构", "全局语义"]
            
            data.append([
                layer['layer'],
                f"{input_points} → {output_points}",
                f"{input_features} → {output_features}",
                f"{input_elements:,} → {output_elements:,}",
                f"{compression_rate:.2f}",
                semantic_levels[i] if i < len(semantic_levels) else "未知"
            ])
        
        # 打印表格
        print(f"{'层':<6} {'点数':<15} {'特征维':<15} {'总元素':<25} {'压缩率':<10} {'语义级别':<10}")
        print("-"*70)
        for row in data:
            print(f"{row[0]:<6} {row[1]:<15} {row[2]:<15} {row[3]:<25} {row[4]:<10} {row[5]:<10}")
        
        # 计算总体压缩
        if sa_layers:
            final_layer = sa_layers[-1]
            final_elements = final_layer['new_points_shape'][0] * \
                            final_layer['new_points_shape'][1] * \
                            final_layer['new_points_shape'][2]
            
            overall_compression = final_elements / total_elements_original if total_elements_original > 0 else 0
            
            print(f"\n总体信息压缩: {total_elements_original:,} → {final_elements:,} 元素")
            print(f"总体压缩率: {overall_compression:.4f} ({1/overall_compression:.1f}x 压缩)")
        
        print("\n三、语义通信应用建议")
        print("-"*70)
        print("基于数据流分析，为语义通信提供的特征提取建议:")
        
        print("\n1. 特征层次选择:")
        print("   a) 高带宽信道: SA1 + SA2 + SA3 (完整层次)")
        print("      - 优点: 信息完整，支持多种下游任务")
        print("      - 缺点: 数据量大，传输成本高")
        print("      - 数据量: ~1.9MB")
        
        print("\n   b) 中带宽信道: SA2 + SA3 (中高层次)")
        print("      - 优点: 平衡信息量和传输成本")
        print("      - 缺点: 丢失局部细节")
        print("      - 数据量: ~648KB")
        
        print("\n   c) 低带宽信道: SA3 (仅高层)")
        print("      - 优点: 数据量小，传输快")
        print("      - 缺点: 只能用于分类等高层任务")
        print("      - 数据量: ~8KB")
        
        print("\n   d) 自适应选择: 根据信道质量动态选择层次")
        print("      - 实时监测信道条件")
        print("      - 动态调整传输的特征层次")
        print("      - 平衡传输可靠性和信息完整性")
        
        print("\n2. 特征编码优化:")
        print("   - 对SA1特征: 使用高压缩率的编码（有损）")
        print("   - 对SA3特征: 使用低压缩率的编码（无损或近无损）")
        print("   - 考虑特征重要性: 对重要特征维度分配更多比特")
        
        print("\n3. 传输策略:")
        print("   - 分层传输: 先传高层特征，再根据需要传输低层特征")
        print("   - 渐进式传输: 从粗糙到精细逐步传输")
        print("   - 错误保护: 对高层特征增加更强的错误保护编码")
        
        # 保存详细报告
        self._save_detailed_report(records, sa_layers)
    
    def _save_detailed_report(self, records, sa_layers):
        """保存详细报告到文件"""
        import json
        from datetime import datetime
        
        report_data = {
            'analysis_time': datetime.now().isoformat(),
            'dataflow_records': records,
            'sa_layers_analysis': sa_layers,
            'summary': {
                'total_layers': len([r for r in records if r['layer'].startswith('SA')]) // 2,
                'point_reduction_path': f"{records[0]['xyz_shape'][-1]} → " + 
                                       " → ".join([str(l['new_xyz_shape'][-1]) for l in sa_layers]),
                'feature_expansion_path': "3 → " + 
                                         " → ".join([str(l['new_points_shape'][1]) for l in sa_layers]),
            }
        }
        
        os.makedirs('logs/dataflow_analysis', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pointnet2_dataflow_detailed_{timestamp}.json"
        filepath = os.path.join('logs/dataflow_analysis', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细数据流报告已保存到: {filepath}")

def visualize_dataflow_evolution(records):
    """可视化数据流演化"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 提取SA层数据
        sa_outputs = [r for r in records if r['layer'].startswith('SA') and r['stage'] == 'output']
        
        if not sa_outputs:
            print("没有SA层数据可可视化")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 点数演化
        layer_names = [layer['layer'] for layer in sa_outputs]
        point_counts = [layer['new_xyz_shape'][-1] for layer in sa_outputs]
        
        axes[0, 0].plot(range(len(point_counts)), point_counts, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_title('点数演化 (下采样过程)')
        axes[0, 0].set_xlabel('网络深度')
        axes[0, 0].set_ylabel('点数')
        axes[0, 0].set_xticks(range(len(layer_names)))
        axes[0, 0].set_xticklabels(layer_names)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, count in enumerate(point_counts):
            axes[0, 0].text(i, count, f'{count}', ha='center', va='bottom', fontsize=10)
        
        # 2. 特征维度演化
        feature_dims = [layer['new_points_shape'][1] for layer in sa_outputs]
        
        axes[0, 1].plot(range(len(feature_dims)), feature_dims, 's-', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_title('特征维度演化 (语义信息增加)')
        axes[0, 1].set_xlabel('网络深度')
        axes[0, 1].set_ylabel('特征维度')
        axes[0, 1].set_xticks(range(len(layer_names)))
        axes[0, 1].set_xticklabels(layer_names)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, dim in enumerate(feature_dims):
            axes[0, 1].text(i, dim, f'{dim}', ha='center', va='bottom', fontsize=10)
        
        # 3. 总元素数演化（点数×特征维）
        total_elements = []
        for layer in sa_outputs:
            elements = layer['new_xyz_shape'][0] * layer['new_points_shape'][1] * layer['new_points_shape'][2]
            total_elements.append(elements)
        
        axes[0, 2].plot(range(len(total_elements)), total_elements, '^-', linewidth=2, markersize=8, color='red')
        axes[0, 2].set_title('总元素数演化 (信息量变化)')
        axes[0, 2].set_xlabel('网络深度')
        axes[0, 2].set_ylabel('总元素数')
        axes[0, 2].set_xticks(range(len(layer_names)))
        axes[0, 2].set_xticklabels(layer_names)
        axes[0, 2].set_yscale('log')  # 对数刻度显示大范围变化
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 压缩率（相对于输入）
        compression_ratios = []
        for i, layer in enumerate(sa_outputs):
            if i == 0:
                # SA1的输入是原始点云
                input_elements = records[0]['xyz_shape'][0] * 3 * records[0]['xyz_shape'][-1]
            else:
                # 后面层的输入是前一层的输出
                prev_layer = sa_outputs[i-1]
                input_elements = prev_layer['new_xyz_shape'][0] * prev_layer['new_points_shape'][1] * prev_layer['new_points_shape'][2]
            
            output_elements = layer['new_xyz_shape'][0] * layer['new_points_shape'][1] * layer['new_points_shape'][2]
            compression = output_elements / input_elements if input_elements > 0 else 0
            compression_ratios.append(compression)
        
        axes[1, 0].bar(range(len(compression_ratios)), compression_ratios, color=['skyblue', 'lightgreen', 'salmon'])
        axes[1, 0].set_title('各层压缩率')
        axes[1, 0].set_xlabel('网络层')
        axes[1, 0].set_ylabel('压缩率')
        axes[1, 0].set_xticks(range(len(layer_names)))
        axes[1, 0].set_xticklabels(layer_names)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 语义级别示意图
        semantic_levels = ['局部几何', '部件结构', '全局语义']
        level_colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        axes[1, 1].barh(range(len(semantic_levels)), [1, 1, 1], color=level_colors)
        axes[1, 1].set_title('语义级别示意')
        axes[1, 1].set_xlabel('网络深度方向')
        axes[1, 1].set_yticks(range(len(semantic_levels)))
        axes[1, 1].set_yticklabels(semantic_levels)
        
        # 添加箭头表示信息流动
        for i in range(len(semantic_levels)-1):
            axes[1, 1].annotate('', xy=(0.5, i+0.8), xytext=(0.5, i+0.2),
                              arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # 6. 数据量对比（用于语义通信）
        data_sizes_kb = [e * 4 / 1024 for e in total_elements]  # 假设32位浮点
        
        # 原始数据量
        original_size = records[0]['xyz_shape'][0] * 3 * records[0]['xyz_shape'][-1] * 4 / 1024
        
        x_pos = range(len(data_sizes_kb))
        axes[1, 2].bar(x_pos, data_sizes_kb, color=['orange', 'green', 'blue'])
        axes[1, 2].axhline(y=original_size, color='red', linestyle='--', label=f'原始数据: {original_size:.1f}KB')
        axes[1, 2].set_title('各层特征数据量 vs 原始数据')
        axes[1, 2].set_xlabel('特征层')
        axes[1, 2].set_ylabel('数据量 (KB)')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(layer_names)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, size in enumerate(data_sizes_kb):
            axes[1, 2].text(i, size, f'{size:.1f}KB', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('PointNet++ 数据流深度分析', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('logs/visualizations', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'logs/visualizations/dataflow_detailed_{timestamp}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"数据流演化图表已保存到: {save_path}")
        
    except ImportError:
        print("需要安装matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"可视化失败: {e}")

def main():
    """主函数"""
    # 设置环境变量避免OpenMP警告
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("="*70)
    print("PointNet++ 数据流深度分析")
    print("="*70)
    
    # 导入模型
    try:
        from pointnet2_cls_msg import get_model
        
        # 创建模型
        model = get_model(num_class=10, normal_channel=False)
        model.eval()
        
        print(f"✓ 成功导入模型: {model.__class__.__name__}")
        
        # 创建测试数据
        batch_size = 2
        num_points = 1024
        test_input = torch.randn(batch_size, 3, num_points)
        
        # 创建分析器
        analyzer = DetailedDataflowAnalyzer()
        
        # 执行深度分析
        records = analyzer.analyze_feature_evolution(model, test_input)
        
        # 生成综合报告
        sa_layers = [r for r in records if r['layer'].startswith('SA') and r['stage'] == 'output']
        analyzer.generate_comprehensive_report(records)
        
        # 可视化
        visualize_dataflow_evolution(records)
        
        print("\n" + "="*70)
        print("分析完成!")
        print("="*70)
        
        print("\n关键发现总结:")
        print("1. PointNet++通过三层SA层实现层次化特征提取")
        print("2. 点数逐步减少: 1024 → 512 → 128 → 1")
        print("3. 特征维度逐步增加: 3 → 320 → 640 → 1024")
        print("4. 总信息量先增加后减少，最终压缩到全局特征")
        print("5. 不同层次的特征具有不同的语义含义")
        
        print("\n对语义通信的启示:")
        print("1. SA1特征: 高分辨率，适合精细任务，但数据量大")
        print("2. SA2特征: 中等分辨率，平衡信息和数据量")
        print("3. SA3特征: 低分辨率，适合分类，数据量最小")
        print("4. 可根据信道条件选择传输不同层次的特征")
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("请确保在正确的目录下运行，并检查模型路径")
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()