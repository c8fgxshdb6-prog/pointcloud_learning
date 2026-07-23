# scripts/pointnet2_dataflow.py
import torch
import torch.nn as nn
import json
import os
from datetime import datetime

class PointNet2FeatureLogger:
    """专门为你的PointNet++模型设计的特征日志器"""
    def __init__(self, log_dir="logs/pointnet2_features"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.feature_maps = {}  # 存储每层的特征
        self.statistics = []    # 存储统计信息
        self.hooks = []         # 存储钩子
        
        print(f"PointNet2特征日志器初始化 - 日志目录: {log_dir}")
    
    def register_hooks(self, model):
        """给模型的所有层注册钩子"""
        print("正在注册特征钩子...")
        
        # 清空之前的钩子
        self._remove_hooks()
        
        # 为每个SA层注册钩子
        for name, module in model.named_children():
            if name.startswith('sa'):
                # 给每个SA层注册前向钩子
                hook = self._create_sa_hook(name)
                self.hooks.append(module.register_forward_hook(hook))
                print(f"  为 {name} 注册钩子")
            
            elif name.startswith('fc') or name.startswith('bn'):
                # 给全连接层和BN层注册钩子
                hook = self._create_fc_hook(name)
                self.hooks.append(module.register_forward_hook(hook))
                print(f"  为 {name} 注册钩子")
        
        print(f"总共注册了 {len(self.hooks)} 个钩子")
    
    def _create_sa_hook(self, layer_name):
        """为SA层创建钩子"""
        def hook(module, input, output):
            # SA层的输出是 (new_xyz, new_points)
            if isinstance(output, tuple) and len(output) == 2:
                new_xyz, new_points = output
                
                # 记录特征统计
                stats = {
                    'layer': layer_name,
                    'type': 'SA',
                    'input_xyz_shape': list(input[0].shape) if input[0] is not None else None,
                    'output_xyz_shape': list(new_xyz.shape),
                    'output_points_shape': list(new_points.shape),
                    'output_points_mean': float(new_points.mean().item()),
                    'output_points_std': float(new_points.std().item()),
                    'output_points_min': float(new_points.min().item()),
                    'output_points_max': float(new_points.max().item()),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.statistics.append(stats)
                
                # 存储特征图（只存储小批量，避免内存占用过大）
                if new_points.shape[0] <= 2:  # batch_size <= 2时保存
                    key = f"{layer_name}_output_points"
                    self.feature_maps[key] = {
                        'data': new_points.detach().cpu().numpy().tolist(),
                        'shape': list(new_points.shape)
                    }
                
                # 打印实时信息
                print(f"[{layer_name}] 点数: {new_xyz.shape[-1]} -> {new_points.shape[-1]}, "
                      f"特征维: {new_points.shape[1]}, "
                      f"均值: {stats['output_points_mean']:.4f}")
        
        return hook
    
    def _create_fc_hook(self, layer_name):
        """为全连接层/BN层创建钩子"""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                stats = {
                    'layer': layer_name,
                    'type': 'FC' if 'fc' in layer_name else 'BN',
                    'input_shape': list(input[0].shape) if input[0] is not None else None,
                    'output_shape': list(output.shape),
                    'output_mean': float(output.mean().item()),
                    'output_std': float(output.std().item()),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.statistics.append(stats)
                
                print(f"[{layer_name}] 输出形状: {output.shape}, "
                      f"均值: {stats['output_mean']:.4f}")
        
        return hook
    
    def _remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def save_statistics(self, filename=None):
        """保存统计信息到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pointnet2_stats_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': self.statistics,
                'feature_shapes': {k: v['shape'] for k, v in self.feature_maps.items()},
                'total_layers': len(self.statistics),
                'save_time': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"统计信息已保存到: {filepath}")
        return filepath
    
    def save_feature_maps(self, filename=None):
        """保存特征图数据（小心，可能很大）"""
        if not self.feature_maps:
            print("没有特征图数据可保存")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pointnet2_features_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        # 只保存前几个特征图，避免文件过大
        features_to_save = {}
        count = 0
        for key, value in self.feature_maps.items():
            # 只保存较小的特征图
            if len(value['data']) * len(str(value['data'][0])) < 1000000:  # 大约1MB
                features_to_save[key] = value
                count += 1
                if count >= 10:  # 最多保存10个
                    break
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'features': features_to_save,
                'save_time': datetime.now().isoformat(),
                'note': '只保存了部分特征图，完整数据请查看统计信息'
            }, f, indent=2, ensure_ascii=False)
        
        print(f"特征图数据已保存到: {filepath}（保存了{len(features_to_save)}个特征）")
        return filepath
    
    def clear(self):
        """清空数据"""
        self.statistics = []
        self.feature_maps = {}
        self._remove_hooks()
        print("已清空所有日志数据")

def analyze_pointnet2_dataflow():
    """分析PointNet++的数据流"""
    print("="*60)
    print("PointNet++ 数据流分析")
    print("="*60)
    
    # 导入你的模型
    import sys
    sys.path.append('Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals')
    
    try:
        from pointnet2_cls_msg import get_model
        
        # 创建模型
        model = get_model(num_class=10, normal_channel=False)
        model.eval()
        print(f"✓ 模型创建成功: {model.__class__.__name__}")
        print(f"  模型结构:")
        for name, module in model.named_children():
            print(f"    - {name}: {module.__class__.__name__}")
        
        # 创建日志器
        logger = PointNet2FeatureLogger()
        
        # 注册钩子
        logger.register_hooks(model)
        
        # 创建测试数据
        batch_size = 2
        num_points = 1024
        xyz = torch.randn(batch_size, 3, num_points)
        print(f"\n测试输入:")
        print(f"  形状: {xyz.shape}")
        print(f"  范围: [{xyz.min():.3f}, {xyz.max():.3f}]")
        
        # 前向传播（会自动记录特征）
        print("\n" + "="*60)
        print("开始前向传播...")
        print("="*60)
        
        with torch.no_grad():
            classification_output, global_features = model(xyz)
        
        print("\n" + "="*60)
        print("前向传播完成!")
        print("="*60)
        
        print(f"分类输出形状: {classification_output.shape}")
        print(f"全局特征形状 (l3_points): {global_features.shape}")
        
        # 保存结果
        stats_file = logger.save_statistics()
        features_file = logger.save_feature_maps()
        
        # 移除钩子
        logger._remove_hooks()
        
        # 生成分析报告
        generate_analysis_report(logger.statistics, global_features)
        
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_analysis_report(statistics, global_features):
    """生成数据分析报告"""
    print("\n" + "="*60)
    print("数据流分析报告")
    print("="*60)
    
    # 分析SA层
    sa_layers = [s for s in statistics if s['type'] == 'SA']
    fc_layers = [s for s in statistics if s['type'] in ['FC', 'BN']]
    
    print(f"1. 网络层次结构:")
    print(f"   - Set Abstraction 层: {len(sa_layers)} 个")
    print(f"   - 全连接/BN 层: {len(fc_layers)} 个")
    print(f"   - 总共记录层数: {len(statistics)} 个")
    
    print(f"\n2. SA层数据流演化:")
    for i, layer in enumerate(sa_layers):
        print(f"   {layer['layer']}:")
        print(f"     输入点数: {layer['input_xyz_shape'][-1] if layer['input_xyz_shape'] else 'N/A'}")
        print(f"     输出点数: {layer['output_xyz_shape'][-1]}")
        print(f"     输出特征维数: {layer['output_points_shape'][1]}")
        print(f"     特征均值: {layer['output_points_mean']:.4f}")
        
        # 计算压缩率
        if i > 0 and sa_layers[i-1]['output_xyz_shape']:
            prev_points = sa_layers[i-1]['output_xyz_shape'][-1]
            curr_points = layer['output_xyz_shape'][-1]
            compression = prev_points / curr_points if curr_points > 0 else 0
            print(f"     点数压缩率: {compression:.1f}x")
    
    print(f"\n3. 全局特征分析 (l3_points):")
    print(f"   形状: {list(global_features.shape)}")
    print(f"   总元素数: {global_features.numel()}")
    print(f"   均值: {global_features.mean().item():.6f}")
    print(f"   标准差: {global_features.std().item():.6f}")
    
    print(f"\n4. 语义通信特征提取建议:")
    print(f"   a) SA1层特征: 局部几何细节 (高分辨率，点数多)")
    print(f"   b) SA2层特征: 部件级语义 (中等分辨率)")
    print(f"   c) SA3层特征: 全局语义特征 (适合分类，低维)")
    print(f"   d) l3_points: 最终全局特征 (1024维，最适合传输)")
    
    print(f"\n5. 自适应传输策略示例:")
    print(f"   - 信道质量好: 传输 SA2 + SA3 + l3_points")
    print(f"   - 信道质量中: 传输 SA3 + l3_points")
    print(f"   - 信道质量差: 只传输 l3_points (1024维)")
    
    # 计算各层特征的数据量（假设32位浮点数）
    print(f"\n6. 各层特征数据量估算 (假设32位浮点):")
    for layer in sa_layers:
        shape = layer['output_points_shape']
        elements = shape[0] * shape[1] * shape[2]  # B × C × N
        bytes_needed = elements * 4  # 4 bytes per float32
        print(f"   {layer['layer']}: {elements:,} 元素, {bytes_needed:,} 字节, {bytes_needed/1024:.1f} KB")
    
    l3_elements = global_features.numel()
    l3_bytes = l3_elements * 4
    print(f"   l3_points: {l3_elements:,} 元素, {l3_bytes:,} 字节, {l3_bytes/1024:.1f} KB")

def visualize_feature_statistics(statistics_file):
    """可视化特征统计"""
    try:
        import matplotlib.pyplot as plt
        import json
        
        with open(statistics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = data['statistics']
        sa_stats = [s for s in stats if s['type'] == 'SA']
        
        if not sa_stats:
            print("没有SA层统计数据可可视化")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 特征均值变化
        layer_names = [s['layer'] for s in sa_stats]
        means = [s['output_points_mean'] for s in sa_stats]
        
        axes[0, 0].bar(range(len(means)), means, color='skyblue')
        axes[0, 0].set_title('SA层特征均值变化')
        axes[0, 0].set_xlabel('SA层')
        axes[0, 0].set_ylabel('均值')
        axes[0, 0].set_xticks(range(len(layer_names)))
        axes[0, 0].set_xticklabels(layer_names)
        
        # 2. 特征维数变化
        feature_dims = [s['output_points_shape'][1] for s in sa_stats]
        axes[0, 1].plot(range(len(feature_dims)), feature_dims, 'o-', linewidth=2)
        axes[0, 1].set_title('特征维度演化')
        axes[0, 1].set_xlabel('SA层')
        axes[0, 1].set_ylabel('特征维度')
        axes[0, 1].set_xticks(range(len(layer_names)))
        axes[0, 1].set_xticklabels(layer_names)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 点数变化
        point_counts = [s['output_xyz_shape'][-1] for s in sa_stats]
        axes[1, 0].bar(range(len(point_counts)), point_counts, color='lightcoral')
        axes[1, 0].set_title('点数变化')
        axes[1, 0].set_xlabel('SA层')
        axes[1, 0].set_ylabel('点数')
        axes[1, 0].set_xticks(range(len(layer_names)))
        axes[1, 0].set_xticklabels(layer_names)
        
        # 4. 压缩率
        compression_rates = []
        for i in range(1, len(point_counts)):
            rate = point_counts[i-1] / point_counts[i] if point_counts[i] > 0 else 0
            compression_rates.append(rate)
        
        axes[1, 1].bar(range(len(compression_rates)), compression_rates, color='lightgreen')
        axes[1, 1].set_title('点数压缩率')
        axes[1, 1].set_xlabel('SA层间')
        axes[1, 1].set_ylabel('压缩率')
        axes[1, 1].set_xticks(range(len(compression_rates)))
        axes[1, 1].set_xticklabels([f"{layer_names[i-1]}→{layer_names[i]}" for i in range(1, len(layer_names))])
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('logs/visualizations', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'logs/visualizations/pointnet2_analysis_{timestamp}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"可视化图表已保存到: {save_path}")
        
    except ImportError:
        print("需要安装matplotlib进行可视化: pip install matplotlib")
    except Exception as e:
        print(f"可视化失败: {e}")

if __name__ == "__main__":
    # 运行数据流分析
    success = analyze_pointnet2_dataflow()
    
    if success:
        print("\n" + "="*60)
        print("✅ 数据流分析完成!")
        print("="*60)
        
        # 找到最新的统计文件进行可视化
        import glob
        stat_files = glob.glob("logs/pointnet2_features/pointnet2_stats_*.json")
        if stat_files:
            latest_file = max(stat_files, key=os.path.getctime)
            print(f"\n正在可视化最新数据: {latest_file}")
            visualize_feature_statistics(latest_file)
        
        print("\n下一步建议:")
        print("1. 查看 logs/pointnet2_features/ 目录下的JSON文件")
        print("2. 查看 logs/visualizations/ 目录下的图表")
        print("3. 现在你理解了PointNet++的数据流，可以开始设计语义通信了")
    else:
        print("\n❌ 数据流分析失败，请检查错误信息")