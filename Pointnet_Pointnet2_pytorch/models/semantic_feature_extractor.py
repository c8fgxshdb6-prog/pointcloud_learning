"""
语义通信专用特征提取器
基于修改后的PointNet++，支持自适应特征选择
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# 确保能导入当前目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pointnet2_cls_msg import get_model


class SemanticFeatureExtractor:
    """
    语义通信特征提取器
    支持根据信道条件和任务需求自适应选择特征层
    """
    
    def __init__(self, num_classes: int = 40, use_normals: bool = False, device: str = 'cuda'):
        """
        初始化特征提取器
        
        Args:
            num_classes: 分类类别数
            use_normals: 是否使用法向量
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model = get_model(num_class=num_classes, normal_channel=use_normals)
        self.model.to(self.device)
        self.model.eval()
        self.use_normals = use_normals
        
        # 特征层元数据
        self.feature_layers = {
            'sa1': {
                'name': '局部几何层',
                'description': '提取局部几何细节，适合分割任务',
                'typical_shape': (320, 512),  # (特征维度, 点数)
                'semantic_level': 'low',
                'bandwidth_kb_per_sample': 640.0,  # 每个样本的KB数
                'suitable_tasks': ['分割', '局部匹配', '细节重建'],
                'channel_condition': 'high_bandwidth',
            },
            'sa2': {
                'name': '部件结构层',
                'description': '提取部件级结构信息，适合目标检测',
                'typical_shape': (640, 128),
                'semantic_level': 'medium',
                'bandwidth_kb_per_sample': 320.0,
                'suitable_tasks': ['目标检测', '部件分类', '姿态估计'],
                'channel_condition': 'medium_bandwidth',
            },
            'sa3': {
                'name': '全局语义层',
                'description': '提取整体对象语义，适合分类任务',
                'typical_shape': (1024, 1),
                'semantic_level': 'high',
                'bandwidth_kb_per_sample': 4.0,
                'suitable_tasks': ['对象分类', '场景理解', '检索'],
                'channel_condition': 'low_bandwidth',
            }
        }
        
        print(f"✅ 语义特征提取器初始化完成")
        print(f"   设备: {self.device}")
        print(f"   使用法向量: {use_normals}")
        print(f"   各层带宽需求:")
        for layer_id, info in self.feature_layers.items():
            print(f"     {info['name']}({layer_id}): {info['bandwidth_kb_per_sample']:.1f} KB/样本")
    
    def extract_all(self, point_cloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取所有三层特征
        
        Args:
            point_cloud: 输入点云，形状为 (B, C, N)
                        C=3 (xyz) 或 C=6 (xyz+normals)
                        
        Returns:
            包含各层特征的字典
        """
        # 验证输入
        B, C, N = point_cloud.shape
        if self.use_normals and C != 6:
            raise ValueError(f"使用法向量时输入应为6维，但得到{C}维")
        elif not self.use_normals and C != 3:
            raise ValueError(f"不使用法向量时输入应为3维，但得到{C}维")
        
        # 移动到设备
        point_cloud = point_cloud.to(self.device)
        
        with torch.no_grad():
            cls_result, sa1, sa2, sa3 = self.model(point_cloud)
            
            # 将特征移回CPU以节省显存
            if self.device.type == 'cuda':
                sa1, sa2, sa3 = sa1.cpu(), sa2.cpu(), sa3.cpu()
                cls_result = cls_result.cpu()
            
            # 组织结果
            result = {
                'classification': cls_result,
                'features': {
                    'sa1': sa1,  # 局部几何特征
                    'sa2': sa2,  # 部件结构特征
                    'sa3': sa3,  # 全局语义特征
                },
                'metadata': {
                    'input_shape': point_cloud.shape,
                    'feature_shapes': {
                        'sa1': sa1.shape,
                        'sa2': sa2.shape,
                        'sa3': sa3.shape,
                    },
                    'bandwidth_requirements_kb': {
                        'sa1': self.feature_layers['sa1']['bandwidth_kb_per_sample'] * B,
                        'sa2': self.feature_layers['sa2']['bandwidth_kb_per_sample'] * B,
                        'sa3': self.feature_layers['sa3']['bandwidth_kb_per_sample'] * B,
                    },
                    'total_bandwidth_kb': (
                        self.feature_layers['sa1']['bandwidth_kb_per_sample'] +
                        self.feature_layers['sa2']['bandwidth_kb_per_sample'] +
                        self.feature_layers['sa3']['bandwidth_kb_per_sample']
                    ) * B,
                    'compression_info': {
                        'original_points_kb': point_cloud.numel() * 4 / 1024,
                        'total_features_kb': (sa1.numel() + sa2.numel() + sa3.numel()) * 4 / 1024,
                        'sa3_only_kb': sa3.numel() * 4 / 1024,
                    }
                }
            }
            
            return result
    
    def adaptive_extract(self,
                        point_cloud: torch.Tensor,
                        available_bandwidth_kb: Optional[float] = None,
                        channel_snr_db: Optional[float] = None,
                        task_type: str = 'auto',
                        latency_constraint_ms: Optional[float] = None) -> Dict:
        """
        自适应特征提取：根据信道条件、任务需求和时延约束选择最优特征
        
        Args:
            point_cloud: 输入点云
            available_bandwidth_kb: 可用带宽(KB)
            channel_snr_db: 信道信噪比(dB)
            task_type: 任务类型 ('classification', 'segmentation', 'detection', 'auto')
            latency_constraint_ms: 时延约束(毫秒)
            
        Returns:
            自适应选择后的特征
        """
        # 提取所有特征
        all_features = self.extract_all(point_cloud)
        B = point_cloud.shape[0]
        
        # 初始化结果
        result = {
            'input_info': {
                'shape': point_cloud.shape,
                'num_points': point_cloud.shape[2],
                'has_normals': self.use_normals,
            },
            'channel_conditions': {
                'available_bandwidth_kb': available_bandwidth_kb,
                'snr_db': channel_snr_db,
                'latency_constraint_ms': latency_constraint_ms,
            },
            'task_type': task_type,
            'selected_features': {},
            'selection_strategy': 'full',
            'performance_metrics': {},
        }
        
        # 获取各层带宽需求
        sa1_kb = self.feature_layers['sa1']['bandwidth_kb_per_sample'] * B
        sa2_kb = self.feature_layers['sa2']['bandwidth_kb_per_sample'] * B
        sa3_kb = self.feature_layers['sa3']['bandwidth_kb_per_sample'] * B
        
        # 基于带宽的自适应策略
        if available_bandwidth_kb is not None:
            if available_bandwidth_kb < sa3_kb:
                # 情况1: 带宽极低，无法传输任何特征
                result['selection_strategy'] = 'bandwidth_insufficient'
                result['error'] = f'带宽不足：至少需要{sa3_kb:.1f}KB传输全局特征'
                return result
            
            elif available_bandwidth_kb < sa2_kb + sa3_kb:
                # 情况2: 低带宽，仅传输全局特征
                result['selected_features']['sa3'] = all_features['features']['sa3']
                result['selection_strategy'] = 'low_bandwidth_optimized'
                result['performance_metrics']['bandwidth_used_kb'] = sa3_kb
                result['performance_metrics']['bandwidth_efficiency'] = sa3_kb / available_bandwidth_kb
                result['recommended_task'] = 'classification'
                
            elif available_bandwidth_kb < sa1_kb + sa2_kb + sa3_kb:
                # 情况3: 中带宽，传输结构和全局特征
                result['selected_features']['sa2'] = all_features['features']['sa2']
                result['selected_features']['sa3'] = all_features['features']['sa3']
                result['selection_strategy'] = 'medium_bandwidth_optimized'
                result['performance_metrics']['bandwidth_used_kb'] = sa2_kb + sa3_kb
                result['performance_metrics']['bandwidth_efficiency'] = (sa2_kb + sa3_kb) / available_bandwidth_kb
                result['recommended_task'] = 'detection'
                
            else:
                # 情况4: 高带宽，传输全部特征
                result['selected_features'] = all_features['features']
                result['selection_strategy'] = 'high_bandwidth_optimized'
                result['performance_metrics']['bandwidth_used_kb'] = sa1_kb + sa2_kb + sa3_kb
                result['performance_metrics']['bandwidth_efficiency'] = (sa1_kb + sa2_kb + sa3_kb) / available_bandwidth_kb
                result['recommended_task'] = 'segmentation'
        
        # 基于任务类型的自适应策略
        elif task_type != 'auto':
            if task_type == 'classification':
                # 分类任务：只需全局特征
                result['selected_features']['sa3'] = all_features['features']['sa3']
                result['selection_strategy'] = 'task_optimized_classification'
                result['performance_metrics']['bandwidth_used_kb'] = sa3_kb
                
            elif task_type == 'detection' or task_type == 'object_detection':
                # 检测任务：需要结构和全局特征
                result['selected_features']['sa2'] = all_features['features']['sa2']
                result['selected_features']['sa3'] = all_features['features']['sa3']
                result['selection_strategy'] = 'task_optimized_detection'
                result['performance_metrics']['bandwidth_used_kb'] = sa2_kb + sa3_kb
                
            elif task_type == 'segmentation':
                # 分割任务：需要局部特征
                result['selected_features']['sa1'] = all_features['features']['sa1']
                result['selection_strategy'] = 'task_optimized_segmentation'
                result['performance_metrics']['bandwidth_used_kb'] = sa1_kb
                
            else:
                # 未知任务，使用全部特征
                result['selected_features'] = all_features['features']
                result['selection_strategy'] = 'full_features_unknown_task'
                result['performance_metrics']['bandwidth_used_kb'] = sa1_kb + sa2_kb + sa3_kb
        
        # 基于信噪比的自适应策略
        elif channel_snr_db is not None:
            if channel_snr_db < 0:
                # 极差信道：仅传输全局特征
                result['selected_features']['sa3'] = all_features['features']['sa3']
                result['selection_strategy'] = 'poor_channel_optimized'
                result['performance_metrics']['bandwidth_used_kb'] = sa3_kb
                
            elif channel_snr_db < 10:
                # 较差信道：传输结构和全局特征
                result['selected_features']['sa2'] = all_features['features']['sa2']
                result['selected_features']['sa3'] = all_features['features']['sa3']
                result['selection_strategy'] = 'medium_channel_optimized'
                result['performance_metrics']['bandwidth_used_kb'] = sa2_kb + sa3_kb
                
            else:
                # 良好信道：传输全部特征
                result['selected_features'] = all_features['features']
                result['selection_strategy'] = 'good_channel_optimized'
                result['performance_metrics']['bandwidth_used_kb'] = sa1_kb + sa2_kb + sa3_kb
        
        # 默认：传输全部特征
        else:
            result['selected_features'] = all_features['features']
            result['selection_strategy'] = 'full_features_default'
            result['performance_metrics']['bandwidth_used_kb'] = sa1_kb + sa2_kb + sa3_kb
        
        # 添加特征形状信息
        result['feature_shapes'] = {
            name: feat.shape for name, feat in result['selected_features'].items()
        }
        
        # 计算时延估计（简化模型）
        if latency_constraint_ms:
            # 简化时延模型：每KB数据传输需要0.1ms（假设）
            transmission_delay_ms = result['performance_metrics']['bandwidth_used_kb'] * 0.1
            processing_delay_ms = 5.0  # 固定处理时延
            total_latency_ms = transmission_delay_ms + processing_delay_ms
            
            result['performance_metrics']['estimated_latency_ms'] = total_latency_ms
            result['performance_metrics']['meets_latency_constraint'] = total_latency_ms <= latency_constraint_ms
        
        return result
    
    def get_layer_info(self, layer_id: str) -> Dict:
        """获取指定特征层的详细信息"""
        if layer_id in self.feature_layers:
            return self.feature_layers[layer_id].copy()
        else:
            raise ValueError(f"未知的特征层: {layer_id}")
    
    def list_available_layers(self) -> List[str]:
        """获取所有可用的特征层ID"""
        return list(self.feature_layers.keys())
    
    def analyze_features(self, features_dict: Dict[str, torch.Tensor]) -> Dict:
        """分析特征统计信息"""
        analysis = {}
        
        for name, tensor in features_dict.items():
            if isinstance(tensor, torch.Tensor):
                tensor_np = tensor.numpy() if tensor.device.type == 'cpu' else tensor.cpu().numpy()
                
                analysis[name] = {
                    'shape': tensor.shape,
                    'dtype': str(tensor.dtype),
                    'min': float(tensor_np.min()),
                    'max': float(tensor_np.max()),
                    'mean': float(tensor_np.mean()),
                    'std': float(tensor_np.std()),
                    'size_kb': tensor.numel() * 4 / 1024,
                    'sparsity': float(np.mean(tensor_np == 0)) if tensor_np.size > 0 else 0.0,
                }
        
        return analysis
    
    def save_features(self, features_dict: Dict, filepath: str):
        """保存特征到文件"""
        save_dict = {}
        
        for name, tensor in features_dict.items():
            if isinstance(tensor, torch.Tensor):
                save_dict[name] = tensor.cpu().numpy()
            else:
                save_dict[name] = tensor
        
        np.savez(filepath, **save_dict)
        print(f"✅ 特征已保存到: {filepath}")
    
    def create_bandwidth_report(self, batch_size: int = 1) -> str:
        """生成带宽需求报告"""
        report = []
        report.append("=" * 60)
        report.append("PointNet++ 语义特征带宽需求报告")
        report.append("=" * 60)
        report.append(f"批次大小: {batch_size}")
        report.append("")
        
        for layer_id, info in self.feature_layers.items():
            bandwidth_kb = info['bandwidth_kb_per_sample'] * batch_size
            report.append(f"{info['name']} ({layer_id}):")
            report.append(f"  特征维度: {info['typical_shape']}")
            report.append(f"  语义级别: {info['semantic_level']}")
            report.append(f"  带宽需求: {bandwidth_kb:.1f} KB")
            report.append(f"  适用任务: {', '.join(info['suitable_tasks'])}")
            report.append(f"  适用信道: {info['channel_condition']}")
            report.append("")
        
        # 推荐配置
        report.append("推荐配置:")
        report.append("  1. 移动网络 (带宽 < 100KB): 仅使用sa3全局特征")
        report.append("  2. WiFi网络 (带宽 100-500KB): 使用sa2+sa3特征")
        report.append("  3. 有线网络 (带宽 > 500KB): 使用全部特征(sa1+sa2+sa3)")
        report.append("")
        report.append("=" * 60)
        
        return '\n'.join(report)


def demo_semantic_communication():
    """演示语义通信场景"""
    print("=" * 70)
    print("语义通信场景演示")
    print("=" * 70)
    
    # 创建特征提取器
    extractor = SemanticFeatureExtractor(use_normals=False, device='cuda')
    
    # 创建测试点云
    point_cloud = torch.randn(2, 3, 1024)  # 2个样本
    
    print("\n1. 提取所有特征:")
    all_features = extractor.extract_all(point_cloud)
    print(f"   输入: {all_features['metadata']['input_shape']}")
    for name, feat in all_features['features'].items():
        print(f"   {name}: {feat.shape}")
    
    print("\n2. 不同通信场景模拟:")
    
    scenarios = [
        {
            'name': '移动网络',
            'bandwidth_kb': 50,
            'snr_db': 5,
            'task': 'classification'
        },
        {
            'name': 'WiFi网络',
            'bandwidth_kb': 300,
            'snr_db': 15,
            'task': 'object_detection'
        },
        {
            'name': '有线网络',
            'bandwidth_kb': 1000,
            'snr_db': 25,
            'task': 'segmentation'
        },
        {
            'name': '卫星通信',
            'bandwidth_kb': 10,
            'snr_db': -5,
            'task': 'classification'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n  场景: {scenario['name']}")
        print(f"    带宽: {scenario['bandwidth_kb']}KB, SNR: {scenario['snr_db']}dB, 任务: {scenario['task']}")
        
        try:
            features = extractor.adaptive_extract(
                point_cloud,
                available_bandwidth_kb=scenario['bandwidth_kb'],
                channel_snr_db=scenario['snr_db'],
                task_type=scenario['task']
            )
            
            if 'error' in features:
                print(f"    ❌ {features['error']}")
            else:
                print(f"    策略: {features['selection_strategy']}")
                print(f"    选择特征: {list(features['selected_features'].keys())}")
                print(f"    使用带宽: {features['performance_metrics']['bandwidth_used_kb']:.1f} KB")
                if 'recommended_task' in features:
                    print(f"    推荐任务: {features['recommended_task']}")
                    
        except Exception as e:
            print(f"    ❌ 错误: {e}")
    
    # 生成带宽报告
    print("\n" + "=" * 70)
    print("带宽需求报告:")
    print("=" * 70)
    print(extractor.create_bandwidth_report(batch_size=2))
    
    # 特征分析
    print("\n特征统计分析:")
    analysis = extractor.analyze_features(all_features['features'])
    for name, stats in analysis.items():
        print(f"  {name}: shape={stats['shape']}, mean={stats['mean']:.4f}, size={stats['size_kb']:.1f}KB")
    
    # 保存示例特征
    extractor.save_features(all_features['features'], 'semantic_features_example.npz')
    
    print("\n" + "=" * 70)
    print("✅ 演示完成")
    print("=" * 70)
    
    return extractor, all_features


if __name__ == "__main__":
    # 运行演示
    extractor, features = demo_semantic_communication()
    
    print("\n🎯 给学长的汇报要点:")
    print("1. ✅ 已完成PointNet++ forward函数修改")
    print("2. ✅ 三层特征提取: SA1(局部), SA2(结构), SA3(全局)")
    print("3. ✅ SA3全局特征仅需4KB/样本，适合低带宽传输")
    print("4. ✅ 实现了自适应特征选择策略")
    print("5. ✅ 支持基于带宽、SNR、任务类型的智能选择")
    print("\n📋 下一步工作:")
    print("1. 讨论与语义通信框架的接口规范")
    print("2. 设计端到端的训练和微调策略")
    print("3. 实现特征压缩和编码模块")
    print("4. 进行实际信道环境测试")