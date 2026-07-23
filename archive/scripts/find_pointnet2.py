# scripts/test_your_pointnet2.py
import sys
import os
import torch

# 添加 Pointnet_Pointnet2_pytorch 到系统路径
sys.path.append('Pointnet_Pointnet2_pytorch')
sys.path.append('.')

def test_import():
    """测试能否导入相关模块"""
    print("测试导入...")
    
    modules_to_test = []
    
    # 尝试导入常见的模块
    try:
        import pointnet2_utils
        modules_to_test.append(("pointnet2_utils", "✓"))
    except ImportError as e:
        modules_to_test.append(("pointnet2_utils", f"✗ {e}"))
    
    try:
        import pointnet2_cls_msg
        modules_to_test.append(("pointnet2_cls_msg", "✓"))
    except ImportError as e:
        modules_to_test.append(("pointnet2_cls_msg", f"✗ {e}"))
    
    # 尝试从日志目录导入（根据你的输出）
    try:
        sys.path.append('Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals')
        from pointnet2_cls_msg import PointNet2
        modules_to_test.append(("PointNet2类", "✓"))
    except ImportError as e:
        modules_to_test.append(("PointNet2类", f"✗ {e}"))
    
    # 显示结果
    for name, result in modules_to_test:
        print(f"  {name:20s}: {result}")

def test_model_creation():
    """测试创建模型"""
    print("\n测试模型创建...")
    
    try:
        # 尝试多种方式
        # 方式1：直接导入
        try:
            from Pointnet_Pointnet2_pytorch.log.classification.pointnet2_msg_normals.pointnet2_cls_msg import PointNet2
            model = PointNet2(num_classes=10)
            print(f"  ✓ 方式1成功: {model.__class__.__name__}")
            print(f"    参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        except Exception as e:
            print(f"  ✗ 方式1失败: {e}")
        
        # 方式2：从其他位置导入
        try:
            # 查看是否有 models 目录
            import models
            print(f"  ✓ 找到 models 包")
        except:
            print(f"  ✗ 未找到 models 包")
            
    except Exception as e:
        print(f"  ✗ 模型创建测试失败: {e}")

def check_data_loading():
    """检查数据加载"""
    print("\n检查数据加载...")
    
    data_dir = Path("data")
    if data_dir.exists():
        print(f"  data目录存在，包含:")
        for item in data_dir.iterdir():
            if item.is_dir():
                num_files = len(list(item.rglob("*")))
                print(f"    - {item.name}: {num_files}个文件/目录")
            else:
                print(f"    - {item.name} (文件)")
    else:
        print("  ✗ data目录不存在")

def main():
    print("="*60)
    print("测试你的PointNet++配置")
    print("="*60)
    
    test_import()
    test_model_creation()
    check_data_loading()
    
    print("\n" + "="*60)
    print("行动建议：")
    
    # 基于你的情况给出具体建议
    print("1. 你的PointNet++代码可能在 Pointnet_Pointnet2_pytorch/log/classification/...")
    print("2. 先尝试运行一个简单的训练：")
    print("   cd Pointnet_Pointnet2_pytorch")
    print("   python train_classification.py --model pointnet2_cls_msg")
    print("\n3. 或者运行测试：")
    print("   python test_classification.py")

if __name__ == "__main__":
    from pathlib import Path
    main()