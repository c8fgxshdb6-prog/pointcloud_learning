# test_model.py - 测试PointNet++修改后的forward函数
import torch
import sys
import os

print("="*60)
print("PointNet++特征提取测试")
print("="*60)

# 打印当前目录
print(f"当前工作目录: {os.getcwd()}")
print(f"Python路径:")
for path in sys.path:
    print(f"  {path}")

# 尝试导入
try:
    # 方法1：直接导入
    import pointnet2_cls_msg
    get_model = pointnet2_cls_msg.get_model
    print("✅ 使用import导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    
    # 方法2：使用importlib
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("pointnet2_cls_msg", "pointnet2_cls_msg.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        get_model = module.get_model
        print("✅ 使用importlib导入成功")
    except Exception as e2:
        print(f"❌ importlib也失败: {e2}")
        exit(1)

# 查看forward函数内容
print(f"\n查看forward函数内容...")
with open('pointnet2_cls_msg.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'def forward' in line and '(self, xyz)' in line:
            print(f"在第 {i+1} 行找到forward函数:")
            # 打印接下来的15行
            for j in range(i, min(i+20, len(lines))):
                print(f"  {j+1}: {lines[j].rstrip()}")
                if lines[j].strip().startswith('return'):
                    print(f"  --> 返回语句在第{j+1}行")
                    break
            break

# 创建模型
print(f"\n创建模型...")
try:
    model = get_model(num_class=40, normal_channel=False)
    model.eval()
    print("✅ 模型创建成功")
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    exit(1)

# 创建测试数据
print(f"\n创建测试数据...")
batch_size = 1
point_cloud = torch.randn(batch_size, 3, 1024)  # (B, 3, N)
print(f"✅ 测试数据: {point_cloud.shape}")

# 测试前向传播
print(f"\n运行前向传播...")
try:
    with torch.no_grad():
        output = model(point_cloud)
    print("✅ 前向传播成功")
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    exit(1)

# 检查输出
print(f"\n检查输出...")
print(f"输出类型: {type(output)}")

if isinstance(output, tuple):
    print(f"输出元组长度: {len(output)}")
    
    if len(output) == 4:
        cls_result, sa1_features, sa2_features, sa3_features = output
        print("✅ 使用新的返回格式 (4个值)")
        print(f"  分类结果维度: {cls_result.shape}")
        print(f"  SA1特征维度: {sa1_features.shape}")
        print(f"  SA2特征维度: {sa2_features.shape}")
        print(f"  SA3特征维度: {sa3_features.shape}")
        
        # 计算数据量
        print(f"\n📊 数据量分析:")
        def calculate_size(tensor):
            return tensor.numel() * 4 / 1024  # KB
        
        sa1_kb = calculate_size(sa1_features)
        sa2_kb = calculate_size(sa2_features)
        sa3_kb = calculate_size(sa3_features)
        total_kb = sa1_kb + sa2_kb + sa3_kb
        
        print(f"  SA1特征: {sa1_kb:.2f} KB")
        print(f"  SA2特征: {sa2_kb:.2f} KB")
        print(f"  SA3特征: {sa3_kb:.2f} KB")
        print(f"  特征总量: {total_kb:.2f} KB")
        
        # 原始点云数据量
        original_kb = point_cloud.numel() * 4 / 1024
        print(f"  原始点云: {original_kb:.2f} KB")
        
        # 压缩比
        compression_ratio = total_kb / original_kb
        print(f"\n  特征/原始数据比: {compression_ratio:.4f}")
        print(f"  信息密度增加: {1/compression_ratio:.2f}倍")
        
        print(f"\n✅ 恭喜！PointNet++修改成功，可以用于语义通信特征提取")
        
    elif len(output) == 2:
        cls_result, l3_points = output
        print("⚠️  使用旧的返回格式 (2个值)")
        print(f"  分类结果维度: {cls_result.shape}")
        print(f"  SA3特征维度: {l3_points.shape}")
        print("\n❌ 需要修改forward函数的返回语句")
        print("  应该返回: x, l1_points, l2_points, l3_points")
        print(f"  当前返回: {len(output)}个值")
    else:
        print(f"❌ 意外的输出长度: {len(output)}")
else:
    print(f"❌ 输出不是元组: {type(output)}")

print("\n" + "="*60)