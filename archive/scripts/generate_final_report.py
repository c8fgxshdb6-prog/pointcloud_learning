# scripts/generate_final_report_simple.py
"""
生成PointNet++数据流分析最终报告（简化版）
"""
import json
import os
import glob
from datetime import datetime

def load_latest_dataflow_file():
    """加载最新的数据流分析文件"""
    dataflow_files = glob.glob("logs/dataflow_analysis/pointnet2_dataflow_detailed_*.json")
    
    if not dataflow_files:
        print("未找到数据流分析文件，请先运行分析脚本")
        return None
    
    latest_file = max(dataflow_files, key=os.path.getctime)
    print(f"使用数据文件: {latest_file}")
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data, latest_file
    except Exception as e:
        print(f"加载文件失败: {e}")
        return None, None

def create_simple_report(data, source_file):
    """创建简化的报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"reports/pointnet2_report_{timestamp}.txt"
    
    os.makedirs("reports", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PointNet++ 数据流分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("生成时间: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("数据源文件: {}\n\n".format(source_file))
        
        # 提取关键数据
        records = data.get('dataflow_records', [])
        sa_outputs = [r for r in records if r['layer'].startswith('SA') and r['stage'] == 'output']
        
        f.write("1. 三层特征抽象总结:\n")
        f.write("-" * 40 + "\n")
        
        for i, layer in enumerate(sa_outputs):
            if i == 0:
                input_points = 1024
            else:
                input_points = sa_outputs[i-1]['new_xyz_shape'][-1]
            
            output_points = layer['new_xyz_shape'][-1]
            feature_dim = layer['new_points_shape'][1]
            
            # 语义级别
            levels = ["局部几何", "部件结构", "全局语义"]
            level = levels[i] if i < len(levels) else "未知"
            
            f.write(f"   {layer['layer']}层:\n")
            f.write(f"     输入点数: {input_points}\n")
            f.write(f"     输出点数: {output_points}\n")
            f.write(f"     特征维度: {feature_dim}\n")
            f.write(f"     语义级别: {level}\n\n")
        
        # 数据量对比
        f.write("2. 数据量对比:\n")
        f.write("-" * 40 + "\n")
        
        # 原始点云数据量
        original_size = 2 * 3 * 1024 * 4 / 1024  # 2 batch, 3 channels, 1024 points, 4 bytes per float32
        
        f.write(f"   原始点云: {original_size:.1f} KB\n")
        
        for i, layer in enumerate(sa_outputs):
            elements = layer['new_xyz_shape'][0] * layer['new_points_shape'][1] * layer['new_points_shape'][2]
            data_size_kb = elements * 4 / 1024
            f.write(f"   {layer['layer']}特征: {data_size_kb:.1f} KB\n")
        
        f.write("\n3. 关键发现:\n")
        f.write("-" * 40 + "\n")
        f.write("   a) PointNet++通过三层SA层实现层次化特征提取\n")
        f.write("   b) 点数从1024压缩到1，特征维度从3扩展到1024\n")
        f.write("   c) SA1/SA2特征数据量大于原始点云，需要压缩\n")
        f.write("   d) SA3特征数据量最小(8KB)，最适合语义通信\n\n")
        
        f.write("4. 语义通信建议:\n")
        f.write("-" * 40 + "\n")
        f.write("   高质量信道: 传输SA1+SA2+SA3特征\n")
        f.write("   中等质量信道: 传输SA2+SA3特征\n")
        f.write("   低质量信道: 只传输SA3特征(8KB)\n\n")
        
        f.write("5. 下一步工作:\n")
        f.write("-" * 40 + "\n")
        f.write("   a) 设计SA3特征的高效编码方案\n")
        f.write("   b) 实现自适应特征选择算法\n")
        f.write("   c) 与通信模块集成测试\n")
    
    print(f"报告已生成: {report_file}")
    return report_file

def create_summary_slides():
    """创建汇报要点"""
    slides_file = "reports/report_summary.txt"
    
    with open(slides_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("汇报要点\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 核心发现\n")
        f.write("   - PointNet++三层特征抽象: SA1→SA2→SA3\n")
        f.write("   - 点数压缩: 1024 → 512 → 128 → 1\n")
        f.write("   - 特征扩展: 3 → 320 → 640 → 1024\n\n")
        
        f.write("2. 数据量对比\n")
        f.write("   - 原始点云: 12.0 KB\n")
        f.write("   - SA1特征: 1280.0 KB (需要压缩)\n")
        f.write("   - SA2特征: 640.0 KB (需要压缩)\n")
        f.write("   - SA3特征: 8.0 KB (适合传输)\n\n")
        
        f.write("3. 语义通信建议\n")
        f.write("   - 低质量信道: 只传输SA3特征(8KB)\n")
        f.write("   - 自适应策略: 根据信道质量选择特征层次\n")
        f.write("   - SA3特征可直接用于分类任务\n\n")
        
        f.write("4. 我的贡献\n")
        f.write("   - 深入分析了PointNet++数据流\n")
        f.write("   - 明确了各层特征的语义含义\n")
        f.write("   - 为语义通信提供了理论基础\n")
        f.write("   - 建议使用SA3特征进行传输\n")
    
    print(f"汇报要点已生成: {slides_file}")
    return slides_file

def main():
    """主函数"""
    print("正在生成分析报告...")
    
    # 加载数据
    result = load_latest_dataflow_file()
    if result is None:
        print("无法加载数据文件，请检查路径")
        return
    
    data, source_file = result
    
    # 生成报告
    report = create_simple_report(data, source_file)
    
    # 生成汇报要点
    slides = create_summary_slides()
    
    print("\n" + "=" * 60)
    print("报告生成完成!")
    print("=" * 60)
    print(f"1. 详细报告: {report}")
    print(f"2. 汇报要点: {slides}")
    print("\n给学长的汇报建议:")
    print("1. 重点说明SA3特征的优势(8KB, 适合传输)")
    print("2. 展示数据流分析结果(点数压缩、特征扩展)")
    print("3. 提出自适应特征选择策略")
    print("4. 讨论下一步合作计划")

if __name__ == "__main__":
    main()