# scripts/view_reports.py
import os

def view_reports():
    """查看生成的报告"""
    print("="*60)
    print("PointNet++ 数据流分析报告汇总")
    print("="*60)
    
    # 查找最新的报告文件
    import glob
    report_files = glob.glob("reports/pointnet2_report_*.txt")
    summary_files = glob.glob("reports/report_summary.txt")
    
    if report_files:
        latest_report = max(report_files, key=os.path.getctime)
        print(f"\n📋 详细报告: {latest_report}")
        print("-"*40)
        with open(latest_report, 'r', encoding='utf-8') as f:
            content = f.read()
            # 显示前30行
            lines = content.split('\n')
            for i, line in enumerate(lines[:30]):
                print(line)
            if len(lines) > 30:
                print("... (完整内容请查看文件)")
    
    if summary_files:
        print(f"\n🎯 汇报要点: {summary_files[0]}")
        print("-"*40)
        with open(summary_files[0], 'r', encoding='utf-8') as f:
            print(f.read())
    
    print("\n" + "="*60)
    print("✅ 报告查看完成！")
    print("="*60)

if __name__ == "__main__":
    view_reports()