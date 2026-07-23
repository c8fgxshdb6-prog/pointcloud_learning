#!/usr/bin/env python3
"""
修复版 Stanford 3D 数据下载脚本
解决 Pylance 警告问题
"""

import os
import tarfile
import urllib.request
import requests
import sys

class SimpleProgressBar:
    """简单的进度条替代方案"""
    def __init__(self, total=100, desc="Downloading"):
        self.total = total
        self.desc = desc
        self.current = 0
    
    def update(self, n=1):
        self.current += n
        percent = (self.current / self.total) * 100
        sys.stdout.write(f"\r{self.desc}: [{('=' * int(percent//2)):<50}] {percent:.1f}%")
        sys.stdout.flush()
    
    def close(self):
        print()  # 换行

class StanfordDataDownloader:
    """Stanford 3D 数据下载器"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.base_url = "http://graphics.stanford.edu/data/3Dscanrep/"
        
        # 创建数据目录
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_file(self, filename, show_progress=True):
        """下载文件"""
        url = self.base_url + filename
        local_path = os.path.join(self.data_dir, filename)
        
        print(f"📥 正在下载: {filename}")
        print(f"🔗 来源: {url}")
        
        try:
            if show_progress:
                # 使用简单进度条
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                progress = SimpleProgressBar(total=total_size, desc=filename)
                
                with open(local_path, 'wb') as file:
                    for data in response.iter_content(chunk_size=1024):
                        file.write(data)
                        progress.update(len(data))
                progress.close()
            else:
                # 简单下载
                urllib.request.urlretrieve(url, local_path)
                
            print(f"✅ 下载完成: {local_path}")
            return local_path
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return None
    
    def extract_tar(self, tar_path, extract_dir=None):
        """解压 tar.gz 文件"""
        if extract_dir is None:
            extract_dir = self.data_dir
            
        print(f"📦 正在解压: {os.path.basename(tar_path)}")
        
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                # 获取文件列表
                members = tar.getmembers()
                print(f"  包含 {len(members)} 个文件")
                
                # 解压所有文件
                tar.extractall(path=extract_dir)
                
                # 打印解压的文件
                for member in members[:5]:  # 只显示前5个文件
                    print(f"    📄 {member.name}")
                if len(members) > 5:
                    print(f"    ... 还有 {len(members) - 5} 个文件")
                    
            print(f"✅ 解压完成: {extract_dir}")
            return True
            
        except Exception as e:
            print(f"❌ 解压失败: {e}")
            return False
    
    def download_stanford_bunny(self):
        """下载 Stanford Bunny 数据"""
        print("🐰 开始下载 Stanford Bunny 数据...")
        
        # 下载 bunny.tar.gz
        tar_path = self.download_file("bunny.tar.gz")
        
        if tar_path:
            # 创建专门的 bunny 目录
            bunny_dir = os.path.join(self.data_dir, "bunny")
            os.makedirs(bunny_dir, exist_ok=True)
            
            # 解压到 bunny 目录
            if self.extract_tar(tar_path, bunny_dir):
                print("🎉 Stanford Bunny 下载和解压完成!")
                return bunny_dir
        
        return None
    
    def download_stanford_dragon(self):
        """下载 Stanford Dragon 数据"""
        print("🐉 开始下载 Stanford Dragon 数据...")
        
        # 下载 dragon.tar.gz
        tar_path = self.download_file("dragon.tar.gz")
        
        if tar_path:
            # 创建专门的 dragon 目录
            dragon_dir = os.path.join(self.data_dir, "dragon")
            os.makedirs(dragon_dir, exist_ok=True)
            
            # 解压到 dragon 目录
            if self.extract_tar(tar_path, dragon_dir):
                print("🎉 Stanford Dragon 下载和解压完成!")
                return dragon_dir
        
        return None

def main():
    """主函数"""
    downloader = StanfordDataDownloader()
    
    print("🚀 Stanford 3D 数据下载器 (修复版)")
    print("=" * 50)
    
    # 下载 Bunny
    bunny_dir = downloader.download_stanford_bunny()
    
    print("\n" + "-" * 30)
    
    # 下载 Dragon
    dragon_dir = downloader.download_stanford_dragon()
    
    print("\n" + "=" * 50)
    
    # 总结
    if bunny_dir or dragon_dir:
        print("🎊 下载完成总结:")
        if bunny_dir:
            print(f"  ✅ Stanford Bunny: {bunny_dir}")
        if dragon_dir:
            print(f"  ✅ Stanford Dragon: {dragon_dir}")
    else:
        print("❌ 下载失败，请检查网络连接")

if __name__ == "__main__":
    main()