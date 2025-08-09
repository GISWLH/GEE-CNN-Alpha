#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上传大文件到GitHub仓库的脚本
由于某些CSV文件较大，可以选择性上传
"""

import subprocess
import os
from pathlib import Path

def run_git_command(command):
    """运行Git命令"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ 成功: {command}")
            if result.stdout:
                print(f"   输出: {result.stdout.strip()}")
        else:
            print(f"❌ 失败: {command}")
            print(f"   错误: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 异常: {command} - {e}")
        return False

def check_file_size(file_path):
    """检查文件大小"""
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return size_mb
    return 0

print("检查大文件并选择性上传...")

# 检查大文件
large_files = [
    "data/water_CNN_with_AlphaEarth_all_merged.csv",
    "data/alphaearth_extracted_values.csv"
]

for file_path in large_files:
    if os.path.exists(file_path):
        size_mb = check_file_size(file_path)
        print(f"\n文件: {file_path}")
        print(f"大小: {size_mb:.2f} MB")
        
        if size_mb > 100:
            print(f"⚠️ 文件过大 ({size_mb:.2f} MB)，建议使用Git LFS或分割文件")
            continue
        elif size_mb > 25:
            print(f"⚠️ 文件较大 ({size_mb:.2f} MB)，GitHub有25MB单文件限制")
            response = input(f"是否仍要上传 {file_path}? (y/n): ")
            if response.lower() != 'y':
                continue
        
        # 添加文件
        if run_git_command(f'git add "{file_path}"'):
            print(f"✅ 已添加: {file_path}")
        else:
            print(f"❌ 添加失败: {file_path}")

# 检查是否有文件需要提交
result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
if result.stdout.strip():
    print(f"\n有文件需要提交:")
    print(result.stdout)
    
    response = input("是否提交并推送这些文件? (y/n): ")
    if response.lower() == 'y':
        # 提交
        commit_msg = "Add large data files"
        if run_git_command(f'git commit -m "{commit_msg}"'):
            # 推送
            if run_git_command("git push origin main"):
                print("🎉 大文件上传完成！")
            else:
                print("❌ 推送失败")
        else:
            print("❌ 提交失败")
else:
    print("✅ 没有新文件需要上传")

print(f"\n📁 GitHub仓库: https://github.com/GISWLH/GEE-CNN-Alpha")
print("🎉 项目上传完成！")
