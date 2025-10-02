#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目清理脚本
删除重复和无用内容，优化项目架构
"""

import os
import shutil
from pathlib import Path

def get_size_mb(path):
    """获取文件或目录大小（MB）"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def cleanup_checkpoints(base_dir="models"):
    """删除所有checkpoint文件夹"""
    print("\n1️⃣ 清理Checkpoint文件...")
    print("-" * 60)
    
    deleted_size = 0
    deleted_count = 0
    
    for root, dirs, files in os.walk(base_dir):
        for dirname in dirs:
            if dirname.startswith("checkpoint-"):
                checkpoint_path = os.path.join(root, dirname)
                size = get_size_mb(checkpoint_path)
                
                print(f"  🗑️  删除: {checkpoint_path} ({size:.1f} MB)")
                shutil.rmtree(checkpoint_path)
                
                deleted_size += size
                deleted_count += 1
    
    print(f"\n  ✅ 删除了 {deleted_count} 个checkpoint")
    print(f"  💾 释放空间: {deleted_size:.1f} MB")
    
    return deleted_size

def cleanup_gradio_temp():
    """清理Gradio临时文件"""
    print("\n2️⃣ 清理Gradio临时文件...")
    print("-" * 60)
    
    gradio_dir = ".gradio"
    if os.path.exists(gradio_dir):
        size = get_size_mb(gradio_dir)
        print(f"  🗑️  删除: {gradio_dir}/ ({size:.1f} MB)")
        shutil.rmtree(gradio_dir)
        print(f"  ✅ 已删除")
        return size
    else:
        print(f"  ℹ️  没有找到临时文件")
        return 0

def cleanup_pycache():
    """清理Python缓存"""
    print("\n3️⃣ 清理Python缓存...")
    print("-" * 60)
    
    deleted_size = 0
    deleted_count = 0
    
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            cache_path = os.path.join(root, "__pycache__")
            size = get_size_mb(cache_path)
            
            print(f"  🗑️  删除: {cache_path}")
            shutil.rmtree(cache_path)
            
            deleted_size += size
            deleted_count += 1
        
        # 删除.pyc文件
        for filename in files:
            if filename.endswith(".pyc"):
                file_path = os.path.join(root, filename)
                os.remove(file_path)
    
    print(f"  ✅ 删除了 {deleted_count} 个缓存目录")
    print(f"  💾 释放空间: {deleted_size:.1f} MB")
    
    return deleted_size

def cleanup_duplicate_models():
    """清理实验中的重复模型（保留最新的）"""
    print("\n4️⃣ 清理实验模型...")
    print("-" * 60)
    
    exp_base = "models/experiments"
    if not os.path.exists(exp_base):
        print("  ℹ️  没有实验模型")
        return 0
    
    experiments = sorted([d for d in os.listdir(exp_base) 
                         if os.path.isdir(os.path.join(exp_base, d))], 
                        reverse=True)
    
    if len(experiments) <= 1:
        print(f"  ℹ️  只有 {len(experiments)} 个实验，保留全部")
        return 0
    
    # 保留最新的实验，删除旧的
    deleted_size = 0
    keep_exp = experiments[0]
    print(f"  ✅ 保留最新实验: {keep_exp}")
    
    for exp_id in experiments[1:]:
        exp_path = os.path.join(exp_base, exp_id)
        size = get_size_mb(exp_path)
        
        print(f"  🗑️  删除旧实验: {exp_id} ({size:.1f} MB)")
        shutil.rmtree(exp_path)
        deleted_size += size
    
    print(f"  💾 释放空间: {deleted_size:.1f} MB")
    
    return deleted_size

def optimize_results_structure():
    """优化results目录结构"""
    print("\n5️⃣ 优化结果目录...")
    print("-" * 60)
    
    # 移动顶层可视化到可视化子目录
    viz_dir = "results/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    files_to_move = [
        ("results/data_visualization.png", "results/visualizations/data_distribution.png"),
        ("results/augmentation_comparison.json", "results/augmentation/comparison.json"),
        ("results/augmentation_comparison.md", "results/augmentation/comparison.md"),
    ]
    
    for src, dst in files_to_move:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            print(f"  📁 移动: {src} -> {dst}")
    
    print("  ✅ 目录结构已优化")

def create_readme_files():
    """为各目录创建README"""
    print("\n6️⃣ 创建目录说明...")
    print("-" * 60)
    
    readmes = {
        "models/README.md": """# 模型目录

## 目录说明

- `sentiment_model/` - 训练好的情感分析模型（用于生产）
- `experiments/` - 实验模型（对比测试用）

## 模型文件

每个模型目录包含：
- `model.safetensors` - 模型权重
- `config.json` - 模型配置
- `tokenizer.json` - 分词器
- `vocab.txt` - 词表

## 使用方法

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('models/sentiment_model')
tokenizer = AutoTokenizer.from_pretrained('models/sentiment_model')
```
""",
        "results/README.md": """# 实验结果目录

## 目录结构

- `visualizations/` - 数据可视化图表
- `augmentation/` - 数据增强对比结果
- `experiments/` - 模型对比实验结果
- `generation_experiments/` - 生成模型实验
- `conditional_generation/` - 条件生成实验

## 文件说明

所有实验都会生成：
- JSON格式的结果数据
- Markdown格式的分析报告
- PNG格式的可视化图表
"""
    }
    
    for path, content in readmes.items():
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ 创建: {path}")

def generate_cleanup_report(total_saved):
    """生成清理报告"""
    print("\n" + "=" * 60)
    print("📊 清理总结")
    print("=" * 60)
    
    report = f"""
✅ 清理完成！

💾 总共释放空间: {total_saved:.1f} MB

📁 优化后的目录结构:
```
project/
├── data/              # 数据集
├── docs/              # 文档
├── models/            # 模型（已删除checkpoint）
│   ├── sentiment_model/
│   └── experiments/   # 只保留最新实验
├── results/           # 结果（已重组）
│   ├── visualizations/
│   ├── augmentation/
│   └── experiments/
└── scripts/           # 脚本
```

🎯 下一步建议:
1. 运行: git status 查看变化
2. 如需恢复checkpoint，可以重新训练
3. 定期运行此脚本保持项目整洁
"""
    
    print(report)
    
    # 保存报告
    with open("CLEANUP_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    print("📄 清理报告已保存到: CLEANUP_REPORT.txt")

def main():
    """主函数"""
    print("=" * 60)
    print("🧹 项目清理与优化")
    print("=" * 60)
    print("\n⚠️  此操作将删除：")
    print("  - 所有checkpoint文件夹")
    print("  - Gradio临时文件")
    print("  - Python缓存")
    print("  - 旧的实验结果（保留最新）")
    
    response = input("\n确认继续? (y/n): ")
    if response.lower() != 'y':
        print("❌ 已取消")
        return
    
    total_saved = 0
    
    # 执行清理
    total_saved += cleanup_checkpoints()
    total_saved += cleanup_gradio_temp()
    total_saved += cleanup_pycache()
    total_saved += cleanup_duplicate_models()
    
    # 优化结构
    optimize_results_structure()
    create_readme_files()
    
    # 生成报告
    generate_cleanup_report(total_saved)

if __name__ == "__main__":
    main()

