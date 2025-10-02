# -*- coding: utf-8 -*-
"""
分类模型优化实验
对比不同预训练模型和超参数配置
"""

import pandas as pd
import torch
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class SentimentDataset(Dataset):
    """情感分析数据集"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class LossHistoryCallback(TrainerCallback):
    """记录训练过程中的loss"""
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])

def map_emotion_to_sentiment(emotion):
    """情感标签映射"""
    emotion_map = {
        '開心語調': 1,
        '悲傷語調': 0,
        '憤怒語調': 0,
        '平淡語氣': 0,
        '驚奇語調': 1,
        '厭惡語調': 0,
        '關切語調': 1,
        '疑問語調': 0
    }
    return emotion_map.get(emotion, 0)

def load_data(data_file='data/data.csv'):
    """加载数据"""
    df = pd.read_csv(data_file)
    df['sentiment'] = df['emotion'].apply(map_emotion_to_sentiment)
    return df['text'].tolist(), df['sentiment'].tolist()

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(model_name, learning_rate, data_file, output_dir, epochs=3):
    """
    训练单个模型配置
    
    Args:
        model_name: 模型名称
        learning_rate: 学习率
        data_file: 数据文件
        output_dir: 输出目录
        epochs: 训练轮数
    
    Returns:
        实验结果字典
    """
    print("\n" + "=" * 60)
    print(f"实验配置:")
    print(f"  模型: {model_name}")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {epochs}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 加载tokenizer和模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None
    
    # 加载数据
    texts, labels = load_data(data_file)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"训练集: {len(train_texts)}, 验证集: {len(val_texts)}")
    
    # 创建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None,
        fp16=torch.cuda.is_available(),
    )
    
    # Loss回调
    loss_callback = LossHistoryCallback()
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[loss_callback]
    )
    
    # 训练
    print("\n开始训练...")
    trainer.train()
    
    # 评估
    eval_results = trainer.evaluate()
    training_time = time.time() - start_time
    
    # 获取预测结果用于混淆矩阵
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    cm = confusion_matrix(val_labels, pred_labels)
    
    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 结果
    results = {
        'model_name': model_name,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'train_size': len(train_texts),
        'val_size': len(val_texts),
        'accuracy': float(eval_results['eval_accuracy']),
        'precision': float(eval_results['eval_precision']),
        'recall': float(eval_results['eval_recall']),
        'f1': float(eval_results['eval_f1']),
        'training_time_seconds': training_time,
        'train_losses': loss_callback.train_losses,
        'eval_losses': loss_callback.eval_losses,
        'steps': loss_callback.steps,
        'confusion_matrix': cm.tolist()
    }
    
    # 保存结果
    with open(os.path.join(output_dir, 'experiment_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 训练完成!")
    print(f"  准确率: {results['accuracy']:.4f}")
    print(f"  F1分数: {results['f1']:.4f}")
    print(f"  训练时间: {training_time:.2f}秒")
    
    return results

def run_all_experiments():
    """运行所有实验配置"""
    
    # 实验配置
    models = [
        'bert-base-chinese',
        'hfl/chinese-roberta-wwm-ext',
        'hfl/chinese-macbert-base'
    ]
    
    learning_rates = [2e-5, 5e-5]
    
    print("=" * 60)
    print("🔬 分类模型优化实验")
    print("=" * 60)
    print(f"\n将测试 {len(models)} 个模型 × {len(learning_rates)} 个学习率 = {len(models) * len(learning_rates)} 个配置")
    print(f"预计总用时: 约 {len(models) * len(learning_rates) * 20} 分钟")
    
    response = input("\n是否继续? (y/n): ")
    if response.lower() != 'y':
        print("已取消。")
        return
    
    all_results = []
    experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for model_name in models:
        for lr in learning_rates:
            # 生成输出目录名
            model_short = model_name.split('/')[-1]
            lr_str = f"{lr:.0e}".replace('-', '_')
            output_dir = f"models/experiments/{experiment_id}/{model_short}_lr{lr_str}"
            
            print(f"\n{'='*60}")
            print(f"实验 {len(all_results) + 1}/{len(models) * len(learning_rates)}")
            print(f"{'='*60}")
            
            # 训练
            result = train_model(
                model_name=model_name,
                learning_rate=lr,
                data_file='data/data.csv',
                output_dir=output_dir,
                epochs=3
            )
            
            if result:
                all_results.append(result)
            
            # 短暂休息
            time.sleep(2)
    
    # 生成对比报告
    generate_comparison_report(all_results, experiment_id)
    
    # 绘制对比图表
    plot_experiment_results(all_results, experiment_id)
    
    print("\n" + "=" * 60)
    print("🎉 所有实验完成!")
    print("=" * 60)
    print(f"\n结果保存在: results/experiments/{experiment_id}/")

def generate_comparison_report(results, experiment_id):
    """生成实验对比报告"""
    
    output_dir = f"results/experiments/{experiment_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整结果
    with open(f"{output_dir}/all_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成Markdown报告
    md_content = f"""# 分类模型优化实验报告

**实验ID**: {experiment_id}  
**实验日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**实验配置数**: {len(results)}

## 📊 实验结果汇总

### 性能对比表

| 模型 | 学习率 | 准确率 | 精确率 | 召回率 | F1分数 | 训练时间 |
|------|--------|--------|--------|--------|--------|----------|
"""
    
    for r in results:
        model_short = r['model_name'].split('/')[-1]
        md_content += f"| {model_short} | {r['learning_rate']:.0e} | "
        md_content += f"{r['accuracy']:.4f} | {r['precision']:.4f} | "
        md_content += f"{r['recall']:.4f} | {r['f1']:.4f} | {r['training_time_seconds']:.1f}s |\n"
    
    # 找出最佳配置
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_f1 = max(results, key=lambda x: x['f1'])
    
    md_content += f"""
## 🏆 最佳配置

### 最高准确率
- **模型**: {best_acc['model_name']}
- **学习率**: {best_acc['learning_rate']:.0e}
- **准确率**: {best_acc['accuracy']:.4f}
- **F1分数**: {best_acc['f1']:.4f}

### 最高F1分数
- **模型**: {best_f1['model_name']}
- **学习率**: {best_f1['learning_rate']:.0e}
- **准确率**: {best_f1['accuracy']:.4f}
- **F1分数**: {best_f1['f1']:.4f}

## 📈 关键发现

### 模型对比
"""
    
    # 按模型分组统计
    model_stats = {}
    for r in results:
        model = r['model_name']
        if model not in model_stats:
            model_stats[model] = {'f1': [], 'acc': []}
        model_stats[model]['f1'].append(r['f1'])
        model_stats[model]['acc'].append(r['accuracy'])
    
    for model, stats in model_stats.items():
        avg_f1 = np.mean(stats['f1'])
        avg_acc = np.mean(stats['acc'])
        md_content += f"- **{model}**: 平均F1 = {avg_f1:.4f}, 平均准确率 = {avg_acc:.4f}\n"
    
    md_content += """
### 学习率影响
"""
    
    # 按学习率分组
    lr_stats = {}
    for r in results:
        lr = r['learning_rate']
        if lr not in lr_stats:
            lr_stats[lr] = {'f1': [], 'acc': []}
        lr_stats[lr]['f1'].append(r['f1'])
        lr_stats[lr]['acc'].append(r['accuracy'])
    
    for lr, stats in sorted(lr_stats.items()):
        avg_f1 = np.mean(stats['f1'])
        avg_acc = np.mean(stats['acc'])
        md_content += f"- **LR = {lr:.0e}**: 平均F1 = {avg_f1:.4f}, 平均准确率 = {avg_acc:.4f}\n"
    
    md_content += f"""
## 💡 结论与建议

1. **最佳模型**: {best_f1['model_name']} 在F1分数上表现最好
2. **推荐学习率**: 根据实验结果，{'较高' if best_f1['learning_rate'] > 3e-5 else '较低'}的学习率更适合此任务
3. **训练效率**: 平均训练时间为 {np.mean([r['training_time_seconds'] for r in results]):.1f} 秒

## 📁 实验文件

- 完整结果: `all_results.json`
- Loss曲线: `loss_curves.png`
- 性能对比: `performance_comparison.png`
- 混淆矩阵: `confusion_matrices.png`

## 🔬 科研建议

1. **可重复性**: 所有实验使用固定随机种子(42)，确保可重复
2. **公平对比**: 所有模型使用相同的数据划分和训练参数
3. **多指标评估**: 不仅看准确率，还需关注F1、精确率、召回率
4. **训练稳定性**: 观察loss曲线，确保模型收敛

---

*本报告由自动化实验系统生成*
"""
    
    # 保存报告
    with open(f"{output_dir}/experiment_report.md", 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n✓ 实验报告已保存: {output_dir}/experiment_report.md")

def plot_experiment_results(results, experiment_id):
    """绘制实验结果图表"""
    
    output_dir = f"results/experiments/{experiment_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Loss曲线对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, result in enumerate(results):
        ax = axes[idx // 2, idx % 2]
        model_short = result['model_name'].split('/')[-1]
        lr_str = f"{result['learning_rate']:.0e}"
        
        if result['steps'] and result['train_losses']:
            ax.plot(result['steps'], result['train_losses'], label='Train Loss', linewidth=2)
        
        ax.set_title(f"{model_short} (LR={lr_str})", fontsize=12, fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curves.png", dpi=300, bbox_inches='tight')
    print(f"✓ Loss曲线已保存: {output_dir}/loss_curves.png")
    plt.close()
    
    # 2. 性能指标对比
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    models = [r['model_name'].split('/')[-1] for r in results]
    lrs = [f"{r['learning_rate']:.0e}" for r in results]
    labels = [f"{m}\n{lr}" for m, lr in zip(models, lrs)]
    
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1'] for r in results]
    
    x = np.arange(len(results))
    
    # 准确率
    axes[0].bar(x, accuracies, color='steelblue', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel('准确率')
    axes[0].set_title('模型准确率对比', fontweight='bold')
    axes[0].set_ylim([min(accuracies) - 0.05, max(accuracies) + 0.05])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    # F1分数
    axes[1].bar(x, f1_scores, color='coral', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('F1分数')
    axes[1].set_title('模型F1分数对比', fontweight='bold')
    axes[1].set_ylim([min(f1_scores) - 0.05, max(f1_scores) + 0.05])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(f1_scores):
        axes[1].text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ 性能对比图已保存: {output_dir}/performance_comparison.png")
    plt.close()
    
    # 3. 混淆矩阵
    n_results = len(results)
    n_plots = min(n_results, 6)  # 最多显示6个
    
    # 计算子图布局
    if n_plots <= 2:
        nrows, ncols = 1, n_plots
    elif n_plots <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    
    # 确保axes是2D数组
    if n_plots == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_plots):
        result = results[idx]
        ax = axes[idx // ncols, idx % ncols]
        model_short = result['model_name'].split('/')[-1]
        lr_str = f"{result['learning_rate']:.0e}"
        
        cm = np.array(result['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax.set_title(f"{model_short} (LR={lr_str})", fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # 隐藏多余的子图
    for idx in range(n_plots, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵已保存: {output_dir}/confusion_matrices.png")
    plt.close()

def regenerate_plots_only(experiment_id=None):
    """
    只重新生成图表，不重新训练
    整合自regenerate_plots.py
    """
    print("=" * 60)
    print("🎨 重新生成实验可视化图表")
    print("=" * 60)
    
    # 查找实验结果
    exp_base = "results/experiments"
    if not os.path.exists(exp_base):
        print(f"❌ 找不到实验目录: {exp_base}")
        return
    
    # 获取所有实验ID
    exp_ids = [d for d in os.listdir(exp_base) if os.path.isdir(os.path.join(exp_base, d))]
    
    if not exp_ids:
        print("❌ 没有找到任何实验结果")
        return
    
    # 使用指定的实验或最新的实验
    if experiment_id is None:
        exp_ids.sort(reverse=True)
        experiment_id = exp_ids[0]
    
    print(f"\n使用实验ID: {experiment_id}")
    
    # 加载结果
    result_dir = f"{exp_base}/{experiment_id}"
    result_file = f"{result_dir}/all_results.json"
    
    if not os.path.exists(result_file):
        print(f"❌ 找不到实验结果: {result_file}")
        return
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"✓ 加载了 {len(results)} 个实验结果")
    
    # 生成图表
    plot_experiment_results(results, experiment_id)
    
    print("\n" + "=" * 60)
    print("✅ 完成！图表已更新:")
    print("=" * 60)
    print(f"📁 {result_dir}/loss_curves.png")
    print(f"📁 {result_dir}/performance_comparison.png")
    print(f"📁 {result_dir}/confusion_matrices.png")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分类模型优化实验')
    parser.add_argument('--regenerate-only', action='store_true',
                       help='只重新生成图表，不重新训练')
    parser.add_argument('--experiment-id', type=str, default=None,
                       help='指定要重新生成图表的实验ID（用于--regenerate-only）')
    
    args = parser.parse_args()
    
    if args.regenerate_only:
        regenerate_plots_only(args.experiment_id)
    else:
        run_all_experiments()

if __name__ == "__main__":
    main()

