#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT中文情感分析微调脚本
使用 bert-base-chinese 模型进行二分类情感分析
"""

import pandas as pd
import torch
import argparse
import os
import json
import time
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import numpy as np

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class SentimentDataset(Dataset):
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
        
        # 使用tokenizer编码文本
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

def map_emotion_to_sentiment(emotion):
    """
    将原始情感标签映射到positive/negative二分类
    
    映射规则:
    - 開心語調 -> positive (1)
    - 悲傷語調 -> negative (0)
    - 憤怒語調 -> negative (0)  
    - 平淡語氣 -> negative (0) (中性倾向负面)
    - 驚奇語調 -> positive (1) (惊奇一般是正面的)
    - 厭惡語調 -> negative (0)
    - 關切語調 -> positive (1) (关心是正面的)
    - 疑問語調 -> negative (0) (疑问倾向中性，归为负面)
    """
    emotion_map = {
        '開心語調': 1,  # positive
        '悲傷語調': 0,  # negative
        '憤怒語調': 0,  # negative
        '平淡語氣': 0,  # negative
        '驚奇語調': 1,  # positive
        '厭惡語調': 0,  # negative
        '關切語調': 1,  # positive
        '疑問語調': 0   # negative
    }
    return emotion_map.get(emotion, 0)  # 默认为negative

def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    print("加载数据...")
    df = pd.read_csv(file_path)
    print(f"原始数据大小: {len(df)}")
    
    # 查看标签分布
    print("原始标签分布:")
    print(df['emotion'].value_counts())
    
    # 映射标签
    df['sentiment'] = df['emotion'].apply(map_emotion_to_sentiment)
    
    # 查看映射后的标签分布
    print("映射后标签分布:")
    print("0 (negative):", sum(df['sentiment'] == 0))
    print("1 (positive):", sum(df['sentiment'] == 1))
    
    return df['text'].tolist(), df['sentiment'].tolist()

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main(args=None):
    """主训练函数"""
    # 解析命令行参数
    if args is None:
        parser = argparse.ArgumentParser(description='BERT中文情感分析训练')
        parser.add_argument('--data_file', type=str, default='data/data.csv', help='训练数据文件路径')
        parser.add_argument('--output_dir', type=str, default='models/sentiment_model', help='模型输出目录')
        parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
        parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
        parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
        args = parser.parse_args()
    
    print("=" * 60)
    print("开始BERT中文情感分析微调")
    print("=" * 60)
    print(f"数据文件: {args.data_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.epochs}")
    
    start_time = time.time()
    
    # 1. 加载tokenizer和模型
    model_name = "bert-base-chinese"
    print(f"\n加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # 二分类
    )
    
    # 2. 加载并预处理数据
    texts, labels = load_and_preprocess_data(args.data_file)
    
    # 3. 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    
    # 4. 创建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    # 5. 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,  # 不使用wandb等
        fp16=torch.cuda.is_available(),  # 如果有GPU就使用混合精度
    )
    
    # 6. 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 7. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 8. 开始训练
    print("开始训练...")
    trainer.train()
    
    # 9. 评估模型
    print("\n评估模型...")
    eval_results = trainer.evaluate()
    
    # 计算训练时间
    training_time = time.time() - start_time
    
    print(f"\n验证集性能:")
    print(f"  准确率: {eval_results['eval_accuracy']:.4f}")
    print(f"  精确率: {eval_results['eval_precision']:.4f}")
    print(f"  召回率: {eval_results['eval_recall']:.4f}")
    print(f"  F1分数: {eval_results['eval_f1']:.4f}")
    print(f"  训练时间: {training_time:.2f}秒")
    
    # 10. 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n模型已保存到 {args.output_dir}")
    
    # 11. 保存训练结果
    results = {
        'data_file': args.data_file,
        'sample_size': len(texts),
        'train_size': len(train_texts),
        'val_size': len(val_texts),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'accuracy': float(eval_results['eval_accuracy']),
        'precision': float(eval_results['eval_precision']),
        'recall': float(eval_results['eval_recall']),
        'f1': float(eval_results['eval_f1']),
        'training_time_seconds': training_time
    }
    
    results_file = os.path.join(args.output_dir, 'training_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"训练结果已保存到 {results_file}")
    
    # 12. 测试模型推理
    print("\n测试模型推理...")
    test_inference(model, tokenizer)
    
    return results

def test_inference(model, tokenizer):
    """测试模型推理功能"""
    # 设置模型为评估模式
    model.eval()
    device = next(model.parameters()).device  # 获取模型所在的设备
    
    # 标签映射
    id2label = {0: "negative", 1: "positive"}
    
    # 测试样例
    test_texts = [
        "我今天很开心，终于完成了这个项目！",
        "我感觉很沮丧，什么都做不好",
        "这家餐厅的食物真的很难吃",
        "今天天气真好，心情不错"
    ]
    
    print("测试推理结果:")
    print("-" * 50)
    
    for text in test_texts:
        # 编码文本
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # 将输入移动到模型所在的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(prediction, dim=-1).item()
            confidence = prediction[0][predicted_class].item()
        
        result = id2label[predicted_class]
        print(f"文本: {text}")
        print(f"预测: {result} (置信度: {confidence:.4f})")
        print("-" * 50)

def predict_sentiment(text, model_path='./sentiment_model'):
    """
    单独的推理函数，用于预测单个文本的情感
    
    Args:
        text (str): 输入文本
        model_path (str): 模型路径
    
    Returns:
        str: "positive" 或 "negative"
    """
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # 编码文本
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(prediction, dim=-1).item()
    
    # 返回结果
    return "positive" if predicted_class == 1 else "negative"

if __name__ == "__main__":
    main()
