#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT中文情感分析推理脚本
输入文本 → 输出 "positive" 或 "negative"
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_sentiment(text, model_path='./sentiment_model'):
    """
    预测单个文本的情感
    
    Args:
        text (str): 输入文本
        model_path (str): 模型路径
    
    Returns:
        str: "positive" 或 "negative"
    """
    try:
        # 加载模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        # 检查设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
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
        
        # 返回结果
        result = "positive" if predicted_class == 1 else "negative"
        print(f"输入文本: {text}")
        print(f"预测结果: {result}")
        print(f"置信度: {confidence:.4f}")
        return result
        
    except Exception as e:
        print(f"预测出错: {e}")
        return "negative"

def main():
    """交互式预测"""
    print("BERT中文情感分析 - 交互式预测")
    print("输入 'quit' 退出程序")
    print("-" * 50)
    
    while True:
        text = input("请输入要分析的文本: ")
        if text.lower() == 'quit':
            print("程序退出")
            break
        
        if text.strip():
            predict_sentiment(text)
            print("-" * 50)

if __name__ == "__main__":
    # 可以直接调用函数测试
    test_texts = [
        "我今天很开心，终于完成了这个项目！",
        "我感觉很沮丧，什么都做不好",
        "这家餐厅的食物真的很难吃",
        "今天天气真好，心情不错"
    ]
    
    print("测试样例:")
    for text in test_texts:
        predict_sentiment(text)
        print()
    
    # 启动交互模式
    main()
