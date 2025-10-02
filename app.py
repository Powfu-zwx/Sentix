# -*- coding: utf-8 -*-
"""
BERT中文情感分析 - Flask Web应用
现代化专业界面
"""

from flask import Flask, render_template, request, jsonify
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

app = Flask(__name__)

# 全局变量
cached_model = None
cached_tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path='models/sentiment_model'):
    """加载模型和tokenizer"""
    global cached_model, cached_tokenizer
    
    if cached_model is None or cached_tokenizer is None:
        try:
            cached_tokenizer = AutoTokenizer.from_pretrained(model_path)
            cached_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            cached_model.eval()
            cached_model = cached_model.to(device)
            print(f"✓ 模型加载成功，使用设备: {device}")
        except Exception as e:
            print(f"✗ 模型加载失败: {str(e)}")
            return None, None
    
    return cached_model, cached_tokenizer

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """情感预测API"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': '请输入有效文本'
            })
        
        model, tokenizer = load_model()
        if model is None:
            return jsonify({
                'success': False,
                'error': '模型未加载'
            })
        
        # 编码输入
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
        
        return jsonify({
            'success': True,
            'sentiment': '积极' if predicted_class == 1 else '消极',
            'sentiment_en': 'positive' if predicted_class == 1 else 'negative',
            'confidence': float(confidence),
            'prob_negative': float(probs[0][0].item()),
            'prob_positive': float(probs[0][1].item()),
            'text_length': len(text)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/experiments', methods=['GET'])
def get_experiments():
    """获取实验结果"""
    try:
        results_path = Path("results/experiments/20251002_130525/all_results.json")
        
        if not results_path.exists():
            return jsonify({
                'success': False,
                'error': '实验结果文件不存在'
            })
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 格式化数据
        formatted_results = []
        for result in results:
            formatted_results.append({
                'model': result['model_name'].replace('hfl/', ''),
                'learning_rate': result['learning_rate'],
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'training_time': result['training_time_seconds'],
                'train_losses': result['train_losses'],
                'eval_losses': result['eval_losses'],
                'steps': result['steps']
            })
        
        return jsonify({
            'success': True,
            'data': formatted_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/system', methods=['GET'])
def get_system_info():
    """获取系统信息"""
    return jsonify({
        'success': True,
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'torch_version': torch.__version__
    })

if __name__ == '__main__':
    # 预加载模型
    print("正在启动应用...")
    load_model()
    print("应用启动成功！")
    
    # 启动Flask服务器
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
