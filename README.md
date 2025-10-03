# Sentix - 中文情感分析系统

基于BERT的中文文本情感二分类系统，提供完整的训练、评估和Web推理解决方案。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.20+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

## 系统特性

- **多模型支持**: BERT、RoBERTa、MacBERT等中文预训练模型
- **自动化实验**: 超参数网格搜索与性能对比
- **数据增强**: 同义词替换、回译等策略提升泛化能力
- **Web部署**: Flask RESTful API与交互式前端界面
- **完整评估**: 混淆矩阵、错误分析、可视化报告

## 性能指标

最佳模型 chinese-macbert-base (LR=2e-05) 在测试集上达到:
- **准确率**: 96.15%
- **F1分数**: 96.15%
- **训练时长**: 252.7秒

## 快速开始

### 环境配置

```bash
# Python 3.8+ 环境
pip install -r requirements.txt
```

可选GPU加速需要CUDA 11.0+

### 启动服务

```bash
python app.py
```

访问 `http://localhost:5000` 使用Web界面

### Python推理

```python
from scripts.inference.inference import predict_sentiment

result = predict_sentiment("这部电影真的太棒了！")
# 返回: "positive" 或 "negative"
```

### API调用

```python
import requests

response = requests.post('http://localhost:5000/api/predict', 
    json={'text': '我今天很开心！'})
data = response.json()
# {'success': True, 'sentiment': '积极', 'confidence': 0.95, ...}
```

## 项目结构

```
.
├── app.py                          # Flask Web应用
├── templates/                      # HTML模板
├── static/                         # CSS/JS静态资源
├── scripts/                        # 核心功能模块
│   ├── training/                   # 模型训练
│   │   └── sentiment_training.py
│   ├── inference/                  # 推理服务
│   │   ├── inference.py
│   │   └── text_generation.py
│   ├── experiments/                # 自动化实验
│   │   ├── classification_experiments.py
│   │   ├── data_augmentation.py
│   │   └── conditional_generation.py
│   ├── evaluation/                 # 模型评估
│   │   └── comprehensive_evaluation.py
│   └── utils/                      # 工具脚本
├── data/                           # 数据集目录
│   ├── data.csv                    # 原始数据 (4,159样本)
│   └── data_augmented.csv          # 增强数据
├── models/                         # 训练模型存储
│   ├── sentiment_model/            # 主模型
│   ├── model_original/             # 原始数据训练模型
│   ├── model_augmented/            # 增强数据训练模型
│   └── experiments/                # 实验模型快照
└── results/                        # 实验结果与报告
    ├── experiments/                # 模型对比结果
    ├── evaluation/                 # 评估报告与错误分析
    ├── augmentation/               # 数据增强对比
    └── visualizations/             # 性能可视化图表
```

## 数据集

### 数据来源
中文情感标注数据集（繁体中文），包含8种原始情感类别，映射为积极/消极二分类。

### 数据统计
- **总样本数**: 4,159
- **平均文本长度**: 20.3字符
- **数据格式**: CSV (text, emotion)
- **数据划分**: 80% 训练集, 20% 验证集

### 原始情感类别
開心語調、悲傷語調、憤怒語調、平淡語氣、驚奇語調、厭惡語調、關切語調、疑問語調

### 数据增强
使用同义词替换和回译策略增强训练数据，性能提升 +0.48%

## 模型训练

### 单模型训练

```bash
python scripts/training/sentiment_training.py
```

### 批量实验

自动进行多模型、多学习率的网格搜索:

```bash
python scripts/experiments/classification_experiments.py
```

### 数据增强

```bash
python scripts/experiments/data_augmentation.py
```

### 综合评估

```bash
python scripts/evaluation/comprehensive_evaluation.py
```

## 实验结果

### 模型性能对比

| 模型 | 学习率 | 准确率 | F1分数 | 训练时间 |
|------|--------|--------|--------|----------|
| **chinese-macbert-base** | **2e-05** | **96.15%** | **96.15%** | **252.7s** |
| chinese-roberta-wwm-ext | 5e-05 | 95.67% | 95.68% | 258.1s |
| bert-base-chinese | 5e-05 | 95.31% | 95.32% | 253.4s |
| chinese-roberta-wwm-ext | 2e-05 | 95.07% | 95.08% | 251.2s |
| bert-base-chinese | 2e-05 | 94.83% | 94.85% | 254.0s |
| chinese-macbert-base | 5e-05 | 94.71% | 94.72% | 253.3s |

### 数据增强影响

- 原始数据集: 94.83%
- 增强数据集: 95.31%
- 性能提升: +0.48%

## 技术栈

### 核心框架
- **PyTorch**: 深度学习框架
- **Transformers**: Hugging Face预训练模型库
- **Accelerate**: 分布式训练加速

### 数据处理
- **Pandas**: 数据处理与分析
- **NumPy**: 数值计算
- **Scikit-learn**: 评估指标与数据划分

### Web服务
- **Flask**: RESTful API服务
- **Chart.js**: 交互式图表
- **Responsive CSS**: 现代化响应式界面

### 可视化
- **Matplotlib**: 静态图表生成
- **Seaborn**: 统计可视化

## Web界面功能

### 情感分析
- 实时文本情感预测
- 置信度与概率分布可视化
- 快速示例测试

### 模型对比
- 6种模型配置性能对比
- 最佳模型高亮显示
- 交互式图表展示

### 可视化分析
- 性能对比图
- 训练/验证损失曲线
- 混淆矩阵热力图

## 训练流程

1. **数据预处理**: 加载CSV数据，清洗和标准化
2. **数据增强**: 应用同义词替换、回译等策略 (可选)
3. **模型初始化**: 加载预训练中文BERT模型
4. **微调训练**: 在情感分类任务上fine-tune
5. **超参数搜索**: 网格搜索最佳学习率配置
6. **性能评估**: 测试集上评估准确率、F1等指标
7. **模型保存**: 持久化最佳checkpoint

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE)

## 引用

主要依赖项目:
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)
- [Flask](https://flask.palletsprojects.com/)

---

© 2025 Sentix. All rights reserved.
