# 🎯 BERT中文情感分析系统

基于预训练BERT模型的中文文本情感分析系统，提供完整的训练、评估和Web部署解决方案。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.20+-yellow.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

## ✨ 项目特点

- 🤖 **多模型支持** - 支持BERT、RoBERTa、MacBERT等多种预训练中文模型
- 📊 **完整实验流程** - 包含数据处理、模型训练、超参数优化、性能评估
- 🎨 **现代化Web界面** - 基于Flask的专业Web应用，实时情感分析
- 📈 **可视化分析** - 丰富的图表展示训练过程和模型性能对比
- 🎯 **高准确率** - 最佳模型准确率达96.15%
- 🔧 **易于扩展** - 模块化设计，便于添加新模型和功能

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动Web应用

```bash
python app.py
```

访问 http://localhost:5000 即可使用Web界面进行情感分析。

## 📂 项目结构

```
.
├── app.py                          # Flask Web应用主程序
├── templates/                      # HTML模板
│   └── index.html                 # 主界面
├── static/                         # 静态资源
│   ├── style.css                  # 样式文件
│   └── script.js                  # 前端脚本
├── scripts/                        # 核心功能脚本
│   ├── training/                  # 模型训练
│   │   └── sentiment_training.py
│   ├── inference/                 # 模型推理
│   │   ├── inference.py
│   │   └── text_generation.py
│   ├── experiments/               # 实验对比
│   │   ├── classification_experiments.py
│   │   ├── data_augmentation.py
│   │   ├── conditional_generation.py
│   │   └── generation_experiments.py
│   ├── evaluation/                # 模型评估
│   │   └── comprehensive_evaluation.py
│   └── utils/                     # 工具函数
│       └── cleanup_project.py
├── data/                          # 数据集
│   ├── data.csv                   # 原始数据
│   └── data_augmented.csv         # 增强数据
├── models/                        # 训练好的模型
│   ├── sentiment_model/           # 主模型
│   ├── model_original/            # 原始数据训练模型
│   ├── model_augmented/           # 增强数据训练模型
│   └── experiments/               # 实验模型
└── results/                       # 实验结果
    ├── experiments/               # 实验对比结果
    ├── evaluation/                # 评估报告
    ├── augmentation/              # 数据增强对比
    └── visualizations/            # 可视化图表
```

## 💡 核心功能

### 1. 情感分析

实时分析中文文本的情感倾向（积极/消极），显示置信度和详细概率分布。

```python
from scripts.inference.inference import predict_sentiment

text = "我今天很开心，终于完成了这个项目！"
result = predict_sentiment(text)
# 输出: "positive"
```

### 2. 模型训练

支持多种预训练模型的微调训练：

```bash
python scripts/training/sentiment_training.py
```

### 3. 批量实验

自动进行多模型、多超参数的对比实验：

```bash
python scripts/experiments/classification_experiments.py
```

### 4. 数据增强

使用多种策略进行数据增强，提升模型泛化能力：

```bash
python scripts/experiments/data_augmentation.py
```

### 5. 综合评估

对模型进行全方位评估，生成详细报告：

```bash
python scripts/evaluation/comprehensive_evaluation.py
```

## 📊 实验结果

### 模型性能对比

| 模型 | 学习率 | 准确率 | F1分数 | 训练时间 |
|------|--------|--------|--------|----------|
| **chinese-macbert-base** | **2e-05** | **96.15%** | **96.15%** | **252.7s** |
| chinese-roberta-wwm-ext | 5e-05 | 95.67% | 95.68% | 258.1s |
| bert-base-chinese | 5e-05 | 95.31% | 95.32% | 253.4s |
| chinese-roberta-wwm-ext | 2e-05 | 95.07% | 95.08% | 251.2s |
| bert-base-chinese | 2e-05 | 94.83% | 94.85% | 254.0s |
| chinese-macbert-base | 5e-05 | 94.71% | 94.72% | 253.3s |

**最佳模型**: chinese-macbert-base (LR=2e-05) 达到 **96.15%** 准确率

### 数据增强效果

- 原始数据集准确率: 94.83%
- 增强数据集准确率: 95.31%
- 性能提升: +0.48%

## 🌐 Web界面功能

### 1. 情感分析页面
- 实时文本情感分析
- 置信度可视化
- 概率分布展示
- 快速示例测试

### 2. 模型对比页面
- 6种模型配置对比
- 性能指标表格
- 最佳模型高亮
- 交互式图表展示

### 3. 可视化分析
- 模型性能对比图
- 训练损失曲线
- 验证损失趋势

### 4. 系统信息
- 运行环境配置
- 技术栈说明
- 项目特点介绍

## 🛠️ 技术栈

### 深度学习
- **PyTorch** - 深度学习框架
- **Transformers** - 预训练模型库
- **Accelerate** - 分布式训练加速

### 数据处理
- **Pandas** - 数据处理
- **NumPy** - 数值计算
- **Scikit-learn** - 机器学习工具

### Web开发
- **Flask** - Web框架
- **Chart.js** - 图表可视化
- **Modern CSS** - 响应式设计

### 可视化
- **Matplotlib** - 静态图表
- **Seaborn** - 统计可视化

## 📖 使用示例

### Python API调用

```python
# 单文本预测
from scripts.inference.inference import predict_sentiment

text = "这部电影真的太棒了！"
sentiment = predict_sentiment(text)
print(f"情感: {sentiment}")

# 批量预测
texts = [
    "我今天很开心",
    "感觉很沮丧",
    "天气不错"
]

for text in texts:
    result = predict_sentiment(text)
    print(f"{text} -> {result}")
```

### Web API调用

```python
import requests

# 情感预测API
response = requests.post('http://localhost:5000/api/predict', 
    json={'text': '我今天很开心！'}
)
result = response.json()
print(result)
# {'success': True, 'sentiment': '积极', 'confidence': 0.95, ...}

# 获取实验结果API
response = requests.get('http://localhost:5000/api/experiments')
experiments = response.json()
print(experiments)
```

## 📈 模型训练流程

1. **数据准备** - 加载并预处理中文情感数据集
2. **数据增强** - 使用同义词替换、回译等方法增强数据
3. **模型选择** - 选择预训练的中文BERT模型
4. **微调训练** - 在情感分类任务上进行微调
5. **超参数优化** - 网格搜索最佳学习率和其他参数
6. **模型评估** - 在测试集上评估性能
7. **模型部署** - 保存最佳模型用于推理

## 🔍 数据集

- **来源**: 中文情感标注数据
- **规模**: 4,159条样本
- **类别**: 8种情感类型（映射为积极/消极二分类）
- **平均长度**: 20.3字符
- **数据划分**: 80% 训练集, 20% 验证集

## 🎯 未来计划

- [ ] 支持更多情感分类类别（多分类）
- [ ] 添加情感强度分析
- [ ] 集成更多预训练模型（ERNIE, RoBERTa-large等）
- [ ] 支持批量文件上传分析
- [ ] 添加API认证和限流
- [ ] Docker容器化部署
- [ ] 移动端适配

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 预训练模型库
- [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) - 中文预训练模型
- [Flask](https://flask.palletsprojects.com/) - Web框架
- [Chart.js](https://www.chartjs.org/) - 图表库

## 📧 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 GitHub Issue
- Email: your.email@example.com

---

<div align="center">
  <p>如果这个项目对您有帮助，请给个 ⭐ Star 支持一下！</p>
  <p>© 2025 BERT中文情感分析系统. All rights reserved.</p>
</div>

