# Sentix - 中文情感分析实验平台

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.20+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Sentix** = **Sent**iment + Analyt**ix** | 专业的中文情感分析与科研实验平台

基于BERT的中文情感分析系统，集成完整的科研实验框架，支持模型对比、数据增强、智能生成。

## 核心功能

- **情感分析**: BERT/RoBERTa/MacBERT模型，准确率96%+
- **智能回复**: GPT-2条件生成，支持情感感知
- **情绪可视化**: 动态颜色球体、能量条、光环效果直观展示情感
- **数据增强**: 同义词替换、语气词插入等策略
- **实验框架**: 模型对比、超参数调优、性能评估
- **Web界面**: Gradio交互式演示，炫酷UI设计

## 快速开始

### 安装

```powershell
# 克隆项目
git clone https://github.com/your-username/sentix.git
cd sentix

# 创建环境
conda create -n sentix python=3.8
conda activate sentix

# 安装依赖
pip install -r requirements.txt
```

### 基础使用

```powershell
# 训练模型
python scripts/sentiment_training.py

# 启动情绪可视化Web界面（推荐）
run_emotion_visualizer.bat

# 或直接启动
python scripts/gradio_demo.py

# 命令行推理
python scripts/inference.py
```

### Python API

```python
from scripts.inference import predict_sentiment

result = predict_sentiment("今天心情很好")
print(result)  # "positive"
```

## 项目结构

```
sentix/
├── data/                        # 数据集 (4,159样本)
├── scripts/                     # 8个核心脚本
│   ├── sentiment_training.py            # 模型训练
│   ├── classification_experiments.py    # 分类实验
│   ├── comprehensive_evaluation.py      # 综合评估
│   ├── data_augmentation.py             # 数据增强
│   ├── generation_experiments.py        # 生成实验
│   ├── conditional_generation.py        # 条件生成
│   ├── gradio_demo.py                   # Web界面
│   └── inference.py                     # 推理接口
├── models/                      # 训练好的模型
├── results/                     # 实验结果
└── README.md
```

## 实验流程

### 1. 模型对比实验

对比3个模型 × 2个学习率 = 6种配置：

```powershell
# 运行完整实验（约2小时）
python scripts/classification_experiments.py

# 只重新生成图表
python scripts/classification_experiments.py --regenerate-only
```

**输出**: Loss曲线、性能对比、混淆矩阵、实验报告

### 2. 数据增强实验

```powershell
# 一键运行完整实验
run_augmentation_experiment.bat
```

或分步执行：

```powershell
# 1. 生成增强数据
python scripts/data_augmentation.py

# 2. 训练原始模型
python scripts/sentiment_training.py --data_file data/data.csv --output_dir models/model_original

# 3. 训练增强模型
python scripts/sentiment_training.py --data_file data/data_augmented.csv --output_dir models/model_augmented

# 4. 对比性能
python scripts/comprehensive_evaluation.py
```

### 3. 生成模型调优

```powershell
# 测试temperature、top_p、repetition_penalty参数
python scripts/generation_experiments.py
```

### 4. 综合评估

```powershell
python scripts/comprehensive_evaluation.py
```

生成性能对比、错误分析、改进建议等报告。

## 训练参数

### sentiment_training.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_file` | 训练数据路径 | `data/data.csv` |
| `--output_dir` | 模型保存目录 | `models/sentiment_model` |
| `--model_name` | 预训练模型 | `bert-base-chinese` |
| `--epochs` | 训练轮数 | 3 |
| `--batch_size` | 批次大小 | 16 |
| `--learning_rate` | 学习率 | 2e-5 |

示例：

```powershell
python scripts/sentiment_training.py --model_name hfl/chinese-macbert-base --epochs 5 --learning_rate 3e-5
```

### classification_experiments.py

| 参数 | 说明 |
|------|------|
| `--regenerate-only` | 只重新生成图表，不重新训练 |
| `--experiment-id` | 指定实验ID（配合--regenerate-only） |

## 性能指标

### 模型对比（验证集）

| 模型 | 学习率 | 准确率 | F1 |
|------|--------|--------|-----|
| MacBERT | 2e-5 | 96.15% | 96.03% |
| MacBERT | 5e-5 | 95.78% | 95.62% |
| RoBERTa | 5e-5 | 95.42% | 95.28% |
| RoBERTa | 2e-5 | 95.18% | 95.04% |
| BERT | 5e-5 | 94.82% | 94.65% |
| BERT | 2e-5 | 94.51% | 94.33% |

**结论**: MacBERT (lr=2e-5) 性能最优

### 系统性能

| 指标 | GPU | CPU |
|------|-----|-----|
| 推理速度 | <200ms | <800ms |
| 内存占用 | ~2.5GB | ~2.5GB |
| 模型大小 | ~400MB | ~400MB |

## 项目精简说明

本项目已完成结构优化，从13个脚本精简至8个核心脚本：

**整合内容**:
- `error_analysis.py` + `evaluate_models.py` → `comprehensive_evaluation.py`
- `regenerate_plots.py` → `classification_experiments.py` (新增`--regenerate-only`参数)
- `run_augmentation_experiment.py` → Windows批处理脚本 (`.bat`文件)

**删除冗余**:
- 4个重复功能脚本
- docs目录（内容整合到README）
- ARCHITECTURE.md（内容整合到README）

**精简效果**:
- 脚本数量: 13个 → 8个 (-38%)
- 文档文件: 5个 → 1个 (-80%)
- 维护复杂度: 显著降低

## 常见问题

### Q: 内存不足

```powershell
# 减小批次大小
python scripts/sentiment_training.py --batch_size 8
```

### Q: 模型下载慢

```powershell
# 使用国内镜像（设置环境变量）
$env:HF_ENDPOINT="https://hf-mirror.com"
```

### Q: 中文字体显示问题

脚本已配置`SimHei`、`Microsoft YaHei`等字体，通常无需额外设置。

### Q: GPU不可用

自动回退到CPU训练，速度较慢但可正常运行。

## 部署

### Docker

```powershell
# 构建镜像
docker build -t sentix:latest .

# 运行容器
docker run -p 7860:7860 sentix:latest
```

### 生产环境

```powershell
# 使用Waitress（Windows推荐）
pip install waitress
waitress-serve --port=7860 scripts.gradio_demo:app
```

## 依赖包

主要依赖：

- `torch>=2.0.0`
- `transformers>=4.20.0`
- `gradio>=5.0.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`

完整列表见 `requirements.txt`

## 开发

### 贡献代码

1. Fork项目
2. 创建特性分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -m 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 提交Pull Request

### 代码规范

- 遵循PEP8
- 添加类型提示
- 编写docstring
- 保持函数单一职责

## 参考文献

1. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", NAACL 2019
2. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", arXiv 2019
3. Cui et al., "Revisiting Pre-Trained Models for Chinese NLP", EMNLP 2020

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 致谢

- [Hugging Face](https://huggingface.co/) - 预训练模型
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Gradio](https://gradio.app/) - Web界面框架
- [HFL](https://github.com/ymcui/Chinese-BERT-wwm) - 中文预训练模型

---

**项目**: Sentix - Sentiment Analysis Experiment Platform  
**版本**: 1.1.0  
**更新**: 2025-10-02  
**协议**: MIT License
