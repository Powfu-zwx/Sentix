# 模型目录

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
