#!/bin/bash
# 数据增强完整实验流程
# 替代原来的 run_augmentation_experiment.py

echo "======================================"
echo "  数据增强完整实验流程"
echo "======================================"

# 步骤1: 生成增强数据
echo ""
echo "步骤 1/4: 生成增强数据..."
python scripts/data_augmentation.py
if [ $? -ne 0 ]; then
    echo "❌ 数据增强失败"
    exit 1
fi

# 步骤2: 训练原始数据模型
echo ""
echo "步骤 2/4: 训练原始数据模型..."
python scripts/sentiment_training.py \
    --data_file data/data.csv \
    --output_dir models/model_original
if [ $? -ne 0 ]; then
    echo "❌ 原始模型训练失败"
    exit 1
fi

# 步骤3: 训练增强数据模型
echo ""
echo "步骤 3/4: 训练增强数据模型..."
python scripts/sentiment_training.py \
    --data_file data/data_augmented.csv \
    --output_dir models/model_augmented
if [ $? -ne 0 ]; then
    echo "❌ 增强模型训练失败"
    exit 1
fi

# 步骤4: 对比性能
echo ""
echo "步骤 4/4: 对比模型性能..."
python scripts/comprehensive_evaluation.py
if [ $? -ne 0 ]; then
    echo "❌ 性能对比失败"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ 实验完成！"
echo "======================================"
echo ""
echo "查看结果:"
echo "  📊 results/augmentation/comparison.md"
echo "  📊 results/augmentation/comparison.json"
echo ""

