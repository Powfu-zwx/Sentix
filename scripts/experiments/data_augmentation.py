# -*- coding: utf-8 -*-
"""
数据增强与性能对比实验
实现多种数据增强策略，并对比模型性能
"""

import pandas as pd
import numpy as np
import random
import re
from typing import List, Tuple
import json
from datetime import datetime

# 设置随机种子
random.seed(42)
np.random.seed(42)

# ========================
# 1. 数据增强策略
# ========================

class DataAugmentation:
    """数据增强类，实现多种增强策略"""
    
    def __init__(self):
        # 同义词词典（繁体中文）
        self.synonym_dict = {
            # 情感词
            '開心': ['高興', '快樂', '愉快', '喜悅', '歡喜'],
            '高興': ['開心', '快樂', '愉快', '喜悅'],
            '快樂': ['開心', '高興', '愉快', '喜悅'],
            '難過': ['傷心', '悲傷', '沮喪', '難受', '痛苦'],
            '傷心': ['難過', '悲傷', '沮喪', '難受'],
            '悲傷': ['難過', '傷心', '沮喪', '痛苦'],
            '生氣': ['憤怒', '火大', '氣憤', '惱火'],
            '憤怒': ['生氣', '火大', '氣憤', '暴怒'],
            '討厭': ['厭惡', '反感', '不喜歡', '痛恨'],
            '厭惡': ['討厭', '反感', '噁心', '嫌棄'],
            
            # 程度副詞
            '很': ['非常', '特別', '十分', '極其', '超級'],
            '非常': ['很', '特別', '十分', '極其'],
            '特別': ['很', '非常', '十分', '格外'],
            '超級': ['很', '非常', '特別', '超'],
            '真的': ['確實', '的確', '實在', '真是'],
            
            # 常用動詞
            '覺得': ['感覺', '認為', '想', '以為'],
            '感覺': ['覺得', '感到', '覺得'],
            '想': ['覺得', '認為', '以為'],
            '看到': ['見到', '瞧見', '看見'],
            '聽到': ['聽見', '聞', '得知'],
            
            # 常用形容詞
            '好': ['不錯', '棒', '優秀', '出色', '讚'],
            '不錯': ['好', '棒', '可以', '還行'],
            '棒': ['好', '不錯', '優秀', '厲害'],
            '差': ['糟糕', '不好', '爛', '不行'],
            '糟糕': ['差', '不好', '糟', '慘'],
            
            # 其他常用詞
            '今天': ['今日', '今兒', '本日'],
            '昨天': ['昨日', '昨兒'],
            '明天': ['明日', '明兒'],
        }
        
        # 語氣詞
        self.tone_particles = ['啊', '呢', '吧', '哦', '啦', '耶', '唷']
        
        # 填充詞
        self.filler_words = ['真的', '其實', '說實話', '老實說', '坦白說']
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """同義詞替換"""
        words = list(text)
        augmented = text
        
        # 尋找可替換的詞
        replaceable = []
        for word, synonyms in self.synonym_dict.items():
            if word in text:
                replaceable.append((word, synonyms))
        
        # 隨機替換n個詞
        if replaceable:
            num_replacements = min(n, len(replaceable))
            for word, synonyms in random.sample(replaceable, num_replacements):
                if word in augmented:
                    synonym = random.choice(synonyms)
                    augmented = augmented.replace(word, synonym, 1)
        
        return augmented
    
    def random_insertion(self, text: str) -> str:
        """隨機插入語氣詞或填充詞"""
        # 70%機率添加語氣詞到結尾
        if random.random() < 0.7 and not any(p in text[-2:] for p in ['！', '？', '。']):
            text = text.rstrip('!?.。！？') + random.choice(self.tone_particles)
        
        # 30%機率在開頭插入填充詞
        if random.random() < 0.3 and len(text) > 10:
            text = random.choice(self.filler_words) + '，' + text
        
        return text
    
    def random_swap(self, text: str) -> str:
        """隨機交換相鄰詞語（保持語義）"""
        # 簡單實現：只對某些可交換的結構進行處理
        # 例如："很高興" <-> "高興極了"
        if len(text) < 5:
            return text
        
        # 這裡只做簡單的同義詞替換
        return self.synonym_replacement(text, n=1)
    
    def back_translation_simulation(self, text: str) -> str:
        """模擬回譯效果（簡化版，不使用真實翻譯API）"""
        # 結合多種策略模擬回譯效果
        augmented = text
        
        # 同義詞替換
        if random.random() < 0.6:
            augmented = self.synonym_replacement(augmented, n=2)
        
        # 調整語序（簡單版）
        if random.random() < 0.3:
            augmented = self.random_swap(augmented)
        
        return augmented
    
    def augment(self, text: str, method: str = 'mixed') -> str:
        """
        執行數據增強
        
        Args:
            text: 原始文本
            method: 增強方法 ('synonym', 'insertion', 'swap', 'backtrans', 'mixed')
        
        Returns:
            增強後的文本
        """
        if method == 'synonym':
            return self.synonym_replacement(text, n=2)
        elif method == 'insertion':
            return self.random_insertion(text)
        elif method == 'swap':
            return self.random_swap(text)
        elif method == 'backtrans':
            return self.back_translation_simulation(text)
        elif method == 'mixed':
            # 混合策略
            augmented = text
            
            # 50%機率使用同義詞替換
            if random.random() < 0.5:
                augmented = self.synonym_replacement(augmented, n=1)
            
            # 40%機率添加語氣詞
            if random.random() < 0.4:
                augmented = self.random_insertion(augmented)
            
            # 如果沒有任何改變，至少做一次同義詞替換
            if augmented == text:
                augmented = self.synonym_replacement(text, n=1)
            
            return augmented
        else:
            return text

# ========================
# 2. 數據增強主函數
# ========================

def create_augmented_dataset(input_csv: str = 'data/data.csv', 
                            output_csv: str = 'data/data_augmented.csv',
                            augmentation_ratio: float = 1.0) -> pd.DataFrame:
    """
    創建增強數據集
    
    Args:
        input_csv: 輸入數據文件
        output_csv: 輸出增強數據文件
        augmentation_ratio: 增強比例（1.0表示每個樣本生成1個增強版本）
    
    Returns:
        增強後的DataFrame
    """
    print("=" * 60)
    print("數據增強實驗")
    print("=" * 60)
    
    # 讀取數據
    df = pd.read_csv(input_csv)
    print(f"\n原始數據集大小: {len(df)}")
    print(f"標籤分布:\n{df['emotion'].value_counts()}")
    
    # 初始化增強器
    augmentor = DataAugmentation()
    
    # 創建增強數據
    augmented_texts = []
    augmented_labels = []
    original_indices = []
    is_augmented = []
    
    print(f"\n開始生成增強數據（比例: {augmentation_ratio}）...")
    
    for idx, row in df.iterrows():
        # 保留原始樣本
        augmented_texts.append(row['text'])
        augmented_labels.append(row['emotion'])
        original_indices.append(idx)
        is_augmented.append(False)
        
        # 生成增強樣本
        num_augmentations = int(augmentation_ratio)
        for _ in range(num_augmentations):
            aug_text = augmentor.augment(row['text'], method='mixed')
            augmented_texts.append(aug_text)
            augmented_labels.append(row['emotion'])
            original_indices.append(idx)
            is_augmented.append(True)
        
        if (idx + 1) % 500 == 0:
            print(f"  已處理 {idx + 1}/{len(df)} 個樣本")
    
    # 創建增強數據集
    df_augmented = pd.DataFrame({
        'text': augmented_texts,
        'emotion': augmented_labels,
        'original_idx': original_indices,
        'is_augmented': is_augmented
    })
    
    print(f"\n增強後數據集大小: {len(df_augmented)}")
    print(f"  - 原始樣本: {sum(~df_augmented['is_augmented'])}")
    print(f"  - 增強樣本: {sum(df_augmented['is_augmented'])}")
    
    # 展示增強樣例
    print("\n" + "=" * 60)
    print("增強樣例展示（前10組）:")
    print("=" * 60)
    
    for i in range(min(10, len(df))):
        original = df.iloc[i]['text']
        augmented_samples = df_augmented[
            (df_augmented['original_idx'] == i) & 
            (df_augmented['is_augmented'] == True)
        ]['text'].tolist()
        
        if augmented_samples:
            aug_text = augmented_samples[0]
            if original != aug_text:  # 只顯示有變化的
                print(f"\n原始 [{df.iloc[i]['emotion']}]: {original}")
                print(f"增強: {aug_text}")
    
    # 保存增強數據集（不包含輔助列）
    df_augmented[['text', 'emotion']].to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✓ 增強數據集已保存至: {output_csv}")
    
    return df_augmented

# ========================
# 3. 性能對比記錄
# ========================

def create_comparison_report(results: dict, output_file: str = 'results/augmentation_comparison.json'):
    """
    創建性能對比報告
    
    Args:
        results: 實驗結果字典
        output_file: 輸出文件路徑
    """
    report = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original_data': results.get('original', {}),
        'augmented_data': results.get('augmented', {}),
        'improvement': {},
        'summary': ""
    }
    
    # 計算提升
    if 'original' in results and 'augmented' in results:
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            if metric in results['original'] and metric in results['augmented']:
                orig = results['original'][metric]
                aug = results['augmented'][metric]
                improvement = ((aug - orig) / orig) * 100 if orig > 0 else 0
                report['improvement'][metric] = {
                    'original': orig,
                    'augmented': aug,
                    'improvement_pct': improvement
                }
    
    # 生成總結
    if report['improvement']:
        acc_imp = report['improvement'].get('accuracy', {}).get('improvement_pct', 0)
        report['summary'] = f"數據增強使準確率提升了 {acc_imp:.2f}%"
    
    # 保存JSON報告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 打印報告
    print("\n" + "=" * 60)
    print("性能對比報告")
    print("=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n✓ 報告已保存至: {output_file}")

# ========================
# 4. 主函數
# ========================

def main():
    """主函數"""
    print("\n" + "=" * 60)
    print("數據增強與性能對比實驗")
    print("=" * 60)
    
    # 1. 創建增強數據集
    df_augmented = create_augmented_dataset(
        input_csv='data/data.csv',
        output_csv='data/data_augmented.csv',
        augmentation_ratio=1.0
    )
    
    # 2. 創建對比說明文件
    comparison_guide = """
# 模型性能對比實驗指南

## 實驗步驟

### 1. 訓練原始數據模型
```powershell
python scripts/sentiment_training.py --data_file data/data.csv --output_dir models/model_original
```

### 2. 訓練增強數據模型  
```powershell
python scripts/sentiment_training.py --data_file data/data_augmented.csv --output_dir models/model_augmented
```

### 3. 評估並對比性能
```powershell
python scripts/comprehensive_evaluation.py
```

## 評估指標

- **準確率 (Accuracy)**: 整體預測正確的比例
- **精確率 (Precision)**: 預測為正的樣本中真正為正的比例
- **召回率 (Recall)**: 真正為正的樣本中被預測為正的比例
- **F1分數**: 精確率和召回率的調和平均

## 預期結果

數據增強通常可以帶來以下改進：
- 提升模型泛化能力
- 減少過擬合
- 提高在不同表達方式下的魯棒性

## 注意事項

- 確保使用相同的訓練參數進行對比
- 使用相同的測試集評估兩個模型
- 記錄訓練時間和資源消耗
"""
    
    with open('docs/AUGMENTATION_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(comparison_guide)
    
    print("\n✓ 實驗指南已保存至: docs/AUGMENTATION_GUIDE.md")
    
    # 3. 創建性能記錄模板
    results_template = {
        'original': {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'training_time': '待測試',
            'sample_size': len(df_augmented[df_augmented['is_augmented'] == False])
        },
        'augmented': {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'training_time': '待測試',
            'sample_size': len(df_augmented)
        }
    }
    
    create_comparison_report(results_template, 'results/augmentation_comparison.json')
    
    print("\n" + "=" * 60)
    print("數據增強完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  1. data/data_augmented.csv - 增強後的數據集")
    print("  2. docs/AUGMENTATION_GUIDE.md - 實驗指南")
    print("  3. results/augmentation_comparison.json - 性能對比模板")
    print("\n下一步：")
    print("  1. 查看 docs/AUGMENTATION_GUIDE.md 了解如何訓練和對比模型")
    print("  2. 運行訓練腳本分別訓練兩個模型")
    print("  3. 更新 results/augmentation_comparison.json 記錄實驗結果")

if __name__ == "__main__":
    main()

