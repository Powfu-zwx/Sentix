# -*- coding: utf-8 -*-
"""
条件生成实验
对比带情感标签的提示与普通提示的生成差异
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalGenerationExperiment:
    """条件生成实验类"""
    
    def __init__(self, model_name="uer/gpt2-chinese-cluecorpussmall"):
        print(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip()
    
    def create_conditional_prompt(self, user_text: str, emotion: str) -> str:
        """创建带情感标签的提示"""
        emotion_label = {
            'positive': '正面',
            'negative': '負面',
            'neutral': '中性'
        }.get(emotion, emotion)
        
        prompt = f"""情感：{emotion_label}
用户说：{user_text}
AI回复："""
        return prompt
    
    def create_normal_prompt(self, user_text: str) -> str:
        """创建普通提示"""
        return f"用户说：{user_text}\nAI回复："
    
    def run_comparison_experiment(self, test_cases: List[Dict]):
        """
        运行对比实验
        
        Args:
            test_cases: 测试用例列表 [{'text': str, 'emotion': str}, ...]
        """
        print("\n" + "=" * 60)
        print("🔬 条件生成对比实验")
        print("=" * 60)
        print(f"\n测试样本数: {len(test_cases)}")
        
        results = []
        
        for idx, case in enumerate(test_cases):
            user_text = case['text']
            emotion = case['emotion']
            
            print(f"\n{'='*60}")
            print(f"样本 {idx+1}/{len(test_cases)}")
            print(f"{'='*60}")
            print(f"用户输入: {user_text}")
            print(f"情感标签: {emotion}")
            
            # 普通提示生成
            normal_prompt = self.create_normal_prompt(user_text)
            normal_generation = self.generate(normal_prompt)
            
            print(f"\n【普通提示】")
            print(f"提示: {normal_prompt}")
            print(f"生成: {normal_generation}")
            
            # 条件提示生成
            conditional_prompt = self.create_conditional_prompt(user_text, emotion)
            conditional_generation = self.generate(conditional_prompt)
            
            print(f"\n【条件提示（带标签）】")
            print(f"提示: {conditional_prompt}")
            print(f"生成: {conditional_generation}")
            
            # 记录结果
            result = {
                'user_text': user_text,
                'emotion': emotion,
                'normal_prompt': normal_prompt,
                'normal_generation': normal_generation,
                'conditional_prompt': conditional_prompt,
                'conditional_generation': conditional_generation,
                'length_diff': len(conditional_generation) - len(normal_generation)
            }
            
            results.append(result)
        
        # 保存和分析结果
        self.save_and_analyze(results)
        
        return results
    
    def save_and_analyze(self, results: List[Dict]):
        """保存并分析结果"""
        experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"results/conditional_generation/{experiment_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON
        with open(f"{output_dir}/results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成对比报告
        md_content = f"""# 条件生成对比实验报告

**实验日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**实验样本数**: {len(results)}

## 🎯 实验目的

对比两种提示方式对生成质量的影响：
1. **普通提示**: 直接输入用户文本
2. **条件提示**: 添加情感标签作为条件

## 📊 实验结果

"""
        
        for idx, result in enumerate(results):
            md_content += f"""### 样本 {idx+1}

**用户输入**: {result['user_text']}  
**情感标签**: {result['emotion']}

#### 普通提示生成
```
{result['normal_prompt']}
{result['normal_generation']}
```

#### 条件提示生成（带标签）
```
{result['conditional_prompt']}
{result['conditional_generation']}
```

**观察**:
- 长度差异: {result['length_diff']} 字符
- 条件生成: {'更长' if result['length_diff'] > 0 else '更短' if result['length_diff'] < 0 else '相同长度'}

---

"""
        
        # 统计分析
        avg_length_diff = sum([r['length_diff'] for r in results]) / len(results)
        longer_count = sum([1 for r in results if r['length_diff'] > 0])
        
        md_content += f"""
## 📈 统计分析

### 长度对比
- 平均长度差异: {avg_length_diff:.1f} 字符
- 条件生成更长的样本数: {longer_count}/{len(results)}
- 条件生成更短的样本数: {len(results)-longer_count}/{len(results)}

### 主要发现

1. **情感一致性**: 
   - 带情感标签的提示是否使生成更符合预期情感？
   - 需要人工评估或使用情感分类器自动评估

2. **生成质量**:
   - 条件提示是否产生更自然、更贴切的回复？
   - 观察是否出现更多重复或不自然的表达

3. **提示工程影响**:
   - 提示的格式和措辞对生成结果有显著影响
   - 添加结构化信息（如情感标签）可能引导模型生成

## 💡 结论

### 条件生成的优势
- ✅ 提供明确的情感导向
- ✅ 可以更好地控制生成内容
- ✅ 适合需要情感一致性的场景

### 条件生成的局限
- ⚠️ 依赖于标签的准确性
- ⚠️ 可能使生成变得机械化
- ⚠️ 需要额外的情感分类步骤

### 推荐使用场景

**使用条件生成**:
- 情感对话系统
- 需要精确控制回复情感的场景
- 多轮对话中保持情感一致性

**使用普通生成**:
- 开放式对话
- 需要更自然、更灵活的回复
- 不强调情感导向的场景

## 🔬 实验改进建议

1. **量化评估**: 使用情感分类器自动评估生成文本的情感倾向
2. **人工评分**: 招募评估者对生成质量打分
3. **更多样本**: 增加测试样本数量提高统计可靠性
4. **A/B测试**: 在实际应用中对比用户偏好
5. **提示优化**: 尝试不同的提示格式找到最优方案

---

*本报告由自动化实验系统生成*
"""
        
        with open(f"{output_dir}/comparison_report.md", 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n✓ 实验结果已保存: {output_dir}/")
        print(f"✓ 对比报告: {output_dir}/comparison_report.md")

def prepare_test_cases():
    """准备测试用例"""
    return [
        {'text': '我今天很开心，考试考得很好！', 'emotion': 'positive'},
        {'text': '我感觉很沮丧，什么都做不好。', 'emotion': 'negative'},
        {'text': '今天天气不错。', 'emotion': 'neutral'},
        {'text': '我对这个结果很满意！', 'emotion': 'positive'},
        {'text': '工作压力太大了，我快撑不住了。', 'emotion': 'negative'},
        {'text': '周末打算做什么？', 'emotion': 'neutral'},
        {'text': '我终于完成了这个项目！', 'emotion': 'positive'},
        {'text': '我今天心情很不好。', 'emotion': 'negative'},
    ]

def main():
    """主函数"""
    print("=" * 60)
    print("🔬 条件生成对比实验")
    print("=" * 60)
    
    experiment = ConditionalGenerationExperiment()
    test_cases = prepare_test_cases()
    
    print(f"\n将测试 {len(test_cases)} 个样本")
    print("预计用时: 约 5-10 分钟")
    
    response = input("\n是否继续? (y/n): ")
    if response.lower() != 'y':
        print("已取消。")
        return
    
    results = experiment.run_comparison_experiment(test_cases)
    
    print("\n" + "=" * 60)
    print("🎉 实验完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()

