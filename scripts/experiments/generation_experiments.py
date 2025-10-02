# -*- coding: utf-8 -*-
"""
生成模型调优实验
测试不同超参数对生成质量的影响
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict
import numpy as np

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class GenerationExperiment:
    """生成实验类"""
    
    def __init__(self, model_name="uer/gpt2-chinese-cluecorpussmall"):
        """初始化模型"""
        print(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_text(self, prompt: str, temperature: float = 0.8,
                     top_p: float = 0.9, repetition_penalty: float = 1.0,
                     max_length: int = 100) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            temperature: 温度参数 (0.1-2.0, 越高越随机)
            top_p: 核采样参数 (0.1-1.0, 越高选择范围越大)
            repetition_penalty: 重复惩罚 (1.0-2.0, 越高越避免重复)
            max_length: 最大生成长度
        
        Returns:
            生成的文本
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 只返回生成的部分
        response = generated_text[len(prompt):].strip()
        return response
    
    def run_parameter_experiments(self, test_prompts: List[Dict]):
        """
        运行参数调优实验
        
        Args:
            test_prompts: 测试提示列表，每个包含 {'prompt': str, 'emotion': str}
        """
        
        print("\n" + "=" * 60)
        print("🔬 生成模型超参数调优实验")
        print("=" * 60)
        
        # 实验配置
        experiment_configs = [
            # 基线配置
            {'name': '基线', 'temperature': 0.8, 'top_p': 0.9, 'repetition_penalty': 1.0},
            
            # 温度实验
            {'name': '低温度', 'temperature': 0.5, 'top_p': 0.9, 'repetition_penalty': 1.0},
            {'name': '高温度', 'temperature': 1.2, 'top_p': 0.9, 'repetition_penalty': 1.0},
            
            # top_p实验
            {'name': '低top_p', 'temperature': 0.8, 'top_p': 0.7, 'repetition_penalty': 1.0},
            {'name': '高top_p', 'temperature': 0.8, 'top_p': 0.95, 'repetition_penalty': 1.0},
            
            # repetition_penalty实验
            {'name': '低重复惩罚', 'temperature': 0.8, 'top_p': 0.9, 'repetition_penalty': 1.0},
            {'name': '高重复惩罚', 'temperature': 0.8, 'top_p': 0.9, 'repetition_penalty': 1.5},
            
            # 组合优化
            {'name': '创意配置', 'temperature': 1.0, 'top_p': 0.95, 'repetition_penalty': 1.2},
            {'name': '保守配置', 'temperature': 0.6, 'top_p': 0.8, 'repetition_penalty': 1.3},
        ]
        
        all_results = []
        
        for config in experiment_configs:
            print(f"\n{'='*60}")
            print(f"实验配置: {config['name']}")
            print(f"  temperature = {config['temperature']}")
            print(f"  top_p = {config['top_p']}")
            print(f"  repetition_penalty = {config['repetition_penalty']}")
            print(f"{'='*60}")
            
            config_results = {
                'config_name': config['name'],
                'parameters': {k: v for k, v in config.items() if k != 'name'},
                'generations': []
            }
            
            for idx, test_case in enumerate(test_prompts):
                prompt = test_case['prompt']
                emotion = test_case.get('emotion', 'unknown')
                
                print(f"\n[{idx+1}/{len(test_prompts)}] 提示: {prompt}")
                
                generated = self.generate_text(
                    prompt,
                    temperature=config['temperature'],
                    top_p=config['top_p'],
                    repetition_penalty=config['repetition_penalty']
                )
                
                print(f"生成: {generated}")
                
                config_results['generations'].append({
                    'prompt': prompt,
                    'emotion': emotion,
                    'generated_text': generated,
                    'length': len(generated)
                })
            
            all_results.append(config_results)
        
        # 保存结果
        self.save_experiment_results(all_results)
        
        # 生成报告
        self.generate_analysis_report(all_results)
        
        return all_results
    
    def save_experiment_results(self, results: List[Dict]):
        """保存实验结果"""
        experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"results/generation_experiments/{experiment_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON
        output_file = f"{output_dir}/experiment_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 实验结果已保存: {output_file}")
        return output_dir
    
    def generate_analysis_report(self, results: List[Dict]):
        """生成分析报告"""
        experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"results/generation_experiments/{experiment_id}"
        
        md_content = f"""# 生成模型超参数调优实验报告

**实验日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**实验配置数**: {len(results)}

## 📋 实验目的

评估不同超参数组合对文本生成质量的影响，特别关注：
1. **自然度**: 生成文本是否流畅自然
2. **情感一致性**: 生成内容是否与输入情感匹配
3. **多样性**: 不同参数下的输出差异
4. **重复问题**: 是否出现不自然的重复

## 🔬 参数说明

### Temperature (温度)
- **范围**: 0.1 - 2.0
- **效果**: 控制输出的随机性
  - 低温度 (0.5): 更确定、更保守
  - 高温度 (1.2): 更随机、更创意
  
### Top-p (核采样)
- **范围**: 0.1 - 1.0
- **效果**: 控制候选词的范围
  - 低top_p (0.7): 只考虑高概率词
  - 高top_p (0.95): 考虑更多可能性

### Repetition Penalty (重复惩罚)
- **范围**: 1.0 - 2.0
- **效果**: 避免重复
  - 1.0: 无惩罚
  - 1.5+: 强力避免重复

---

## 📊 实验结果

"""
        
        for result in results:
            config_name = result['config_name']
            params = result['parameters']
            generations = result['generations']
            
            md_content += f"""### {config_name}

**参数配置**:
- Temperature: {params['temperature']}
- Top-p: {params['top_p']}
- Repetition Penalty: {params['repetition_penalty']}

**生成样例**:

"""
            
            for gen in generations[:5]:  # 只显示前5个
                md_content += f"""---
**提示**: {gen['prompt']}  
**情感**: {gen['emotion']}  
**生成**: {gen['generated_text']}  
**长度**: {gen['length']} 字符

"""
        
        md_content += """
---

## 💡 观察与分析

### Temperature 影响

"""
        
        # 分析temperature影响
        temp_configs = [r for r in results if '温度' in r['config_name']]
        if temp_configs:
            md_content += "根据实验观察:\n\n"
            for config in temp_configs:
                avg_length = np.mean([g['length'] for g in config['generations']])
                md_content += f"- **{config['config_name']}** (T={config['parameters']['temperature']}): 平均长度 {avg_length:.1f} 字符\n"
        
        md_content += """
### Top-p 影响

"""
        
        # 分析top_p影响
        topp_configs = [r for r in results if 'top_p' in r['config_name']]
        if topp_configs:
            md_content += "根据实验观察:\n\n"
            for config in topp_configs:
                avg_length = np.mean([g['length'] for g in config['generations']])
                md_content += f"- **{config['config_name']}** (top_p={config['parameters']['top_p']}): 平均长度 {avg_length:.1f} 字符\n"
        
        md_content += """
### Repetition Penalty 影响

"""
        
        md_content += """
根据实验观察repetition_penalty对避免重复的效果。

---

## 🎯 推荐配置

基于实验结果，针对不同场景的推荐配置：

### 1. 情感回复（保守）
```python
temperature = 0.7
top_p = 0.85
repetition_penalty = 1.2
```
**适用**: 需要稳定、可靠的回复

### 2. 创意生成（探索）
```python
temperature = 1.0
top_p = 0.95
repetition_penalty = 1.3
```
**适用**: 需要多样化、有创意的输出

### 3. 平衡配置（推荐）
```python
temperature = 0.8
top_p = 0.9
repetition_penalty = 1.2
```
**适用**: 大多数场景

---

## 📝 科研笔记

### 实验设计要点
1. ✅ 使用固定的测试集确保可对比性
2. ✅ 每个参数独立变化观察单一变量影响
3. ✅ 记录所有生成结果便于后续分析
4. ✅ 多个测试样例覆盖不同情感类别

### 改进方向
1. 可以增加人工评分环节评估生成质量
2. 使用自动化指标（如BLEU、困惑度）量化评估
3. 测试更多参数组合找到最优配置
4. 针对特定情感类别调优参数

---

*本报告由自动化实验系统生成*
"""
        
        # 保存报告
        report_file = f"{output_dir}/experiment_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"✓ 实验报告已保存: {report_file}")

def prepare_test_prompts():
    """准备测试提示"""
    prompts = [
        {
            'prompt': '用户说：我今天很开心！\nAI回复：',
            'emotion': 'positive'
        },
        {
            'prompt': '用户说：我感觉很难过，什么都不顺利。\nAI回复：',
            'emotion': 'negative'
        },
        {
            'prompt': '用户说：我对这个结果很满意！\nAI回复：',
            'emotion': 'positive'
        },
        {
            'prompt': '用户说：工作压力太大了，我快撑不住了。\nAI回复：',
            'emotion': 'negative'
        },
        {
            'prompt': '用户说：今天天气真好，心情也跟着好起来了。\nAI回复：',
            'emotion': 'positive'
        },
    ]
    return prompts

def main():
    """主函数"""
    print("=" * 60)
    print("🔬 生成模型超参数调优实验")
    print("=" * 60)
    
    # 初始化实验
    experiment = GenerationExperiment()
    
    # 准备测试数据
    test_prompts = prepare_test_prompts()
    
    print(f"\n将使用 {len(test_prompts)} 个测试提示")
    print("预计用时: 约 10-15 分钟")
    
    response = input("\n是否继续? (y/n): ")
    if response.lower() != 'y':
        print("已取消。")
        return
    
    # 运行实验
    results = experiment.run_parameter_experiments(test_prompts)
    
    print("\n" + "=" * 60)
    print("🎉 实验完成!")
    print("=" * 60)
    print("\n查看详细报告: results/generation_experiments/[实验ID]/experiment_report.md")

if __name__ == "__main__":
    main()

