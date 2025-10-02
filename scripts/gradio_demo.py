#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感AI助手 - Gradio网页界面
集成情感分析和文本生成功能
"""

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

class EmotionAI:
    def __init__(self):
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.generation_model = None
        self.generation_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.load_models()
    
    def load_models(self):
        """加载情感分析和文本生成模型"""
        print("正在加载模型...")
        
        try:
            # 加载情感分析模型
            sentiment_model_path = './sentiment_model'
            print("加载情感分析模型...")
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
            self.sentiment_model = self.sentiment_model.to(self.device)
            self.sentiment_model.eval()
            
            # 加载文本生成模型
            generation_model_name = "uer/gpt2-chinese-cluecorpussmall"
            print("加载文本生成模型...")
            self.generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
            self.generation_model = AutoModelForCausalLM.from_pretrained(generation_model_name)
            self.generation_model = self.generation_model.to(self.device)
            self.generation_model.eval()
            
            print(f"模型加载完成！使用设备: {self.device}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
        
        return True
    
    def predict_sentiment(self, text):
        """
        预测文本情感
        
        Args:
            text (str): 输入文本
        
        Returns:
            tuple: (sentiment, confidence) - 情感标签和置信度
        """
        try:
            # 编码文本
            inputs = self.sentiment_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # 将输入移动到模型所在的设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(prediction, dim=-1).item()
                confidence = prediction[0][predicted_class].item()
            
            # 返回结果
            sentiment = "正面" if predicted_class == 1 else "负面"
            return sentiment, confidence
            
        except Exception as e:
            print(f"情感分析出错: {e}")
            return "负面", 0.5
    
    def generate_response(self, sentiment, original_text):
        """
        根据情感生成回复
        
        Args:
            sentiment (str): 情感标签
            original_text (str): 原始文本
        
        Returns:
            str: 生成的回复
        """
        try:
            # 根据情感选择不同的提示词模板
            if sentiment == "正面":
                prompt = f"用户说：{original_text}\nAI鼓励回复："
            else:
                prompt = f"用户说：{original_text}\nAI安慰回复："
            
            # 编码输入
            inputs = self.generation_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.generation_model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 50,  # 限制生成长度
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.generation_tokenizer.eos_token_id,
                    eos_token_id=self.generation_tokenizer.eos_token_id,
                    repetition_penalty=1.2  # 减少重复
                )
            
            # 解码输出
            generated_text = self.generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 处理中文文本格式，去除不必要的空格
            generated_text = generated_text.replace(" ", "")
            
            # 提取AI回复部分
            if "AI鼓励回复：" in generated_text:
                reply = generated_text.split("AI鼓励回复：")[1].strip()
            elif "AI安慰回复：" in generated_text:
                reply = generated_text.split("AI安慰回复：")[1].strip()
            else:
                # 如果没有找到标记，使用备用回复
                if sentiment == "正面":
                    reply = "太好了！你的想法很积极，继续保持这种心态！"
                else:
                    reply = "我理解你现在的感受。每个人都会有低落的时候，这很正常。记住，困难总是暂时的，你一定能度过这个难关。"
            
            return reply[:100]  # 限制回复长度
            
        except Exception as e:
            print(f"文本生成出错: {e}")
            # 备用回复
            if sentiment == "正面":
                return "你的想法很棒！继续保持积极的心态！"
            else:
                return "我能理解你的感受。记住，一切都会好起来的。"

# 创建AI助手实例
print("初始化情感AI助手...")
ai_assistant = EmotionAI()

def analyze_and_reply(text):
    """
    分析文本情感并生成相应回复
    
    Args:
        text (str): 用户输入的文本
    
    Returns:
        str: 包含情感分析结果和AI回复的文本
    """
    if not text or not text.strip():
        return "请输入一些文本！"
    
    try:
        # 1. 情感判断
        sentiment, confidence = ai_assistant.predict_sentiment(text)
        
        # 2. 生成回复（正面时生成鼓励，负面时生成安慰）
        ai_reply = ai_assistant.generate_response(sentiment, text)
        
        # 格式化输出
        result = f"🔍 情感判断：{sentiment} (置信度: {confidence:.2f})\n\n🤖 AI回复：{ai_reply}"
        
        return result
        
    except Exception as e:
        return f"处理出错: {e}"

# 创建更多示例
examples = [
    ["我今天考试考得很好，太开心了！"],
    ["今天心情很糟糕，什么都不顺心"],
    ["刚刚完成了一个很棒的项目，感觉很有成就感"],
    ["我觉得很沮丧，感觉自己什么都做不好"],
    ["天气真好，心情也变得愉快起来"],
    ["工作压力太大了，我快受不了了"]
]

# 创建Gradio界面
def create_interface():
    """创建Gradio网页界面"""
    
    interface = gr.Interface(
        fn=analyze_and_reply,
        inputs=gr.Textbox(
            label="输入你的文本", 
            placeholder="请输入你想表达的内容...",
            lines=3
        ),
        outputs=gr.Textbox(
            label="分析结果", 
            lines=6
        ),
        title="🤖 情感AI助手",
        description="""
        这个AI助手可以：
        1. 📊 分析你的文本情感（正面/负面）
        2. 💬 根据情感提供相应的回复（鼓励或安慰）
        
        试着输入一些文本，看看AI如何回应！
        """,
        examples=examples,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .gr-button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }
        """,
        allow_flagging="never"
    )
    
    return interface

if __name__ == "__main__":
    print("启动Gradio网页界面...")
    
    # 创建并启动界面
    demo = create_interface()
    
    # 启动服务
    demo.launch(
        share=True,  # 创建公共链接
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,  # 指定端口
        show_error=True  # 显示错误信息
    )
