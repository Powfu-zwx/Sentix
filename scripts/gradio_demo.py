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
            sentiment_model_path = 'models/sentiment_model'
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

def get_emotion_color(sentiment, confidence):
    """
    根据情感和置信度返回颜色
    
    Args:
        sentiment (str): 情感标签
        confidence (float): 置信度
    
    Returns:
        tuple: (主颜色, 渐变颜色, 光环颜色, 情绪描述)
    """
    # 根据置信度调整颜色饱和度
    intensity = int(confidence * 100)
    
    if sentiment == "正面":
        if confidence > 0.8:
            return f"rgba(255, 215, 0, {confidence})", f"rgba(255, 165, 0, {confidence*0.6})", "rgba(255, 215, 0, 0.3)", "😊 非常开心"
        elif confidence > 0.6:
            return f"rgba(135, 206, 250, {confidence})", f"rgba(100, 149, 237, {confidence*0.6})", "rgba(135, 206, 250, 0.3)", "😌 愉悦平静"
        else:
            return f"rgba(144, 238, 144, {confidence})", f"rgba(60, 179, 113, {confidence*0.6})", "rgba(144, 238, 144, 0.3)", "🙂 轻松"
    else:
        if confidence > 0.8:
            return f"rgba(220, 20, 60, {confidence})", f"rgba(178, 34, 34, {confidence*0.6})", "rgba(220, 20, 60, 0.3)", "😤 愤怒/沮丧"
        elif confidence > 0.6:
            return f"rgba(138, 43, 226, {confidence})", f"rgba(75, 0, 130, {confidence*0.6})", "rgba(138, 43, 226, 0.3)", "😔 失落"
        else:
            return f"rgba(169, 169, 169, {confidence})", f"rgba(128, 128, 128, {confidence*0.6})", "rgba(169, 169, 169, 0.3)", "😐 平淡"

def analyze_and_reply(text):
    """
    分析文本情感并生成相应回复，返回HTML格式的可视化结果
    
    Args:
        text (str): 用户输入的文本
    
    Returns:
        str: 包含情感可视化和AI回复的HTML
    """
    if not text or not text.strip():
        return "<div style='text-align: center; padding: 20px; color: #888;'>请输入一些文本！</div>"
    
    try:
        # 1. 情感判断
        sentiment, confidence = ai_assistant.predict_sentiment(text)
        
        # 2. 生成回复
        ai_reply = ai_assistant.generate_response(sentiment, text)
        
        # 3. 获取情绪颜色
        main_color, gradient_color, glow_color, emotion_desc = get_emotion_color(sentiment, confidence)
        
        # 4. 计算能量条百分比
        energy_percent = int(confidence * 100)
        
        # 5. 生成HTML可视化
        html_output = f"""
        <div style="font-family: 'Arial', 'Microsoft YaHei', sans-serif; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
            <!-- 情绪球体区域 -->
            <div style="text-align: center; margin-bottom: 30px;">
                <h2 style="margin-bottom: 20px; font-size: 24px;">💫 情绪波动分析</h2>
                
                <!-- 动态情绪球体 -->
                <div style="position: relative; width: 200px; height: 200px; margin: 0 auto 20px;">
                    <!-- 外层光环 -->
                    <div style="
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        width: 200px;
                        height: 200px;
                        background: radial-gradient(circle, {glow_color} 0%, transparent 70%);
                        border-radius: 50%;
                        animation: pulse 2s ease-in-out infinite;
                    "></div>
                    
                    <!-- 中层球体 -->
                    <div style="
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        width: 140px;
                        height: 140px;
                        background: linear-gradient(135deg, {main_color} 0%, {gradient_color} 100%);
                        border-radius: 50%;
                        box-shadow: 0 0 40px {glow_color}, inset 0 0 30px rgba(255,255,255,0.3);
                        animation: float 3s ease-in-out infinite;
                    "></div>
                    
                    <!-- 内层高光 -->
                    <div style="
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%) translate(-20px, -20px);
                        width: 50px;
                        height: 50px;
                        background: radial-gradient(circle, rgba(255,255,255,0.8) 0%, transparent 70%);
                        border-radius: 50%;
                        animation: float 3s ease-in-out infinite;
                    "></div>
                </div>
                
                <!-- 情绪描述 -->
                <div style="font-size: 28px; font-weight: bold; margin-bottom: 10px; animation: fadeIn 1s;">
                    {emotion_desc}
                </div>
                <div style="font-size: 16px; opacity: 0.9;">
                    情感倾向：<span style="font-weight: bold; font-size: 20px;">{sentiment}</span>
                </div>
            </div>
            
            <!-- 情绪能量条 -->
            <div style="margin: 30px 0; background: rgba(255,255,255,0.2); border-radius: 15px; padding: 20px;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 18px; font-weight: bold; margin-right: 10px;">⚡ 情绪能量</span>
                    <span style="font-size: 24px; font-weight: bold; color: {main_color}; text-shadow: 0 0 10px {glow_color};">{energy_percent}%</span>
                </div>
                
                <!-- 能量条背景 -->
                <div style="
                    width: 100%;
                    height: 30px;
                    background: rgba(0,0,0,0.3);
                    border-radius: 15px;
                    overflow: hidden;
                    box-shadow: inset 0 2px 5px rgba(0,0,0,0.3);
                ">
                    <!-- 能量条填充 -->
                    <div style="
                        width: {energy_percent}%;
                        height: 100%;
                        background: linear-gradient(90deg, {gradient_color} 0%, {main_color} 100%);
                        border-radius: 15px;
                        box-shadow: 0 0 20px {glow_color};
                        animation: slideIn 1s ease-out;
                        position: relative;
                        overflow: hidden;
                    ">
                        <!-- 流动光效 -->
                        <div style="
                            position: absolute;
                            top: 0;
                            left: -100%;
                            width: 100%;
                            height: 100%;
                            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
                            animation: shine 2s infinite;
                        "></div>
                    </div>
                </div>
            </div>
            
            <!-- AI回复区域 -->
            <div style="
                background: rgba(255,255,255,0.95);
                color: #333;
                padding: 20px;
                border-radius: 15px;
                margin-top: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                animation: fadeInUp 1s;
            ">
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #667eea;">
                    🤖 AI回复
                </div>
                <div style="font-size: 16px; line-height: 1.6; color: #555;">
                    {ai_reply}
                </div>
            </div>
        </div>
        
        <!-- CSS动画 -->
        <style>
            @keyframes pulse {{
                0%, 100% {{
                    transform: translate(-50%, -50%) scale(1);
                    opacity: 0.6;
                }}
                50% {{
                    transform: translate(-50%, -50%) scale(1.1);
                    opacity: 0.9;
                }}
            }}
            
            @keyframes float {{
                0%, 100% {{
                    transform: translate(-50%, -50%) translateY(0px);
                }}
                50% {{
                    transform: translate(-50%, -50%) translateY(-10px);
                }}
            }}
            
            @keyframes fadeIn {{
                from {{
                    opacity: 0;
                }}
                to {{
                    opacity: 1;
                }}
            }}
            
            @keyframes fadeInUp {{
                from {{
                    opacity: 0;
                    transform: translateY(20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            
            @keyframes slideIn {{
                from {{
                    width: 0;
                }}
            }}
            
            @keyframes shine {{
                to {{
                    left: 100%;
                }}
            }}
        </style>
        """
        
        return html_output
        
    except Exception as e:
        return f"<div style='color: red; padding: 20px;'>处理出错: {e}</div>"

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
            label="✨ 输入你的文本", 
            placeholder="请输入你想表达的内容...",
            lines=3
        ),
        outputs=gr.HTML(
            label="🎨 情绪可视化"
        ),
        title="🤖 情感AI助手 - 情绪可视化增强版",
        description="""
        <div style='text-align: center; font-size: 16px;'>
            <p><b>这个AI助手可以：</b></p>
            <p>💫 通过<b>动态颜色球体</b>直观展示情绪波动</p>
            <p>⚡ 用<b>情绪能量条</b>显示情感强度</p>
            <p>🌈 不同情绪对应不同颜色（开心→暖黄、愤怒→红色、平静→蓝色、失落→紫色）</p>
            <p>💬 根据情感提供相应的回复（鼓励或安慰）</p>
            <p><i>试着输入一些文本，看看AI如何呈现你的情绪！</i></p>
        </div>
        """,
        examples=examples,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Arial', 'Microsoft YaHei', sans-serif;
            max-width: 900px !important;
        }
        .gr-button-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            color: white !important;
            padding: 12px 32px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }
        .gr-button-primary:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
        }
        .gr-input {
            border-radius: 10px !important;
            border: 2px solid #667eea !important;
        }
        .gr-input:focus {
            border-color: #764ba2 !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
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
