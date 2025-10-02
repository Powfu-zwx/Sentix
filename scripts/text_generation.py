# -*- coding: utf-8 -*-
"""
中文GPT-2文本生成脚本
使用uer/gpt2-chinese-cluecorpussmall模型进行中文文本生成
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_chinese_gpt2_model():
    """
    加载中文GPT-2模型
    
    Returns:
        model: 文本生成模型
        tokenizer: 分词器
    """
    print("正在加载中文GPT-2模型...")
    
    # 加载模型和分词器
    model_name = "uer/gpt2-chinese-cluecorpussmall"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        gen_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 检查设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen_model = gen_model.to(device)
        gen_model.eval()
        
        print(f"模型加载成功！使用设备: {device}")
        return gen_model, tokenizer, device
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None, None

def generate_text(model, tokenizer, device, prompt, max_length=100, temperature=0.8, top_p=0.9, do_sample=True):
    """
    根据提示词生成文本
    
    Args:
        model: 生成模型
        tokenizer: 分词器
        device: 设备
        prompt (str): 输入提示词
        max_length (int): 最大生成长度
        temperature (float): 温度参数，控制随机性
        top_p (float): nucleus采样参数
        do_sample (bool): 是否采样
    
    Returns:
        str: 生成的文本
    """
    try:
        # 编码输入
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # 生成文本
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 处理中文文本格式，去除不必要的空格
        generated_text = generated_text.replace(" ", "")
        
        return generated_text
        
    except Exception as e:
        print(f"生成文本时出错: {e}")
        return ""

def main():
    """主函数"""
    print("中文GPT-2文本生成系统")
    print("=" * 50)
    
    # 加载模型
    model, tokenizer, device = load_chinese_gpt2_model()
    
    if model is None:
        print("无法加载模型，程序退出")
        return
    
    # 预设提示词
    prompt = "用户说：今天心情很糟糕。\nAI回复："
    
    print(f"\n使用提示词: {prompt}")
    print("-" * 50)
    
    # 生成文本
    generated_text = generate_text(
        model, tokenizer, device, prompt,
        max_length=150,
        temperature=0.8,
        top_p=0.9
    )
    
    if generated_text:
        print("生成的完整文本:")
        print(generated_text)
        print("\n" + "-" * 50)
        
        # 提取AI回复部分
        if "AI回复：" in generated_text:
            ai_reply = generated_text.split("AI回复：")[1].strip()
            print("AI回复内容:")
            print(ai_reply)
    
    # 交互式生成
    print("\n" + "=" * 50)
    print("交互式文本生成 (输入 'quit' 退出)")
    print("=" * 50)
    
    while True:
        user_prompt = input("\n请输入提示词: ")
        
        if user_prompt.lower() == 'quit':
            print("程序退出")
            break
            
        if user_prompt.strip():
            print("\n生成中...")
            generated = generate_text(
                model, tokenizer, device, user_prompt,
                max_length=100,
                temperature=0.8,
                top_p=0.9
            )
            
            if generated:
                print(f"\n生成结果:\n{generated}")
            
            print("-" * 30)

def demo_examples():
    """演示不同的生成示例"""
    print("演示模式 - 不同类型的文本生成")
    print("=" * 50)
    
    # 加载模型
    model, tokenizer, device = load_chinese_gpt2_model()
    
    if model is None:
        print("无法加载模型，程序退出")
        return
    
    # 测试不同的提示词
    test_prompts = [
        "用户说：今天心情很糟糕。\nAI回复：",
        "今天天气很好，",
        "人工智能的发展",
        "故事：从前有一个",
        "科技改变生活，"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n示例 {i}:")
        print(f"提示词: {prompt}")
        
        generated = generate_text(
            model, tokenizer, device, prompt,
            max_length=80,
            temperature=0.7,
            top_p=0.9
        )
        
        if generated:
            print(f"生成结果: {generated}")
        
        print("-" * 40)

if __name__ == "__main__":
    # 运行主程序
    main()
    
    # 如果想看更多示例，可以取消下面的注释
    # print("\n" + "=" * 50)
    # demo_examples()
