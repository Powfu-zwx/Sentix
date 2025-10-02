# -*- coding: utf-8 -*-
"""
推理脚本模块
包含模型推理和文本生成相关的脚本
"""

from .inference import predict_sentiment
from .text_generation import generate_text, load_chinese_gpt2_model

__all__ = ['predict_sentiment', 'generate_text', 'load_chinese_gpt2_model']

