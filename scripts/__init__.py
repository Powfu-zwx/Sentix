# -*- coding: utf-8 -*-
"""
Scripts包
包含所有训练、推理、实验、评估和工具脚本
"""

# 可以从这里直接导入各个模块
from . import training
from . import inference
from . import experiments
from . import evaluation
from . import utils

__all__ = ['training', 'inference', 'experiments', 'evaluation', 'utils']

