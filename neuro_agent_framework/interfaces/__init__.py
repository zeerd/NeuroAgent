"""
NeuroAgent Framework - 接口层

核心接口定义
所有后续实现都必须实现这些接口
"""

from .execution_strategy import IExecutionStrategy
from .reviewer import IReviewer
from .confidence_calculator import IConfidenceCalculator

__all__ = [
    'IExecutionStrategy',
    'IReviewer',
    'IConfidenceCalculator',
]
