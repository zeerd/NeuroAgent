"""
ExecutionStrategy - 执行策略基类
定义所有执行策略的基础接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict

from ..core.dataclasses import RegisteredModel, ModelResult


class ExecutionStrategy(ABC):
    """
    执行策略接口
    
    所有执行策略都必须实现这个接口，
    定义如何并行执行多个模型
    """
    
    @abstractmethod
    def execute(self, 
               models: List[RegisteredModel], 
               request: str,
               context: Dict) -> List[ModelResult]:
        """
        并行执行模型
        
        Args:
            models: 要执行的模型列表
            request: 用户请求
            context: 任务上下文信息
        
        Returns:
            所有模型的执行结果列表
        """
        pass
    
    @abstractmethod
    def should_diversify(self, num_models: int) -> bool:
        """
        判断是否需要对不同模型分发差异化任务提示
        
        Args:
            num_models: 模型数量
        
        Returns:
            True 表示需要差异化提示，False 表示使用相同提示
        """
        pass
