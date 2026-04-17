"""
执行策略接口 - IExecutionStrategy

确定性的接口定义，不确定的实现通过具体类实现
"""

from abc import ABC, abstractmethod
from typing import List, Dict

from ..core.dataclasses import ModelResult
from ..core.enums import ModelRole


class IExecutionStrategy(ABC):
    """
    执行策略接口
    
    核心职责：
    - 根据任务需求分发执行任务给各个模型
    - 决定并行还是串行执行
    
    设计原则：
    - 接口确定，实现可变
    - 不依赖具体模型实现
    - 可配置的策略选择
    """

    @abstractmethod
    def execute(
        self,
        models: List['RegisteredModel'],
        request: str,
        context: Dict,
        task_complexity: float = None
    ) -> List[ModelResult]:
        """
        执行任务的核心逻辑
        
        Args:
            models: 可执行的模型列表
            request: 用户请求
            context: 任务上下文配置
            task_complexity: 任务复杂度 (0-1)
            
        Returns:
            所有模型的执行结果列表
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """获取策略名称，用于识别"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        获取策略支持的能力列表
        
        Returns:
            支持的能力列表，如：['parallel', 'sequential', 'diverse_prompts']
        """
        pass
    
    @abstractmethod
    def should_diversify(self, num_models: int) -> bool:
        """
        判断是否应该使用差异化提示
        
        Args:
            num_models: 模型数量
            
        Returns:
            True 如果使用差异化提示
        """
        pass
