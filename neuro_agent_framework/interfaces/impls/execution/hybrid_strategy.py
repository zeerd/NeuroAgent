"""
HybridStrategy - 混合策略

自动根据模型数量选择最优策略：
- 2 个模型：基础策略（相同提示）
- 3+ 个模型：多样策略（差异提示）
"""

import logging
from typing import List, Dict

from neuro_agent_framework.interfaces.impls.execution.base_strategy import ExecutionStrategy
from neuro_agent_framework.core.dataclasses import RegisteredModel, ModelResult
from neuro_agent_framework.interfaces.impls.execution.basic_strategy import BasicParallelStrategy
from neuro_agent_framework.interfaces.impls.execution.diversified_strategy import DiversifiedParallelStrategy


logger = logging.getLogger(__name__)


class HybridStrategy(ExecutionStrategy):
    """
    混合策略 - 自动选择最优策略
    
    根据执行模型数量动态选择最合适的执行方式
    """
    
    def __init__(self):
        self.basic = BasicParallelStrategy()
        self.diversified = DiversifiedParallelStrategy()
        logger.info("HybridStrategy initialized")
    
    def execute(self, 
               models: List[RegisteredModel], 
               request: str,
               context: Dict) -> List[ModelResult]:
        """根据模型数量自动选择策略"""
        if len(models) <= 2:
            logger.debug(f"Using basic strategy for {len(models)} models")
            return self.basic.execute(models, request, context)
        else:
            logger.debug(f"Using diversified strategy for {len(models)} models")
            return self.diversified.execute(models, request, context)
    
    def should_diversify(self, num_models: int) -> bool:
        """根据模型数量决定是否多样化"""
        return num_models > 2
