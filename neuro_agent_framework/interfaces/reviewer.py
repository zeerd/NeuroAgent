"""
评审器接口 - IReviewer

确定性的接口定义，不确定的评审逻辑通过具体类实现
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..core.dataclasses import ModelResult


class IReviewer(ABC):
    """
    评审器接口
    
    核心职责：
    - 综合分析多个执行器的输出
    - 评估答案的一致性
    - 判断是否需要专家升级
    - 生成综合答案
    
    设计原则：
    - 接口确定，实现可变（可以是 LLM、规则引擎、混合等）
    - 配置化实现选择
    """

    @abstractmethod
    def review(
        self,
        results: List[ModelResult],
        request: str,
        llm: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        执行评审的核心逻辑
        
        Args:
            results: 所有执行器的结果
            request: 原始用户请求
            llm: 用于评审的 LLM 实例（可选）
            
        Returns:
            评审结果字典，包含：
            - combined_answer: 综合答案
            - confidence: 一致性评分 (0-1)
            - needs_expert: 是否需要专家升级 (bool)
            - rationale: 评审理由
        """
        pass

    @abstractmethod
    def get_reviewer_type(self) -> str:
        """获取评审器类型，用于识别"""
        pass
    
    @abstractmethod
    def can_review(self, num_results: int) -> bool:
        """
        判断这个评审器是否能评审给定数量的结果
        
        Returns:
            True 如果评审器支持评审
        """
        pass
