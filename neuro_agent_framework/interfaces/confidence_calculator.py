"""
置信度计算器接口 - IConfidenceCalculator

真实数据驱动的置信度评估接口

设计理念：
- 接口契约确定，实现可变
- 使用 Reviewer 的真实评分，而非杜撰数据
- 支持多种实现：LLM 分析、规则引擎、混合、占位
- 明确标注哪些是"已验证"、哪些是"待探索"
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..core.dataclasses import ModelResult


class IConfidenceCalculator(ABC):
    """
    置信度计算器接口
    
    核心职责：
    - 基于多模型输出和评审结果，评估任务完成质量
    - 决定是否需要专家升级
    - 提供详细的置信度分解
    
    关键原则：
    1. **使用真实数据** - 不杜撰任何数值
    2. **数据驱动决策** - 依赖 Reviewer 的真实评分
    3. **可验证性** - 每个维度的评分都有明确计算依据
    4. **透明性** - 明确标注哪些是启发式规则，哪些是实证分析
    
    设计说明：
    - 这个组件是"待探索"领域
    - 当前实现应明确标注"占位/示例"
    - 未来可以开发：LLM 分析、规则引擎、混合模式等
    """

    @abstractmethod
    def compute(
        self,
        results: List[ModelResult],
        context: Dict[str, Any],
        review_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算置信度评分
        
        Args:
            results: 所有模型执行结果
            context: 上下文配置，必须包含：
                - 'complexity': float (任务复杂度 0-1)
                - 其他可选配置
            review_result: Reviewer 的评审结果，必须包含：
                - 'confidence': float (0-1, 真实一致性评分)
                - 'combined_answer': str
                - 'needs_expert': bool (Reviewer 建议)
                - 'rationale': str
        
        Returns:
            置信度评估结果，包含：
            {
                'overall': float (综合评分，0-1),
                'needs_expert': bool (是否必须专家升级),
                'details': {
                    'consistency': {
},  # 来自 Reviewer 的真实评分
                    'coverage': {  # 覆盖度分析
                        'score': float,
                        'source': str,  # 数据来源，"prompt_analysis" / "llm_analysis" / ...
                        'method': str,
                    },
                    'quality': {  # 质量分析
                        'score': float,
                        'source': str,
                        'method': str,
                    },
                    # 其他维度...
                },
                'confidence_source': {
                    'is_data_driven': bool,  # 是否数据驱动
                    'has_hueristics': bool,  # 是否使用启发式规则
                    'notes': str  # 补充说明
                }
            }
            
        重要说明：
        - 必须使用 review_result['confidence'] 作为一致性依据
        - 不能杜撰任何置信度数值
        - 每个评分维度必须有明确的数据来源说明
        """
        pass
    
    @abstractmethod
    def get_calculator_type(self) -> str:
        """
        获取计算器类型标识
        
        Returns:
            类型字符串，如：'basic', 'llm', 'rule', 'placeholder'
        """
        pass
    
    @abstractmethod
    def is_data_driven(self) -> bool:
        """
        判断是否数据驱动实现
        
        Returns:
            True 如果计算基于真实数据
            False 如果包含启发式规则或默认值
        """
        pass
    
    @abstractmethod
    def can_calculate(self, num_results: int) -> bool:
        """
        判断是否能计算给定数量的结果
        
        Args:
            num_results: 结果数量
            
        Returns:
            True 如果可以计算
        """
        pass
