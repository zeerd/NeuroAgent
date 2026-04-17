"""
RuleBasedConfidenceCalculator - 基于规则置信度计算器
"""

from typing import List, Dict, Any

from neuro_agent_framework.interfaces.confidence_calculator import IConfidenceCalculator
from neuro_agent_framework.core.dataclasses import ModelResult


class RuleBasedConfidenceCalculator(IConfidenceCalculator):
    TYPE = "rule_based"

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._logger = __import__('logging').getLogger(__name__)

    def compute(self, results, context, review_result):
        consistency_from_review = review_result['confidence']
        self._logger.info(f"consistency: {consistency_from_review}")

        quality_score = 0.9 if results else 0.5
        coverage_score = 0.8

        overall = consistency_from_review * 0.5 + quality_score * 0.25 + coverage_score * 0.25
        needs_expert = consistency_from_review < 0.7 or quality_score < 0.7

        return {
            'overall': overall,
            'needs_expert': needs_expert,
            'details': {
                'consistency': {'score': consistency_from_review, 'source': 'reviewer_analysis'},
                'quality': {'score': quality_score, 'source': 'output_length'},
                'coverage': {'score': coverage_score, 'source': 'task_complexity'}
            },
            'confidence_source': {'is_data_driven': False, 'has_hueristics': False}
        }

    def get_calculator_type(self) -> str:
        return self.TYPE

    def is_data_driven(self) -> bool:
        return False

    def can_calculate(self, num_results: int) -> bool:
        return True
