"""
V2 框架集成测试 - 端到端测试

验证：
1. 接口正确使用
2. 数据流正确
3. 真实数据而非默认值
4. 决策逻辑正确
"""

import pytest
import sys
from pathlib import Path

# 添加项目路径
project_path = Path(__file__).parent.parent
sys.path.insert(0, str(project_path))

from neuro_agent_framework.interfaces.execution_strategy import IExecutionStrategy
from neuro_agent_framework.interfaces.reviewer import IReviewer
from neuro_agent_framework.interfaces.confidence_calculator import IConfidenceCalculator
from neuro_agent_framework.interfaces.impls.execution.basic_parallel_strategy import BasicParallelStrategy
from neuro_agent_framework.interfaces.impls.reviewer.llm_reviewer import LLMBasedReviewer
from neuro_agent_framework.interfaces.impls.confidence.placeholder_confidence_calculator import PlaceholderConfidenceCalculator
from neuro_agent_framework.interfaces.impls.confidence.rule_confidence_calculator import RuleBasedConfidenceCalculator
from neuro_agent_framework.core.dataclasses import RegisteredModel, ModelResult
from neuro_agent_framework.core.enums import ModelRole


class TestFrameworkInterfaces:
    """测试 v2 框架的接口使用"""

    def test_all_implements_interface(self):
        """所有实现类都正确实现相应接口"""
        # 执行策略
        strategy = BasicParallelStrategy()
        assert isinstance(strategy, IExecutionStrategy)
        assert hasattr(strategy, 'execute')
        assert hasattr(strategy, 'get_strategy_type')

        # 评审器
        # 注意：LLMBasedReviewer 需要模型才能初始化，这里只测试接口
        assert hasattr(LLMBasedReviewer, 'review')
        assert hasattr(LLMBasedReviewer, 'get_reviewer_type')

        # 置信度计算器
        calculator = PlaceholderConfidenceCalculator()
        assert isinstance(calculator, IConfidenceCalculator)
        assert hasattr(calculator, 'compute')
        assert hasattr(calculator, 'is_data_driven')


class TestConfidenceCalculatorRealData:
    """测试置信度计算器使用真实数据"""

    def test_placeholder_usess_reviewer_confidence(self):
        """Placeholder 计算器使用 Reviewer 的真实评分"""
        calculator = PlaceholderConfidenceCalculator()

        # Reviewer 评分为 0.95
        review_result = {'confidence': 0.95}
        result = calculator.compute([], {'complexity': 0.5}, review_result)

        # 应使用真实值 0.95，而非默认值 0.5
        assert result['details']['consistency']['score'] == 0.95
        assert result['details']['consistency']['source'] == 'reviewer_analysis'

    def test_rule_based_uses_reviewer_confidence(self):
        """RuleBased 计算器使用 Reviewer 的真实评分"""
        calculator = RuleBasedConfidenceCalculator()

        review_result = {'confidence': 0.85}
        result = calculator.compute([], {'complexity': 0.3}, review_result)

        assert result['details']['consistency']['score'] == 0.85


class TestDecisionLogic:
    """测试决策逻辑"""

    def test_high_consistency_no_expert(self):
        """高一致性应不触发专家"""
        calculator = PlaceholderConfidenceCalculator()

        review_result = {
            'confidence': 0.95,  # 高一致性
            'needs_expert': False
        }

        result = calculator.compute([], {'complexity': 0.5}, review_result)

        assert result['needs_expert'] is False
        assert result['overall'] >= 0.7

    def test_low_consistency_with_low_quality_needs_expert(self):
        """低一致性 + 低质量应触发专家"""
        calculator = PlaceholderConfidenceCalculator()

        review_result = {
            'confidence': 0.40,  # 低一致性
            'needs_expert': True
        }

        result = calculator.compute([], {'complexity': 0.5}, review_result)

        # 综合低质应触发专家
        assert result['needs_expert'] is True


class TestInterfacePolymorphism:
    """测试接口多态性"""

    def test_can_swap_calculators(self):
        """可以互换使用不同的计算器实现"""
        calculators = [
            PlaceholderConfidenceCalculator(),
            RuleBasedConfidenceCalculator()
        ]

        for cal in calculators:
            # 都满足接口契约
            assert isinstance(cal, IConfidenceCalculator)
            assert cal.get_calculator_type() is not None
            assert isinstance(cal.can_calculate(1), bool)

    def test_can_swap_strategies(self):
        """可以互换使用不同的策略实现"""
        strategies = [
            BasicParallelStrategy()
        ]

        for strategy in strategies:
            assert isinstance(strategy, IExecutionStrategy)
            assert hasattr(strategy, 'get_strategy_type')


class TestDataFlowsCorrectly:
    """测试数据流正确性"""

    def test_reviewer_confidence_propagates(self):
        """Reviewer 评分正确传播"""
        calculator = PlaceholderConfidenceCalculator()

        # 模拟 Reviewer 的 0.90 评分
        review_result = {'confidence': 0.90}
        result = calculator.compute([], {'complexity': 0.5}, review_result)

        assert result['details']['consistency']['score'] == 0.90

    def test_complexity_affects_quality(self):
        """复杂度影响质量分析"""
        calculator = PlaceholderConfidenceCalculator()

        review_result = {'confidence': 0.80}

        # 高复杂度
        high_complexity = calculator.compute([], {'complexity': 0.9}, review_result)
        # 低复杂度
        low_complexity = calculator.compute([], {'complexity': 0.1}, review_result)

        # 两者都应该有意义（不是默认的 0.5）
        assert high_complexity['details']['quality']['score'] != 0.5
        assert low_complexity['details']['quality']['score'] != 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
