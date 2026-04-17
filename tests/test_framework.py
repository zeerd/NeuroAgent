"""
测试 Framework V2 - 接口驱动的架构

测试场景：
1. 高一致性场景（Reviewer 0.95）
2. 低一致性场景（Reviewer 0.40）

确保：
- 接口正确使用
- 真实数据流
- 决策逻辑正确
"""

import pytest
import sys
sys.path.insert(0, '..')

from neuro_agent_framework.interfaces.execution_strategy import IExecutionStrategy
from neuro_agent_framework.interfaces.reviewer import IReviewer
from neuro_agent_framework.interfaces.confidence_calculator import IConfidenceCalculator
from neuro_agent_framework.interfaces.impls.execution.basic_parallel_strategy import BasicParallelStrategy
from neuro_agent_framework.interfaces.impls.reviewer.llm_reviewer import LLMBasedReviewer
from neuro_agent_framework.interfaces.impls.confidence.placeholder_confidence_calculator import PlaceholderConfidenceCalculator
from neuro_agent_framework.core.dataclasses import TaskResult, ModelResult, RegisteredModel
from neuro_agent_framework.core.enums import ModelType, ModelRole


class TestFrameworkInterface:
    """测试 v2 版本的接口使用"""

    def test_basic_strategy_implements_interface(self):
        """测试 BasicParallelStrategy 实现 IExecutionStrategy"""
        strategy = BasicParallelStrategy()

        assert isinstance(strategy, IExecutionStrategy)
        assert strategy.get_strategy_type() == "basic_parallel"
        assert "parallel" in strategy.get_capabilities()
        assert strategy.should_diversify(2) is False

    def test_placeholder_confidence_implements_interface(self):
        """测试 PlaceholderConfidenceCalculator 实现 IConfidenceCalculator"""
        calculator = PlaceholderConfidenceCalculator()

        assert isinstance(calculator, IConfidenceCalculator)
        assert calculator.get_calculator_type() == "placeholder"
        assert calculator.is_data_driven() is False
        assert calculator.can_calculate(1)

    def test_interface_compatible(self):
        """测试接口兼容性"""
        # 可以互换使用任何实现
        strategy1 = BasicParallelStrategy()

        # 支持多态调用
        assert isinstance(strategy1, IExecutionStrategy)

        # 可以调用接口方法
        result = strategy1.get_strategy_type()
        assert result in ["basic_parallel"]


class TestConfidenceCalculationFlow:
    """测试置信度计算流程"""

    def test_high_consistency_no_expert_needed(self):
        """
        测试高一致性场景
        Reviewer 评分 0.95, 应不触发专家
        """
        calculator = PlaceholderConfidenceCalculator()

        # 模拟 Reviewer 的高一致性评分
        review_result = {
            'confidence': 0.95,
            'needs_expert': False,
            'combined_answer': '综合答案'
        }

        # 模拟中等质量的任务
        confidence = calculator.compute(
            [],
            {'complexity': 0.5},
            review_result
        )

        assert confidence['overall'] >= 0.7
        assert confidence['needs_expert'] is False, "高一致性不应触发专家"

    def test_low_consistency_with_low_quality_triggers_expert(self):
        """
        测试低一致性 + 低质量场景
        Reviewer 评分 0.40, 应触发专家
        """
        calculator = PlaceholderConfidenceCalculator()

        review_result = {
            'confidence': 0.40,
            'needs_expert': True,
            'combined_answer': '综合答案'
        }

        confidence = calculator.compute(
            [],
            {'complexity': 0.5},
            review_result
        )

        assert confidence['needs_expert'] is True, "低一致性应触发专家升级"
        assert confidence['details']['consistency']['score'] == 0.40

    def test_data_flows_correctly(self):
        """测试数据正确传递"""
        calculator = PlaceholderConfidenceCalculator()

        review_result = {'confidence': 0.85}
        result = calculator.compute([], {'complexity': 0.3}, review_result)

        # 验证 Reviewer 的评分传入
        assert result['details']['consistency']['score'] == 0.85
        assert result['confidence_source']['is_data_driven'] is False


class TestImplementationDetails:
    """测试实现细节"""

    def test_no_default_values(self):
        """测试没有使用默认值 0.5"""
        calculator = PlaceholderConfidenceCalculator()

        # 使用真实 Reviewer 评分
        review_result = {'confidence': 0.90}
        result = calculator.compute([], {}, review_result)

        # 应使用真实值，而非默认 0.5
        assert result['details']['consistency']['score'] == 0.90

        # 覆盖率和质量应基于分析
        assert 'output_length_analysis' in result['details']['coverage']['source']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
