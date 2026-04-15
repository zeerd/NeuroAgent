"""
测试执行策略
"""

import pytest
from neuro_agent_framework.core.dataclasses import RegisteredModel, ModelResult
from neuro_agent_framework.core.enums import ModelType, ModelRole
from neuro_agent_framework.strategy.basic_strategy import BasicParallelStrategy
from neuro_agent_framework.strategy.hybrid_strategy import HybridStrategy
from neuro_agent_framework.strategy.diversified_strategy import DiversifiedParallelStrategy
from neuro_agent_framework.registry import model_registry

class TestBasicParallelStrategy:
    """测试基础并行策略"""

    def test_init(self):
        """测试初始化"""
        strategy = BasicParallelStrategy()
        assert strategy is not None
        assert strategy.STANDARD_PROMPT is not None

    def test_should_diversify(self):
        """测试是否应该分发的逻辑"""
        strategy = BasicParallelStrategy()
        # 基础策略不分发差异化提示
        assert strategy.should_diversify(2) is False
        assert strategy.should_diversify(4) is False


class TestDiversifiedParallelStrategy:
    """测试多元化策略"""

    def test_init(self):
        """测试初始化"""
        strategy = DiversifiedParallelStrategy()
        assert strategy is not None

    def test_should_diversify_small(self):
        """测试小规模的模型数量应该分发"""
        strategy = DiversifiedParallelStrategy()
        assert strategy.should_diversify(2) is True
        assert strategy.should_diversify(3) is True

    def test_should_diversify_large(self):
        """测试大规模的模型数量都应该分发"""
        strategy = DiversifiedParallelStrategy()
        # DiversifiedParallelStrategy 总是使用差异化策略
        assert strategy.should_diversify(2) is True
        assert strategy.should_diversify(3) is True
        assert strategy.should_diversify(5) is True
        assert strategy.should_diversify(8) is True


class TestHybridStrategy:
    """测试混合策略"""

    def test_init(self):
        """测试初始化"""
        strategy = HybridStrategy()
        assert strategy is not None


class TestStrategyExecution:
    """测试策略执行"""

    def create_mock_model(self, model_id, role):
        """创建模拟模型"""
        return RegisteredModel(
            model_id=model_id,
            name=f"Model {model_id}",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=role,
            estimated_cost=0.001,
            estimated_latency=1.0,
            config={}
        )

    @pytest.mark.integration
    def test_basic_strategy_with_real_llm(self, model_registry):
        """测试基本策略与真实 LLM 执行"""
        strategy = BasicParallelStrategy()

        executors = model_registry.list_models(model_type=ModelType.CHEAP_EXECUTOR)

        results = strategy.execute(
            executors=executors,
            request="测试请求",
            context={ "key": "value" },
            task_complexity=0.5
        )

        assert len(results) == len(executors)
        assert all(isinstance(r, ModelResult) for r in results)

    @pytest.mark.integration
    def test_hybrid_strategy_with_real_llm(self, model_registry):
        """测试混合策略与真实 LLM 执行"""
        strategy = HybridStrategy()

        executors = model_registry.list_models(model_type=ModelType.CHEAP_EXECUTOR)

        results = strategy.execute(
            executors=executors,
            request="测试请求",
            context={ "key": "value" },
            task_complexity=0.5
        )

        assert len(results) == len(executors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
