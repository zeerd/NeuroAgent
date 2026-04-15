"""
测试 NeuroAgent Framework 初始化
"""

import pytest
from unittest.mock import MagicMock, Mock
from neuro_agent_framework.framework.framework import NeuroAgentFramework
from neuro_agent_framework.framework.config import FrameworkConfig
from neuro_agent_framework.strategy.basic_strategy import BasicParallelStrategy
from neuro_agent_framework.core.enums import ModelType, ModelRole
from neuro_agent_framework.core.dataclasses import RegisteredModel, ModelResult, TaskResult
from neuro_agent_framework.llm.base import BaseLLM, LLMResponse
from neuro_agent_framework.registry.model_registry import ModelRegistry
from conftest import registry_with_executors

class TestNeuroAgentFrameworkInit:
    """测试框架初始化"""

    @pytest.fixture
    def mock_registry(self):
        """创建空注册表"""
        return ModelRegistry()

    @pytest.fixture
    def registry_with_executors(self, mock_registry):
        """创建包含执行器的注册表"""
        mock = MagicMock(spec=BaseLLM)
        mock.chat = Mock(return_value=Mock(spec=LLMResponse))

        # 注册至少 2 个执行器
        for i, role in enumerate([ModelRole.rACC_STANDARD, ModelRole.rACC_ALTERNATIVE]):
            mock_registry.register(RegisteredModel(
                model_id=f"executor{i+1}",
                name=f"Executor{i+1}",
                model_type=ModelType.CHEAP_EXECUTOR,
                primary_role=role,
                estimated_cost=0.001,
                estimated_latency=30.0,
                is_active=True,
                config={}
            ))

        # 注册评审器
        mock_registry.register(RegisteredModel(
            model_id="reviewer",
            name="Reviewer",
            model_type=ModelType.CHEAP_REVIEWER,
            primary_role=ModelRole.rTPJ_REVIEWER,
            estimated_cost=0.0005,
            estimated_latency=20.0,
            is_active=True,
            config={}
        ))

        return mock_registry

    @pytest.fixture
    def mock_reviewer_llm(self):
        """创建 mock 评审器 LLM"""
        mock = MagicMock(spec=BaseLLM)
        mock.model_id = "reviewer_mock"
        mock.role = "rTPJ_REVIEWER"

        mock_response = Mock(spec=LLMResponse)
        mock_response.success = True
        mock_response.content = '{"evaluation": "good", "confidence": 0.75}'
        mock_response.usage = {}
        mock.chat.return_value = mock_response
        return mock

    def test_init_minimal(self, registry_with_executors):
        """测试最小初始化"""
        framework = NeuroAgentFramework(
            model_registry=registry_with_executors,
        )

        assert framework is not None
        assert framework.strategy is not None

    def test_init_with_custom_strategy(self, registry_with_executors):
        """测试自定义策略初始化"""
        strategy = BasicParallelStrategy()

        framework = NeuroAgentFramework(
            model_registry=registry_with_executors,
            execution_strategy=strategy,
        )

        assert framework.strategy == strategy


class TestFrameworkValidation:
    """测试框架验证"""

    def test_framework_requires_registry(self):
        """测试框架要求注册表"""
        with pytest.raises(TypeError):
            NeuroAgentFramework()


class TestFrameworkBasicOperations:
    """测试框架基本操作"""

    @pytest.fixture
    def framework(self, registry_with_executors):
        """创建框架实例"""
        return NeuroAgentFramework(
            model_registry=registry_with_executors,
        )

    def test_framework_initialization(self, framework):
        """测试框架初始化"""
        assert framework is not None
        assert hasattr(framework, 'strategy')
        assert hasattr(framework, 'registry')
        assert framework.registry is not None

    def test_framework_has_execute_method(self, framework):
        """测试框架有 execute 方法"""
        assert hasattr(framework, 'execute')
        assert callable(framework.execute)

    def test_framework_has_reviewer(self, framework):
        """测试框架有 reviewer 属性"""
        assert hasattr(framework, 'reviewer')
        assert framework.reviewer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
