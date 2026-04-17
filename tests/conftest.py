"""
pytest fixtures and configuration
"""

import pytest
from unittest.mock import MagicMock, Mock
from neuro_agent_framework.interfaces.impls.reviewer.llm_reviewer import LLMBasedReviewer 
from neuro_agent_framework.llm.base import BaseLLM, LLMResponse
from neuro_agent_framework.llm.config_loader import ConfigLoader, LLMConfig
from neuro_agent_framework.core.enums import ModelType, ModelRole
from neuro_agent_framework.core.dataclasses import RegisteredModel, TaskResult, ModelResult
from neuro_agent_framework.registry.model_registry import ModelRegistry
from neuro_agent_framework.interfaces.impls.execution.basic_parallel_strategy import BasicParallelStrategy
from neuro_agent_framework.interfaces.impls.execution.hybrid_strategy import HybridStrategy
from neuro_agent_framework.interfaces.impls.execution.diversified_strategy import DiversifiedParallelStrategy


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )


def pytest_configure(config):
    """配置 pytest 插件"""
    config.addinivalue_line("markers", "integration: mark test as an integration test")


def is_integration_allowed():
    """检查是否允许运行集成测试"""
    return True


@pytest.fixture
def model_registry():
    """创建基本模型注册表"""
    return ModelRegistry()


@pytest.fixture
def mock_llm():
    """创建基本 Mock LLM"""
    from neuro_agent_framework.llm.base import MessageRole
    
    mock = MagicMock(spec=BaseLLM)
    mock.model_id = "test_model"
    mock.model = "test:model"
    
    response = MagicMock(spec=LLMResponse)
    response.success = True
    response.content = "test output"
    response.model_id = "test_model"
    response.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    response.error = None
    
    mock.chat.return_value = response
    return mock


@pytest.fixture
def mock_llm_reviewer():
    """创建 mock 评审器 LLM (使用 qwen3.5_08b)"""
    from neuro_agent_framework.llm.base import LLMResponse
    
    mock = MagicMock(spec=['chat', 'model_id', 'model', 'api_type', 'temperature', 'top_p', 'max_tokens', 'role', 'name'])
    mock.model_id = "qwen3.5_08b"
    mock.model = "qwen3.5:0.8b"
    mock.api_type = "openai"
    mock.temperature = 0.5
    mock.top_p = 0.9
    mock.max_tokens = 1024
    mock.role = "rTPJ_REVIEWER"
    mock.name = "qwen3.5:0.8b"

    # Mock response
    response = MagicMock(spec=LLMResponse)
    response.success = True
    response.content = '{"evaluation": "good", "confidence": 0.75}'
    response.model_id = "qwen3.5_08b"
    response.usage = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
    response.error = None

    mock.chat.return_value = response
    return mock


@pytest.fixture
def mock_reviewer_llm():
    """创建 mock 评审器 LLM (别名，为了向后兼容)"""
    return mock_llm_reviewer()


@pytest.fixture
def mock_llm_expert():
    """创建 mock 专家 LLM (使用 qwen3.5_35b)"""
    mock = MagicMock(spec=BaseLLM)
    mock.model_id = "qwen3.5_35b"
    mock.model = "qwen3.5:35b"
    mock.api_type = "openai"
    mock.temperature = 0.3
    mock.top_p = 0.95
    mock.max_tokens = 4096
    mock.role = "rDLPFC_UPGRADER"
    mock.name = "qwen3.5:35b"

    # Mock expert response
    response = MagicMock(spec=LLMResponse)
    response.success = True
    response.content = '{"reasoning": "这是深入分析", "final_answer": "专家提供的最终答案"}'
    response.model_id = "qwen3.5_35b"
    response.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    response.error = None

    mock.chat.return_value = response
    return mock


@pytest.fixture
def registry_with_executors():
    """创建带有执行器的注册表"""
    from neuro_agent_framework.registry.model_registry import ModelRegistry
    from neuro_agent_framework.llm.base import BaseLLM
    
    registry = ModelRegistry()
    
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.model_id = "test"
    mock_llm.model = "test:model"
    
    # 添加 mock 的 chat 方法
    response = MagicMock(spec=LLMResponse)
    response.success = False
    mock_llm.chat.return_value = response
    
    # 注册测试执行器
    registry.register(RegisteredModel(
        model_id="cheap_exec_1",
        name="Test Executor 1",
        model_type=ModelType.CHEAP_EXECUTOR,
        primary_role=ModelRole.rACC_STANDARD,
        estimated_cost=0.001,
        estimated_latency=1.0,
        is_active=True,
        config={"llm_instance": mock_llm}
    ))
    
    registry.register(RegisteredModel(
        model_id="cheap_exec_2",
        name="Test Executor 2",
        model_type=ModelType.CHEAP_EXECUTOR,
        primary_role=ModelRole.rACC_ALTERNATIVE,
        estimated_cost=0.001,
        estimated_latency=1.0,
        is_active=True,
        config={"llm_instance": mock_llm}
    ))
    
    # 注册评审器
    registry.register(RegisteredModel(
        model_id="reviewer_1",
        name="Test Reviewer",
        model_type=ModelType.CHEAP_REVIEWER,
        primary_role=ModelRole.rTPJ_REVIEWER,
        estimated_cost=0.001,
        estimated_latency=1.0,
        is_active=True,
        config={"llm_instance": mock_llm}
    ))
    
    return registry


@pytest.fixture
def basic_strategy():
    """创建基础策略实例"""
    return BasicParallelStrategy()


@pytest.fixture
def hybrid_strategy():
    """创建混合策略实例"""
    return HybridStrategy()


@pytest.fixture
def diversified_strategy():
    """创建多样化策略实例"""
    return DiversifiedParallelStrategy()


@pytest.fixture
def config_loader():
    """创建配置加载器实例"""
    return ConfigLoader()
