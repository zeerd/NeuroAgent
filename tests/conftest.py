"""
pytest fixtures and configuration
"""

import pytest
from unittest.mock import MagicMock, Mock
from neuro_agent_framework.reviewer.reviewer import Reviewer
from neuro_agent_framework.llm.base import BaseLLM, LLMResponse
from neuro_agent_framework.llm.config_loader import ConfigLoader, LLMConfig
from neuro_agent_framework.core.enums import ModelType, ModelRole
from neuro_agent_framework.core.dataclasses import RegisteredModel, TaskResult, ModelResult
from neuro_agent_framework.registry.model_registry import ModelRegistry
from neuro_agent_framework.strategy.basic_strategy import BasicParallelStrategy
from neuro_agent_framework.strategy.hybrid_strategy import HybridStrategy
from neuro_agent_framework.strategy.diversified_strategy import DiversifiedParallelStrategy


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="运行集成测试"
    )


def pytest_configure(config):
    """配置自定义标记"""
    config.addinivalue_line(
        "markers", "integration: marks test as requiring LLM integration"
    )
    config.option.integration = False


def pytest_collection_modifyitems(config, items):
    """跳过集成测试 (默认)"""
    skip_integration = pytest.mark.skip(reason="需要 --run-integration 标记")
    if not config.option.integration:
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture
def mock_reviewer():
    """创建模拟 Reviewer 实例 (实际 Reviewer 类)"""
    mock_model = MagicMock(spec=BaseLLM)
    mock_model.model_id = "reviewer_mock"
    mock_model.model = "reviewer_mock"
    mock_model.role = "rTPJ_REVIEWER"
    mock_model.name = "Reviewer Mock"
    mock_model.chat = Mock(return_value=Mock(spec=LLMResponse))

    try:
        return Reviewer(mock_model)
    except Exception:
        return MagicMock(spec=Reviewer)


@pytest.fixture
def mock_registry():
    """返回空 ModelRegistry 实例 (不需要额外参数)"""
    return ModelRegistry()


@pytest.fixture
def model_registry():
    """返回空 ModelRegistry 实例 (别名)"""
    return ModelRegistry()


@pytest.fixture
def registry_with_executors():
    """创建包含执行器的注册表，供测试使用"""
    registry = ModelRegistry()
    mock = MagicMock(spec=BaseLLM)
    mock.chat = Mock(return_value=Mock(spec=LLMResponse))

    # 注册至少 2 个执行器
    for i, role in enumerate([ModelRole.rACC_STANDARD, ModelRole.rACC_ALTERNATIVE]):
        registry.register(RegisteredModel(
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
    registry.register(RegisteredModel(
        model_id="reviewer",
        name="Reviewer",
        model_type=ModelType.CHEAP_REVIEWER,
        primary_role=ModelRole.rTPJ_REVIEWER,
        estimated_cost=0.0005,
        estimated_latency=20.0,
        is_active=True,
        config={}
    ))

    return registry


@pytest.fixture
def mock_llm_reviewer():
    """创建 mock 评审器 LLM (使用 qwen3.5_08b)"""
    mock = MagicMock(spec=BaseLLM)
    mock.model_id = "qwen3.5_08b"
    mock.model = "qwen3.5:0.8b"
    mock.api_type = "openai"
    mock.temperature = 0.5
    mock.top_p = 0.9
    mock.max_tokens = 1024
    mock.role = "rTPJ_REVIEWER"
    mock.name = "qwen3.5:0.8b"

    # Mock LLM.chat 响应
    response = Mock(spec=LLMResponse)
    response.success = True
    response.content = '{"evaluation": "good", "confidence": 0.75}'
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

    # Mock LLM.chat 返回详细答案
    response = Mock(spec=LLMResponse)
    response.success = True
    response.content = '{"reasoning": "这是推理步骤", "final_answer": "最终答案"}'
    response.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    response.error = None

    mock.chat.return_value = response
    return mock


@pytest.fixture
def mock_llm_executor():
    """创建 mock 执行器 LLM"""
    mock = MagicMock(spec=BaseLLM)
    mock.model_id = "qwen3.5_2b"
    mock.model = "qwen3.5:2b"
    mock.api_type = "openai"
    mock.temperature = 0.7
    mock.top_p = 1.0
    mock.max_tokens = 2048
    mock.role = "rACC_STANDARD"
    mock.name = "qwen3.5:2b"

    # Mock LLM.chat 返回答案
    response = Mock(spec=LLMResponse)
    response.success = True
    response.content = '{"answer": "模拟答案", "status": "complete"}'
    response.usage = {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
    response.error = None

    mock.chat.return_value = response
    return mock


@pytest.fixture
def mock_llm_factory():
    """创建 mock LLMFactory"""
    factory = Mock()
    factory.get_or_create = Mock(side_effect=[MagicMock(spec=BaseLLM), MagicMock(spec=BaseLLM), MagicMock(spec=BaseLLM)])
    return factory


@pytest.fixture
def loader():
    """配置加载器 fixture"""
    loader = ConfigLoader()
    loader.load(force=True)
    return loader


@pytest.fixture
def llm_config():
    """配置加载器 fixture"""
    loader = ConfigLoader()
    return loader


@pytest.fixture
def registered_model(model_registry):
    """注册模型 fixture"""
    model = RegisteredModel(
        model_id="model1",
        name="Model 1",
        model_type=ModelType.CHEAP_EXECUTOR,
        primary_role=ModelRole.rACC_STANDARD,
        estimated_cost=0.001,
        estimated_latency=1.0,
        config={}
    )
    return model


@pytest.fixture
def mock_llm_config():
    """创建 LLMConfig 对象"""
    return LLMConfig(model="test-model", temperature=0.7)


@pytest.fixture
def mock_model_result(model_registry):
    """创建 ModelResult 对象"""
    result = ModelResult(
        model_id="model1",
        model_name="Model 1",
        role=ModelRole.rACC_STANDARD,
        output="test output",
        confidence=0.75,
        latency=1.2
    )
    return result


@pytest.fixture
def mock_task_complexity_low():
    """模拟低复杂度任务"""
    return 0.3


@pytest.fixture
def mock_task_complexity_high():
    """模拟高复杂度任务"""
    return 0.9


@pytest.fixture
def mock_task_complexity_medium():
    """模拟中等复杂度任务"""
    return 0.6


@pytest.fixture(params=[0.1, 0.3, 0.5, 0.7, 0.9])
def task_complexity(request):
    """提供不同复杂度的参数化 fixtures"""
    return request.param


@pytest.fixture
def mock_context():
    """模拟上下文"""
    return {
        "key1": "value1",
        "key2": "value2"
    }


@pytest.fixture
def mock_request():
    """模拟请求"""
    return "这是什么？"


@pytest.fixture
def mock_llm_response():
    """模拟 LLM 响应"""
    return LLMResponse(
        success=True,
        content="测试响应",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )


@pytest.fixture
def mock_model_registry():
    """创建空模型注册表"""
    return ModelRegistry()


@pytest.fixture
def mock_basic_strategy():
    """创建基础策略"""
    return BasicParallelStrategy()


@pytest.fixture
def mock_expert_strategy():
    """创建混合策略"""
    return HybridStrategy()


@pytest.fixture
def mock_diversified_strategy():
    """创建多样化策略"""
    return DiversifiedParallelStrategy()


# 旧 fixture 为了向后兼容
@pytest.fixture
def mock_llm():
    """创建 mock LLM"""
    mock = MagicMock(spec=BaseLLM)
    mock.chat = MagicMock(return_value=Mock(spec=LLMResponse))
    return mock


@pytest.fixture
def config():
    """加载配置"""
    return loader()
