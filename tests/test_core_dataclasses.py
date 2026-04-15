"""
测试核心数据结构
"""

import pytest
from neuro_agent_framework.core.dataclasses import (
    RegisteredModel,
    ModelResult,
    TaskResult
)
from neuro_agent_framework.llm.base import LLMConfig, LLMResponse
from neuro_agent_framework.core.enums import ModelType, ModelRole


class TestRegisteredModel:
    """测试注册模型"""

    def test_create_minimal(self):
        """测试创建最小化模型"""
        model = RegisteredModel(
            model_id="test",
            name="Test",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD
        )

        assert model.model_id == "test"
        assert model.name == "Test"

    def test_create_all_fields(self):
        """测试创建所有字段"""
        model = RegisteredModel(
            model_id="full_test",
            name="Full Test",
            model_type=ModelType.EXPERT,
            primary_role=ModelRole.rDLPFC_UPGRADER,
            estimated_cost=0.01,
            estimated_latency=10.0,
            capabilities=["test1", "test2"],
            is_active=True,
            weight=1.0
        )

        assert model.model_id == "full_test"
        assert model.estimated_cost == 0.01
        assert model.estimated_latency == 10.0
        assert model.capabilities == ["test1", "test2"]
        assert model.is_active is True

    def test_default_values(self):
        """测试默认值"""
        model = RegisteredModel(
            model_id="test",
            name="Test",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD
        )

        assert model.estimated_cost == 0.001  # 默认是 0.001,不是 0.0
        assert model.capabilities == []  # 默认
        assert model.weight == 1.0


class TestModelResult:
    """测试模型结果"""

    def test_create_minimal(self):
        """测试创建最小化结果"""
        result = ModelResult(
            model_id="test",
            model_name="Test",
            role=ModelRole.rACC_STANDARD,
            output="Test output"
        )

        assert result.model_id == "test"
        assert result.output == "Test output"

    def test_create_with_all_fields(self):
        """测试创建所有字段"""
        result = ModelResult(
            model_id="test",
            model_name="Test",
            role=ModelRole.rACC_STANDARD,
            output="Test output",
            latency=2.5,
            confidence=0.9,
            metadata={'key': 'value'}
        )

        assert result.latency == 2.5
        assert result.confidence == 0.9
        assert result.metadata == {'key': 'value'}

    def test_metadata_default(self):
        """测试元数据默认值"""
        result = ModelResult(
            model_id="test",
            model_name="Test",
            role=ModelRole.rACC_STANDARD,
            output="Test output"
        )

        assert result.metadata == {}


class TestTaskResult:
    """测试任务结果"""

    def test_create_minimal(self):
        """测试创建最小化任务结果"""
        result = TaskResult(
            success=True,
            combined_answer="Answer",
            confidence=0.9,
            num_executors=1,
            used_expert=False,
            total_time=0.0
        )

        assert result.success is True
        assert result.combined_answer == "Answer"

    def test_create_full(self):
        """测试创建完整任务结果"""
        result = TaskResult(
            success=True,
            combined_answer="Comprehensive answer",
            confidence=0.95,
            num_executors=3,
            used_expert=True,
            total_time=5.5,
            metadata={
                'details': 'info'
            }
        )

        assert result.success is True
        assert result.num_executors == 3
        assert result.used_expert is True
        assert result.total_time == 5.5
        assert result.metadata == {'details': 'info'}
        assert result.metadata == {'details': 'info'}


class TestLLMConfig:
    """测试 LLM 配置"""

    def test_create_minimal(self):
        """测试创建最小化配置"""
        config = LLMConfig(model="test-model")

        assert config.model == "test-model"
        assert config.api_type == "openai"  # 默认
        assert config.temperature == 0.7  # 默认

    def test_create_all_fields(self):
        """测试创建所有字段"""
        config = LLMConfig(
            model="gpt-4",
            api_type="openai",
            api_base="http://api.openai.com/v1",
            api_key="sk-test",
            temperature=0.7,
            top_p=0.9,
            max_tokens=4096,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            n=1,
            stream=False,
            timeout=60.0
        )

        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 60.0

    def test_defaults(self):
        """测试默认值"""
        config = LLMConfig(model="test")
        
        assert config.api_type == "openai"
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.max_tokens == 4096  # 默认是 4096
        assert config.timeout == 60.0  # 默认 timeout


class TestLLMResponse:
    """测试 LLM 响应"""

    def test_create_success(self):
        """测试创建成功响应"""
        response = LLMResponse(
            success=True,
            content="Test response",
            model_id="test-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "total_tokens": 60
            }
        )

        assert response.success is True
        assert response.content == "Test response"
        assert response.usage['total_tokens'] == 60

    def test_create_error(self):
        """测试创建错误响应"""
        response = LLMResponse.from_error("Test error", "test-model")

        assert response.success is False
        assert response.error == "Test error"
        assert response.model_id == "test-model"

    def test_success_false(self):
        """测试成功=False 的响应"""
        response = LLMResponse(
            success=False,
            content="",
            model_id="test-model",
            error="Failed"
        )

        assert response.success is False


class TestModelTypes:
    """测试模型类型"""

    def test_all_model_types(self):
        """测试所有模型类型"""
        assert ModelType.CHEAP_EXECUTOR.name == 'CHEAP_EXECUTOR'
        assert ModelType.EXPERT.name == 'EXPERT'
        assert ModelType.CHEAP_REVIEWER.name == 'CHEAP_REVIEWER'

    def test_all_roles(self):
        """测试所有角色"""
        assert ModelRole.rACC_STANDARD.name == 'rACC_STANDARD'
        assert ModelRole.rACC_ALTERNATIVE.name == 'rACC_ALTERNATIVE'
        assert ModelRole.rACC_DIVERSE.name == 'rACC_DIVERSE'
        assert ModelRole.rACC_CRITICAL.name == 'rACC_CRITICAL'
        assert ModelRole.rDLPFC_UPGRADER.name == 'rDLPFC_UPGRADER'
        assert ModelRole.rTPJ_REVIEWER.name == 'rTPJ_REVIEWER'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
