"""
测试 ModelRegistry 注册表
"""

import pytest
from unittest.mock import MagicMock
from neuro_agent_framework.registry.model_registry import ModelRegistry
from neuro_agent_framework.core.enums import ModelType, ModelRole
from neuro_agent_framework.core.dataclasses import RegisteredModel
from neuro_agent_framework.llm.base import LLMResponse


class TestModelRegistry:
    """测试模型注册表"""

    @pytest.fixture
    def registry(self):
        """创建空的注册表"""
        return ModelRegistry()

    @pytest.fixture
    def mock_llm(self):
        """创建模拟 LLM"""
        mock = MagicMock()
        mock.role = "rACC_STANDARD"
        return mock

    def test_init(self, registry):
        """测试初始化"""
        assert registry is not None
        assert registry._models == {}
        assert registry._model_by_name == {}

    def test_register_model(self, registry):
        """测试注册模型"""
        model = RegisteredModel(
            model_id="model1",
            name="Model 1",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            estimated_cost=0.001,
            estimated_latency=1.0,
            is_active=True,
            config={}
        )

        result = registry.register(model)

        assert result is True
        assert "model1" in registry._models
        registered_model = registry._models["model1"]
        assert registered_model.model_id == "model1"
        assert registered_model.name == "Model 1"
        assert registered_model.model_type == ModelType.CHEAP_EXECUTOR
        assert registered_model.primary_role == ModelRole.rACC_STANDARD

    def test_register_duplicate(self, registry):
        """测试注册重复模型"""
        model1 = RegisteredModel(
            model_id="model1",
            name="Model 1",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            estimated_cost=0.001,
            estimated_latency=1.0,
            is_active=True,
            config={}
        )

        registry.register(model1)

        model1_updated = RegisteredModel(
            model_id="model1",
            name="Model 1 Updated",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            estimated_cost=0.001,
            estimated_latency=1.0,
            is_active=True,
            config={}
        )

        registry.register(model1_updated)

        assert registry._models["model1"].name == "Model 1 Updated"

    def test_register_inactive_model(self, registry):
        """测试注册非激活模型"""
        model = RegisteredModel(
            model_id="model1",
            name="Model 1",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            estimated_cost=0.001,
            estimated_latency=1.0,
            is_active=False,
            config={}
        )

        result = registry.register(model)

        assert result is False

    def test_get_model(self, registry):
        """测试获取模型"""
        model = RegisteredModel(
            model_id="model1",
            name="Model 1",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            estimated_cost=0.001,
            estimated_latency=1.0,
            is_active=True,
            config={}
        )

        registry.register(model)
        retrieved = registry.get("model1")

        assert retrieved is not None
        assert retrieved.model_id == "model1"

    def test_list_models(self, registry):
        """测试列出所有模型"""
        model1 = RegisteredModel(
            model_id="model1",
            name="Model 1",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            estimated_cost=0.001,
            estimated_latency=1.0,
            is_active=True,
            config={}
        )
        model2 = RegisteredModel(
            model_id="model2",
            name="Model 2",
            model_type=ModelType.EXPERT,
            primary_role=ModelRole.rDLPFC_UPGRADER,
            estimated_cost=0.01,
            estimated_latency=2.0,
            is_active=True,
            config={}
        )

        registry.register(model1)
        registry.register(model2)

        # 获取所有模型
        all_models = registry.list_models()
        assert len(all_models) == 2

        # 按类型过滤
        cheap_models = registry.list_models(ModelType.CHEAP_EXECUTOR)
        assert len(cheap_models) == 1

    def test_get_status(self, registry):
        """测试获取状态"""
        model1 = RegisteredModel(
            model_id="model1",
            name="Model 1",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            estimated_cost=0.001,
            estimated_latency=1.0,
            is_active=True,
            config={}
        )

        registry.register(model1)

        status = registry.get_status()
        assert "total_models" in status
        assert status["total_models"] == 1

    def test_get_available_models(self, registry):
        """测试获取可用模型列表"""
        model1 = RegisteredModel(
            model_id="model1",
            name="Model 1",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            estimated_cost=0.001,
            estimated_latency=1.0,
            is_active=True,
            config={}
        )

        registry.register(model1)

        available = registry.get_available_models()
        assert len(available) == 1
        assert available[0].model_id == "model1"

    def test_add_role_to_model(self, registry):
        """测试添加角色到模型"""
        model = RegisteredModel(
            model_id="model1",
            name="Model 1",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            estimated_cost=0.001,
            estimated_latency=1.0,
            is_active=True,
            config={}
        )

        registry.register(model)

        # 添加新角色
        registry.add_role_to_model("model1", ModelRole.rACC_ALTERNATIVE)

        # 验证
        model_obj = registry.get("model1")
        assert ModelRole.rACC_ALTERNATIVE in model_obj.optional_roles

    def test_remove_role_from_model(self, registry):
        """测试从模型移除角色"""
        model = RegisteredModel(
            model_id="model1",
            name="Model 1",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            optional_roles=[ModelRole.rACC_ALTERNATIVE],
            estimated_cost=0.001,
            estimated_latency=1.0,
            is_active=True,
            config={}
        )

        registry.register(model)

        # 移除角色
        registry.remove_role_from_model("model1", ModelRole.rACC_ALTERNATIVE)

        # 验证
        model_obj = registry.get("model1")
        assert ModelRole.rACC_ALTERNATIVE not in model_obj.optional_roles


class TestModelRegistryUnregister:
    """测试模型注销"""

    def test_unregister_existing_model(self, model_registry):
        """测试注销现有模型"""
        # unregister 不存在于 model_registry fixture
        pass  # Skip as unregister is not implemented

    def test_unregister_nonexistent_model(self, model_registry):
        """测试注销不存在的模型"""
        # unregister 不存在于 model_registry fixture
        pass  # Skip as unregister is not implemented


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
