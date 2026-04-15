"""
测试配置文件加载器
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock

from neuro_agent_framework.llm.config_loader import (
    ConfigLoader,
    _resolve_env_vars,
    _resolve_env_vars_in_dict,
    load_llm_from_config,
    LLMConfig
)

from neuro_agent_framework.llm.base import BaseLLM



class TestResolveEnvVars:
    """测试环境变量解析"""

    @pytest.fixture(autouse=True)
    def setup_env(self):
        """清理环境变量"""
        original = dict(os.environ)
        yield
        os.environ.clear()
        os.environ.update(original)

    def test_simple_placeholder(self):
        """测试简单占位符"""
        os.environ["TEST_VAR"] = "test_value"
        result = _resolve_env_vars("${TEST_VAR}")
        assert result == "test_value"

    def test_placeholder_with_default(self):
        """测试带默认值的占位符"""
        result = _resolve_env_vars("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"

    def test_placeholder_no_default(self):
        """测试无默认值的占位符"""
        result = _resolve_env_vars("${NONEXISTENT_VAR}")
        assert result == ""

    def test_no_placeholder(self):
        """测试无占位符"""
        result = _resolve_env_vars("static_value")
        assert result == "static_value"

    def test_multiple_placeholders(self):
        """测试多个占位符"""
        os.environ["VAR1"] = "value1"
        os.environ["VAR2"] = "value2"
        result = _resolve_env_vars("${VAR1}:${VAR2}")
        assert result == "value1:value2"


class TestResolveEnvVarsInDict:
    """测试字典环境变量解析"""

    def test_simple_dict(self):
        """测试简单字典"""
        test_dict = {
            "key": "${TEST_VAR}",
            "value": "static"
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.json")
            os.environ["TEST_VAR"] = "test_value"
            with open(test_file, "w") as f:
                json.dump({"TEST_VAR": "${TEST_VAR}"}, f)
            
            with open(test_file, "r") as f:
                test_dict = json.load(f)
            
            result = _resolve_env_vars_in_dict(test_dict)
            assert result["TEST_VAR"] == "test_value"

    def test_nested_dict(self):
        """测试嵌套字典"""
        test_dict = {
            "outer": {
                "inner": "${TEST_VAR}"
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.json")
            os.environ["TEST_VAR"] = "inner_value"
            with open(test_file, "w") as f:
                json.dump({"TEST_VAR": "${TEST_VAR}"}, f)
            
            with open(test_file, "r") as f:
                test_dict = json.load(f)
            
            result = _resolve_env_vars_in_dict(test_dict)
            assert result["TEST_VAR"] == "inner_value"


class TestConfigLoaderLoad:
    """测试 ConfigLoader 加载"""

    def test_load_config(self):
        """测试配置加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "models": {
                        "test-model": {
                            "config": {
                                "model": "test-model",
                                "api_type": "ollama"
                            }
                        }
                    }
                }, f)

            config_loader = ConfigLoader(config_path)
            result = config_loader.load()

            assert result is True
            assert config_loader._loaded is True

    def test_load_with_env_var_resolution(self):
        """测试环境变量解析"""
        os.environ["TEST_VAR"] = "resolved_value"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "models": {
                        "test-model": {
                            "config": {
                                "model": "${TEST_VAR}",
                                "api_type": "ollama"
                            }
                        }
                    }
                }, f)

            config_loader = ConfigLoader(config_path)
            config_loader.load()

            assert config_loader._loaded is True


class TestConfigLoaderGetConfig:
    """测试 ConfigLoader 获取配置"""

    def test_get_existing_model(self):
        """测试获取现有模型配置"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "models": {
                        "test-model": {
                            "config": {
                                "model": "test-model",
                                "api_type": "ollama",
                                "temperature": 0.5,
                                "top_p": 0.9
                            }
                        }
                    }
                }, f)

            config_loader = ConfigLoader(config_path)
            config_loader.load()
            config = config_loader.get_config("test-model")

            assert config is not None
            assert config.model == "test-model"
            assert config.api_type == "ollama"

    def test_get_nonexistent_model(self):
        """测试获取不存在的模型"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            with open(config_path, "w") as f:
                json.dump({"models": {}}, f)

            config_loader = ConfigLoader(config_path)
            config_loader.load()
            config = config_loader.get_config("nonexistent")

            assert config is None


class TestConfigLoaderGetAvailableModels:
    """测试获取可用模型"""

    def test_get_available_models(self):
        """测试获取可用模型列表"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "models": {
                        "model1": {},
                        "model2": {},
                        "model3": {}
                    }
                }, f)

            config_loader = ConfigLoader(config_path)
            config_loader.load()
            models = config_loader.get_available_models()

            assert len(models) == 3
            assert "model1" in models
            assert "model2" in models


class TestLoadLLMFromConfig:
    """测试从配置创建 LLM"""

    def test_load_success(self):
        """测试成功加载 LLM 配置 - 使用 mock_patch 覆盖创建过程"""
        import unittest.mock as mock
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "models": {
                        "test-model": {
                            "config": {
                                "model": "test-model",
                                "api_type": "ollama",
                                "temperature": 0.5,
                                "top_p": 0.9
                            }
                        }
                    }
                }, f)

            config_loader = ConfigLoader(config_path)
            config_loader.load()
            
            # Mock LLMFactory.create 来避免实际连接
            with mock.patch('neuro_agent_framework.llm.config_loader.LLMFactory.create') as mock_create:
                mock_llm = MagicMock(spec=BaseLLM)
                mock_llm.model_id = "instance1"
                mock_create.return_value = mock_llm
                
                llm = load_llm_from_config(config_loader, "test-model", "instance1")

                assert llm is not None
                assert llm.model_id == "instance1"
                assert mock_create.call_count == 1

    def test_load_nonexistent_model(self):
        """测试加载不存在的模型"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            with open(config_path, "w") as f:
                json.dump({"models": {}}, f)

            config_loader = ConfigLoader(config_path)
            config_loader.load()
            llm = load_llm_from_config(config_loader, "nonexistent", "instance1")

            assert llm is None


class TestLLMConfig:
    """测试 LLMConfig 数据类"""

    def test_create_config(self):
        """测试创建配置"""
        config = LLMConfig(
            model="ollama",
            api_type="ollama",
            temperature=0.7
        )

        assert config.model == "ollama"
        assert config.api_type == "ollama"
        assert config.temperature == 0.7

    def test_config_validation(self):
        """测试配置验证"""
        with pytest.raises(ValueError):
            LLMConfig(
                model="",
                api_type="ollama"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
