"""
配置文件加载器

从 JSON 配置文件加载 LLM 配置
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .base import LLMConfig, Message, MessageRole
from .factory import LLMFactory

logger = logging.getLogger(__name__)


def _resolve_env_vars(value: str) -> str:
    """
    解析环境变量占位符，如 ${VAR:-default}

    Args:
        value: 可能包含占位符的字符串

    Returns:
        解析后的值
    """
    import re

    if not isinstance(value, str):
        return value

    def replacer_with_default(match):
        """替换 ${VAR:-default} 格式"""
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default)

    def replacer_simple(match):
        """替换 ${VAR} 格式"""
        var_name = match.group(1)
        return os.environ.get(var_name, '')

    # 先处理带默认值格式 ${VAR:-default}
    # 使用更精确的正则：变量名后跟 :-
    value = re.sub(r'\$\{([^:]+):-([^}]*)\}', replacer_with_default, value)

    # 再处理简单格式 ${VAR}
    value = re.sub(r'\$\{([^}]+)\}', replacer_simple, value)

    return value


def _resolve_env_vars_in_dict(d: Dict) -> Dict:
    """
    递归解析字典中的所有环境变量占位符

    Args:
        d: 字典

    Returns:
        替换后的字典
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, str):
            result[k] = _resolve_env_vars(v)
        elif isinstance(v, dict):
            result[k] = _resolve_env_vars_in_dict(v)
        elif isinstance(v, list):
            result[k] = [_resolve_env_vars(item) if isinstance(item, str) else item for item in v]
        else:
            result[k] = v
    return result


class ConfigLoader:
    """
    配置文件加载器

    从 JSON 文件加载 LLM 配置，支持：
    - 模型配置加载
    - 提供商配置加载
    - 策略配置加载
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，默认为项目根目录下的 config/llm_config.json
        """
        if config_path is None:
            # 默认搜索路径
            config_paths = [
                Path(__file__).parent.parent / "config" / "llm_config.json",
                Path("config/llm_config.json"),
                Path("../config/llm_config.json"),
            ]
            for path in config_paths:
                if path.exists():
                    self.config_path = str(path)
                    break
            else:
                raise FileNotFoundError("找不到配置文件")
        else:
            self.config_path = config_path

        self._loaded = False
        self._config: Dict[str, Any] = {}

    def load(self, force: bool = False) -> bool:
        """
        加载配置文件

        Args:
            force: 是否强制重新加载

        Returns:
            是否成功加载
        """
        if self._loaded and not force:
            return True

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)

            # 解析环境变量
            self._config = _resolve_env_vars_in_dict(self._config)

            self._loaded = True
            logger.info(f"配置加载成功：{self.config_path}")
            return True
        except Exception as e:
            logger.error(f"配置加载失败：{e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return self._config.get(key, default)

    def get_config(self, model_key: str) -> Optional[LLMConfig]:
        """
        获取模型配置

        Args:
            model_key: 模型键

        Returns:
            LLMConfig 对象或 None
        """
        if not self._loaded:
            logger.warning("配置尚未加载")
            return None

        model_config = self._config.get("models", {}).get(model_key)
        if not model_config:
            logger.warning(f"未找到模型配置：{model_key}")
            return None

        # 提取配置
        config_dict = model_config.get("config", {})

        # 创建 LLMConfig
        return LLMConfig(
            model=config_dict.get("model"),
            api_type=config_dict.get("api_type"),
            api_base=config_dict.get("api_base"),
            api_key=config_dict.get("api_key"),
            api_version=config_dict.get("api_version"),
            temperature=config_dict.get("temperature"),
            top_p=config_dict.get("top_p"),
            max_tokens=config_dict.get("max_tokens"),
            frequency_penalty=config_dict.get("frequency_penalty"),
            presence_penalty=config_dict.get("presence_penalty"),
            n=config_dict.get("n"),
            stream=config_dict.get("stream"),
            timeout=config_dict.get("timeout"),
        )

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """
        获取模型信息

        Args:
            model_key: 模型键

        Returns:
            模型信息字典或 None
        """
        if not self._loaded:
            logger.warning("配置尚未加载")
            return None

        return self._config.get("models", {}).get(model_key)

    def get_available_models(self) -> list:
        """
        获取所有可用模型的 key 列表

        Returns:
            模型 key 列表
        """
        if not self._loaded:
            logger.warning("配置尚未加载")
            return []

        return list(self._config.get("models", {}).keys())

    def model_exists(self, model_key: str) -> bool:
        """
        检查模型是否配置

        Args:
            model_key: 模型键

        Returns:
            是否配置
        """
        if not self._loaded:
            return False

        return model_key in self._config.get("models", {})


def load_llm_from_config(config_loader: ConfigLoader,
                         model_key: str,
                         instance_id: str) -> Optional[BaseLLM]:
    """
    从配置创建 LLM 实例

    Args:
        config_loader: 配置加载器
        model_key: 模型键
        instance_id: 实例 ID

    Returns:
        BaseLLM 实例或 None
    """
    config = config_loader.get_config(model_key)
    if config:
        config.model = config.model
        # 创建 LLM 实例
        llm = LLMFactory.create("openai", config, instance_id)
        return llm
    return None


def create_llm_from_json(config_path: str,
                         models: Dict[str, Any],
                         instance_id: str,
                         provider: str = "openai") -> LLMFactory:
    """
    从 JSON 配置创建 LLM 实例

    Args:
        config_path: 配置路径
        models: 模型字典
        instance_id: 实例 ID
        provider: 提供商类型

    Returns:
        LLMFactory 实例
    """
    loader = ConfigLoader(config_path)
    return LLMFactory(loader=loader)


# 别名，为了向后兼容
create_llm_from_config = create_llm_from_json
