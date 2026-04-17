"""
LLM 工厂类

提供统一接口来创建和管理 LLM 实例
"""

import logging
from typing import Dict, Type, Optional, Any

from .base import BaseLLM, LLMConfig
from .openai_adapter import OpenAILLM

logger = logging.getLogger(__name__)


class LLMProvider:
    """LLM 提供商标识"""
    name: str
    class_path: str

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'LLMProvider':
        """从字典创建提供者"""
        return cls(
            name=data.get("name", "unknown"),
            class_path=data.get("class_path", "")
        )


class LLMRegistry:
    """
    LLM 实现注册表

    注册不同提供商的 LLM 实现，支持动态扩展
    """

    def __init__(self):
        self._providers: Dict[str, Type[BaseLLM]] = {}
        self._instances: Dict[str, BaseLLM] = {}
        self._register_default_providers()

    def _register_default_providers(self):
        """注册默认提供者"""
        self.register_provider("openai", OpenAILLM)
        logger.info(f"✓ Registered {len(self._providers)} default providers")

    def register_provider(self, name: str, llm_class: Type[BaseLLM]):
        """注册新的 LLM 提供者

        Args:
            name: 提供者名称
            llm_class: LLM 实现类
        """
        if name in self._providers:
            logger.warning(f"Provider '{name}' already exists, overwriting...")

        self._providers[name] = llm_class
        logger.info(f"✓ Registered provider: {name}")

    def get_provider(self, name: str) -> Type[BaseLLM]:
        """获取提供者类"""
        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}")

        return self._providers[name]

    def list_providers(self) -> list:
        """列出所有注册的提供者"""
        return list(self._providers.keys())

    def has_provider(self, name: str) -> bool:
        """检查是否存在提供者"""
        return name in self._providers


class LLMFactory:
    """
    LLM 工厂类

    提供统一的 LLM 创建和管理接口
    """

    _registry: LLMRegistry = None
    _instances: Dict[str, BaseLLM] = {}

    @classmethod
    def get_registry(cls) -> LLMRegistry:
        """获取 LLM 注册表"""
        if cls._registry is None:
            cls._registry = LLMRegistry()
        return cls._registry

    @classmethod
    def register_provider(cls, name: str, llm_class: Type[BaseLLM]):
        """注册新的提供者

        Args:
            name: 提供者名称
            llm_class: LLM 实现类
        """
        registry = cls.get_registry()
        registry.register_provider(name, llm_class)

    @classmethod
    def create(
        cls,
        provider: str,
        config: LLMConfig,
        instance_id: Optional[str] = None
    ) -> BaseLLM:
        """
        创建 LLM 实例

        Args:
            provider: 提供者名称 (如 "openai")
            config: LLM 配置
            instance_id: 实例 ID, 用于缓存

        Returns:
            BaseLLM 实例

        Raises:
            ValueError: 如果提供者不存在
        """
        registry = cls.get_registry()

        if provider not in registry._providers:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {registry.list_providers()}"
            )

        llm_class = registry._providers[provider]

        # 如果启用了实例缓存
        if instance_id and instance_id in cls._instances:
            # 检查是否可以使用缓存实例
            cached = cls._instances[instance_id]
            if (cached.config.model == config.model
                    and cached.config.api_base == config.api_base):
                logger.info(f"✓ Reusing cached instance: {instance_id}")
                return cached

        # 创建新实例
        llm = llm_class(config)

        # 缓存新实例
        if instance_id:
            cls._instances[instance_id] = llm

        logger.info(f"✓ Created {provider} LLM instance: {config.model}")
        return llm

    @classmethod
    def create_from_config(
        cls, config: Dict[str, Any],
        instance_id: Optional[str] = None
    ) -> BaseLLM:
        """
        从配置字典创建 LLM 实例

        Args:
            config: 配置字典
            instance_id: 实例 ID

        Returns:
            BaseLLM 实例
        """
        provider = config.get("provider", "openai")

        # 构建 LLMConfig
        llm_config = LLMConfig(
            model=config.get("model", "gpt-3.5-turbo"),
            api_type=config.get("api_type", "openai"),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            api_version=config.get("api_version"),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 1.0),
            max_tokens=config.get("max_tokens", 4096),
            frequency_penalty=config.get("frequency_penalty", 0.0),
            presence_penalty=config.get("presence_penalty", 0.0),
            n=config.get("n", 1),
            stream=config.get("stream", False),
            timeout=config.get("timeout", 60.0)
        )

        return cls.create(provider, llm_config, instance_id)

    @classmethod
    def get_instance(cls, instance_id: str) -> Optional[BaseLLM]:
        """获取已缓存的实例"""
        return cls._instances.get(instance_id)

    @classmethod
    def close_instance(cls, instance_id: str):
        """关闭并移除缓存实例"""
        if instance_id in cls._instances:
            instance = cls._instances[instance_id]
            instance.close()
            del cls._instances[instance_id]
            logger.info(f"✗ Closed cached instance: {instance_id}")


# 全局实例
factory = LLMFactory()
