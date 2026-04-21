"""
LLM 模块 - Language Model Module

提供统一的 LLM 接口和适配器实现。
"""

try:
    from .base import (
        BaseLLM,
        LLMConfig,
        LLMResponse,
        Message as FrameworkMessage,
        MessageRole as FrameworkMessageRole,
    )
    from .factory import (
        LLMFactory,
        LLMRegistry,
        LLMProvider,
    )
    from .openai_adapter import OpenAILLM
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import base LLM modules: {e}")
    BaseLLM = None  # type: ignore
    LLMConfig = None  # type: ignore
    LLMResponse = None  # type: ignore
    FrameworkMessage = None  # type: ignore
    FrameworkMessageRole = None  # type: ignore
    LLMFactory = None  # type: ignore

# 从 CopilotLLM 导入所有组件
from neuro_agent_framework.llm.copilot_llm_adapter import (
    COPILOT_AVAILABLE,
    CopilotLLM,
    CopilotSessionWrapper,
    CopilotMessage,
    get_async_copilot_client,
)

__all__ = [
    # 基础类型
    "BaseLLM",
    "LLMConfig",
    "LLMResponse",
    "FrameworkMessageRole",
    "FrameworkMessage",
    
    # Factory
    "LLMFactory",
    "LLMRegistry",
    "LLMProvider",
    
    # OpenAI 适配器
    "OpenAILLM",
    
    # Copilot 适配器
    "CopilotLLM",
    "CopilotMessage",
    "CopilotMessageRole",
    "CopilotBaseLLM",
    "CopilotLLMConfig",
    "CopilotLLMResponse",
    "CopilotSessionWrapper",
    "COPILOT_AVAILABLE",
    "get_async_copilot_client",
]
