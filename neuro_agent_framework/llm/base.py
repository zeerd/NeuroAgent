"""
LLM 基础定义 - Language Model Base Definitions

Define abstract interface for language model implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class MessageRole(Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """单条消息"""
    role: MessageRole
    content: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式（兼容 OpenAPI 格式）"""
        return {
            "role": self.role.value,
            "content": self.content
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """从字典创建消息"""
        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content", "")
        )


@dataclass
class LLMConfig:
    """
    LLM 配置类

    用于定义 LLM 的基本参数
    """
    model: str
    api_type: str = "openai"
    api_base: str = None
    api_key: str = None
    api_version: str = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    stream: bool = False
    timeout: float = 60.0  # API call timeout in seconds
    role: str = None  # Framework-specific role (rACC_STANDARD, rDLPFC_upgrader, etc.)
    # Additional OpenAI-specific parameters
    tools: Any = None
    tool_choice: Any = None
    functions: Any = None
    logit_bias: Dict = field(default_factory=dict)
    user: str = ""
    parallel_tool_calls: bool = True  # OpenAI parameter
    estimated_cost: float = 0.0
    estimated_latency: float = 0.0
    capabilities: List[str] = field(default_factory=list)

    def __post_init__(self):
        """验证配置参数"""
        if not self.model:
            raise ValueError("Model must be specified")
        # Allow openai, copilot, and 'ollama' (Ollama's OpenAI-compatible API)
        if self.api_type not in ["openai", "copilot", "ollama"]:
            raise ValueError(f"Invalid api_type: {self.api_type}. Must be one of 'openai', 'copilot', or 'ollama'")
        if not (0 <= self.temperature <= 2):
            raise ValueError("Temperature must be between 0 and 2")
        if not (0 <= self.top_p <= 1):
            raise ValueError("Top-p must be between 0 and 1")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")


@dataclass
class LLMResponse:
    """
    LLM 响应

    用于保存 LLM 调用的结果
    """
    success: bool
    content: str
    model_id: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency: float = 0.0
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None
    finish_reason: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        """检查调用是否成功"""
        return self.success

    @classmethod
    def from_error(cls, error_message: str, model_id: str = "unknown") -> 'LLMResponse':
        """创建错误响应"""
        return cls(
            success=False,
            content="",
            model_id=model_id,
            error=error_message
        )


class BaseLLM(ABC):
    """
    抽象基类

    定义语言模型的基本接口
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model_id = config.model
        self._setup()

    def _setup(self):
        """初始化时的自定义设置"""
        pass

    @abstractmethod
    def _call(self, messages: List[Message]) -> LLMResponse:
        """
        内部调用方法

        Args:
            messages: 消息列表

        Returns:
            LLMResponse 对象
        """
        pass

    async def _async_call(self, messages: List[Message]) -> LLMResponse:
        """异步调用（默认同步）"""
        return self._call(messages)

    @abstractmethod
    def _stream_call(self, messages: List[Message]):
        """
        流式调用（生成器）

        Args:
            messages: 消息列表

        Yields:
            字符串块
        """
        pass

    def chat(self, messages: List[Message]) -> LLMResponse:
        """
        发送消息并获得响应

        Args:
            messages: 消息列表

        Returns:
            LLMResponse 对象
        """
        try:
            return self._call(messages)
        except Exception as e:
            return LLMResponse.from_error(str(e), self.model_id)

    async def chat_async(self, messages: List[Message]) -> LLMResponse:
        """异步发送消息并获得响应"""
        try:
            return await self._async_call(messages)
        except Exception as e:
            return LLMResponse.from_error(str(e), self.model_id)

    def stream_chat(self, messages: List[Message]):
        """
        流式发送消息

        Args:
            messages: 消息列表

        Yields:
            字符串块
        """
        try:
            for chunk in self._stream_call(messages):
                yield chunk
        except Exception as e:
            yield f"[ERROR] {e}"

    def count_tokens(self, messages: List[Message]) -> int:
        """估算 token 数量"""
        return sum(len(msg.content) for msg in messages) // 4

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """是否支持流式接口"""
        pass

    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """最大上下文长度"""
        pass

    def close(self):
        """关闭连接"""
        pass
