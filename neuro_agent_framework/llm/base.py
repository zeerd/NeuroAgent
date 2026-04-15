"""
LLM 基础定义 - Language Model Base Definitions

Define abstract interface for language model implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式（兼容 OpenAPI 格式）"""
        return {
            "role": self.role.value,
            "content": self.content
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            metadata=data.get("metadata", {})
        )


@dataclass
class LLMConfig:
    """
    LLM 配置
    
    统一的配置接口，支持不同提供商
    """
    # 基础配置
    model: str                                # 模型标识符
    
    # API 配置
    api_type: str = "openai"                  # API 类型：openai, azure, etc.
    api_base: Optional[str] = None            # API 端点 URL
    api_key: Optional[str] = None             # API 密钥
    api_version: Optional[str] = None         # API 版本（Azure 等需要）
    
    # 请求配置
    temperature: float = 0.7                  # 采样温度
    top_p: float = 1.0                        # Top-p 采样
    max_tokens: int = 4096                    # 最大生成长度
    frequency_penalty: float = 0.0            # 频率惩罚
    presence_penalty: float = 0.0             # 存在惩罚
    n: int = 1                                # 生成数量
    
    # 功能配置
    stream: bool = False                      # 是否流式输出
    tools: Optional[List[Dict[str, Any]]] = None  # 工具定义
    tool_choice: Optional[str] = None         # 工具选择策略
    
    # 超时配置
    timeout: float = 60.0                     # 请求超时（秒）
    
    def __post_init__(self):
        """验证配置"""
        if not self.model:
            raise ValueError("Model must be specified")
        
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("Top-p must be between 0 and 1")


@dataclass
class LLMResponse:
    """
    LLM 响应
    
    统一的响应格式
    """
    success: bool
    content: str
    model_id: str
    usage: Dict[str, int] = field(default_factory=dict)  # token 使用统计
    latency: float = 0.0                        # 延迟（秒）
    metadata: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None         # 结束原因
    error: Optional[str] = None                 # 错误信息（如果失败）
    
    @property
    def is_successful(self) -> bool:
        """便捷的访问方法"""
        return self.success
    
    @classmethod
    def from_error(cls, error_message: str, model_id: str = "unknown") -> 'LLMResponse':
        """从错误创建响应"""
        return cls(
            success=False,
            content="",
            model_id=model_id,
            error=error_message
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        status = "✓" if self.success else "✗"
        return f"{status} LLMResponse: {self.model_id} ({self.latency:.2f}s) - {len(self.content)} chars"


class BaseLLM(ABC):
    """
    LLM 抽象基类
    
    所有 LLM 实现必须继承此类并提供相应的接口
    """
    
    def __init__(self, config: LLMConfig):
        """
        初始化 LLM
        
        Args:
            config: LLM 配置对象
        """
        self.config = config
        self.model_id = config.model
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """
        设置方法
        
        子类实现初始化逻辑（如连接验证等）
        """
        pass
    
    @abstractmethod
    def _call(self, messages: List[Message]) -> LLMResponse:
        """
        核心调用方法
        
        Args:
            messages: 消息列表
            
        Returns:
            LLM 响应
        """
        pass
    
    @abstractmethod
    def _stream_call(self, messages: List[Message]) -> List[str]:
        """
        流式调用方法
        
        返回生成片段序列
        
        Args:
            messages: 消息列表
            
        Returns:
            片段列表
        """
        pass
    
    def chat(self, messages: List[Message]) -> LLMResponse:
        """
        聊天接口
        
        Args:
            messages: 消息列表
            
        Returns:
            响应结果
        """
        start_time = self._get_time()
        
        try:
            response = self._call(messages)
            response.latency = self._get_time() - start_time
            return response
        except Exception as e:
            latency = self._get_time() - start_time
            return LLMResponse.from_error(str(e), self.model_id)
    
    def stream_chat(self, messages: List[Message]) -> List[str]:
        """
        流式聊天接口
        
        Args:
            messages: 消息列表
            
        Returns:
            响应片段序列
        """
        start_time = self._get_time()
        
        try:
            fragments = self._stream_call(messages)
            for i, fragment in enumerate(fragments):
                yield fragment
        except Exception as e:
            logging.error(f"Stream chat error: {e}")
            yield f"[ERROR] {e}"
    
    def _get_time(self) -> float:
        """获取当前时间（秒）"""
        import time
        return time.time()
    
    def validate_connection(self) -> bool:
        """
        验证连接是否可用
        
        Returns:
            连接状态
        """
        test_message = [
            Message(role=MessageRole.USER, content="Hello, are you there?")
        ]
        response = self.chat(test_message)
        return response.success
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """是否支持流式输出"""
        pass
    
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """最大上下文长度"""
        pass
    
    @abstractmethod
    def count_tokens(self, messages: List[Message]) -> int:
        """
        计算消息 token 数量
        
        Args:
            messages: 消息列表
            
        Returns:
            token 数量
        """
        pass
