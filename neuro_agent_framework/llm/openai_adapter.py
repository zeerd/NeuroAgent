"""
OpenAI SDK 适配器

使用 OpenAI 官方 SDK 调用兼容接口
支持：
- OpenAI 官方 API
- Azure OpenAI Service
- LocalAI, Ollama, vLLM 等兼容服务
"""

import logging
import os
import time
from typing import List, Dict, Any, Iterator, Optional

from openai import OpenAI, AzureOpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .base import BaseLLM, LLMConfig, LLMResponse, Message, MessageRole

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """
    OpenAI SDK 兼容 LLM 实现

    基于官方 openai SDK，支持：
    - OpenAI 官方 API (GPT-3.5, GPT-4 等)
    - Azure OpenAI Service
    - 任何兼容 OpenAI Chat API 的本地服务（Ollama, LM Studio, vLLM 等）
    """

    def __init__(self, config: LLMConfig):
        """
        初始化 OpenAI LLM

        Args:
            config: LLM 配置对象
        """
        self._client: Optional[OpenAI] = None
        self._azure: bool = False
        self._initialized = False

        super().__init__(config)

    def _setup(self):
        """初始化 HTTP 客户端"""
        # 自动检测 API 密钥来源
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY", "")

        # 检测到本地网络 IP，使用 dummy key
        if not api_key:
            raise ValueError("API key is required for remote services")
        else:
            logger.info(f"Using API key: {'*' * min(8, len(api_key))}...")

        self.config.api_key = api_key

        # 检测是否为 Azure
        if "api.azure.com" in (self.config.api_base or ""):
            self._azure = True
            logger.info("Using Azure OpenAI")

            self._client = AzureOpenAI(
                api_key=self.config.api_key,
                api_version=self.config.api_version or "2024-02-01",
                azure_endpoint=self.config.api_base
            )
        else:
            self._azure = False
            logger.info("Using OpenAI-compatible API")

            self._client = OpenAI(
                base_url=self.config.api_base,
                api_key=self.config.api_key
            )

        logger.info(f"✓ OpenAILLM initialized with model: {self.model_id}")
        logger.info(f"  API Base: {self.config.api_base or 'https://api.openai.com/v1'}")
        logger.info(f"  Azure: {self._azure}")

        self._initialized = True

    def _to_openai_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """转换消息格式到 OpenAI 格式"""
        return [msg.to_dict() for msg in messages]

    def _call(self, messages: List[Message]) -> LLMResponse:
        """
        同步调用 OpenAI API

        Args:
            messages: 消息列表

        Returns:
            LLM 响应
        """
        if not self._initialized:
            raise RuntimeError("LLM not initialized")

        if not self._client:
            raise RuntimeError("Client not initialized")

        logger.info(f"[OPENAI] Sending request to model: {self.model_id}")
        logger.info(f"[OPENAI] Messages: {len(messages)} messages")
        for i, msg in enumerate(messages):
            logger.info(f"[OPENAI] Message {i} [{msg.role.value}]: {msg.content}...")

        # 构建请求参数
        params = {
            "model": self.model_id,
            "messages": self._to_openai_messages(messages),
            "temperature": self.config.temperature if self.config.temperature else 0.7,
            "top_p": self.config.top_p if self.config.top_p else 1.0,
            "max_tokens": self.config.max_tokens,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "n": self.config.n,
            "stream": False
        }

        # 添加工具定义（如果提供）
        if self.config.tools:
            params["tools"] = self.config.tools

        if self.config.tool_choice:
            params["tool_choice"] = self.config.tool_choice

        start_time = time.time()

        try:
            response: ChatCompletion = self._client.chat.completions.create(**params)

            latency = time.time() - start_time

            logger.info(f"[OPENAI] Model: {response.model}")
            logger.info(f"[OPENAI] Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}")
            logger.info(f"[OPENAI] Finish reason: {response.choices[0].finish_reason}")
            logger.info(f"[OPENAI] Latency: {latency:.2f}s")

            content = response.choices[0].message.content or ""
            logger.info(f"[OPENAI] Response sent to framework: {len(content)} chars")

            return LLMResponse(
                success=True,
                content=content,
                model_id=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason
            )

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"[OPENAI] API call failed after {latency:.2f}s: {e}")
            return LLMResponse.from_error(str(e), self.model_id)

    async def _async_call(self, messages: List[Message]) -> LLMResponse:
        """
        异步调用 OpenAI API

        Args:
            messages: 消息列表

        Returns:
            LLM 响应
        """
        if not self._initialized:
            raise RuntimeError("LLM not initialized")

        if not self._client:
            raise RuntimeError("Client not initialized")

        # 构建请求参数
        params = {
            "model": self.model_id,
            "messages": self._to_openai_messages(messages),
            "temperature": self.config.temperature or 0.7,
            "top_p": self.config.top_p or 1.0,
            "max_tokens": self.config.max_tokens,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "n": self.config.n,
            "stream": False
        }

        if self.config.tools:
            params["tools"] = self.config.tools

        if self.config.tool_choice:
            params["tool_choice"] = self.config.tool_choice

        try:
            response: ChatCompletion = await self._client.chat.completions.create(**params)

            return LLMResponse(
                success=True,
                content=response.choices[0].message.content or "",
                model_id=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason
            )

        except Exception as e:
            logger.error(f"Async API call failed: {e}")
            return LLMResponse.from_error(str(e), self.model_id)

    def _stream_call(self, messages: List[Message]) -> Iterator[str]:
        """
        流式调用 OpenAI API

        Args:
            messages: 消息列表

        Yields:
            生成片段
        """
        if not self._initialized:
            raise RuntimeError("LLM not initialized")

        if not self._client:
            raise RuntimeError("Client not initialized")

        # 构建请求参数
        params = {
            "model": self.model_id,
            "messages": self._to_openai_messages(messages),
            "temperature": self.config.temperature or 0.7,
            "top_p": self.config.top_p or 1.0,
            "max_tokens": self.config.max_tokens,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "n": self.config.n,
            "stream": True
        }

        if self.config.tools:
            params["tools"] = self.config.tools

        if self.config.tool_choice:
            params["tool_choice"] = self.config.tool_choice

        try:
            stream: Stream[ChatCompletionChunk] = self._client.chat.completions.create(**params)

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Stream API call failed: {e}")
            yield f"[ERROR] {e}"

    def count_tokens(self, messages: List[Message]) -> int:
        """
        计算 token 数量（估算）

        Args:
            messages: 消息列表

        Returns:
            估计的 token 数量
        """
        # 简单估算：每 4 个字符 ≈ 1 个 token
        total_chars = sum(len(msg.content) for msg in messages)
        return max(1, total_chars // 4)

    @property
    def supports_streaming(self) -> bool:
        """支持流式输出"""
        return True

    @property
    def max_context_length(self) -> int:
        """最大上下文长度"""
        if self.model_id in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"]:
            return 16384
        elif self.model_id in ["gpt-4", "gpt-4-32k", "gpt-4o", "gpt-4o-mini"]:
            return 128000
        return 8192  # 默认值

    def close(self):
        """关闭客户端连接"""
        if self._client:
            # OpenAI SDK 自动管理连接，不需要显式关闭
            pass
        self._initialized = False


# 向后兼容别名
BaseOpenAILLM = OpenAILLM
