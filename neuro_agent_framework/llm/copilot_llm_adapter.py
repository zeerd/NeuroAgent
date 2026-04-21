"""
CopilotLLM - GitHub Copilot SDK LLM Adapter
"""

import time
import logging
import os
from typing import List, Dict, Optional, Iterator, Any
from dataclasses import dataclass, field
from enum import Enum
from neuro_agent_framework.llm.base import LLMConfig, BaseLLM, LLMResponse, Message, MessageRole

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ===== Copilot SDK 导入 =====
COPILOT_AVAILABLE = False

try:
    from copilot import (
        CopilotClient,
        CopilotSession,
        SubprocessConfig,
        ExternalServerConfig,
    )
    from copilot.session import PermissionHandler
    COPILOT_AVAILABLE = True
    logger.info("✓ Copilot SDK imported successfully")
except ImportError as e:
    logger.warning(f"Copilot SDK import failed: {e}")
    CopilotClient = None
    CopilotSession = None
    SubprocessConfig = None



# ===== Copilot CLI Message Wrapper =====

class CopilotMessage:
    """Copilot CLI 消息格式"""
    def __init__(self, role: str, content: str, name: Optional[str] = None):
        self._setup()

    def _setup(self):
        pass

    def _call(self, messages: List[Message]) -> LLMResponse:
        raise NotImplementedError

    async def _async_call(self, messages: List[Message]) -> LLMResponse:
        return self._call(messages)

    def _stream_call(self, messages: List[Message]) -> Iterator[str]:
        raise NotImplementedError

    def chat(self, messages: List[Message]) -> LLMResponse:
        try:
            return self._call(messages)
        except Exception as e:
            return LLMResponse.from_error(str(e), self.model_id)

    def stream_chat(self, messages: List[Message]):
        try:
            for chunk in self._stream_call(messages):
                yield chunk
        except Exception as e:
            yield f"[ERROR] {e}"

    def count_tokens(self, messages: List[Message]) -> int:
        return sum(len(msg.content) for msg in messages) // 4

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def max_context_length(self) -> int:
        return 8192


class CopilotMessage:
    """Copilot SDK 消息格式"""
    def __init__(self, role: str, content: str, name: Optional[str] = None):
        self.role = role
        self.content = content
        self.name = name

    def to_dict(self) -> Dict:
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


class CopilotSessionWrapper:
    """Copilot CLI Session wrapper（每个线程独立实例）"""

    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = None
        self._session = None
        self._initialized = False
        self._created = False

    async def _connect_and_create(self):
        import asyncio
        if COPILOT_AVAILABLE:
            try:
                github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
                subprocess_config = SubprocessConfig(github_token=github_token)
                self._client = CopilotClient(subprocess_config)
                await self._client.start()
                self._session = await self._client.create_session(
                    model=self._config.model,
                    on_permission_request=PermissionHandler.approve_all,
                )
                self._initialized = True
                self._created = True
                logger.info(f"✓ Session created: {self._session.session_id[:8]}..., model={self._config.model}")
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                self._initialized = False
                self._created = False
                raise
        else:
            self._initialized = True
            self._created = True

    def is_ready(self) -> bool:
        """检查 session 是否已准备好"""
        return (hasattr(self, '_initialized') and self._initialized and
                hasattr(self, '_created') and self._created and
                hasattr(self, '_session') and self._session is not None and
                getattr(self._session, 'session_id', None) is not None)

    def close(self):
        """关闭连接"""
        self._session = None
        self._initialized = False
        self._created = False
        if self._client:
            try:
                self._client.stop()
            except:
                pass

    async def send_and_wait(self, prompt: str, timeout: float = 60.0) -> str:
        """发送消息并等待响应"""
        import asyncio
        try:
            if self._session:
                event = await self._session.send_and_wait(prompt, timeout=timeout)
                if event and hasattr(event, 'data') and hasattr(event.data, 'content'):
                    content = event.data.content if hasattr(event.data, 'content') else ""
                    return content or f"[Received response from Copilot]"
            return f"[Mock Response to: {prompt[:100]}...]"
        except TimeoutError:
            raise TimeoutError(f"Copilot CLI timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Copilot SDK error: {e}")
            raise


class CopilotLLM(BaseLLM):
    """
    GitHub Copilot CLI LLM Adapter - 线程安全版本

    关键改进:
    1. 不在 __init__ 中创建 session
    2. 每次 chat() 调用时创建独立 session
    3. 每个 Worker 线程拥有完整生命周期
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._config = config
        self._session_created = False

    def _setup(self):
        """初始化时不创建 session - 这是关键修改"""
        pass

    def _convert_messages(self, messages) -> str:
        """将消息转换为 Copilot CLI 提示"""
        if isinstance(messages, str):
            return messages
        lines = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                lines.append(f"System: {msg.content}")
            elif msg.role == MessageRole.USER:
                lines.append(f"User: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                lines.append(f"Assistant: {msg.content}")
        return "\n".join(lines)

    def _call(self, messages) -> LLMResponse:
        """每次调用创建独立 session(线程安全)"""
        import time
        import threading

        start_time = time.time()
        worker_name = threading.current_thread().name

        try:
            prompt = self._convert_messages(messages)
            logger.info(f"Worker {worker_name}: Creating new session for {self.model_id}")
            logger.info(f"Worker {worker_name}: Prompt:\n{prompt}")

            # 创建新的 session(每次调用都是独立的)
            import asyncio

            def create_and_call():
                loop = asyncio.new_event_loop()
                try:
                    wrapper = CopilotSessionWrapper(self._config)

                    # 创建 session
                    loop.run_until_complete(wrapper._connect_and_create())
                    if not wrapper.is_ready():
                        raise RuntimeError(f"Session creation failed for {self.model_id}")

                    logger.info(f"Worker {worker_name}: Session ready, sending request")

                    # 发送消息
                    result = loop.run_until_complete(wrapper.send_and_wait(prompt))

                    return result, wrapper
                finally:
                    loop.close()

            response_text, wrapper = create_and_call()

            latency = time.time() - start_time

            logger.info(f"Worker {worker_name}: Got response, latency={latency:.2f}s")
            logger.info(f"Worker {worker_name}: Response:\n{response_text}")

            return LLMResponse(
                success=True,
                content=response_text,
                model_id=self.model_id,
                latency=latency,
                usage={
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(response_text) // 4 if response_text else 0,
                    "total_tokens": max(len(prompt) // 4 + len(response_text) // 4, 1) if response_text else len(prompt) // 4
                }
            )

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Worker {worker_name}: Copilot call failed: {e}")
            return LLMResponse(
                success=False,
                content="",
                model_id=self.model_id,
                error=str(e),
                latency=latency
            )

    def _stream_call(self, messages) -> Iterator[str]:
        """Streaming interface (chunks by words)"""
        response = self._call(messages)
        if not response.success:
            yield f"[ERROR] {response.error}"
            return
        for word in response.content.split():
            yield word + " "

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def max_context_length(self) -> int:
        return 8192

    def close(self):
        """Close connection"""
        pass


async def get_async_copilot_client(config: LLMConfig):
    """Async Copilot client factory"""
    llm = CopilotLLM(config)
    return llm


__all__ = [
    "COPILOT_AVAILABLE",
    "BaseLLM",
    "LLMConfig",
    "LLMResponse",
    "Message",
    "MessageRole",
    "CopilotMessage",
    "CopilotSessionWrapper",
    "CopilotLLM",
    "get_async_copilot_client",
    "PermissionHandler",
]
