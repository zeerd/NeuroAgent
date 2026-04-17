#!/usr/bin/env python3
"""
LLM 模块使用示例

展示如何使用 NeuroAgent 框架的 LLM 功能
"""

# import sys  # noqa: F401 - unused import kept for compatibility
import os  # needed for os.getenv

# 添加项目路径

from neuro_agent_framework.llm.base import LLMConfig, Message, MessageRole
from neuro_agent_framework.llm.factory import LLMFactory


def create_simple_llm():
    """创建简单的 LLM 实例"""
    api_key = os.getenv("OPENAI_API_KEY", "sk-your-api-key-here")

    config = LLMConfig(
        model="gpt-3.5-turbo",  # 可选：gpt-3.5-turbo, gpt-4o, gpt-4o-mini
        temperature=0.7,
        max_tokens=2000,
        api_key=api_key
    )

    llm = LLMFactory.create("openai", config, "example_01")

    # 准备消息
    messages = [
        Message(role=MessageRole.SYSTEM, content="你是一位有帮助的智能助手。"),
        Message(role=MessageRole.USER, content="请介绍一下你自己。"),
    ]

    # 调用 LLM
    response = llm.chat(messages)

    if response.success:
        print("\n[响应]")
        print(f"  模型：{response.model_id}")
        print(f"  延迟：{response.latency:.2f}s")
        print(f"  Token 使用：{response.usage}")
        print(f"  Token 使用：{response.usage}")
        print("\n  " + response.content)
    else:
        print(f"\n[错误]{response.error}")

    llm.close()
    return response


def create_azure_llm():
    """创建 Azure OpenAI 实例"""
    config = LLMConfig(
        model="gpt-35-turbo",  # Azure 的模型 ID
        api_type="azure",
        api_base="https://your-resource-name.openai.azure.com",
        api_key="your-api-key",
        api_version="2024-02-01",
        temperature=0.7,
        max_tokens=1000
    )

    llm = LLMFactory.create("openai", config, "azure_example")

    messages = [
        Message(role=MessageRole.USER, content="你好"),
    ]

    response = llm.chat(messages)
    print(f".Azure OpenAI response: {response.content}")

    llm.close()
    return response


def stream_chat_example():
    """展示流式聊天功能"""
    api_key = os.getenv("OPENAI_API_KEY", "sk-your-api-key-here")

    config = LLMConfig(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=api_key,
        stream=True
    )

    llm = LLMFactory.create("openai", config, "stream_example")

    messages = [
        Message(role=MessageRole.SYSTEM, content="用简短的句子回答问题。"),
        Message(role=MessageRole.USER, content="什么是人工智能？请逐步解释。"),
    ]

    print("\n[流式输出]")
    chunks = []
    for chunk in llm.stream_chat(messages):
        chunks.append(chunk)
        print(chunk, end="", flush=True)

    print(f"\n\n完整响应：{''.join(chunks)}")
    llm.close()


def show_llm_registry():
    """展示 LLM 注册表信息"""
    registry = LLMFactory.get_registry()

    print("\n[LLM 注册表]")
    print(f"  已注册提供者：{registry.list_providers()}")

    for provider_name in registry.list_providers():
        provider_class = registry.get_provider(provider_name)
        print(f"\n  {provider_name}:")
        print(f"    类：{provider_class.__name__}")
        print(f"    模块：{provider_class.__module__}")
        has_attr = hasattr(provider_class.supports_streaming, 'fget')
        if has_attr:
            stream_support = provider_class.supports_streaming.fget(None)
        else:
            stream_support = 'N/A'
        print(f"    支持流式：{stream_support}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("NeuroAgent LLM 模块使用示例")
    print("="*60)

    # 获取 API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("\n⚠  提示：设置 OPENAI_API_KEY 环境变量以获得真实 API 响应")
        print("export OPENAI_API_KEY=sk-your-key-here\n")

    print("\n" + "="*60)
    print("示例 1: 创建并使用 OpenAI LLM")
    print("="*60)
    _ = create_simple_llm()  # noqa: F841

    print("\n" + "="*60)
    print("示例 2: 查看 LLM 注册表")
    print("="*60)
    show_llm_registry()

    print("\n" + "="*60)
    print("示例 3: 流式聊天")
    print("="*60)
    if api_key:
        stream_chat_example()
    else:
        print("\n⚠  需要真实 API key 才能演示流式聊天")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
