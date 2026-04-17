#!/usr/bin/env python3
"""
LLM 模块快速开始示例

这个脚本展示如何使用 NeuroAgent 框架的 LLM 功能
"""

import os
import sys
from pathlib import Path
from neuro_agent_framework.llm.base import (
    LLMConfig, Message, MessageRole
)
from neuro_agent_framework.llm.factory import LLMFactory


# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """快速开始示例"""

    print("\n" + "=" * 60)
    print("NeuroAgent LLM 模块 - 快速开始")
    print("=" * 60)

    # 步骤 1: 准备 API key
    api_key = os.getenv("OPENAI_API_KEY", "sk-your-api-key")

    if api_key == "sk-your-api-key":
        print("\n⚠  提示：设置 API key")
        print("export OPENAI_API_KEY=sk-your-key-here\n")

    # 步骤 2: 创建 LLM 配置
    print("\n步骤 1: 创建 LLM 配置")
    print("-" * 60)

    config = LLMConfig(
        model="gpt-3.5-turbo",  # GPT-3.5 Turbo
        temperature=0.7,  # 生成温度 (0-2)
        max_tokens=1000,  # 最大生成长度
        api_key=api_key  # API 密钥
    )

    print("  ✓ 配置创建完成")
    print(f"    - Model: {config.model}")
    print(f"    - Temperature: {config.temperature}")
    print(f"    - Max tokens: {config.max_tokens}")

    # 步骤 3: 创建 LLM 实例
    print("\n步骤 2: 创建 LLM 实例")
    print("-" * 60)

    llm = LLMFactory.create("openai", config, "quickstart")
    print("  ✓ LLM 实例创建成功")
    print(f"    - Model ID: {llm.model_id}")
    print(f"    - Supports streaming: {llm.supports_streaming}")

    # 步骤 4: 准备消息
    print("\n步骤 3: 准备对话消息")
    print("-" * 60)

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="你是一位专业、有帮助的智能助手。请用中文回答。"
        ),
        Message(
            role=MessageRole.USER,
            content="你好！请介绍一下你自己。"
        )
    ]

    print(f"  ✓ 消息列表已准备（{len(messages)}条消息）")

    # 步骤 5: 调用 LLM
    print("\n步骤 4: 调用 LLM")
    print("-" * 60)

    response = llm.chat(messages)

    if response.success:
        print("  ✓ 调用成功")
        print(f"    - Model: {response.model_id}")
        print(f"    - Latency: {response.latency:.2f}s")
        print(f"    - Tokens: {response.usage}")
        print(f"    - Finish reason: {response.finish_reason}")
        print()
        print("  输出内容:")
        print("  " + "-" * 56)
        for line in response.content.split('\n'):
            print(f"  {line}")
        print("  " + "-" * 56)
    else:
        print("  ✗ 调用失败")
        print(f"    Error: {response.error}")

    # 步骤 6: 关闭连接
    print("\n步骤 5: 清理资源")
    print("-" * 60)

    llm.close()
    print("✓ LLM 实例已关闭")

    print("\n" + "=" * 60)
    print("✅ 快速开始完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
