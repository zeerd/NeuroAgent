"""
BasicParallelStrategy - 基础并行执行策略

特点：
- 所有模型接收相同的提示
- 真正的并行执行（使用 ThreadPoolExecutor）
- 适合 2 个执行器的简单场景
"""

import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .base_strategy import ExecutionStrategy
from neuro_agent_framework.core.dataclasses import RegisteredModel, ModelResult
from neuro_agent_framework.core.enums import ModelRole


logger = logging.getLogger(__name__)


class BasicParallelStrategy(ExecutionStrategy):
    """
    基础并行执行策略

    所有模型接收相同的提示，真正并行执行

    适合场景：
    - 快速验证
    - 简单任务
    - 2 个执行器
    """

    STANDARD_PROMPT = """
【标准方法执行者】
使用最直接的解决方法，遵循最佳实践。

用户任务：{request}
背景信息：{context}

请按标准格式输出答案。
"""

    def execute(self,
                executors: List[RegisteredModel],
                request: str,
                context: Dict,
                task_complexity: float = None) -> List[ModelResult]:
        """真正并行执行所有模型"""
        logger.info("="*70)
        logger.info("🚀 BasicParallelStrategy executing (PARALLEL)")
        logger.info(f"Number of executors: {len(executors)}")
        logger.info(f"Request: {request}...")
        logger.info(f"Context: {context}")
        logger.info(f"Task complexity: {task_complexity}")
        logger.info("="*70)

        # 构建共享的提示模板
        prompt_template = self.STANDARD_PROMPT

        def _execute_executor(model: RegisteredModel) -> ModelResult:
            """执行单个执行器的辅助函数"""
            start = time.time()
            model_name = model.name
            model_id = model.model_id

            # 构建提示
            prompt = prompt_template.format(
                request=request,
                context=context
            )

            logger.info(f"\n📍 启动执行器：{model_name}")

            from neuro_agent_framework.llm.base import Message, MessageRole

            # 构建消息
            system_msg = Message(
                role=MessageRole.SYSTEM,
                content="你是一个智能助手，请直接回答用户的问题。"
            )
            user_msg = Message(role=MessageRole.USER, content=prompt)
            messages = [system_msg, user_msg]

            # 调用 LLM
            llm_instance = model.config.get('llm_instance')
            llm_start = time.time()
            response = llm_instance.chat(messages)
            llm_latency = time.time() - llm_start

            if response.success:
                output = response.content
                logger.info(f"✓ {model_name} 响应完成")
                logger.info(f"  Tokens: {response.usage}")
                logger.info(f"  耗时：{llm_latency:.2f}s")
                confidence = 0.8 if len(output) > 100 else 0.6
            else:
                logger.error(f"✗ {model_name} 调用失败：{response.error}")
                raise RuntimeError(f"LLM execution failed: {response.error}")

            total_latency = time.time() - start

            return ModelResult(
                model_id=model_id,
                model_name=model_name,
                role=model.primary_role,
                output=output,
                latency=total_latency
            )

        # ===真正并行执行===
        results = []
        total_start = time.time()

        logger.info("\n🚀 同步启动所有执行器...")

        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=len(executors)) as executor:
            # 提交所有任务
            future_to_model = {executor.submit(_execute_executor, model): model for model in executors}

            # 收集结果
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"✅ {model.name} 完成")
                except Exception as e:
                    logger.error(f"❌ {model.name} 执行失败：{e}")
                    # 失败时创建一个错误结果
                    results.append(ModelResult(
                        model_id=model.model_id,
                        model_name=model.name,
                        role=model.primary_role,
                        output=f"错误：{str(e)}",
                        latency=0
                    ))

        total_time = time.time() - total_start

        logger.info(f"\n📍 策略执行完成：{len(results)} 个结果")
        logger.info(f"   总耗时：{total_time:.2f}s")
        logger.info(f"   平均耗时：{(total_time/len(results)):.2f}s")

        return results

    def should_diversify(self, num_models: int) -> bool:
        """基础策略不分发差异化提示"""
        return False
