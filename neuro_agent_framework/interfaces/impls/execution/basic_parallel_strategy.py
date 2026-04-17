"""
Basic 基础并行执行策略
"""

import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from neuro_agent_framework.interfaces.execution_strategy import IExecutionStrategy
from neuro_agent_framework.core.dataclasses import RegisteredModel, ModelResult
from neuro_agent_framework.core.enums import ModelRole


logger = logging.getLogger(__name__)


class BasicParallelStrategy(IExecutionStrategy):
    """
    基础并行执行策略
    """

    TYPE = "basic_parallel"
    NAME = "basic_parallel"

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers

    def execute(
        self,
        models: List[RegisteredModel],
        request: str,
        context: Dict[str, Any],
        task_complexity: float = 0.5
    ) -> List[ModelResult]:
        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_model = {}
            for model in models:
                future = executor.submit(self._execute_model, model, request, task_complexity)
                future_to_model[future] = model

            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    result.role = model.primary_role
                    results.append(result)
                except Exception as e:
                    logger.error(f"模型执行失败：{model.name}，错误：{e}")

        execution_time = time.time() - start_time
        logger.info(f"执行完成，耗时 {execution_time:.2f}s")
        return results

    def _execute_model(self, model: RegisteredModel, request: str, task_complexity: float) -> ModelResult:
        import neuro_agent_framework.llm.factory as llm_factory
        # Try model_id first, then model_id + _instance suffix
        instance_id = model.model_id
        llm_instance = llm_factory.LLMFactory.get_instance(instance_id)
        if llm_instance is None:
            llm_instance = llm_factory.LLMFactory.get_instance(f"{instance_id}_instance")
        if llm_instance is None:
            raise RuntimeError(f"LLM instance not found for model: {model.model_id}")
        response = llm_instance.chat(request)

        if not response.success:
            raise RuntimeError(f"执行失败：{response.error}")

        return ModelResult(
            model_id=model.model_id,
            model_name=model.name,
            output=response.content,
            latency=0.5,
            confidence=0.7,
            role=model.primary_role
        )

    def get_strategy_name(self) -> str:
        return self.NAME

    def get_strategy_type(self) -> str:
        return self.TYPE

    def get_capabilities(self) -> List[str]:
        return ["parallel_execution", "consistent_prompt", "parallel"]

    def should_diversify(self, num_models: int) -> bool:
        return num_models > 3
