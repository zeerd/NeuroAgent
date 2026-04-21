"""
任务执行器模块

提供简单的任务执行函数，用于调用 NeuroAgent Framework
"""

import logging
from neuro_agent_framework.core.dataclasses import TaskResult, RegisteredModel
from neuro_agent_framework.core.enums import ModelType, ModelRole
from neuro_agent_framework.registry.model_registry import ModelRegistry

from ..llm.factory import LLMFactory
from ..llm.config_loader import load_llm_from_config
from ..framework.config import FrameworkConfig
from ..framework.framework import NeuroAgentFramework
from ..interfaces.impls.execution.basic_parallel_strategy import BasicParallelStrategy
from ..interfaces.impls.confidence.rule_confidence_calculator import RuleBasedConfidenceCalculator
from ..interfaces.impls.reviewer.llm_reviewer import LLMBasedReviewer


logger = logging.getLogger(__name__)


def run_task(task: str, framework: NeuroAgentFramework = None, complexity: float = 0.5):
    """
    执行一个简单任务
    
    Args:
        task: 任务描述
        framework: 可选的框架实例。如果为 None，将创建默认框架
        complexity: 任务复杂度 (0.0-1.0)
    
    Returns:
        TaskResult: 任务执行结果
    """
    try:
        # 如果没有提供框架，创建默认框架
        if framework is None:
            framework = create_default_framework()
        
        # 执行任务
        context = {'complexity': complexity}
        result = framework.execute(task, context)
        
        logger.info(f"\n=== 任务执行完成 ===")
        logger.info(f"复杂度：{complexity}")
        logger.info(f"执行器数量：{result.num_executors}")
        logger.info(f"是否使用专家：{result.used_expert}")
        logger.info(f"置信度：{result.confidence:.3f}")
        logger.info(f"总耗时：{result.total_time:.2f}秒")
        
        return result
        
    except Exception as e:
        logger.exception(f"\n❌ 任务执行失败：{e}")
        return TaskResult(
            success=False,
            combined_answer=f"错误：{str(e)}",
            confidence=0.0,
            num_executors=0,
            used_expert=False,
            total_time=0.0
        )


def create_default_framework() -> NeuroAgentFramework:
    """
    创建一个默认配置的 NeuroAgent Framework
    
    Returns:
        NeuroAgentFramework: 默认框架实例
    """
    # 初始化模型注册表
    registry = ModelRegistry()
    
    # 注册模型
    registry.register(
        model=RegisteredModel(
            model_id="chat_llama",
            name="Chat Llama 3.1 70B",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD
        )
    )
    registry.register(
        model=RegisteredModel(
            model_id="deepseek_r1",
            name="DeepSeek R1",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD
        )
    )
    registry.register(
        model=RegisteredModel(
            model_id="gpt_4o",
            name="GPT-4o",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD
        )
    )
    registry.register(
        model=RegisteredModel(
            model_id="qwen_2.5_max",
            name="Qwen 2.5 Max",
            model_type=ModelType.EXPERT,
            primary_role=ModelRole.rDLPFC_UPGRADER
        )
    )
    
    # 获取模型 - 根据 primary_role 的名称判断角色
    all_models = registry.get_available_models()
    executor_models = []
    for model in all_models:
        # rACC_*开头的角色是执行器
        if model.primary_role.value.startswith('racc_'):
            executor_models.append(model)
    
    expert_model = registry.get("qwen_2.5_max")
    
    if not executor_models or not expert_model:
        logger.warning("模型注册失败，使用空框架")
    
    # 创建组件
    strategy = BasicParallelStrategy()
    
    # LLMBasedReviewer 需要一个 model，使用 qwen_2.5_max 作为评审 LLM
    reviewer_model = registry.get("qwen_2.5_max")
    reviewer = LLMBasedReviewer(model=reviewer_model)
    
    confidence_calculator = RuleBasedConfidenceCalculator()
    
    # 创建框架
    framework = NeuroAgentFramework(
        executor_models=executor_models,
        expert_model=expert_model,
        execution_strategy=strategy,
        reviewer=reviewer,
        confidence_calculator=confidence_calculator
    )
    
    return framework
