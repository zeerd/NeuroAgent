"""
NeuroAgent Framework - 核心框架 v2

演示接口驱动的设计模式
"""

import logging
from typing import List, Dict, Any, Type
from datetime import datetime

from ..interfaces.execution_strategy import IExecutionStrategy
from ..interfaces.reviewer import IReviewer
from ..interfaces.confidence_calculator import IConfidenceCalculator
from ..core.dataclasses import TaskResult, ModelResult, RegisteredModel


logger = logging.getLogger(__name__)


class NeuroAgentFramework:
    """
    NeuroAgent Framework 核心类 v2

    核心改进：
    - 使用接口而非具体类
    - 支持运行时配置各组件
    - 清晰的数据流设计
    """

    def __init__(
        self,
        executor_models: List[RegisteredModel],
        expert_model: RegisteredModel,
        execution_strategy: IExecutionStrategy,
        reviewer: IReviewer,
        confidence_calculator: IConfidenceCalculator
    ):
        """
        初始化框架

        Args:
            executor_models: 执行器模型列表
            expert_model: 专家模型
            execution_strategy: 执行策略（接口）
            reviewer: 评审器（接口）
            confidence_calculator: 置信度计算器（接口）
        """
        self.executor_models = executor_models
        self.expert_model = expert_model

        self.execution_strategy = execution_strategy
        self.reviewer = reviewer
        self.confidence_calculator = confidence_calculator

        logger.info(f"✅ NeuroAgent Framework V2 initialized")
        logger.info(f"   执行器：{len(executor_models)} 个")
        logger.info(f"   策略类型：{execution_strategy.get_strategy_type()}")
        logger.info(f"   评审器：{reviewer.get_reviewer_type()}")
        logger.info(f"   置信度计算器：{confidence_calculator.get_calculator_type()}")

    def _get_llm_for_reviewer(self):
        """为评审器获取 LLM 实例"""
        # 优先使用 reviewer 自己的 model
        if hasattr(self.reviewer, 'model') and self.reviewer.model and self.reviewer.model.config:
            llm = self.reviewer.model.config.get('llm_instance')
            if llm:
                return llm
        # 回退：从执行器/专家模型获取
        for model in self.executor_models:
            if hasattr(model, 'config') and model.config and 'llm_instance' in model.config:
                llm = model.config['llm_instance']
                if llm:
                    return llm
        if hasattr(self.expert_model, 'config') and self.expert_model.config and 'llm_instance' in self.expert_model.config:
            return self.expert_model.config.get('llm_instance')
        return None

    def execute(self, request: str, context: Dict[str, Any] = None) -> TaskResult:
        """
        执行任务的核心流程

        1. 使用策略执行所有模型
        2. 使用 Reviewer 评审
        3. 使用 ConfidenceCalculator 评估
        4. 决定是否升级专家
        """
        if context is None:
            context = {}

        start_time = datetime.now()
        complexity = context.get('complexity', 0.5)

        logger.info("\n" + "="*80)
        logger.info("🚀 EXECUTION START")
        logger.info(f"  任务：{request}")
        logger.info(f"  复杂度：{complexity}")
        logger.info("="*80)

        # ===== PHASE 1: 执行策略 =====
        logger.info("\n📍 PHASE 1: Execution")
        results = self.execution_strategy.execute(
            self.executor_models,
            request,
            context,
            complexity
        )
        logger.info(f"Phase 1 completed: {len(results)} results")

        # ===== PHASE 2: 评审合成 =====
        logger.info("\n📍 PHASE 2: Review & Synthesis")
        review_result = self.reviewer.review(results, request, self._get_llm_for_reviewer())
        logger.info(f"  Reviewer confidence: {review_result['confidence']}")

        # ===== PHASE 3: 置信度评估 =====
        logger.info("\n📍 PHASE 3: Confidence Assessment")
        confidence_result = self.confidence_calculator.compute(
            results,
            {'complexity': complexity},
            review_result
        )
        logger.info(f"  Overall: {confidence_result['overall']:.2f}")
        logger.info(f"  Needs expert: {confidence_result['needs_expert']}")

        # ===== PHASE 4: 专家升级决策 =====
        needs_expert = confidence_result['needs_expert']
        used_expert = False

        if needs_expert:
            logger.info("\n⚡ TRIGGERING EXPERT UPGRADE")
            logger.info(f"  使用专家模型：{self.expert_model.name}")

            # 调用专家执行
            from ..llm.factory import LLMFactory
            from ..llm.base import Message, MessageRole
            expert_llm = LLMFactory.get_instance(self.expert_model.model_id)
            if expert_llm is None:
                expert_llm = LLMFactory.get_instance(f"{self.expert_model.model_id}_instance")
            if expert_llm:
                messages = [Message(role=MessageRole.USER, content=request)]
                expert_response = expert_llm.chat(messages)

                if expert_response.success:
                    answers = review_result['combined_answer'] + "\n\n---\n专家回复:\n" + expert_response.content

                    used_expert = True
                    logger.info(f"  专家完成，tokens: {expert_response.usage}")
                else:
                    logger.error(f"  专家执行失败：{expert_response.error}")
                    answers = review_result['combined_answer']  # 回退到 Reviewer 答案
            else:
                logger.error("  无法获取专家 LLM 实例")
                answers = review_result['combined_answer']
        else:
            logger.info("\n✅ No expert upgrade needed")
            answers = review_result['combined_answer']

        total_time = (datetime.now() - start_time).total_seconds()

        return TaskResult(
            success=True,
            combined_answer=answers,
            confidence=confidence_result['overall'],
            num_executors=len(results),
            used_expert=used_expert,
            total_time=total_time,
            metadata={
                'execution_strategy': self.execution_strategy.get_strategy_type(),
                'reviewer_type': self.reviewer.get_reviewer_type(),
                'conf_calculator_type': self.confidence_calculator.get_calculator_type(),
                'reviewer_confidence': review_result['confidence'],
                'task_complexity': complexity
            }
        )
