"""
NeuroAgentFramework - 主框架实现

整合所有模块，提供统一的服务接口
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from ..core.enums import ModelType, ModelRole
from ..core.dataclasses import RegisteredModel, ModelResult, TaskResult
from ..registry.model_registry import ModelRegistry
from ..strategy.base_strategy import ExecutionStrategy
from ..strategy.hybrid_strategy import HybridStrategy
from ..calculator.neuro_confidence import NeuroConfidenceCalculator
from ..reviewer.reviewer import Reviewer

logger = logging.getLogger(__name__)


class NeuroAgentFramework:
    """
    NeuroAgent Framework v2.0

    神经科学启发的灵活多模型协作框架

    设计理念：
    1. Harness Engineering: 多模型并行 + 评审
    2. The Advisor Strategy: 置信度驱动的专家升级
    3. 神经心理学: rACC/rTPJ/rDLPFC 三脑机制
    """

    def __init__(self,
                 model_registry: ModelRegistry,
                 execution_strategy: ExecutionStrategy = None,
                 thresholds: Dict = None):
        """初始化框架"""
        logger.info("🔧 NeuroAgent Framework initializing...")

        self.registry = model_registry
        self.strategy = execution_strategy or HybridStrategy()
        self.thresholds = thresholds or {
            'combined_threshold': 0.80,
            'consistency_threshold': 0.75,
            'completeness_threshold': 0.70,
            'reliability_threshold': 0.80
        }

        self.reviewer = None
        self.neuro_calculator = None

        # 验证配置并初始化组件
        self._validate_setup()

        logger.info("✅ NeuroAgent Framework initialized successfully")

    def _validate_setup(self):
        """验证配置有效性"""
        executors = self.registry.list_models(model_type=ModelType.CHEAP_EXECUTOR)

        if len(executors) < 2:
            raise ValueError(f"需要至少 2 个执行模型，当前只有 {len(executors)} 个")

        reviewers = self.registry.list_models(model_type=ModelType.CHEAP_REVIEWER)
        if not reviewers:
            raise ValueError("必须注册至少一个评审模型 (ModelType.CHEAP_REVIEWER)")

        # 初始化组件
        self.reviewer = Reviewer(reviewers[0])
        self.neuro_calculator = NeuroConfidenceCalculator(self.thresholds)

        logger.info(f"✓ Setup validated: {len(executors)} executors, {reviewers[0].name} as reviewer")

    def execute(self,
               request: str,
               task_context: Dict = None,
               task_complexity: float = None) -> TaskResult:
        """
        执行完整的工作流程

        包含 4 个阶段:
        1. PHASE 1: 模型并行执行
        2. PHASE 2: 评审与综合
        3. PHASE 3: 神经置信度评估
        4. PHASE 4: 决策与专家升级 (如有必要)

        Args:
            request: 用户请求
            task_context: 任务背景信息
            task_complexity: 任务复杂度 (0-1)

        Returns:
            TaskResult: 任务执行结果
        """
        import time
        start_time = time.time()

        if task_context is None:
            task_context = {}

        if task_complexity is None:
            task_complexity = 0.5

        logger.info("\n" + "="*70)
        logger.info("🚀 NEUROAGENT FRAMEWORK v2.0 - EXECUTION START")
        logger.info("="*70)
        logger.info(f"Request: {request}...")
        logger.info(f"Context: {task_context}")
        logger.info(f"Complexity: {task_complexity}")
        logger.info("="*70)

        # ===== PHASE 1: 模型并行执行 =====
        logger.info("\n📍 PHASE 1: Multi-Model Execution")
        logger.info("="*70)

        phase1_start = time.time()

        results = self.strategy.execute(self.registry.list_models(model_type=ModelType.CHEAP_EXECUTOR), request, task_context)

        phase1_duration = time.time() - phase1_start

        logger.info(f"\nPhase 1 complete: {len(results)} results in {phase1_duration:.2f}s")

        for result in results:
            logger.info(f"\n  ✓ {result.model_name} [{result.role.value}]")
            logger.info(f"      Output: {result.output}...")
            logger.info(f"      Latency: {result.latency:.2f}s")

        # ===== PHASE 2: 评审与综合 =====
        logger.info("\n📍 PHASE 2: Review and Synthesis")
        logger.info("="*70)

        phase2_start = time.time()

        # 获取 reviewer 的 LLM 实例
        reviewer_llm = None
        for model in self.registry.list_models(model_type=ModelType.CHEAP_REVIEWER):
            reviewer_llm = model.config.get('llm_instance')
            break

        review_result = self.reviewer.review(results, request, reviewer_llm)

        phase2_duration = time.time() - phase2_start

        logger.info(f"\nReview complete: confidence={review_result['confidence']:.2f}")
        logger.info(f"Combined answer: {review_result['combined_answer']}...")

        # ===== PHASE 3: 神经置信度评估 =====
        logger.info("\n📍 PHASE 3: Neuro Confidence Assessment")
        logger.info("="*70)

        phase3_start = time.time()

        # 调用置信度计算器
        task_complexity_context = {'complexity': task_complexity}
        confidence_breakdown = self.neuro_calculator.compute(results, task_complexity_context)

        phase3_duration = time.time() - phase3_start

        logger.info("\nConfidence Assessment (Neuro-Centric):")
        logger.info(f"  Overall: {confidence_breakdown['overall']:.2f}")
        logger.info(f"  Needs upgrade: {confidence_breakdown['needs_upgrade']}")

        for component, details in confidence_breakdown.get('details', {}).items():
            logger.info(f"\n  {component} ({details['name']}): {details['score']:.2f}")
            logger.info(f"      Threshold: {details['threshold']:.2f}")
            logger.info(f"      Passed: {details['passed']}")

        # ===== PHASE 4: 专家升级 =====
        needs_expert = confidence_breakdown['needs_upgrade']

        if needs_expert:
            logger.info("\n📍 PHASE 4: Quality Assessment and Upgrade Decision")
            logger.info("="*70)
            logger.info("⚠️  Expert upgrade required - calling expert model...")

            phase4_start = time.time()

            # 调用专家模型
            expert_results = self._call_expert_model(request, results)

            phase4_duration = time.time() - phase4_start

            logger.info("\nExpert complete in " + f"{phase4_duration:.2f}s")
            logger.info(f"Expert output: {expert_results[0].output}...")

            # 重新评估 - 传递 reviewer 的 LLM
            final_review = self.reviewer.review(expert_results, request, reviewer_llm)
            logger.info(f"\nFinal review: confidence={final_review['confidence']:.2f}")

        total_time = time.time() - start_time

        # 返回结果
        return TaskResult(
            success=True,
            combined_answer=review_result['combined_answer'],
            confidence=review_result['confidence'],
            num_executors=len(results),
            used_expert=needs_expert,
            total_time=total_time,
            metadata={
                'phase1_duration': phase1_duration,
                'phase2_duration': phase2_duration,
                'phase3_duration': phase3_duration,
                'confidence_breakdown': confidence_breakdown,
                'total_steps': 4 if needs_expert else 3
            }
        )

    def _call_expert_model(self, request: str, results: List[ModelResult]) -> List[ModelResult]:
        """调用专家模型"""
        import time

        logger.info(f"\n[Expert] Calling expert model for request: {request}...")

        start_time = time.time()

        # 获取专家模型
        expert_models = self.registry.list_models(model_type=ModelType.EXPERT)
        if not expert_models:
            logger.warning("No expert model registered!")
            return []

        expert = expert_models[0]
        llm_instance = expert.config.get('llm_instance')

        if not llm_instance:
            logger.warning("No LLM instance found for expert model!")
            return []

        # 构建专家提示
        expert_prompt = f"""请作为专家对以下问题进行分析和解答。

【原始请求】
{request}

【执行器的初步分析结果】
"""
        for r in results:
            expert_prompt += f"\n---\n模型：{r.model_name} ({r.role.value})\n{r.output}..."

        expert_prompt += "\n\n请给出专业、深入的分析和解答。"

        # 调用 LLM
        from neuro_agent_framework.llm.base import Message, MessageRole
        messages = [
            Message(role=MessageRole.SYSTEM, content="你是一位专家级 AI，需要提供专业、深入的分析和解答。"),
            Message(role=MessageRole.USER, content=expert_prompt)
        ]

        response = llm_instance.chat(messages)

        latency = time.time() - start_time

        logger.info(f"[Expert] LLM call complete: {latency:.2f}s")
        # 处理 token 统计
        if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
            token_count = response.usage.total_tokens
            logger.info(f"[Expert] Tokens: {token_count}")
        else:
            token_count = 0

        return [ModelResult(
            model_id=expert.model_id,
            model_name=expert.name,
            role=expert.primary_role,
            output=response.content,
            latency=latency
        )]
