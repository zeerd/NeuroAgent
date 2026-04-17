"""
DiversifiedParallelStrategy - 多样化并行执行策略

特点：
- 根据不同模型的角色分发差异化任务提示
- 适合 3 个及以上执行器的场景
- 类似大脑 rACC: 不同情境激活不同经验
"""

import logging
from typing import List, Dict
import time

from .base_strategy import ExecutionStrategy
from neuro_agent_framework.core.dataclasses import RegisteredModel, ModelResult
from neuro_agent_framework.core.enums import ModelRole
from neuro_agent_framework.llm.base import Message, MessageRole


logger = logging.getLogger(__name__)


class DiversifiedParallelStrategy(ExecutionStrategy):
    """
    多样化并行执行策略
    
    根据每个模型的角色分发差异化任务提示
    """
    
    ROLE_PROMPTS = {
        ModelRole.rACC_STANDARD: """
【标准方法执行者】
使用最直接的解决方法，遵循最佳实践。
""",
        ModelRole.rACC_ALTERNATIVE: """
【创新方法探索者】
挑战常规做法，考虑更有创新性的解决方案。
""",
        ModelRole.rACC_DIVERSE: """
【多元化视角执行者】
从多个角度分析问题，考虑各种可能的方案。
""",
        ModelRole.rACC_CRITICAL: """
【批判性思考者】
质疑隐含假设，找出潜在的问题和漏洞。
"""
    }
    
    def execute(self, 
               models: List[RegisteredModel], 
               request: str,
               context: Dict,
               task_complexity: float = None) -> List[ModelResult]:
        """根据角色分发差异化提示执行"""
        logger.info("="*70)
        logger.info("🚀 DiversifiedParallelStrategy executing (PARALLEL)")
        logger.info(f"Number of models: {len(models)}")
        logger.info(f"Task complexity: {task_complexity}")
        logger.info("="*70)
        
        results = []
        
        for model in models:
            # 获取对应的提示模板
            prompt_template = self.ROLE_PROMPTS.get(
                model.primary_role, 
                self.ROLE_PROMPTS[ModelRole.rACC_STANDARD]
            )
            
            prompt = self._build_prompt(prompt_template, request, context)
            
            result = self._execute_model(model, prompt)
            results.append(result)
        
        return results
    
    def _build_prompt(self, template: str, request: str, context: Dict) -> str:
        """构建完整提示"""
        return f"""{template}

用户任务：{request}
背景信息：{context}

请按标准格式输出答案。
"""
    
    def _execute_model(self, model: RegisteredModel, prompt: str) -> ModelResult:
        """真正执行单个模型"""
        start = time.time()
        
        model_name = model.name
        model_id = model.model_id
        
        logger.info(f"\n📍 启动执行器：{model_name} ({model.primary_role.value})")
        
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
            confidence = 0.8 if len(output) > 100 else 0.6
            logger.info(f"✓ {model_name} 响应完成")
            logger.info(f"  Tokens: {response.usage}")
            logger.info(f"  耗时：{llm_latency:.2f}s")
        else:
            logger.error(f"✗ {model_name} 调用失败：{response.error}")
            raise RuntimeError(f"LLM execution failed: {response.error}")
        
        latency = time.time() - start
        
        return ModelResult(
            model_id=model_id,
            model_name=model_name,
            role=model.primary_role,
            output=output,
            confidence=confidence,
            latency=latency
        )
    
    def should_diversify(self, num_models: int) -> bool:
        """总是使用差异化策略"""
        return True
