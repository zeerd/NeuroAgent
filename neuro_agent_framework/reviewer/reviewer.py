"""
Reviewer - 评审器模块

对应：rTPJ 机制
功能：对多个执行结果进行综合评审和合成
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.dataclasses import ModelResult, RegisteredModel
from ..core.enums import ModelType


logger = logging.getLogger(__name__)


class Reviewer:
    """
    评审器 - 对执行结果进行综合和评审

    核心功能：
    1. 分析多个执行器的输出差异
    2. 识别各视角的核心洞见
    3. 生成综合性的最终答案

    类似大脑 rTPJ: 模拟对手机制，评估各方观点
    """

    def __init__(self, model: RegisteredModel):
        self.model = model
        logger.info(f"📝 Reviewer initialized: {model.name}")

    def review(self, results: List[ModelResult], request: str, llm: Optional[Any] = None) -> Dict:
        """
        评审多个执行结果，生成综合结论

        Args:
            results: 执行结果列表
            request: 原始请求
            llm: LLM 实例用于真实评审

        Returns:
            包含综合答案、置信度、评审摘要的字典

        Raises:
            RuntimeError: 如果未提供 llm 参数
        """
        logger.info("\n" + "="*70)
        logger.info("📝 REVIEWER: Starting review for execution results")
        logger.info("="*70)
        logger.info(f"Number of execution results: {len(results)}")
        logger.info(f"Original request: {request}...")
        logger.info(f"Number of messages: {len(results)}")

        # 显示各执行结果摘要
        for i, result in enumerate(results):
            logger.info(f"Result #{i+1}:")
            logger.info(f"  Model: {result.model_name} ({result.role.value})")
            logger.info(f"  Latency: {result.latency:.2f}s")
            logger.info(f"  Text: {result.output}...")

        # 构建评审提示
        review_prompt = self._build_review_prompt(results, request)

        logger.info("\n📝 REVIEWER: Building review prompt...")
        logger.info(f"Review prompt preview: {review_prompt}...")

        # 调用 LLM 进行评审（如果提供了）
        if llm:
            logger.info("\n📝 REVIEWER: Calling LLM for review...")
            review_output, review_latency, call_success = self._call_llm_review(review_prompt, llm)
        else:
            raise RuntimeError("Reviewer requires an LLM instance to generate review output. Please pass 'llm' parameter to review() method.")

        logger.info(f"\n📝 REVIEWER: Review output preview: {review_output}...")
        logger.info(f"📝 REVIEWER: Review complete in {review_latency:.2f}s")
        logger.info("="*70)

        # 解析评审结果
        combined_answer, confidence, needs_upgrade = self._parse_review_output(review_output)

        rationale = self._generate_rationale(results, review_output, needs_upgrade)

        return {
            'combined_answer': combined_answer,
            'confidence': confidence,
            'rationale': rationale,
            'review_latency': review_latency,
            'reviewer_model': self.model.name,
            'needs_expert': needs_upgrade,
            'review_output': review_output,
            'review_with_llm': call_success
        }

    def _build_review_prompt(self, results: List[ModelResult], request: str) -> str:
        """构建完整的评审提示"""
        role_summaries = []
        for r in results:
            role_name = r.role.value.replace('_', ' ')
            # 使用完整输出而不是摘要
            key_sentence = r.output
            role_summaries.append(f"**{r.model_name}** ({role_name}):\n{key_sentence}")

        prompt = f"""【评审任务】分析多个视角并给出综合建议

【原始请求】
{request}

【执行结果】

""" + '\n---\n'.join(role_summaries) + f"""

【分析要求】
1. 分析各视角的核心差异和一致性
2. 如果答案差异很大或置信度低，标记 '需要专家升级'
3. 给出置信度评分（0.0-1.0）
4. 给出最终综合答案

请按照以下格式输出：
置信度：XX.XX (等级：低/中/高)
综合答案：(你的综合答案)
专家升级：是/否
分析：简要说明评审理由"""

        return prompt

    def _call_llm_review(self, prompt: str, llm: Any) -> tuple:
        """调用 LLM 进行评审"""
        from neuro_agent_framework.llm.base import Message, MessageRole
        import time

        logger.info("📝 REVIEWER: LLM 调用详情")
        logger.info("📝 REVIEWER: Creating system message...")
        logger.info("📝 REVIEWER: Creating user message...")

        system_msg = Message(
            role=MessageRole.SYSTEM,
            content="""你是评审专家，能够：
1. 分析多视角答案的一致性和差异
2. 给出准确的置信度评分（0.0-1.0）
3. 判断是否需要专家升级

置信度分级规则：
- 高 (0.8+): 多视角一致，无需专家
- 中 (0.6-0.8): 部分一致，视情况决定
- 低 (<0.6): 差异大，必须专家升级"""
        )

        logger.info(f"System message length: {len(system_msg.content)}")
        logger.info(f"User message length: {len(prompt)}")

        user_msg = Message(role=MessageRole.USER, content=prompt)

        full_prompt = f"System:\n{system_msg.content}\n\nUser:\n{user_msg.content}"
        logger.info(f"Full prompt total length: {len(full_prompt)}")

        # 准备消息列表用于 LLM 调用
        messages = [system_msg, user_msg]

        # 调用 LLM
        logger.info(f"[OPENAI] Sending request to model: {llm.model_id}")
        logger.info(f"[OPENAI] Total messages: {len(messages)}")

        llm_start = time.time()

        response = llm.chat(messages)

        llm_latency = time.time() - llm_start

        if response.success:
            logger.info(f"[OPENAI] Model: {response.model_id}")
            logger.info(f"[OPENAI] Tokens: prompt={response.usage.get('prompt_tokens', 0)}, completion={response.usage.get('completion_tokens', 0)}, total={response.usage.get('total_tokens', 0)}")
            logger.info(f"[OPENAI] Latency: {llm_latency:.2f}s")
            logger.info(f"[OPENAI] Response received: {len(response.content)} chars")

            return response.content, llm_latency, True
        else:
            logger.error(f"[OPENAI] Call failed: {response.error}")
            raise RuntimeError(f"LLM call failed: {response.error}")

    def _parse_review_output(self, output: str) -> tuple:
        """解析评审输出"""
        logger.info("📝 REVIEWER: Parsing review output...")

        confidence = 0.70
        combined_answer = "基于多视角分析的综合结论。"
        needs_expert = False

        # 尝试提取置信度
        if "置信度：" in output:
            try:
                conf_part = output.split("置信度：")[1].split("\n")[0].strip()
                # 提取数字
                import re
                match = re.search(r'([\d.]+)', conf_part)
                if match:
                    confidence = float(match.group(1))
                    logger.info(f"  Extracted confidence: {confidence}")
            except Exception as e:
                logger.error(f"Error parsing confidence: {e}")

        # 提取专家升级
        if "专家升级" in output:
            part = output.split("专家升级")[1].split("\n")[0].strip()
            needs_expert = "是" in part
            logger.info(f"  Expert upgrade needed: {needs_expert}")

        # 提取综合答案
        if "综合答案：" in output:
            part = output.split("综合答案：")[1].split("\n")[0].strip()
            combined_answer = part
            logger.info(f"  Extracted answer: {combined_answer}...")

        logger.info(f"📝 REVIEWER: Parsed results - confidence: {confidence}, needs_expert: {needs_expert}")

        return combined_answer, confidence, needs_expert

    def _generate_rationale(self, results: List[ModelResult], review_output: str, needs_expert: bool) -> str:
        """生成评审理由"""
        if needs_expert:
            return "多视角存在差异，置信度低，建议专家升级"
        elif "高" in review_output:
            return "多视角一致，置信度高，无需升级"
        else:
            return "部分一致，置信度中等"
