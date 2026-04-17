"""
LLMBasedReviewer - LLM 驱动的评审器实现

对应：rTPJ 机制

核心功能：
1. 分析多个执行器的输出差异
2. 识别各视角的核心洞见
3. 生成综合性的最终答案
4. 评估整体一致性

类似大脑 rTPJ: 模拟对手机制，评估各方观点
"""

from typing import List, Dict, Any, Optional

from neuro_agent_framework.interfaces.reviewer import IReviewer
from neuro_agent_framework.core.dataclasses import ModelResult, RegisteredModel


class LLMBasedReviewer(IReviewer):
    """
    LLM 驱动的评审器实现

    核心功能：
    1. 分析多个执行器的输出差异
    2. 识别各视角的核心洞见
    3. 生成综合性的最终答案
    4. 评估整体一致性

    ⚠️ 注意：这是一个"已确定"的实现，使用真实 LLM 生成评分
    """

    TYPE = "llm"

    def __init__(self, model: RegisteredModel):
        """
        初始化 LLM 评审器

        Args:
            model: 用于评审的 LLM 实例
        """
        self.model = model
        import logging
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"📝 LLMBasedReviewer initialized: {model.name}")

    def review(
        self,
        results: List[ModelResult],
        request: str,
        llm: Optional[Any] = None
    ) -> Dict:
        """
        评审多个执行结果，生成综合结论

        Args:
            results: 执行结果列表
            request: 原始请求
            llm: LLM 实例用于真实评审

        Returns:
            包含综合答案、置信度、评审摘要的字典
        """
        self._logger.info("\n" + "="*70)
        self._logger.info("📝 REVIEWER: Starting review for execution results")
        self._logger.info("="*70)
        self._logger.info(f"Number of execution results: {len(results)}")
        self._logger.info(f"Original request: {request}")

        for i, result in enumerate(results):
            self._logger.info(f"Result #{i+1}:")
            self._logger.info(f"  Model: {result.model_name} ({result.role})")
            self._logger.info(f"  Text length: {len(result.output)} characters")

        review_prompt = self._build_review_prompt(results, request)

        self._logger.info("\n📝 REVIEWER: Building review prompt...")

        if llm:
            self._logger.info("\n📝 REVIEWER: Calling LLM for review...")
            llm_instance = llm if isinstance(llm, type(llm)) \
                else __import__('logging').getLogger(__name__)
            review_output = self._call_llm_review(review_prompt, llm)
        else:
            # 没有 LLM 时，基于评审结果直接生成综合答案
            self._logger.info("\n📝 REVIEWER: No LLM available, generating review based on execution results...")
            if not results:
                # 没有执行结果，基于原始请求生成回答
                review_output = f"\n【评审】\n由于没有执行器的输出结果，我直接基于原始问题生成答案。\n\n最终综合答案：\n{request}\n\n一致性评分：0.30\n\n是否需要专家升级：是"
            else:
                review_output = self._synthesize_from_results(results, request)

        self._logger.info(f"\n📝 REVIEWER: Review output preview: {review_output[:100]}...")

        combined_answer, confidence, needs_upgrade = self._parse_review_output(review_output)

        rationale = self._generate_rationale(results, review_output, needs_upgrade)

        return {
            'combined_answer': combined_answer,
            'confidence': confidence,
            'rationale': rationale,
            'needs_expert': needs_upgrade,
            'review_output': review_output
        }

    def _synthesize_from_results(self, results, request):
        """当没有 LLM 时，基于执行结果综合答案"""
        self._logger.info("Synthesizing from execution results...")
        combined = "\n---\n".join([
            f"[模型: {r.model_name} ({r.role.value})]\n{r.output[:500]}"
            for r in results
        ])
        return f"""
【评审】
基于多个执行器的结果进行综合分析。\n\n综合答案：\n{combined}\n\n一致性评分：0.50\n\n是否需要专家升级：是"""

    def _build_review_prompt(
        self,
        results: List[ModelResult],
        request: str
    ) -> str:
        """构建评审提示词"""
        self._logger.info("Building review prompt...")

        # 构建各模型输出的拼接
        outputs_summary = ""
        for i, result in enumerate(results):
            outputs_summary += f"""
--- 视角 {i+1} ({result.role}) ---
输出内容:
{result.output[:500]}...

"""

        prompt = f"""
请作为专业的评审器，对您下方的多个 AI 模型回答进行综合分析。

【原始问题】
{request}

【各模型回答摘要】
{outputs_summary}

【评审任务】
1. 分析各模型回答的核心异同
2. 找出共识和矛盾
3. 综合形成最终答案
4. 评估整体一致性（输出时给出 0-1 的置信度评分）

请严格按照以下格式输出：

【评审】
[对各个回答的详细对比分析]

最终综合答案：[综合后的完整回答]

一致性评分：0.XX

是否需要专家升级：是/否

【注意事项】
- 不要复制粘贴任何原始回答，要综合成自己的分析
- 一致性评分必须是 0 到 1 之间的数字
- 专家升级决策必须基于一致性评分和答案质量

"""
        return prompt

    def _call_llm_review(self, review_prompt: str, llm: Any) -> str:
        """
        调用 LLM 进行评审

        Args:
            review_prompt: 评审提示词
            llm: LLM 实例

        Returns:
            LLM 生成的评审结果
        """
        try:
            from neuro_agent_framework.llm.factory import LLMFactory
            if llm:
                llm_instance = llm
            else:
                llm_instance = LLMFactory.get_instance(self.model.model_id)
                if llm_instance is None:
                    llm_instance = LLMFactory.get_instance(f"{self.model.model_id}_instance")
            response = llm_instance.chat(review_prompt)

            if response.success:
                if not response.content:
                    self._logger.error(f"  LLM 评审返回空内容：model={response.model_id}, finish={response.finish_reason}")
                    raise RuntimeError("LLM review returned empty content")
                self._logger.info(f"  LLM 评审调用成功 ({len(response.content)} chars)")
                return response.content
            else:
                raise RuntimeError(f"LLM call failed: {response.error}")

        except Exception as e:
            self._logger.error(f"LLM 评审调用失败：{e}")
            raise

    def _parse_review_output(self, output: str) -> tuple:
        """
        解析评审输出

        Args:
            output: LLM 生成的评审结果

        Returns:
            (combined_answer, confidence, needs_expert)
        """
        # 解析一致性评分
        confidence = 0.5  # 默认值
        for line in output.split('\n'):
            if '一致性评分：' in line or '一致性评分 ' in line:
                try:
                    confidence = float(line.split('：')[-1].strip())
                    confidence = max(0.0, min(1.0, confidence))  # 裁剪到 0-1
                    break
                except (ValueError, IndexError):
                    pass

        # 解析专家升级决策
        needs_expert = False
        for marker in ["是否需要专家升级：", "是否需要专家升级", "专家升级：", "专家升级"]:
            if marker in output:
                part = output.split(marker)[1].split('\n')[0].strip()
                if "是" in part:
                    needs_expert = True
                break

        # 解析综合答案
        combined_answer = ""
        for marker in ["最终综合答案：", "最终综合答案"]:
            if marker in output:
                part = output.split(marker)[1].split('\n\n')[0].strip()
                combined_answer = part
                break

        return combined_answer, confidence, needs_expert

    def _generate_rationale(self, results: List[ModelResult],
                          review_output: str, needs_expert: bool) -> str:
        """生成评审理由"""
        if needs_expert:
            return "多视角存在差异，置信度低，建议专家升级"
        elif "高" in review_output:
            return "多视角一致，置信度高，无需升级"
        else:
            return "部分一致，置信度中等"

    def get_reviewer_type(self) -> str:
        """获取评审器类型"""
        return self.TYPE

    def can_review(self, num_results: int) -> bool:
        """能否评审给定数量的结果"""
        return num_results >= 2
