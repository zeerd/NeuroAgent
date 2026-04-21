"""
LLMBasedReviewer - LLM 驱动的评审器实现

角色标识：rTPJ_REVIEWER（内部命名）

核心功能：
1. 分析多个执行器的输出差异
2. 识别各视角的核心洞见
3. 生成综合性的最终答案
4. 评估整体一致性

⚠️ 注意：这是一个"已确定"的实现，使用真实 LLM 生成评分
"""

from typing import List, Dict, Any, Optional

from neuro_agent_framework.interfaces.reviewer import IReviewer
from neuro_agent_framework.core.dataclasses import ModelResult, RegisteredModel
from neuro_agent_framework.core.template_loader import get_template_loader
from neuro_agent_framework.llm import FrameworkMessage, FrameworkMessageRole
from neuro_agent_framework.llm.base import MessageRole


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
        self._logger = logging.getLogger("reviewer")

    def review(self, results: List[ModelResult], request: str, llm: Any = None) -> Dict[str, Any]:
        """
        评审多个执行结果，生成综合结论

        Args:
            results: 执行结果列表
            request: 原始请求问题
            llm: 可选的 LLM 实例，如果为 None 则使用 self.model

        Returns:
            包含综合答案、置信度、评审摘要的字典

        Raises:
            RuntimeError: 当缺少执行器结果或 LLM 调用失败时抛出
        """
        self._logger.info("📝 REVIEWER: Starting review for execution results")
        self._logger.info(f"   Number of execution results: {len(results)}")

        if len(results) == 0:
            # 没有执行器结果，必须抛出异常
            raise RuntimeError("Reviewer requires at least one execution result to review")

        # 构建评审提示词
        review_prompt = self._build_review_prompt(results, request)
        self._logger.info("\n📝 REVIEWER: Building review prompt...")

        # 调用 LLM 进行评审
        self._logger.info("\n📝 REVIEWER: Calling LLM for review...")
        review_output = self._call_llm_review(review_prompt, llm)
        self._logger.info(f"LLM review done ({len(review_output)} chars)")

        # 解析评审输出
        combined_answer, confidence, needs_upgrade = self._parse_review_output(review_output)

        rationale = self._generate_rationale(results, review_output, needs_upgrade)

        return {
            'combined_answer': combined_answer,
            'confidence': confidence,
            'needs_expert': needs_upgrade,
            'rationale': rationale
        }

    def _build_review_prompt(self, results: List[ModelResult], request: str) -> str:
        """构建评审提示词 - 从模板文件加载"""
        self._logger.info("Building review prompt from template...")

        # 构建各模型输出的拼接
        outputs_summary = ""
        for i, result in enumerate(results):
            outputs_summary += f"""
--- 视角 {i+1} ({result.role}) ---
输出内容:
{result.output}

"""

        # 从模板文件加载提示词
        loader = get_template_loader()
        prompt = loader.load_template(
            "reviewer",
            variables={
                "原始问题": request,
                "各模型回答摘要": outputs_summary
            }
        )
        self._logger.info(f"Loaded prompt from template, length: {len(prompt)}")
        return prompt

    def _call_llm_review(self, review_prompt: str, llm: Any) -> str:
        """
        调用 LLM 进行评审

        Args:
            review_prompt: 评审提示词
            llm: LLM 实例

        Returns:
            LLM 生成的评审结果

        Raises:
            RuntimeError: 当 LLM 调用失败或返回空内容时抛出
        """
        from neuro_agent_framework.llm.factory import LLMFactory
        if llm:
            llm_instance = llm
        else:
            llm_instance = LLMFactory.get_instance(self.model.model_id)
            if llm_instance is None:
                llm_instance = LLMFactory.get_instance(f"{self.model.model_id}_instance")

        response = llm_instance.chat([FrameworkMessage(role=FrameworkMessageRole.USER, content=review_prompt)])

        if response.success:
            if not response.content:
                self._logger.error(f"LLM 评审返回空内容：model={response.model_id}, finish={response.finish_reason}")
                raise RuntimeError("LLM review returned empty content")
            self._logger.info(f"LLM 评审调用成功 ({len(response.content)} chars)")
            return response.content
        else:
            raise RuntimeError(f"LLM call failed: {response.error}")

    def _parse_review_output(self, output: str) -> tuple:
        """
        解析评审输出

        Args:
            output: 包含评审结果的字符串

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
