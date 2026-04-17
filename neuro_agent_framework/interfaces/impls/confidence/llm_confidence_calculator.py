"""
LLMConfidenceCalculator - LLM 驱动置信度计算器
"""

from typing import List, Dict, Any
import re
import json

from neuro_agent_framework.interfaces.confidence_calculator import IConfidenceCalculator
from neuro_agent_framework.core.dataclasses import ModelResult


class LLMConfidenceCalculator(IConfidenceCalculator):
    TYPE = "llm"

    def __init__(self, model_id: str, config: Dict[str, Any] = None):
        self.model_id = model_id
        self._logger = __import__('logging').getLogger(__name__)

    def compute(self, results, context, review_result):
        prompt = self._build_analysis_prompt(results, context, review_result)
        self._logger.info(f"Calling LLM for confidence analysis...")

        import neuro_agent_framework.llm.factory as llm_factory
        llm_instance = llm_factory.LLMFactory.get_instance(self.model_id)
        if llm_instance is None:
            llm_instance = llm_factory.LLMFactory.get_instance(f"{self.model_id}_instance")
        response = llm_instance.chat(prompt)

        if not response.success:
            raise RuntimeError(f"LLM analysis failed: {response.error}")

        analysis = self._parse_llm_analysis(response.content)
        return {**analysis, 'confidence_source': {'is_data_driven': True}}

    def _build_analysis_prompt(self, results, context, review_result):
        results_summary = ""
        for i, result in enumerate(results):
            results_summary += f"--- 结果 #{i+1} ({result.role}) ---\n{result.output[:300]}\n\n"

        prompt = f"""
分析执行结果置信度。一致性评分:{review_result.get('confidence', 'N/A')}

任务复杂度：{context.get('complexity', 0.5)}
执行结果:\n{results_summary if results else '无结果'}

请输出 JSON:
{{
    "overall": 0.5-1.0,
    "needs_expert": true/false,
    "details": {{
        "consistency": {{ "score": 0-1 }},
        "quality": {{ "score": 0-1 }},
        "coverage": {{ "score": 0-1 }}
    }}
}}
"""
        return prompt

    def _parse_llm_analysis(self, output):
        try:
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    'overall': data.get('overall', 0.5),
                    'needs_expert': data.get('needs_expert', False),
                    'details': data.get('details', {'consistency': {'score': 0.5}})
                }
        except:
            pass
        return {'overall': 0.5, 'needs_expert': False, 'details': {'consistency': {'score': 0.5}}}

    def get_calculator_type(self) -> str:
        return self.TYPE

    def is_data_driven(self) -> bool:
        return True

    def can_calculate(self, num_results: int) -> bool:
        return True
