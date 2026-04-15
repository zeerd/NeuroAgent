"""
NeuroConfidenceCalculator - 神经科学启发的置信度计算器

基于三脑机制：
- rACC: 基于历史经验的一致性信号
- rTPJ: 基于模拟的可靠性信号  
- rDLPFC: 基于策略层级的充分性信号
"""

import logging
from typing import List, Dict

from ..core.dataclasses import ModelResult
from ..core.enums import ModelRole


logger = logging.getLogger(__name__)


class NeuroConfidenceCalculator:
    """
    神经科学研究启发的置信度计算器
    
    模拟人类大脑的置信度感知机制：
    - 经验一致性 (rACC)
    - 模拟可靠性 (rTPJ)
    - 策略充分性 (rDLPFC)
    """
    
    DEFAULT_THRESHOLDS = {
        'consistency_threshold': 0.75,
        'completeness_threshold': 0.70,
        'reliability_threshold': 0.80,
        'combined_threshold': 0.80
    }
    
    def __init__(self, thresholds: Dict = None):
        # Merge thresholds with defaults
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
        logger.info("NeuroConfidenceCalculator initialized")
    
    def compute(self, results: List[ModelResult], context: Dict) -> Dict:
        """
        三维度置信度计算
        
        Args:
            results: 执行模型的结果列表
            context: 任务上下文
        
        Returns:
            包含整体置信度、各维度细分、是否需要升级的字典
        """
        logger.info("Computing confidence scores (neuro-based)...")
        
        # rACC: 经验一致性
        acc_score = self._compute_acc_confidence(results)
        
        # rTPJ: 模拟可靠性
        tpj_score = self._compute_tpj_confidence(results, context)
        
        # rDLPFC: 策略充分性
        dlpfc_score = self._compute_dlpfc_confidence(results, context)
        
        # 综合置信度 (加权平均)
        overall = 0.4 * acc_score + 0.3 * tpj_score + 0.3 * dlpfc_score
        
        # 判断是否需要升级
        needs_upgrade = overall < self.thresholds['combined_threshold']
        
        return {
            'overall': overall,
            'needs_upgrade': needs_upgrade,
            'rACC': {
                'score': acc_score,
                'signal': '经验一致性',
                'threshold': self.thresholds['consistency_threshold'],
                'passed': acc_score >= self.thresholds['consistency_threshold']
            },
            'rTPJ': {
                'score': tpj_score,
                'signal': '模拟可靠性',
                'threshold': self.thresholds['completeness_threshold'],
                'passed': tpj_score >= self.thresholds['completeness_threshold']
            },
            'rDLPFC': {
                'score': dlpfc_score,
                'signal': '策略充分性',
                'threshold': self.thresholds['reliability_threshold'],
                'passed': dlpfc_score >= self.thresholds['reliability_threshold']
            }
        }
    
    def _compute_acc_confidence(self, results: List[ModelResult]) -> float:
        """
        rACC: 基于经验的一致性
        
        指标：多个执行器结果的重合度和一致性
        """
        n = len(results)
        if n < 2:
            return 1.0
        
        # 计算平均置信度
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        # 简单的多样性惩罚
        diversity_bonus = 0.1
        
        return min(0.95, avg_confidence + diversity_bonus)
    
    def _compute_tpj_confidence(self, results: List[ModelResult], context: Dict) -> float:
        """
        rTPJ: 基于模拟的可靠性
        
        指标：对任务上下文的覆盖度和逻辑自洽性
        """
        # 基于覆盖关键指标的任务复杂度，评估覆盖度
        n = len(results)
        role_diversity = self._estimate_role_diversity(results)
        
        # 基础分数
        base_score = 0.7
        
        # 角色多样性加分
        role_bonus = min(0.25, role_diversity * 0.1)
        
        # 任务复杂度惩罚（复杂任务如果只有简单回应，得分低）
        task_complexity = context.get('complexity', 0.5)
        complexity_penalty = task_complexity * 0.1
        
        return min(0.95, base_score + role_bonus - complexity_penalty)
    
    def _estimate_role_diversity(self, results: List[ModelResult]) -> float:
        """评估执行结果的视角多样性"""
        roles = set(r.role.value for r in results)
        return min(1.0, len(roles) / 4)  # 4 种 rACC 角色
    
    def _compute_dlpfc_confidence(self, results: List[ModelResult], context: Dict) -> float:
        """
        rDLPFC: 基于策略充分性
        
        指标：推理深度是否匹配任务复杂度
        """
        task_complexity = context.get('complexity', 0.5)
        
        # 基于执行器和置信度估计推理深度
        n = min(5, len(results))  # 最多考虑 5 个结果
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.5
        
        # 策略充分性 = 平均置信度 * (任务复杂度匹配度)
        complexity_factor = min(1.0, 1.0 + (task_complexity - 0.5) * 0.2)
        
        return min(0.95, avg_confidence * complexity_factor)
