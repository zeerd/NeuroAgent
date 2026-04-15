"""
测试 Reviewer 解析器
"""

import pytest
from unittest.mock import MagicMock
from neuro_agent_framework.reviewer.reviewer import Reviewer
from neuro_agent_framework.core.dataclasses import ModelResult
from neuro_agent_framework.core.enums import ModelRole


class TestReviewer:
    """测试评审器"""
    
    @pytest.fixture
    def model(self):
        """创建模拟模型"""
        mock = MagicMock()
        mock.model_id = "reviewer_mock"
        mock.name = "Review Mock"
        mock.model_type = "executor"
        mock.primary_role = "reviewer"
        mock.is_active = True
        mock.config = {}
        mock.optional_roles = []
        mock.estimated_cost = 0.001
        mock.estimated_latency = 1.0
        mock.capabilities = []
        return mock
    
    @pytest.fixture
    def reviewer(self, model):
        """创建 Reviewer 实例"""
        return Reviewer(model)
    
    def test_init(self, reviewer):
        """测试初始化"""
        assert reviewer is not None
        assert reviewer.model is not None
    
    def test_parse_review_with_confidence(self, reviewer):
        """测试解析带置信度的评审输出"""
        output = "评估结果：good\n置信度：0.75\n"
        combined_answer, confidence, needs_expert = reviewer._parse_review_output(output)
        
        assert confidence == 0.75
        assert needs_expert is False
    
    def test_parse_review_with_expert(self, reviewer):
        """测试解析带专家升级的评审输出"""
        output = "评估结果：需要改进\n置信度：0.6\n专家升级：是\n"
        combined_answer, confidence, needs_expert = reviewer._parse_review_output(output)
        
        assert confidence == 0.6
        assert needs_expert is True
    
    def test_parse_review_default(self, reviewer):
        """测试解析默认结果"""
        output = "简单的评审结果\n"
        combined_answer, confidence, needs_expert = reviewer._parse_review_output(output)
        
        assert confidence == 0.7  # 默认值
        assert needs_expert is False


class TestModelResult:
    """测试 ModelResult 数据类"""
    
    def test_model_result_creation(self):
        """测试 ModelResult 创建"""
        result = ModelResult(
            model_id="model1",
            model_name="Model 1",
            role=ModelRole.rACC_STANDARD,
            output="test output",
            confidence=0.75,
            latency=1.2
        )
        
        assert result.model_id == "model1"
        assert result.output == "test output"
        assert result.confidence == 0.75
        assert result.latency == 1.2
    
    def test_model_result_zero_latency(self):
        """测试零延迟"""
        result = ModelResult(
            model_id="model1",
            model_name="Model 1",
            role=ModelRole.rACC_STANDARD,
            output="test",
            confidence=0.5,
            latency=0.0
        )
        assert result.latency == 0.0


class TestImportanceCalculatorUnimplemented:
    """测试重要性计算器 - 当前不存在"""
    
    def test_importance_not_implemented(self):
        """测试重要性计算器当前不存在"""
        with pytest.raises(ImportError):
            from neuro_agent_framework.reviewer.reviewer import ImportanceCalculator


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
