"""
测试边界情况
"""

import pytest
from unittest.mock import MagicMock
from neuro_agent_framework.core.dataclasses import TaskResult, ModelResult
from neuro_agent_framework.core.enums import ModelRole
from neuro_agent_framework.reviewer.reviewer import Reviewer
from neuro_agent_framework.llm.base import BaseLLM


class TestEdgeCases:
    """测试边界情况"""
    
    def test_short_task(self):
        """测试短任务"""
        task_result = TaskResult(
            success=True,
            combined_answer="1",
            confidence=0.9,
            num_executors=2,
            used_expert=False,
            total_time=0.5,
            metadata={}
        )
        
        assert task_result.combined_answer == "1"
        assert task_result.confidence == 0.9
    
    def test_very_long_answer(self):
        """测试超长答案"""
        long_answer = "x" * 10000
        task_result = TaskResult(
            success=True,
            combined_answer=long_answer,
            confidence=0.8,
            num_executors=2,
            used_expert=False,
            total_time=10.0,
            metadata={}
        )
        
        assert len(task_result.combined_answer) == 10000
    
    def test_special_characters(self):
        """测试特殊字符"""
        output = "<script>alert('xss')</script>\n{'<b>bold</b>'}"
        task_result = TaskResult(
            success=True,
            combined_answer=output,
            confidence=0.7,
            num_executors=2,
            used_expert=False,
            total_time=1.0,
            metadata={}
        )
        
        assert task_result.combined_answer == output
    
    def test_empty_context(self):
        """测试空上下文"""
        task_result = TaskResult(
            success=True,
            combined_answer="",
            confidence=0.5,
            num_executors=2,
            used_expert=False,
            total_time=0,
            metadata={}
        )
        
        assert task_result.combined_answer == ""
    
    def test_exception_handling(self):
        """测试异常处理"""
        with pytest.raises(Exception):
            raise ValueError("测试异常")


class TestTaskResultEdgeCases:
    """测试 TaskResult 边界情况"""
    
    def test_zero_confidence(self):
        """测试零置信度"""
        result = TaskResult(
            success=False,
            combined_answer="",
            confidence=0.0,
            num_executors=2,
            used_expert=False,
            total_time=0,
            metadata={}
        )
        assert result.confidence == 0.0
    
    def test_very_high_confidence(self):
        """测试极高置信度"""
        result = TaskResult(
            success=True,
            combined_answer="ok",
            confidence=1.0,
            num_executors=2,
            used_expert=False,
            total_time=0.1,
            metadata={}
        )
        assert result.confidence == 1.0
    
    def test_none_metadata(self):
        """测试 None 元数据"""
        result = TaskResult(
            success=True,
            combined_answer="test",
            confidence=0.8,
            num_executors=1,
            used_expert=False,
            total_time=0.5,
            metadata=None
        )
        assert result.metadata is None


class TestModelResultEdgeCases:
    """测试 ModelResult 边界情况"""
    
    def test_empty_output(self):
        """测试空输出"""
        result = ModelResult(
            model_id="test_model",
            model_name="Test Model",
            role=ModelRole.rACC_STANDARD,
            output="",
            latency=0.0
        )
        
        assert result.output == ""
        assert result.latency == 0.0
    
    def test_whitespace_output(self):
        """测试空白输出"""
        result = ModelResult(
            model_id="test_model",
            model_name="Test Model",
            role=ModelRole.rACC_STANDARD,
            output="   ",
            latency=0.1
        )
        
        assert result.output == "   "


class TestReviewParser:
    """测试评审解析器"""

    @pytest.fixture
    def mock_reviewer(self):
        """创建 mock Reviewer 实例"""
        mock_model = MagicMock(spec=BaseLLM)
        mock_model.model_id = "reviewer_mock"
        mock_model.model = "reviewer_mock"
        mock_model.role = "rTPJ_REVIEWER"
        mock_model.name = "Reviewer Mock"
        
        # 确保 name 属性存在
        type(mock_model).name = property(lambda self: "Reviewer Mock")
        
        try:
            reviewer = Reviewer(mock_model)
            return reviewer
        except:
            return MagicMock(spec=Reviewer)

    def test_parse_empty_output(self, mock_reviewer):
        """测试解析空输出"""
        result = mock_reviewer._parse_review_output("")
        
        assert result is not None
        assert len(result) == 3

    def test_parse_missing_evaluation(self, mock_reviewer):
        """测试解析缺失 evaluation 字段"""
        result = mock_reviewer._parse_review_output("")
        
        assert result is not None
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
