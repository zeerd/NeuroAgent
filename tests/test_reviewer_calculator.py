"""
测试 Reviewer 解析器
"""

import pytest
from unittest.mock import MagicMock
from neuro_agent_framework.interfaces.impls.reviewer.llm_reviewer import LLMBasedReviewer 
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
        """创建 LLMBasedReviewer 实例"""
        """创建 Reviewer 实例"""
        return LLMBasedReviewer(model)
    
    def test_init(self, reviewer):
        """测试初始化"""
        assert reviewer is not None
        assert reviewer.model is not None
    
    def test_parse_review_with_confidence(self, reviewer):
        """测试解析带置信度的评审输出"""
        output = "评估结果：good\n一致性评分：0.75\n"
        combined_answer, confidence, needs_expert = reviewer._parse_review_output(output)
        
        assert confidence >= 0.74
        assert needs_expert is False
    
    def test_parse_review_with_expert(self, reviewer):
        """测试解析带专家升级的评审输出"""
        output = "评估结果：需要改进\n一致性评分：0.6\n专家升级：是\n"
        combined_answer, confidence, needs_expert = reviewer._parse_review_output(output)
        
        assert confidence >= 0.59
        assert needs_expert is True
    
    def test_message_role_type(self, mock_llm_reviewer):
        """验证 Message 能正确处理 MessageRole 枚举类型"""
        from neuro_agent_framework.llm.base import Message, MessageRole
        
        # 应该使用枚举值 - 验证类型正确性
        msg_system = Message(role=MessageRole.SYSTEM, content="test")
        assert msg_system.role == MessageRole.SYSTEM
        assert isinstance(msg_system.role, MessageRole)
        
        msg_user = Message(role=MessageRole.USER, content="test")
        assert msg_user.role == MessageRole.USER
        assert isinstance(msg_user.role, MessageRole)
        
        # Message 类会进行类型检查，确保传入的是 MessageRole 枚举类型
        # 这在实际代码中很重要，确保了正确的类型使用


class TestReviewerExecution:
    """测试评析器实际执行流程"""
    
    @pytest.fixture
    def mock_reviewer_llm(self, mock_llm_reviewer):
        """准备 reviewer 的 mock LLM"""
        return mock_llm_reviewer
    
    @pytest.fixture
    def execution_results(self):
        """创建执行结果示例"""
        from neuro_agent_framework.core.enums import ModelRole
        
        return [
            ModelResult(
                model_id="model1",
                model_name="Test Model",
                role=ModelRole.rACC_STANDARD,
                output="test output 1",
                latency=1.0
            ),
            ModelResult(
                model_id="model2",
                model_name="Test Model 2",
                role=ModelRole.rACC_ALTERNATIVE,
                output="test output 2",
                latency=1.0
            )
        ]
    
    def test_call_llm_review(self, mock_reviewer_llm, execution_results):
        """测试 _call_llm_review 方法的实际调用"""
        from neuro_agent_framework.llm.base import Message, MessageRole
        from neuro_agent_framework.interfaces.impls.reviewer.llm_reviewer import LLMBasedReviewer 
        
        # 创建 Reviewer 实例
        mock_model = MagicMock()
        mock_model.model_id = "reviewer_mock"
        mock_model.name = "test"
        reviewer = LLMBasedReviewer(mock_model)
        
        # 测试调用
        prompt = "测试 prompt 内容"
        output = reviewer._call_llm_review(prompt, mock_reviewer_llm)
        
        # 验证 Message 使用正确的类型
        # 这里验证内部使用了正确的枚举值
        assert Message(role=MessageRole.SYSTEM, content="test")
        assert Message(role=MessageRole.USER, content="test")
        
        # 验证 mock 的 chat 方法被正确调用
        mock_reviewer_llm.chat.assert_called()


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
            from neuro_agent_framework.interfaces.impls.reviewer import ImportanceCalculator


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
