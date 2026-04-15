"""
FrameworkConfig - 框架配置生成器

提供快速配置模板
"""

from ..core.enums import ModelType, ModelRole
from ..core.dataclasses import RegisteredModel
from ..registry.model_registry import ModelRegistry


class FrameworkConfig:
    """
    框架配置生成器
    
    提供常用的配置组合模板
    """
    
    @staticmethod
    def create_simple_config():
        """
        创建简单配置（2 个执行器 + 1 个评审 + 1 个专家）
        
        Returns:
            ModelRegistry: 已配置的注册中心
        """
        registry = ModelRegistry()
        
        # 添加执行器
        registry.register(RegisteredModel(
            model_id="model_a",
            name="GPT-4o-Mini-A",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            estimated_cost=0.001,
            estimated_latency=2.0
        ))
        
        registry.register(RegisteredModel(
            model_id="model_b",
            name="Claude-3.5-Sonnet-B",
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_ALTERNATIVE,
            estimated_cost=0.001,
            estimated_latency=2.2
        ))
        
        # 添加评审
        registry.register(RegisteredModel(
            model_id="model_c",
            name="GPT-4o-Mini-Reviewer",
            model_type=ModelType.CHEAP_REVIEWER,
            primary_role=ModelRole.rTPJ_REVIEWER,
            estimated_cost=0.001,
            estimated_latency=1.8
        ))
        
        # 添加专家
        registry.register(RegisteredModel(
            model_id="model_expert",
            name="DeepSeek-V3-Expert",
            model_type=ModelType.EXPERT,
            primary_role=ModelRole.rDLPFC_UPGRADER,
            estimated_cost=0.02,
            estimated_latency=5.0
        ))
        
        return registry
    
    @staticmethod
    def create_advanced_config():
        """
        创建高级配置（4 个执行器 + 1 个评审 + 1 个专家）
        
        Returns:
            ModelRegistry: 已配置的注册中心
        """
        registry = ModelRegistry()
        
        # 多视角执行器
        roles = [
            (ModelRole.rACC_STANDARD, "GPT-4o-Mini-A"),
            (ModelRole.rACC_ALTERNATIVE, "GPT-4o-Mini-B"),
            (ModelRole.rACC_DIVERSE, "Claude-3.5-A"),
            (ModelRole.rACC_CRITICAL, "Claude-3.5-B"),
        ]
        
        for i, (role, name) in enumerate(roles, 1):
            registry.register(RegisteredModel(
                model_id=f"model_a{i}",
                name=name,
                model_type=ModelType.CHEAP_EXECUTOR,
                primary_role=role,
                estimated_cost=0.001,
                estimated_latency=2.0
            ))
        
        # 评审
        registry.register(RegisteredModel(
            model_id="model_c",
            name="Expert-Reviewer",
            model_type=ModelType.CHEAP_REVIEWER,
            primary_role=ModelRole.rTPJ_REVIEWER,
            estimated_cost=0.001,
            estimated_latency=2.0
        ))
        
        # 专家
        registry.register(RegisteredModel(
            model_id="model_expert",
            name="DeepSeek-V3-Expert",
            model_type=ModelType.EXPERT,
            primary_role=ModelRole.rDLPFC_UPGRADER,
            estimated_cost=0.02,
            estimated_latency=5.0
        ))
        
        return registry
