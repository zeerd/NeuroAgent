"""
核心枚举定义
Core Enumerations
"""

from enum import Enum, auto


class ModelType(Enum):
    """
    模型类型枚举定义系统中所有可能的模型角色
    """
    CHEAP_EXECUTOR = auto()  # 便宜执行模型
    CHEAP_REVIEWER = auto()  # 便宜评审模型
    EXPERT = auto()          # 专家模型


class ModelRole(Enum):
    """
    模型角色枚举对应神经机制

    基于神经心理学发现：
    - rACC: 适应性学习机制
    - rTPJ: 模拟对手机制
    - rDLPFC: 复杂推理机制
    """

    # rACC 机制：执行器角色
    rACC_STANDARD = "racc_standard"       # 标准执行（习惯性）
    rACC_ALTERNATIVE = "racc_alternative" # 替代执行（探索性）
    rACC_DIVERSE = "racc_diverse"         # 多样化执行（发散性）
    rACC_CRITICAL = "racc_critical"       # 批判视角（质疑性）

    # rTPJ 机制：评审角色
    rTPJ_REVIEWER = "rtpj_reviewer"       # 评审与合成

    # rDLPFC 机制：复杂推理
    rDLPFC_UPGRADER = "rDLPFC_upgrader"   # 复杂推理升级
    rDLPFC_VALIDATOR = "rDLPFC_validator" # 专家验证
