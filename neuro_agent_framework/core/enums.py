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
    模型角色枚举定义系统中所有可能的模型角色

    命名说明：
    - rACC_* : 执行器角色（rACC 为内部命名）
    - rTPJ_* : 评审器角色（rTPJ 为内部命名）
    - rDLPFC_* : 专家模型角色（rDLPFC 为内部命名）

    ⚠️ 这些命名仅为内部标识符，不代表真实的神经科学机制。
    框架灵感来源于：
    - Harness Engineering（多模型并行 + 评审）
    - The Advisor Strategy（置信度驱动的专家升级）
    """

    # 执行器角色
    rACC_STANDARD = "racc_standard"       # 标准执行（习惯性）
    rACC_ALTERNATIVE = "racc_alternative" # 替代执行（探索性）
    rACC_DIVERSE = "racc_diverse"         # 多样化执行（发散性）
    rACC_CRITICAL = "racc_critical"       # 批判视角（质疑性）

    # 评审器角色
    rTPJ_REVIEWER = "rtpj_reviewer"       # 评审与合成

    # 专家模型角色
    rDLPFC_UPGRADER = "rDLPFC_upgrader"   # 复杂推理升级
    rDLPFC_VALIDATOR = "rDLPFC_validator" # 专家验证
