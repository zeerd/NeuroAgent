"""
核心数据类定义
Core Data Classes
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime
import time


@dataclass
class RegisteredModel:
    """
    注册模型配置
    
    用于表示一个已注册的 AI 模型及其属性
    """
    model_id: str                         # 唯一标识
    name: str                             # 模型名称
    model_type: Any                       # 模型类型 (ModelType)
    primary_role: Any                     # 主要角色 (ModelRole)
    optional_roles: List[Any] = field(default_factory=list)
    estimated_cost: float = 0.001         # 估算成本/请求 ($)
    estimated_latency: float = 2.0        # 估算延迟 (秒)
    capabilities: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True                # 是否活跃
    weight: float = 1.0                   # 在并行执行中的权重
    
    def __str__(self):
        roles = [self.primary_role.value] + self.optional_roles
        return f"{self.name} [{', '.join(roles)}]"


@dataclass
class ModelResult:
    """
    模型执行结果
    
    用于记录单个模型执行的完整结果
    """
    model_id: str                         # 模型 ID
    model_name: str                       # 模型名称
    role: Any                             # 执行角色 (ModelRole)
    output: str                           # 输出内容
    confidence: float = 0.5               # 置信度评分
    latency: float = 0.0                  # 执行延迟 (秒)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """
    任务整体结果
    
    用于记录整个框架执行任务的最终结果
    """
    success: bool                         # 执行是否成功
    combined_answer: str                  # 综合答案
    confidence: float                     # 最终置信度
    num_executors: int                    # 使用的执行器数量
    used_expert: bool                     # 是否使用了专家模型
    total_time: float                     # 总执行时间 (秒)
    metadata: Dict[str, Any] = field(default_factory=dict)
