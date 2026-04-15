"""
ModelRegistry - 模型注册中心
核心功能：动态注册、查询、管理 AI 模型
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from ..core.enums import ModelType, ModelRole
from ..core.dataclasses import RegisteredModel


logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    模型注册中心 - 支持动态注册管理
    
    类似大脑的"模型库"，可以动态添加/移除/调整
    提供统一的接口查询不同角色和类型的模型
    """
    
    def __init__(self):
        self._models: Dict[str, RegisteredModel] = {}
        self._model_by_name: Dict[str, RegisteredModel] = {}
        logger.info("ModelRegistry initialized")
    
    def register(self, model: RegisteredModel) -> bool:
        """注册新模型"""
        if not model.is_active:
            logger.warning(f"Skipping inactive model: {model.name}")
            return False
        
        if model.model_id in self._models:
            logger.warning(f"Model {model.model_id} already registered, overwriting...")
        
        # 创建副本
        model_copy = RegisteredModel(
            model_id=model.model_id,
            name=model.name,
            model_type=model.model_type,
            primary_role=model.primary_role,
            optional_roles=list(model.optional_roles),
            estimated_cost=model.estimated_cost,
            estimated_latency=model.estimated_latency,
            capabilities=list(model.capabilities) if model.capabilities else [],
            config=dict(model.config) if model.config else {},
            is_active=model.is_active,
            weight=model.weight
        )
        
        self._models[model.model_id] = model_copy
        self._model_by_name[model.name] = model_copy
        logger.info(f"✓ Registered: {model.name} ({model.model_id}) - {model.primary_role.value}")
        return True
    
    def unregister(self, model_id: str) -> bool:
        """取消注册模型"""
        if model_id not in self._models:
            logger.warning(f"Model {model_id} not found")
            return False
        
        del self._models[model_id]
        logger.info(f"✗ Unregistered: {model_id}")
        return True
    
    def get(self, model_id: str) -> RegisteredModel:
        """获取已注册模型"""
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found")
        return self._models[model_id]
    
    def get_by_name(self, name: str) -> RegisteredModel:
        """通过名称获取模型"""
        if name not in self._model_by_name:
            raise ValueError(f"Model {name} not found")
        return self._model_by_name[name]
    
    def list_models(self, 
                   model_type: ModelType = None,
                   role: ModelRole = None,
                   active_only: bool = True) -> List[RegisteredModel]:
        """列出已注册模型，支持过滤"""
        models = list(self._models.values())
        
        if active_only:
            models = [m for m in models if m.is_active]
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if role:
            models = [m for m in models if role in [m.primary_role] + m.optional_roles]
        
        return models
    
    def get_default_model(self, role: ModelRole) -> RegisteredModel:
        """获取默认模型（基于角色）"""
        candidates = self.list_models(role=role)
        
        if not candidates:
            raise ValueError(f"No model available for role: {role.value}")
        
        # 返回权重最高的模型
        return max(candidates, key=lambda m: m.weight)
    
    def get_available_models(self) -> List[RegisteredModel]:
        """获取所有可用模型"""
        return self.list_models(active_only=True)
    
    def get_status(self) -> Dict[str, int]:
        """获取注册状态"""
        return {
            'total_models': len(self._models),
            'cheap_executors': len(self.list_models(ModelType.CHEAP_EXECUTOR)),
            'cheap_reviewers': len(self.list_models(ModelType.CHEAP_REVIEWER)),
            'experts': len(self.list_models(ModelType.EXPERT))
        }
    
    def add_role_to_model(self, model_id: str, new_role: ModelRole):
        """为模型添加额外角色"""
        model = self.get(model_id)
        model.optional_roles.append(new_role)
    
    def remove_role_from_model(self, model_id: str, role: ModelRole):
        """从模型移除角色"""
        model = self.get(model_id)
        if role in model.optional_roles:
            model.optional_roles.remove(role)
