"""
提示词模板加载器

从独立文件加载模板，支持变量替换
"""

import os
import logging
import re
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TemplateLoader:
    """模板加载器 - 从独立文件加载提示词模板"""
    
    def __init__(self, base_dir: str = None):
        """
        初始化模板加载器
        
        Args:
            base_dir: 模板目录基础路径，默认从环境变量或当前项目目录查找
        """
        if base_dir is None:
            # 从环境变量获取，或默认使用/prompts/目录
            self._base_dir = os.environ.get('PROMPTS_DIR', '/home/node/.openclaw/workspace/neuro_agent_framework/prompts')
        else:
            self._base_dir = base_dir
        
        logger.debug(f"TemplateLoader initialized with base_dir: {self._base_dir}")
    
    def load_template(self, template_name: str, variables: Dict[str, Any] = None) -> str:
        """
        加载模板并填充变量
        
        Args:
            template_name: 模板文件名（不含扩展名）
            variables: 模板变量字典
            
        Returns:
            填充变量后的完整提示词
            
        Raises:
            FileNotFoundError: 模板文件不存在
        """
        # 模板文件扩展名
        template_path = str(Path(self._base_dir) / f"{template_name}.md")
        
        if not os.path.exists(template_path):
            logger.warning(f"Template not found: {template_path}")
            raise FileNotFoundError(f"模板文件不存在：{template_path}")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
        except Exception as e:
            logger.error(f"Failed to load template {template_path}: {e}")
            raise
        
        logger.debug(f"Loaded template: {template_name}")
        
        # 填充变量 {{变量名}}
        if variables:
            template_content = self._fill_variables(template_content, variables)
            logger.debug(f"Filled variables in template: {template_name}")
        
        return template_content
    
    def _fill_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """
        填充模板变量
        
        Args:
            text: 包含 {{变量名}} 的文本
            variables: 变量字典
            
        Returns:
            填充后的文本
        """
        # 支持 {{变量名}} 格式的变量替换
        def replace_var(match):
            var_name = match.group(1)
            if var_name in variables:
                value = variables[var_name]
                if isinstance(value, str):
                    return value
                elif value is None:
                    return ""
                else:
                    return str(value)
            else:
                # 如果变量不存在，保留原占位符
                logger.warning(f"Variable not found: {var_name}")
                return match.group(0)
        
        # 匹配 {{变量名}} 格式
        pattern = r'\{\{(\w+)\}\}'
        result = re.sub(pattern, replace_var, text)
        return result
    
    def get_available_templates(self) -> list:
        """
        获取所有可用的模板列表
        
        Returns:
            模板名称列表 (包含子目录路径)
        """
        templates = []
        try:
            # 检查顶层模板
            for file in Path(self._base_dir).glob('*.md'):
                if file.name.startswith('_') or file.name.startswith('.') or file.name == 'README':
                    continue
                templates.append(file.stem)
            
            # 检查 subdirectories 模板
            subdir = Path(self._base_dir) / 'executors'
            if subdir.exists():
                for file in subdir.glob('*.md'):
                    templates.append(f"executors/{file.stem}")
            
            subdir = Path(self._base_dir) / 'system'
            if subdir.exists():
                for file in subdir.glob('*.md'):
                    templates.append(f"system/{file.stem}")
            
            subdir = Path(self._base_dir) / 'reviewer'
            if subdir.exists():
                for file in subdir.glob('*.md'):
                    templates.append(f"reviewer/{file.stem}")
        except Exception as e:
            logger.warning(f"Failed to list templates: {e}")
        
        return sorted(templates)
    
    def load_all(self, template_name: str) -> Dict[str, str]:
        """
        加载模板的所有可用变量占位符（用于测试）
        
        Args:
            template_name: 模板文件名
            
        Returns:
            所有变量占位符列表
        """
        template_path = str(Path(self._base_dir) / f"{template_name}.md")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取所有 {{变量名}}
        pattern = r'\{\{(\w+)\}\}'
        matches = re.findall(pattern, content)
        return list(set(matches))


# 全局模板加载器实例
_template_loader = None

def get_template_loader() -> TemplateLoader:
    """获取全局模板加载器实例"""
    global _template_loader
    if _template_loader is None:
        _template_loader = TemplateLoader()
    return _template_loader

def load_prompt(template_name: str, variables: Dict[str, Any] = None) -> str:
    """
    便捷函数：加载模板
    
    Args:
        template_name: 模板文件名
        variables: 模板变量
        
    Returns:
        填充变量后的完整提示词
    """
    loader = get_template_loader()
    return loader.load_template(template_name, variables)
