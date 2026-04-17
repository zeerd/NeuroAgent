"""
Prompt Loader - 提示词加载器

从外部文件加载和管理提示词模板
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from jinja2 import Template, StrictUndefined

logger = logging.getLogger(__name__)


class PromptLoader:
    """提示词加载器"""

    def __init__(self, prompt_dir: Optional[Path] = None):
        """
        初始化提示词加载器

        Args:
            prompt_dir: 提示词文件目录，默认为项目根目录下的 prompts 目录
        """
        self.prompt_dir = prompt_dir or Path(__file__).parent.parent.parent / 'prompts'
        self._prompts: Dict[str, str] = {}
        self._ensure_prompt_dir()

    def _ensure_prompt_dir(self):
        """确保提示词目录存在"""
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Prompt directory: {self.prompt_dir}")

    def load_prompt(self, name: str) -> Template:
        """
        加载单个提示词模板

        Args:
            name: 提示词名称（不带扩展名）

        Returns:
            Jinja2 Template 对象

        Raises:
            FileNotFoundError: 如果提示词文件不存在
        """
        if name in self._prompts:
            return Template(self._prompts[name], undefined=StrictUndefined)

        file_path = self.prompt_dir / f"{name}.j2"
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        try:
            content = file_path.read_text(encoding='utf-8')
            self._prompts[name] = content
            return Template(content, undefined=StrictUndefined)
        except Exception as e:
            logger.error(f"Failed to load prompt {name}: {e}")
            raise

    def load_prompts(self, names: list, required: bool = True) -> Dict[str, Template]:
        """
        批量加载提示词模板

        Args:
            names: 提示词名称列表
            required: 是否要求所有提示词都必须存在

        Returns:
            提示词模板字典
        """
        templates = {}
        for name in names:
            try:
                templates[name] = self.load_prompt(name)
            except FileNotFoundError as e:
                if required:
                    raise
                logger.warning(f"Optional prompt not found: {name}")
        return templates

    def get_available_prompts(self) -> list:
        """获取所有可用的提示词名称"""
        if not self.prompt_dir.exists():
            return []
        return [
            f.stem for f in self.prompt_dir.glob("*.j2")
        ]

    def render(self, name: str, **kwargs) -> str:
        """
        渲染单个提示词

        Args:
            name: 提示词名称
            **kwargs: 变量值

        Returns:
            渲染后的提示词文本
        """
        template = self.load_prompt(name)
        return template.render(**kwargs)

    def render_batch(self, name: str, **kwargs) -> Dict[str, str]:
        """
        渲染批量提示词（返回多个变量）

        Args:
            name: 提示词批次名称
            **kwargs: 变量值

        Returns:
            渲染后的变量字典
        """
        # 实现批量渲染逻辑
        # ...
        return {f"var_{i}": str(i).format(**kwargs) for i in range(3)}


# 全局提示词加载器示例
_default_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """获取或创建全局提示词加载器"""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader


def load_prompt(name: str, **kwargs) -> str:
    """
    加载并渲染单个提示词

    Args:
        name: 提示词名称
        **kwargs: 变量值

    Returns:
        渲染后的提示词文本
    """
    loader = get_prompt_loader()
    return loader.render(name, **kwargs)


# 预定义的标准提示词（向后兼容）
STANDARD_PROMPT = """
【标准方法执行者】
使用最直接的解决方法，遵循最佳实践。

用户任务：{request}
背景信息：{context}

请按标准格式输出答案。
"""

ROLE_PROMPTS = {
    "rACC_STANDARD": """【标准方法执行者】
使用最直接的解决方法，遵循最佳实践。

用户任务：{request}
背景信息：{context}

请按标准格式输出答案。""",
    "rACC_ALTERNATIVE": """【创新方法探索者】
挑战常规做法，考虑更有创新性的解决方案。

用户任务：{request}
背景信息：{context}

请提供替代方案。""",
    "rACC_DIVERSE": """【多元化视角执行者】
从多个角度分析问题，考虑各种可能的方案。

用户任务：{request}
背景信息：{context}

请按标准格式输出答案。""",
    "rACC_CRITICAL": """【批判性思考者】
质疑隐含假设，找出潜在的问题和漏洞。

用户任务：{request}
背景信息：{context}

请给出批判性分析。"""
}
