# NeuroAgent Framework - 独立提示词模板

所有与 LLM 交互的提示词（Prompts）都在此目录中，采用独立文件管理，便于维护。

## 📁 目录结构

```
prompts/
├── README.md                      # 本文件
├── reviewer.md                    # 评审器系统提示词
└── executors/                     # 执行器提示词模板
    ├── standard.md                # 标准方法执行者 (rACC_STANDARD)
    ├── alternative.md             # 创新方法探索者 (rACC_ALTERNATIVE)
    ├── diverse.md                 # 多元化视角执行者 (rACC_DIVERSE)
    └── critical.md                # 批判性思考者 (rACC_CRITICAL)
└── system/                        # 系统级提示词模板
    ├── assistant.md               # 通用助手系统提示词
    └── expert.md                  # 专家级 AI 系统提示词
```

## 🔧 使用方式

### 1. 代码中使用

```python
from neuro_agent_framework.core.template_loader import get_template_loader

# 加载用户提示词模板
loader = get_template_loader()
prompt = loader.load_template(
    "executors/standard",  # 模板名称
    variables={
        "request": "设计一个推广方案",
        "context": "目标用户：年轻人"
    }
)

# 加载系统提示词模板
system_prompt = loader.load_template("system/assistant")
```

### 2. 模板变量格式

使用 `{变量名}` 或 `{{变量名}}` 作为占位符。

| 变量 | 示例值 | 说明 |
|------|--------|------|
| `{request}` | "设计一个推广方案" | 用户原始任务 |
| `{context}` | "目标用户：年轻人" | 背景信息 |

## 📋 模板列表

### 执行器模板 (executors/*)

1. **executors/standard.md** - 标准方法执行者
   - 角色：rACC_STANDARD
   - 用途：使用最直接的解决方法

2. **executors/alternative.md** - 创新方法探索者
   - 角色：rACC_ALTERNATIVE
   - 用途：挑战常规，提供创新方案

3. **executors/diverse.md** - 多元化视角执行者
   - 角色：rACC_DIVERSE
   - 用途：从多个角度分析问题

4. **executors/critical.md** - 批判性思考者
   - 角色：rACC_CRITICAL
   - 用途：质疑假设，找出问题

### 系统模板 (system/*)

1. **system/assistant.md** - 通用助手系统提示词
   - 用途：所有执行器的默认系统提示词

2. **system/expert.md** - 专家级 AI 系统提示词
   - 用途：专家升级模块使用

### 评审器模板 (reviewer.md)

1. **reviewer.md** - 评审器系统提示词
   - 角色：rTPJ_REVIEWER
   - 用途：分析多个执行器结果，进行综合评估

## 🚀 添加新模板

1. 在 `prompts/` 下创建新的 `.md` 文件
2. 使用 `{变量名}` 定义占位符
3. 在代码中通过 `TemplateLoader` 加载

## 📚 设计文档参考

完整的设计文档请参考：
- `docs/PROMPTS.md` - 提示词设计原则和架构

## 🔄 维护说明

- ✅ 所有提示词使用 Markdown 格式
- ✅ 所有提示词都使用变量化设计
- ✅ 所有提示词都易于人类阅读和编辑
- ✅ 所有提示词都有清晰的角色定义

---

**最后更新**: 2026-04-20
