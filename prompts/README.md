# NeuroAgent Framework - 提示词目录

这个目录包含所有与 LLM 交互的提示词模板，使用 Jinja2 格式。

## 📁 目录结构

```
prompts/
├── README.md                     # 本文件
├── executor_standard.j2          # 标准方法执行者
├── executor_alternative.j2       # 创新方法探索者
├── executor_diverse.j2           # 多元化视角执行者
├── executor_critical.j2          # 批判性思考者
├── reviewer_system.j2            # 评审器系统提示
├── reviewer_user.j2              # 评审器用户提示
└── expert_task.j2                # 专家升级任务提示
```

## 📝 提示词模板说明

### 执行器提示词 (Executor Prompts)

所有执行器提示词都支持以下变量：

- `{{ request }}` - 用户原始任务
- `{{ context }}` - 背景信息

| 文件名 | 角色 | 用途 |
|--------|------|------|
| `executor_standard.j2` | 标准方法执行者 | 遵循最佳实践，直接解决问题 |
| `executor_alternative.j2` | 创新方法探索者 | 挑战常规，提供创新方案 |
| `executor_diverse.j2` | 多元化视角执行者 | 多角度的综合解决方案 |
| `executor_critical.j2` | 批判性思考者 | 质疑假设，识别问题 |

### 评审器提示词 (Reviewer Prompts)

| 文件名 | 类型 | 用途 |
|--------|------|------|
| `reviewer_system.j2` | System | 评审专家身份设定 |
| `reviewer_user.j2` | User | 评审任务详细说明 |

**变量：**
- `{{ request }}` - 原始用户请求
- `{{ execution_results }}` - 执行结果摘要列表

### 专家升级提示词 (Expert Prompts)

| 文件名 | 类型 | 用途 |
|--------|------|------|
| `expert_task.j2` | User | 专家升级任务任务 |

**变量：**
- `{{ request }}` - 原始用户请求
- `{{ results }}` - 执行结果列表（Jinja2 循环）

## 🔄 使用说明

### 加载提示词

```python
from neuro_agent_framework.prompts.prompt_loader import PromptLoader

# 创建加载器
loader = PromptLoader()

# 加载单个提示词
template = loader.load_prompt("executor_standard")

# 渲染提示词
prompt = template.render(
    request="设计一个推广方案",
    context="目标用户：年轻人"
)

# 批量加载
templates = loader.load_prompts([
    "executor_standard",
    "executor_alternative"
])
```

### 使用全局加载器

```python
from neuro_agent_framework.prompts.prompt_loader import load_prompt

# 加载并渲染提示词
prompt = load_prompt(
    "executor_standard",
    request="测试任务",
    context="测试背景"
)
```

## 📖 文档

完整的提示词文档请查看 `[docs/PROMPTS.md](../docs/PROMPTS.md)`。

## 🛠️ 维护说明

### 添加新提示词

1. 创建新的 `.j2` 文件
2. 使用 Jinja2 变量语法定义变量
3. 在 `docs/PROMPTS.md` 中补充文档
4. 添加测试验证

### 提示词变量规范

- 使用 `{{ variable }}` 语法
- 变量名使用蛇形命名（如 `user_request`）
- 避免特殊字符和空格

### 测试提示词

```python
from neuro_agent_framework.prompts import PromptLoader

loader = PromptLoader()
template = loader.load_prompt("executor_standard")

result = template.render(
    request="测试",
    context="测试上下文"
)

print(result)
```
