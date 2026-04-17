# NeuroAgent Framework v2.0

神经科学启发的灵活多模型协作框架

基于 **Harness Engineering**、**The Advisor Strategy** 和 **神经心理学（rACC/rTPJ/rDLPFC 三脑机制）**，实现成本效率最优的 AI Agent 系统。

## 架构概览

```
用户请求
  │
  ▼
┌─────────────────────┐
│  PHASE 1: 并行执行   │  ← 2~N 个廉价执行器同时回答
│  BasicParallelStrategy │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  PHASE 2: 评审合成   │  ← LLM Reviewer 综合分析
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  PHASE 3: 置信度评估 │  ← 规则/LLM 置信度计算器
└─────────┬───────────┘
          │
     ┌────┴────┐
     │         │
  置信度达标   置信度不足
     │         │
     │         ▼
     │   ┌─────────────┐
     │   │ PHASE 4:    │  ← 调用昂贵专家模型
     │   │ 专家升级    │
     │   └──────┬──────┘
     │          │
     │          ▼
     │     专家答案 + 评审答案
     │
     ▼
最终答案 ✓
```

## 快速开始

### 安装

```bash
pip install neuro-agent-framework
```

### 使用

```python
from neuro_agent_framework.llm.config_loader import ConfigLoader
from neuro_agent_framework.llm.factory import LLMFactory
from neuro_agent_framework.framework.config import FrameworkConfig
from neuro_agent_framework.registry.model_registry import ModelRegistry
from neuro_agent_framework.framework.framework import NeuroAgentFramework
from neuro_agent_framework.interfaces.impls.execution.basic_parallel_strategy import BasicParallelStrategy
from neuro_agent_framework.interfaces.impls.reviewer.llm_reviewer import LLMBasedReviewer
from neuro_agent_framework.interfaces.impls.confidence.rule_confidence_calculator import RuleBasedConfidenceCalculator

# 方式 1: 使用 JSON 配置文件
config_loader = ConfigLoader("config/llm_config.json")
config_loader.load()

# 方式 2: 使用预置配置模板
registry = FrameworkConfig.create_simple_config()  # 或 create_advanced_config()

# 使用 ModelRegistry 手动构建
registry = ModelRegistry()
# ... 注册模型 ...

# 创建框架
framework = NeuroAgentFramework(
    executor_models=registry.list_models(model_type=ModelType.CHEAP_EXECUTOR),
    expert_model=registry.list_models(model_type=ModelType.EXPERT)[0],
    execution_strategy=BasicParallelStrategy(),
    reviewer=LLMBasedReviewer(reviewer_models[0]),
    confidence_calculator=RuleBasedConfidenceCalculator()
)

# 执行任务
result = framework.execute(
    request="什么是 AI?",
    context={"complexity": 0.3}
)

print(f"置信度: {result.confidence:.2f}")
print(f"答案: {result.combined_answer}")
```

### CLI 使用

```bash
# 直接传入任务
python main.py "设计一个电商推广方案"

# 从文件读取任务
python main.py task.txt

# 指定配置文件
python main.py "AI 方案" --config=my_config.json

# 运行内置测试
python main.py --test
```

## 🧠 LLM 模块

支持通过 OpenAI 兼容接口调用各种 LLM：

| 提供商 | 支持情况 | 说明 |
|--------|----------|------|
| OpenAI 官方 | ✅ | GPT-3.5, GPT-4 系列 |
| Azure OpenAI | ✅ | 企业级部署 |
| OpenRouter | ✅ | Claude, Gemini 等 |
| LocalAI/Ollama | ✅ | 本地部署 |

### LLM 配置

配置文件 `config/llm_config.json` 格式：

```json
{
  "_metadata": {
    "version": "2.1.0",
    "description": "LLM 配置 - 支持 OpenAI 兼容接口"
  },
  "models": {
    "qwen3.5_2b": {
      "name": "qwen3.5:2b",
      "description": "2B 模型 - 便宜执行器 A",
      "role": "rACC_STANDARD",
      "config": {
        "model": "qwen3.5:2b",
        "api_base": "http://192.168.2.23:11434/v1",
        "api_key": "",
        "temperature": 0.7,
        "max_tokens": 2048
      }
    },
    "qwen3.5_08b_reviewer": {
      "name": "qwen3.5:0.8b",
      "description": "评审器",
      "role": "rTPJ_REVIEWER",
      "config": {
        "model": "qwen3.5:0.8b",
        "api_base": "http://192.168.2.23:11434/v1",
        "temperature": 0.3,
        "max_tokens": 2048
      }
    },
    "qwen_35b_expert": {
      "name": "qwen3.6:35b-a3b",
      "description": "专家模型",
      "role": "rDLPFC_UPGRADER",
      "config": {
        "model": "qwen3.6:35b-a3b",
        "api_base": "http://192.168.2.23:11434/v1",
        "temperature": 0.3,
        "max_tokens": 4096
      }
    }
  }
}
```

**role 可选值：**
- `rACC_STANDARD` — 标准执行器
- `rACC_ALTERNATIVE` — 替代执行器
- `rTPJ_REVIEWER` — 评审器
- `rDLPFC_UPGRADER` — 专家升级

## 模块结构

```
neuro_agent_framework/
├── core/                      # 核心数据结构
│   ├── enums.py               # ModelType, ModelRole
│   └── dataclasses.py         # RegisteredModel, ModelResult
├── framework/                 # 框架核心
│   ├── framework.py           # NeuroAgentFrameworkV2 主类
│   └── config.py              # FrameworkConfig 配置生成器
├── interfaces/                # 接口定义
│   ├── execution_strategy.py  # 执行策略接口
│   ├── confidence_calculator.py # 置信度计算器接口
│   ├── reviewer.py            # 评审器接口
│   └── impls/                 # 实现
│       ├── execution/
│       │   ├── base_strategy.py
│       │   ├── basic_parallel_strategy.py
│       │   ├── basic_strategy.py
│       │   ├── diversified_strategy.py
│       │   └── hybrid_strategy.py
│       ├── confidence/
│       │   ├── rule_confidence_calculator.py
│       │   ├── llm_confidence_calculator.py
│       │   └── placeholder_confidence_calculator.py
│       └── reviewer/
│           └── llm_reviewer.py
├── llm/                       # LLM 模块
│   ├── base.py                # BaseLLM, LLMConfig, LLMResponse
│   ├── config_loader.py       # JSON 配置加载
│   ├── factory.py             # LLMFactory 工厂模式
│   └── openai_adapter.py      # OpenAI 兼容适配器
├── prompts/                   # Jinja2 提示词模板
│   ├── executor_standard.j2
│   ├── executor_alternative.j2
│   ├── executor_diverse.j2
│   ├── executor_critical.j2
│   ├── reviewer_system.j2
│   ├── reviewer_user.j2
│   └── expert_task.j2
└── registry/
    └── model_registry.py      # 模型注册中心
```

## 神经科学启发

| 机制 | 对应大脑 | 框架角色 |
|------|----------|----------|
| rACC | 前扣带回（适应性学习） | 廉价执行器，快速响应 |
| rTPJ | 颞顶联合区（模拟对手机制） | LLM Reviewer，综合评审 |
| rDLPFC | 背外侧前额叶（复杂推理） | 专家模型，深度推理 |

## 🔧 配置选项

```python
# 自定义置信度阈值
thresholds = {
    'consistency_threshold': 0.75,  # rACC 一致性
    'completeness_threshold': 0.70,  # rTPJ 完整性
    'reliability_threshold': 0.80,   # rDLPFC 可靠性
    'combined_threshold': 0.80,      # 总体综合
}
```

## 📈 路线图

- [x] **v2.0** — 接口驱动架构 + 配置加载器
- [x] **v2.1** — JSON 配置文件 + 模型自动分类
- [ ] **v2.2** — 多模型并行策略优化
- [ ] **v3.0** — 企业版特性

## 🔗 参考

- [Harness Engineering](https://martinfowler.com/articles/harness-engineering.html)
- [The Advisor Strategy](https://claude.com/blog/the-advisor-strategy)
