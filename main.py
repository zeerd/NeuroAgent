#!/usr/bin/env python3
"""
NeuroAgent Framework v2.0 - 接口驱动版本
神经科学启发的灵活多模型协作框架
基于：
- Harness Engineering（多模型并行 + 评审）
- The Advisor Strategy（置信度驱动的专家升级）
运行方式：
    # 直接传入任务字符串
    python main.py "设计一个电商推广方案"
    # 从文件读取任务
    python main.py task.txt
    # 指定配置文件
    python main.py "任务描述" --config=my_config.json
    # 指定置信度计算器
    python main.py "任务描述" --calculator=rule
    # 运行内置测试
    python main.py --test
"""
import sys
import argparse
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
# 设置日志目录
logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)
# 设置日志格式
log_file = logs_dir / f'framework_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
# Import framework components
from neuro_agent_framework.core.dataclasses import RegisteredModel
from neuro_agent_framework.core.enums import ModelType, ModelRole
from neuro_agent_framework.registry.model_registry import ModelRegistry
from neuro_agent_framework.framework.framework import NeuroAgentFramework
from neuro_agent_framework.llm.config_loader import ConfigLoader, load_llm_from_config
from neuro_agent_framework.llm.factory import LLMFactory
from neuro_agent_framework.interfaces.impls.execution.basic_parallel_strategy import BasicParallelStrategy
from neuro_agent_framework.interfaces.impls.execution.hybrid_strategy import HybridStrategy
from neuro_agent_framework.interfaces.impls.reviewer.llm_reviewer import LLMBasedReviewer
from neuro_agent_framework.interfaces.impls.confidence.rule_confidence_calculator import RuleBasedConfidenceCalculator
from neuro_agent_framework.interfaces.impls.confidence.llm_confidence_calculator import LLMConfidenceCalculator
from neuro_agent_framework.interfaces.impls.confidence.placeholder_confidence_calculator import PlaceholderConfidenceCalculator
def create_framework_from_config(config_path: str = None) -> tuple:
    """
    从配置文件创建框架
    自动识别 CHEAP_EXECUTOR, EXPERT, CHEAP_REVIEWER 类型模型
    Args:
        config_path: 配置文件路径
    Returns:
        (framework, executor_list, reviewer_models)
    """
    print("\n" + "="*70)
    print("NEUROAGENT FRAMEWORK v2.0 - 神经科学启发的多模型协作框架")
    print("="*70)
    print(f"日志文件：{log_file}")
    print("="*70)
    # 加载配置
    print(f"\n步骤 1: 加载 LLM 配置文件")
    print("-" * 70)
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_path}")
        config_loader = ConfigLoader(str(config_path))
    else:
        default_config = Path(project_root) / 'config' / 'llm_config.json'
        config_loader = ConfigLoader(str(default_config))
    if not config_loader.load():
        raise RuntimeError("配置文件加载失败")
    logger.info(f"Configuration loaded from: {config_loader.config_path}")
    print(f"✓ 配置文件已加载：{config_loader.config_path}")
    # 获取所有可用模型
    all_model_keys = config_loader.get_available_models()
    print(f"\n✓ 可用模型：{', '.join(all_model_keys)}\n")
    # 动态识别模型类型
    executors = []
    experts = []
    reviewers = []
    for model_key in all_model_keys:
        model_info = config_loader.get_model_info(model_key)
        role = model_info.get('config', {}).get('role', '')
        if role == 'rACC_STANDARD' or role == 'rACC_ALTERNATIVE':
            executors.append(model_key)
        elif role == 'rDLPFC_UPGRADER':
            experts.append(model_key)
        elif role == 'rTPJ_REVIEWER':
            reviewers.append(model_key)
    print(f"\n自动识别的模型分类:")
    print(f"  执行器 (executors): {len(executors)} 个 - {', '.join(executors) if executors else '无'}")
    print(f"  专家 (experts): {len(experts)} 个 - {', '.join(experts) if experts else '无'}")
    print(f"  评审器 (reviewers): {len(reviewers)} 个 - {', '.join(reviewers) if reviewers else '无'}")
    # 创建并注册模型
    print("\n步骤 2: 创建并注册模型")
    print("-" * 70)
    registry = ModelRegistry()
    for model_key in executors:
        model_info = config_loader.get_model_info(model_key)
        llm = load_llm_from_config(config_loader, model_key, f"{model_key}_instance")
        # 注册到 LLMFactory 以便线程获取
        LLMFactory.register_llm_instance(llm, model_key)
        registry.register(RegisteredModel(
            model_id=model_key,
            name=model_info.get('name', model_key),
            model_type=ModelType.CHEAP_EXECUTOR,
            primary_role=ModelRole.rACC_STANDARD,
            estimated_cost=model_info.get('config', {}).get('estimated_cost', 0.001),
            estimated_latency=model_info.get('config', {}).get('estimated_latency', 30.0),
            capabilities=model_info.get('config', {}).get('capabilities', ['推理']),
            is_active=True,
            weight=1.0,
            config={
                'config_path': str(config_loader.config_path),
                'llm_config': model_info.get('config', {})
            }
        ))
        logger.info(f"✓ Registered executor: {model_key}")
    for model_key in experts:
        model_info = config_loader.get_model_info(model_key)
        llm = load_llm_from_config(config_loader, model_key, f"{model_key}_instance")
        # 注册到 LLMFactory 以便线程获取
        LLMFactory.register_llm_instance(llm, model_key)
        registry.register(RegisteredModel(
            model_id=model_key,
            name=model_info.get('name', model_key),
            model_type=ModelType.EXPERT,
            primary_role=ModelRole.rDLPFC_UPGRADER,
            estimated_cost=model_info.get('config', {}).get('estimated_cost', 0.02),
            estimated_latency=model_info.get('config', {}).get('estimated_latency', 45.0),
            capabilities=model_info.get('config', {}).get('capabilities', ['复杂推理']),
            is_active=True,
            weight=1.0,
            config={
                'config_path': str(config_loader.config_path),
                'llm_config': model_info.get('config', {})
            }
        ))
        logger.info(f"✓ Registered expert: {model_key}")
    for model_key in reviewers:
        model_info = config_loader.get_model_info(model_key)
        llm = load_llm_from_config(config_loader, model_key, f"{model_key}_instance")
        # 注册到 LLMFactory 以便线程获取
        LLMFactory.register_llm_instance(llm, model_key)
        registry.register(RegisteredModel(
            model_id=model_key,
            name=model_info.get('name', model_key),
            model_type=ModelType.CHEAP_REVIEWER,
            primary_role=ModelRole.rTPJ_REVIEWER,
            estimated_cost=model_info.get('config', {}).get('estimated_cost', 0.001),
            estimated_latency=model_info.get('config', {}).get('estimated_latency', 30.0),
            capabilities=model_info.get('config', {}).get('capabilities', ['评审']),
            is_active=True,
            weight=1.0,
            config={
                'config_path': str(config_loader.config_path),
                'llm_config': model_info.get('config', {})
            }
        ))
        logger.info(f"✓ Registered reviewer: {model_key}")
    # 创建框架
    print("\n步骤 3: 创建 NeuroAgent 框架")
    print("-" * 70)
    executor_list = registry.list_models(model_type=ModelType.CHEAP_EXECUTOR)
    reviewer_list = registry.list_models(model_type=ModelType.CHEAP_REVIEWER)
    expert_list = registry.list_models(model_type=ModelType.EXPERT)
    framework = NeuroAgentFramework(
        executor_models=executor_list,
        expert_model=expert_list[0] if expert_list else None,
        execution_strategy=BasicParallelStrategy(),
        reviewer=LLMBasedReviewer(reviewer_list[0]) if reviewer_list else None,
        confidence_calculator=RuleBasedConfidenceCalculator()
    )
    return framework, executor_list, reviewer_list
def run_task(
    task: str,
    framework,
    complexity: float = None,
    context: dict = None
) -> dict:
    """执行任务"""
    import time
    print("\n" + "="*70)
    print("🚀 开始执行任务")
    print("="*70)
    if complexity is None:
        if len(task) < 100:
            complexity = 0.3
        elif len(task) < 500:
            complexity = 0.5
        elif len(task) < 1000:
            complexity = 0.7
        else:
            complexity = 0.9
    print(f"任务长度：{len(task)} 字符")
    print(f"任务复杂度：{complexity}")
    print(f"任务预览：{task}")
    print("="*70)
    start_time = time.time()
    result = framework.execute(
        request=task,
        context={**context, "complexity": complexity} if context else {"complexity": complexity},
    )
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("📊 执行结果")
    print("="*70)
    print(f"  执行成功：{'✓' if result.success else '✗'}")
    print(f"  最终置信度：{result.confidence:.2f}")
    print(f"  使用的执行器数量：{result.num_executors}")
    print(f"  使用了专家：{'✓' if result.used_expert else '✗'}")
    print(f"  总执行时间：{result.total_time:.2f}s")
    print("="*70)
    print("\n🎯 综合回答:")
    print("-" * 70)
    print(result.combined_answer)
    # 如果内容过长，提示完整内容在日志中
    if len(result.combined_answer) > 500:
        print(f"\n(综合回答超过 500 字符，完整内容请查看日志文件：{log_file})")
    return result
def run_framework_test():
    """运行内置测试"""
    print("\n" + "="*70)
    print("NEUROAGENT FRAMEWORK v2.0 - 内置测试运行")
    print("="*70)

    # 使用默认配置创建框架
    framework, executor_list, reviewer_list = create_framework_from_config()
    # 测试案例
    test_cases = [
        {"name": "简单问答", "task": "什么是 AI?", "complexity": 0.3},
        {"name": "技术解释", "task": "请解释 Transformer 架构原理。", "complexity": 0.8},
        {"name": "创意任务", "task": "设计一个推广方案。", "complexity": 0.6}
    ]
    for i, test in enumerate(test_cases, 1):
        print(f"\n测试 {i}/{len(test_cases)}: {test['name']}")
        print("-" * 70)
        try:
            result = run_task(
                task=test['task'],
                framework=framework,
                complexity=test['complexity']
            )
        except Exception as e:
            # LLM 连接失败时继续测试下一个
            logger.warning(f"测试 {test['name']} 发生错误：{e}")
            print(f"⚠️  测试 {test['name']} 跳过 (LLM 不可用): {e}")
            continue
        print("-" * 70)
    print("\n" + "="*70)
    print("✅ 所有测试完成!")
    print("="*70)
def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="NeuroAgent Framework v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py "设计推广方案"
  python main.py task.txt
  python main.py "AI 方案" --config=config.json
  python main.py "AI 方案" --calculator=rule
  python main.py --test
        """
    )
    parser.add_argument("task", nargs="?", help="任务描述或文件")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--complexity", type=float, help="任务复杂度")
    parser.add_argument("--calculator", type=str, default="rule", choices=["rule", "llm", "placeholder"], help="置信度计算器类型")
    parser.add_argument("--test", action="store_true", help="运行内置测试")
    parser.add_argument("--verbose", action="store_true", help="详细模式")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    try:
        if args.test:
            run_framework_test()
            return
        if not args.task:
            print(parser.format_help())
            print("\n❌ 请提供任务描述或 --test")
            sys.exit(1)
        task_input = args.task
        is_file = os.path.isfile(args.task)
        task = Path(task_input).read_text().strip() if is_file else task_input
        framework, _, _ = create_framework_from_config(args.config)
        complexity = args.complexity if args.complexity else None
        result = run_task(task=task, framework=framework, complexity=complexity)
        print("\n" + "="*70)
        print("✅ 任务执行完成!")
        print("="*70)
    except Exception as e:
        logger.exception(f"执行失败：{e}")
        print(f"\n❌ 执行失败：{e}")
        sys.exit(1)
if __name__ == "__main__":
    main()
