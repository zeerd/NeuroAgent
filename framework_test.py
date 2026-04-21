#!/usr/bin/env python3
"""
NeuroAgent Framework 测试脚本

通过调用 main.py 来执行测试
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)

TEST_CASES = [
    {"name": "简单问答", "task": "什么是 AI?", "complexity": 0.3},
    {"name": "技术解释", "task": "请解释 Transformer 架构原理。", "complexity": 0.8},
    {"name": "创意任务", "task": "设计一个推广方案。", "complexity": 0.6},
    {"name": "CopilotLLM 集成测试", "task": "请展示 GitHub Copilot SDK 的 model 参数用法", "complexity": 0.5, "provider": "copilot", "config_name": "copilot_config.json"}
]


def run_single_test(test_number: int):
    """通过调用 main.py 来运行单个测试"""
    if test_number < 1 or test_number > len(TEST_CASES):
        raise ValueError(f"测试编号 {test_number} 超出范围 (1-{len(TEST_CASES)})")

    test_case = TEST_CASES[test_number - 1]
    task = test_case['task']

    print(f"\n{'='*70}")
    print(f"测试 {test_number}/{len(TEST_CASES)}: {test_case['name']}")
    print(f"{'='*70}")
    print(f"任务：{task}")
    print(f"复杂度：{test_case['complexity']}")
    print(f"{'='*70}\n")

    main_script = project_root / 'main.py'
    subprocess_args = [
        sys.executable,
        str(main_script),
        task,
        f"--complexity={test_case['complexity']}"
    ]

    # 如果测试用例指定了配置文件，添加 --config 参数
    config_name = test_case.get('config_name')
    if config_name:
        config_path = project_root / 'config' / config_name
        subprocess_args.extend(["--config", str(config_path)])

    return subprocess.run(subprocess_args, cwd=project_root, capture_output=False, check=False)


def run_tests(test_numbers: list = None):
    """运行一个或多个测试"""
    print("\n" + "="*70)
    print("NEUROAGENT FRAMEWORK v2.0 - 测试脚本运行")
    print("="*70)

    if test_numbers:
        selected_tests = [test_numbers[i - 1] for i in range(len(test_numbers))]
        print(f"运行选定的测试：{', '.join(map(str, test_numbers))}")
    else:
        selected_tests = list(range(1, len(TEST_CASES) + 1))
        print("运行所有测试")

    print(f"\n测试脚本路径：{project_root / 'main.py'}")
    print(f"使用解释器：{sys.executable}")
    print("="*70)

    results = []
    for test_num in selected_tests:
        try:
            result = run_single_test(test_num)
            results.append({
                'number': test_num,
                'name': TEST_CASES[test_num - 1]['name'],
                'success': result.returncode == 0,
                'returncode': result.returncode
            })
        except Exception as e:
            print(f"测试 {test_num} 运行时发生错误：{e}")
            results.append({
                'number': test_num,
                'name': TEST_CASES[test_num - 1]['name'],
                'success': False,
                'error': str(e)
            })

    print("\n" + "="*70)
    print("📊 测试汇总")
    print("="*70)

    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} 测试 {result['number']}: {result['name']} - 返回值：{result['returncode']}")

    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)

    print("="*70)
    print(f"✅ 完成：{success_count}/{total_count} 个测试成功")
    print("="*70)


def main():
    """测试脚本主入口"""
    parser = argparse.ArgumentParser(
        description="NeuroAgent Framework 测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="使用方式:\n  python framework_test.py                    # 运行所有测试\n  python framework_test.py -n 1               # 仅运行第 1 个测试\n  python framework_test.py -n 1,2             # 运行第 1、2 个测试"
    )

    parser.add_argument(
        "-n", "--numbers",
        type=str,
        help="指定要运行的测试编号，用逗号分隔（如 1,2,3），不指定则运行全部"
    )

    args = parser.parse_args()

    test_numbers = None
    if args.numbers:
        try:
            test_numbers = [int(int_num.strip()) for int_num in args.numbers.split(',')]
            for num in test_numbers:
                if num < 1 or num > len(TEST_CASES):
                    print(f"错误：测试编号 {num} 超出范围 (1-{len(TEST_CASES)})")
                    sys.exit(1)
        except ValueError:
            print("错误：测试编号必须是整数，用逗号分隔")
            sys.exit(1)

    try:
        run_tests(test_numbers=test_numbers)
    except Exception as e:
        print(f"\n❌ 测试执行失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
