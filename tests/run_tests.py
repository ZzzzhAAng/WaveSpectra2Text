#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本
运行所有测试并生成报告
"""

import sys
import os
import unittest
from pathlib import Path
import time

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def run_all_tests():
    """运行所有测试"""
    print("🧪 开始运行WaveSpectra2Text测试套件...")
    print("=" * 60)
    
    # 测试模块列表
    test_modules = [
        'test_core',
        'test_data', 
        'test_training',
        'test_inference'
    ]
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 加载所有测试
    for module_name in test_modules:
        try:
            module = __import__(module_name)
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)
            print(f"✅ 加载测试模块: {module_name}")
        except Exception as e:
            print(f"❌ 加载测试模块失败: {module_name} - {e}")
    
    # 运行测试
    print("\n🚀 开始运行测试...")
    print("-" * 60)
    
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"运行时间: {duration:.2f}秒")
    
    if result.failures:
        print("\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # 返回测试是否全部通过
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息")
    
    return success


def run_specific_test(test_name):
    """运行特定测试"""
    print(f"🧪 运行特定测试: {test_name}")
    print("=" * 60)
    
    try:
        module = __import__(test_name)
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except Exception as e:
        print(f"❌ 运行测试失败: {e}")
        return False


def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 运行特定测试
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # 运行所有测试
        success = run_all_tests()
    
    # 退出码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
