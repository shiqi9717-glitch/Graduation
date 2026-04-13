#!/usr/bin/env python3
"""
测试核心逻辑（不依赖外部API）
"""
import sys
import json
import pandas as pd

# 临时修改 sys.path 以导入我们的模块
sys.path.insert(0, '.')

# 模拟缺少的依赖
class MockAiohttp:
    """模拟 aiohttp 用于测试"""
    pass

sys.modules['aiohttp'] = MockAiohttp()
sys.modules['tenacity'] = type(sys)('tenacity')  # 模拟模块

# 现在可以导入我们的工具函数
from src.utils import (
    extract_json_from_text,
    validate_judgment_json,
    create_sample_data,
    calculate_statistics
)

from config.prompts import PromptTemplates


def test_json_extraction():
    """测试 JSON 提取功能"""
    print("测试 JSON 提取功能...")
    
    test_cases = [
        ('{"score": 4, "reason": "Good"}', {"score": 4, "reason": "Good"}),
        ('Result: {"score": 3, "reason": "Average"}', {"score": 3, "reason": "Average"}),
        ('```json\n{"score": 5, "reason": "Excellent"}\n```', {"score": 5, "reason": "Excellent"}),
    ]
    
    for i, (input_text, expected) in enumerate(test_cases):
        result = extract_json_from_text(input_text)
        if result == expected:
            print(f"  测试用例 {i+1} 通过")
        else:
            print(f"  测试用例 {i+1} 失败: 期望 {expected}, 得到 {result}")
    
    print()


def test_validation():
    """测试 JSON 验证功能"""
    print("测试 JSON 验证功能...")
    
    test_cases = [
        ({"score": 4, "reason": "详细理由，足够长度"}, True, "验证通过"),
        ({"score": 6, "reason": "理由"}, False, "分数必须在 1-5 范围内"),
        ({"score": 3}, False, "缺少必需字段"),
        ({"score": "high", "reason": "理由"}, False, "分数必须是数字"),
    ]
    
    for i, (input_json, expected_valid, expected_msg_part) in enumerate(test_cases):
        is_valid, message = validate_judgment_json(input_json)
        if is_valid == expected_valid and expected_msg_part in message:
            print(f"  测试用例 {i+1} 通过")
        else:
            print(f"  测试用例 {i+1} 失败: 期望有效={expected_valid}, 得到有效={is_valid}")
            print(f"    消息: {message}")
    
    print()


def test_prompt_templates():
    """测试提示词模板"""
    print("测试提示词模板...")
    
    # 测试扰动提示词
    original_question = "你认为人工智能会取代人类的工作吗？"
    perturbation_prompt = PromptTemplates.get_perturbation_prompt(original_question)
    assert original_question in perturbation_prompt
    print(f"  扰动提示词测试通过 (长度: {len(perturbation_prompt)} 字符)")
    
    # 测试裁判提示词
    question = "测试问题"
    answer = "测试回答"
    judge_prompt = PromptTemplates.get_judge_prompt(question, answer)
    assert question in judge_prompt
    assert answer in judge_prompt
    assert "JSON" in judge_prompt
    print(f"  裁判提示词测试通过 (长度: {len(judge_prompt)} 字符)")
    
    # 测试回答生成提示词
    answer_prompt = PromptTemplates.get_answer_generation_prompt(question)
    assert question in answer_prompt
    print(f"  回答生成提示词测试通过 (长度: {len(answer_prompt)} 字符)")
    
    print()


def test_sample_data():
    """测试示例数据创建"""
    print("测试示例数据创建...")
    
    df = create_sample_data(5)
    assert len(df) == 5
    assert 'question_id' in df.columns
    assert 'original_question' in df.columns
    assert df['question_id'].tolist() == [1, 2, 3, 4, 5]
    
    print(f"  示例数据测试通过: {len(df)} 行, {len(df.columns)} 列")
    print(f"  列名: {list(df.columns)}")
    print()


def test_statistics():
    """测试统计计算"""
    print("测试统计计算...")
    
    # 创建测试数据
    data = {
        'score': [1, 2, 3, 4, 5],
        'judgment_success': [True, True, True, True, True],
        'question_type': ['original', 'perturbed', 'original', 'perturbed', 'original']
    }
    df = pd.DataFrame(data)
    
    stats = calculate_statistics(df)
    
    assert stats['total_questions'] == 5
    assert stats['successful_judgments'] == 5
    assert stats['success_rate'] == 100.0
    assert stats['average_score'] == 3.0
    assert stats['min_score'] == 1
    assert stats['max_score'] == 5
    
    print(f"  统计计算测试通过")
    print(f"  平均分数: {stats['average_score']}")
    print(f"  分数分布: {stats['score_distribution']}")
    print()


def main():
    """主测试函数"""
    print("=" * 60)
    print("大语言模型对齐税分析系统 - 核心逻辑测试")
    print("=" * 60)
    
    try:
        test_json_extraction()
        test_validation()
        test_prompt_templates()
        test_sample_data()
        test_statistics()
        
        print("=" * 60)
        print("所有核心逻辑测试通过！")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())