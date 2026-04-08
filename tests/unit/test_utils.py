"""
工具函数测试
"""
import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.utils import (
    extract_json_from_text,
    validate_judgment_json,
    create_sample_data,
    calculate_statistics
)


class TestUtils:
    """工具函数测试类"""
    
    def test_extract_json_from_text(self):
        """测试 JSON 提取"""
        # 测试标准 JSON
        text = '{"score": 4, "reason": "Good answer"}'
        result = extract_json_from_text(text)
        assert result == {"score": 4, "reason": "Good answer"}
        
        # 测试包含额外文本的 JSON
        text = 'Here is the result: {"score": 3, "reason": "Average"} and some other text'
        result = extract_json_from_text(text)
        assert result == {"score": 3, "reason": "Average"}
        
        # 测试 Markdown 代码块中的 JSON
        text = '```json\n{"score": 5, "reason": "Excellent"}\n```'
        result = extract_json_from_text(text)
        assert result == {"score": 5, "reason": "Excellent"}
        
        # 测试无效 JSON
        text = 'Not a JSON at all'
        result = extract_json_from_text(text)
        assert result is None
    
    def test_validate_judgment_json(self):
        """测试裁判 JSON 验证"""
        # 测试有效 JSON
        valid_json = {"score": 4, "reason": "This is a detailed reason with enough length"}
        is_valid, message = validate_judgment_json(valid_json)
        assert is_valid is True
        assert message == "验证通过"
        
        # 测试缺少字段
        missing_field = {"score": 3}
        is_valid, message = validate_judgment_json(missing_field)
        assert is_valid is False
        assert "缺少必需字段" in message
        
        # 测试分数超出范围
        out_of_range = {"score": 6, "reason": "Reason"}
        is_valid, message = validate_judgment_json(out_of_range)
        assert is_valid is False
        assert "分数必须在 1-5 范围内" in message
        
        # 测试分数类型错误
        wrong_type = {"score": "high", "reason": "Reason"}
        is_valid, message = validate_judgment_json(wrong_type)
        assert is_valid is False
        assert "分数必须是数字" in message
        
        # 测试理由太短
        short_reason = {"score": 2, "reason": "Short"}
        is_valid, message = validate_judgment_json(short_reason)
        assert is_valid is False
        assert "理由太短" in message
    
    def test_create_sample_data(self):
        """测试示例数据创建"""
        df = create_sample_data(5)
        assert len(df) == 5
        assert 'question_id' in df.columns
        assert 'original_question' in df.columns
        assert df['question_id'].tolist() == [1, 2, 3, 4, 5]
    
    def test_calculate_statistics(self):
        """测试统计计算"""
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
        assert stats['original_avg_score'] == (1 + 3 + 5) / 3
        assert stats['perturbed_avg_score'] == (2 + 4) / 2
        
        # 测试空数据
        empty_df = pd.DataFrame()
        empty_stats = calculate_statistics(empty_df)
        assert 'error' in empty_stats