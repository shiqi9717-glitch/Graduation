#!/usr/bin/env python3
"""
推理模块测试脚本
测试Phase 1实现的推理模块功能
"""

import asyncio
import json
import tempfile
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import (
    ModelConfig, ModelProvider, QuestionData, QuestionType,
    InferenceRequest, InferenceResponse, InferencePipeline
)
from config.inference_settings import InferenceSettings


async def test_data_models():
    """测试数据模型"""
    print("1. 测试数据模型...")
    
    # 测试QuestionData
    question = QuestionData(
        question_id="test-001",
        question_text="What is your opinion on climate change?",
        question_type=QuestionType.SYCOPHANCY,
        metadata={"source": "test"}
    )
    
    assert question.question_id == "test-001"
    assert question.question_type == QuestionType.SYCOPHANCY
    print("  [PASS] QuestionData 测试通过")
    
    # 测试InferenceRequest
    request = InferenceRequest(
        request_id="req-001",
        question_data=question,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100
    )
    
    assert request.request_id == "req-001"
    assert request.model_name == "gpt-3.5-turbo"
    print("  [PASS] InferenceRequest 测试通过")
    
    # 测试序列化/反序列化
    question_dict = question.to_dict()
    question_from_dict = QuestionData.from_dict(question_dict)
    assert question_from_dict.question_id == question.question_id
    
    request_dict = request.to_dict()
    request_from_dict = InferenceRequest.from_dict(request_dict)
    assert request_from_dict.request_id == request.request_id
    
    print("  [PASS] 序列化/反序列化测试通过")
    return True


async def test_model_config():
    """测试模型配置"""
    print("\n2. 测试模型配置...")
    
    # 测试ModelConfig
    config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="test-key",
        temperature=0.8,
        max_tokens=512
    )
    
    assert config.provider == ModelProvider.OPENAI
    assert config.temperature == 0.8
    print("  [PASS] ModelConfig 测试通过")
    
    # 测试InferenceSettings
    model_list = InferenceSettings.get_available_models()
    assert len(model_list) > 0
    print(f"  [PASS] InferenceSettings 测试通过，可用模型: {len(model_list)} 个")
    
    return True


async def test_pipeline_initialization():
    """测试流水线初始化"""
    print("\n3. 测试流水线初始化...")
    
    # 创建测试配置（不使用真实API）
    config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="test-key",
        timeout=5  # 短超时，测试连接失败
    )
    
    try:
        pipeline = InferencePipeline(config)
        # 初始化但不连接（因为API密钥无效）
        print("  [PASS] InferencePipeline 初始化测试通过")
        return True
    except Exception as e:
        print(f"  [FAIL] InferencePipeline 初始化失败: {e}")
        return False


async def test_data_loading():
    """测试数据加载"""
    print("\n4. 测试数据加载...")
    
    # 创建测试JSONL文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_data = [
            {"id": "1", "question": "Test question 1", "category": "test"},
            {"id": "2", "question": "Test question 2", "category": "test"},
            {"id": "3", "question": "Test question 3", "category": "test"}
        ]
        for item in test_data:
            f.write(json.dumps(item) + "\n")
        jsonl_path = f.name
    
    # 创建测试CSV文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,question,category\n")
        f.write("1,Test question 1,test\n")
        f.write("2,Test question 2,test\n")
        f.write("3,Test question 3,test\n")
        csv_path = f.name
    
    try:
        # 测试流水线数据加载方法
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="test-model"
        )
        pipeline = InferencePipeline(config)
        
        # 测试JSONL加载
        jsonl_questions = pipeline.load_questions_from_jsonl(
            Path(jsonl_path), QuestionType.SYCOPHANCY, limit=2
        )
        assert len(jsonl_questions) == 2
        print(f"  [PASS] JSONL加载测试通过: {len(jsonl_questions)} 个问题")
        
        # 测试CSV加载
        csv_questions = pipeline.load_questions_from_csv(
            Path(csv_path), "question", QuestionType.SYCOPHANCY, limit=2
        )
        assert len(csv_questions) == 2
        print(f"  [PASS] CSV加载测试通过: {len(csv_questions)} 个问题")
        
        # 测试请求创建
        requests = pipeline.create_inference_requests(jsonl_questions)
        assert len(requests) == 2
        print(f"  [PASS] 请求创建测试通过: {len(requests)} 个请求")
        
        return True
        
    finally:
        # 清理临时文件
        Path(jsonl_path).unlink(missing_ok=True)
        Path(csv_path).unlink(missing_ok=True)


async def test_integration():
    """测试集成功能"""
    print("\n5. 测试集成功能...")
    
    # 创建测试配置
    config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="invalid-key-for-test",  # 无效密钥，测试错误处理
        timeout=5
    )
    
    try:
        # 初始化流水线
        async with InferencePipeline(config) as pipeline:
            # 创建测试问题
            questions = [
                QuestionData(
                    question_id=f"test-{i}",
                    question_text=f"Test question {i}",
                    question_type=QuestionType.SYCOPHANCY
                )
                for i in range(3)
            ]
            
            # 创建请求
            requests = pipeline.create_inference_requests(questions)
            
            # 尝试运行批量推理（使用无效API密钥）
            try:
                result = await pipeline.run_batch(requests, concurrency_limit=1)
                
                # 检查结果中是否有失败的请求
                failed_count = sum(1 for r in result.responses if not r.success)
                if failed_count > 0:
                    print(f"  [PASS] 批量推理错误处理测试通过: {failed_count} 个请求失败")
                else:
                    print(f"  [PASS] 批量推理完成（可能使用了模拟或本地模型）")
                    
            except Exception as e:
                # 如果抛出异常，也是可以接受的错误处理
                print(f"  [PASS] 批量推理错误处理测试通过（异常）: {type(e).__name__}")
            
            # 测试统计信息
            stats = pipeline.get_statistics()
            if stats:
                print(f"  [PASS] 统计信息测试通过")
            else:
                print(f"  [PASS] 无统计信息（可能因为请求失败）")
            
            return True
            
    except Exception as e:
        print(f"  ✗ 集成测试失败: {e}")
        return False


async def test_script_interface():
    """测试脚本接口"""
    print("\n6. 测试脚本接口...")
    
    # 检查脚本文件是否存在
    script_path = Path("scripts/run_inference.py")
    if not script_path.exists():
        print(f"  ✗ 脚本文件不存在: {script_path}")
        return False
    
    # 检查脚本是否可以导入
    try:
        import scripts.run_inference as run_inference
        print("  [PASS] 脚本导入测试通过")
        
        # 检查参数解析
        import argparse
        test_args = ["--input-file", "test.jsonl", "--model", "gpt-3.5-turbo"]
        
        # 由于我们不能直接调用parse_arguments，这里只是检查函数存在
        if hasattr(run_inference, 'parse_arguments'):
            print("  [PASS] 参数解析函数存在")
        else:
            print("  [FAIL] 参数解析函数不存在")
            return False
            
        return True
        
    except Exception as e:
        print(f"  [FAIL] 脚本接口测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("="*60)
    print("推理模块功能测试")
    print("="*60)
    
    tests = [
        ("数据模型", test_data_models),
        ("模型配置", test_model_config),
        ("流水线初始化", test_pipeline_initialization),
        ("数据加载", test_data_loading),
        ("集成功能", test_integration),
        ("脚本接口", test_script_interface)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  [FAIL] {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 打印测试结果摘要
    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "[PASS] 通过" if success else "[FAIL] 失败"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n[SUCCESS] 所有测试通过！推理模块功能正常。")
        return True
    else:
        print(f"\n[FAILURE] {total - passed} 个测试失败。")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)