#!/usr/bin/env python3
"""
推理模块主脚本
用于执行批量推理任务
"""

import asyncio
import argparse
import sys
from pathlib import Path
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import InferencePipeline, ModelConfig, QuestionType, ModelProvider
from config.inference_settings import InferenceSettings
from src.logging_config import setup_logger


logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="大模型推理流水线")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                       help="模型名称 (默认: gpt-3.5-turbo)")
    parser.add_argument("--provider", type=str, default="openai",
                       choices=["openai", "anthropic", "deepseek", "qwen", "custom"],
                       help="模型提供商 (默认: openai)")
    parser.add_argument("--api-key", type=str, default="",
                       help="API密钥 (默认从环境变量读取)")
    parser.add_argument("--api-base", type=str, default="",
                       help="API基础URL (默认使用提供商默认值)")
    
    # 数据参数
    parser.add_argument("--input-file", type=str, required=True,
                       help="输入文件路径 (JSONL或CSV格式)")
    parser.add_argument("--input-format", type=str, default="auto",
                       choices=["auto", "jsonl", "csv"],
                       help="输入文件格式 (默认: auto)")
    parser.add_argument("--question-column", type=str, default="question",
                       help="CSV文件中问题列名 (默认: question)")
    parser.add_argument("--question-type", type=str, default="sycophancy",
                       choices=["sycophancy", "control", "perturbed"],
                       help="问题类型 (默认: sycophancy)")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制处理的问题数量")
    
    # 推理参数
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数 (默认: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                       help="最大token数 (默认: 2048)")
    parser.add_argument("--system-prompt", type=str, default="",
                       help="系统提示词")
    parser.add_argument("--concurrency", type=int, default=5,
                       help="并发请求数 (默认: 5)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="批量大小 (默认: 10)")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="",
                       help="输出目录 (默认: data/results/inference/<timestamp>)")
    parser.add_argument("--output-format", type=str, default="all",
                       choices=["jsonl", "csv", "json", "all"],
                       help="输出格式 (默认: all)")
    parser.add_argument("--no-save", action="store_true",
                       help="不保存结果")
    
    # 日志参数
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别 (默认: INFO)")
    parser.add_argument("--log-file", type=str, default="",
                       help="日志文件路径 (默认: logs/inference.log)")
    
    return parser.parse_args()


def detect_file_format(file_path: Path) -> str:
    """检测文件格式"""
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    elif suffix == ".csv":
        return "csv"
    else:
        # 尝试通过内容检测
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith("{") and first_line.endswith("}"):
                    return "jsonl"
                elif "," in first_line:
                    return "csv"
        except:
            pass
        
        raise ValueError(f"无法检测文件格式: {file_path}")


async def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志
    log_file = args.log_file or InferenceSettings.LOG_FILE
    if log_file:
        log_file = Path(log_file)
    setup_logger(name="inference_pipeline", log_file=log_file, level=args.log_level)
    
    logger.info("="*60)
    logger.info("开始推理流水线")
    logger.info("="*60)
    
    try:
        # 1. 准备模型配置
        logger.info(f"准备模型配置: {args.model} ({args.provider})")
        
        model_config = ModelConfig(
            provider=ModelProvider(args.provider),
            model_name=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=InferenceSettings.DEFAULT_MODELS.get(args.model, {}).get("timeout", 30),
            max_retries=InferenceSettings.DEFAULT_MODELS.get(args.model, {}).get("max_retries", 3)
        )
        
        # 2. 初始化流水线
        logger.info("初始化推理流水线...")
        pipeline = InferencePipeline(model_config)
        await pipeline.initialize()
        
        # 3. 加载数据
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        # 检测文件格式
        if args.input_format == "auto":
            file_format = detect_file_format(input_path)
            logger.info(f"检测到文件格式: {file_format}")
        else:
            file_format = args.input_format
        
        # 加载问题
        logger.info(f"从 {input_path} 加载问题...")
        question_type = QuestionType(args.question_type)
        
        if file_format == "jsonl":
            questions = pipeline.load_questions_from_jsonl(
                input_path, question_type, args.limit
            )
        elif file_format == "csv":
            questions = pipeline.load_questions_from_csv(
                input_path, args.question_column, question_type, args.limit
            )
        else:
            raise ValueError(f"不支持的格式: {file_format}")
        
        if not questions:
            raise ValueError("未加载到任何问题")
        
        logger.info(f"成功加载 {len(questions)} 个问题")
        
        # 4. 创建推理请求
        logger.info("创建推理请求...")
        requests = pipeline.create_inference_requests(
            questions, args.system_prompt, args.temperature, args.max_tokens
        )
        
        # 5. 分批处理
        total_requests = len(requests)
        batch_size = args.batch_size
        batches = [requests[i:i + batch_size] for i in range(0, total_requests, batch_size)]
        
        logger.info(f"将 {total_requests} 个请求分为 {len(batches)} 批，每批 {batch_size} 个")
        
        # 6. 执行推理
        all_results = []
        for i, batch_requests in enumerate(batches, 1):
            logger.info(f"处理第 {i}/{len(batches)} 批 ({len(batch_requests)} 个请求)...")
            
            batch_result = await pipeline.run_batch(batch_requests, args.concurrency)
            all_results.extend(batch_result.responses)
            
            # 打印批次摘要
            logger.info(f"批次 {i} 完成: {batch_result.successful_requests} 成功, "
                       f"{batch_result.failed_requests} 失败, "
                       f"平均延迟: {batch_result.avg_latency_ms:.2f}ms")
            
            # 保存中间结果（如果启用）
            if InferenceSettings.SAVE_INTERMEDIATE and i % 5 == 0:
                intermediate_dir = Path(InferenceSettings.get_output_path("intermediate"))
                pipeline.save_results(intermediate_dir, "jsonl")
        
        # 7. 保存最终结果
        if not args.no_save:
            output_dir = Path(args.output_dir) if args.output_dir else None
            saved_files = pipeline.save_results(output_dir, args.output_format)
            
            logger.info("结果保存完成:")
            for file_type, file_path in saved_files.items():
                logger.info(f"  {file_type}: {file_path}")
        
        # 8. 打印最终摘要
        pipeline.print_summary()
        
        # 9. 获取统计信息
        stats = pipeline.get_statistics()
        if stats:
            logger.info(f"推理完成: {stats.successful_requests}/{stats.total_requests} "
                       f"成功 ({stats.success_rate:.2%})")
        
        logger.info("推理流水线执行完成")
        
    except Exception as e:
        logger.error(f"推理流水线执行失败: {e}", exc_info=True)
        raise
    
    finally:
        # 清理资源
        await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())