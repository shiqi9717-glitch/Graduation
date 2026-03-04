#!/usr/bin/env python3
"""
数据扰动流水线运行脚本
"""
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline import DataPerturbationPipeline
from src.logging_config import logger


async def main():
    """主函数"""
    logger.info("启动数据扰动流水线...")
    
    try:
        # 创建流水线实例
        pipeline = DataPerturbationPipeline()
        
        # 运行流水线（可以调整样本大小）
        result_df = await pipeline.run_pipeline(sample_size=5)  # 测试用5个样本
        
        # 显示结果摘要
        print("\n" + "="*60)
        print("数据扰动流水线完成")
        print("="*60)
        
        total_original = len(result_df[result_df['question_type'] == 'original'])
        total_perturbed = len(result_df[result_df['question_type'] == 'perturbed'])
        success_rate = total_perturbed / total_original * 100
        
        print(f"原始问题数: {total_original}")
        print(f"成功扰动数: {total_perturbed}")
        print(f"扰动成功率: {success_rate:.1f}%")
        print(f"总记录数: {len(result_df)}")
        
        # 显示前几个示例
        print("\n示例数据:")
        sample = result_df[['pair_id', 'question_type', 'question']].head(6)
        for _, row in sample.iterrows():
            question_preview = row['question'][:50] + "..." if len(row['question']) > 50 else row['question']
            print(f"  {row['pair_id']} ({row['question_type']}): {question_preview}")
        
        return 0
        
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        print(f"\n错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # 运行异步主函数
    exit_code = asyncio.run(main())
    sys.exit(exit_code)