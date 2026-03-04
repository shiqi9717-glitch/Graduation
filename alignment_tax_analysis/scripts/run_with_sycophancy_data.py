#!/usr/bin/env python3
"""
直接使用 sycophancy_database 中的 JSONL 文件运行数据扰动流水线
"""
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline import DataPerturbationPipeline
from src.logging_config import logger


async def run_with_jsonl(jsonl_path: Path, sample_size: int = 10):
    """
    使用 JSONL 文件运行数据扰动流水线
    
    Args:
        jsonl_path: JSONL 文件路径
        sample_size: 样本大小
    """
    logger.info("=" * 60)
    logger.info(f"使用 sycophancy 数据运行数据扰动流水线")
    logger.info(f"数据文件: {jsonl_path}")
    logger.info(f"样本大小: {sample_size}")
    logger.info("=" * 60)
    
    try:
        # 创建流水线实例
        pipeline = DataPerturbationPipeline()
        
        # 直接使用 JSONL 文件运行流水线
        result_df = await pipeline.run_pipeline(
            input_path=jsonl_path,
            sample_size=sample_size
        )
        
        # 显示结果摘要
        print("\n" + "=" * 60)
        print("数据扰动流水线完成 (使用 sycophancy 数据)")
        print("=" * 60)
        
        total_original = len(result_df[result_df['question_type'] == 'original'])
        total_perturbed = len(result_df[result_df['question_type'] == 'perturbed'])
        
        if total_original > 0:
            success_rate = total_perturbed / total_original * 100
        else:
            success_rate = 0
        
        print(f"原始问题数: {total_original}")
        print(f"成功扰动数: {total_perturbed}")
        print(f"扰动成功率: {success_rate:.1f}%")
        print(f"总记录数: {len(result_df)}")
        
        # 显示示例
        print("\n示例数据 (原始题 vs 扰动题):")
        sample_pairs = result_df[result_df['pair_id'].isin(['pair_001', 'pair_002', 'pair_003'])]
        
        for pair_id in ['pair_001', 'pair_002', 'pair_003']:
            pair_data = sample_pairs[sample_pairs['pair_id'] == pair_id]
            if len(pair_data) == 2:  # 应该有原始题和扰动题
                original = pair_data[pair_data['question_type'] == 'original'].iloc[0]
                perturbed = pair_data[pair_data['question_type'] == 'perturbed'].iloc[0] if len(pair_data[pair_data['question_type'] == 'perturbed']) > 0 else None
                
                print(f"\n{pair_id}:")
                print(f"  原始题: {original['question'][:80]}...")
                if perturbed is not None:
                    print(f"  扰动题: {perturbed['question'][:80]}...")
                else:
                    print(f"  扰动题: [生成失败]")
        
        # 保存路径信息
        output_path = pipeline.settings.PROCESSED_DATA_PATH
        print(f"\n输出文件: {output_path}")
        print("=" * 60)
        
        return result_df
        
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        raise


def list_sycophancy_files():
    """列出可用的 sycophancy 数据文件"""
    data_dir = Path("../../sycophancy_database").resolve()
    
    if not data_dir.exists():
        print(f"错误: sycophancy_database 目录不存在于 {data_dir}")
        return []
    
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    print("\n可用的 sycophancy 数据文件:")
    for i, file in enumerate(jsonl_files, 1):
        # 获取文件大小
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {i}. {file.name} ({size_mb:.1f} MB)")
    
    return jsonl_files


async def main():
    """主函数"""
    print("=" * 60)
    print("sycophancy_database 数据扰动流水线")
    print("=" * 60)
    
    # 列出可用文件
    jsonl_files = list_sycophancy_files()
    
    if not jsonl_files:
        print("没有找到 sycophancy 数据文件")
        return 1
    
    try:
        # 选择第一个文件进行测试
        selected_file = jsonl_files[0]
        print(f"\n选择文件: {selected_file.name}")
        
        # 设置样本大小（小样本测试）
        sample_size = 5
        
        print(f"使用样本大小: {sample_size} (测试用)")
        print("注意: 完整运行可能需要较长时间，建议先小样本测试")
        
        # 运行流水线
        result_df = await run_with_jsonl(selected_file, sample_size)
        
        print("\n下一步:")
        print("1. 检查输出文件中的数据质量")
        print("2. 调整样本大小进行完整运行")
        print("3. 运行裁判流水线进行评估")
        print(f"   命令: python scripts/run_judge_pipeline.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # 运行异步主函数
    exit_code = asyncio.run(main())
    sys.exit(exit_code)