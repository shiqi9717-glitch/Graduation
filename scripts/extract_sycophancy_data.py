#!/usr/bin/env python3
"""
从 sycophancy_database 中抽取数据并转换为项目格式
"""
import sys
import json
import pandas as pd
from pathlib import Path
import random

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import extract_question_core
from src.logging_config import logger


def extract_data_from_jsonl(jsonl_path: Path, sample_size: int = None, random_seed: int = 42) -> pd.DataFrame:
    """
    从 JSONL 文件中抽取数据
    
    Args:
        jsonl_path: JSONL 文件路径
        sample_size: 样本大小，如果为 None 则抽取所有数据
        random_seed: 随机种子
        
    Returns:
        抽取的数据 DataFrame
    """
    logger.info(f"从 {jsonl_path} 抽取数据...")
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 如果需要抽样，随机选择行
    if sample_size and sample_size < len(lines):
        random.seed(random_seed)
        selected_indices = random.sample(range(len(lines)), sample_size)
        selected_lines = [lines[i] for i in selected_indices]
    else:
        selected_lines = lines
        sample_size = len(lines)
    
    logger.info(f"处理 {len(selected_lines)} 条记录...")
    
    for i, line in enumerate(selected_lines, 1):
        try:
            item = json.loads(line.strip())
            
            # 提取核心问题
            core_question = extract_question_core(item.get('question', ''))
            
            data.append({
                'question_id': i,
                'original_question': item.get('question', ''),
                'core_question': core_question,
                'answer_matching_behavior': item.get('answer_matching_behavior', ''),
                'answer_not_matching_behavior': item.get('answer_not_matching_behavior', ''),
                'source_file': jsonl_path.name,
                'source_index': i,
                'category': jsonl_path.stem.replace('sycophancy_on_', ''),
                'difficulty': 'medium'  # 默认难度
            })
            
            if i % 1000 == 0:
                logger.info(f"已处理 {i}/{len(selected_lines)} 条记录")
                
        except json.JSONDecodeError as e:
            logger.warning(f"第 {i} 行 JSON 解析失败: {e}")
            continue
    
    df = pd.DataFrame(data)
    logger.info(f"成功抽取 {len(df)} 条记录从 {jsonl_path}")
    
    return df


def merge_datasets(data_dir: Path, sample_per_file: int = 100) -> pd.DataFrame:
    """
    合并多个数据集
    
    Args:
        data_dir: 数据目录路径
        sample_per_file: 每个文件抽取的样本数
        
    Returns:
        合并后的 DataFrame
    """
    all_data = []
    
    # 查找所有 JSONL 文件
    jsonl_files = list(data_dir.glob("*.jsonl"))
    logger.info(f"找到 {len(jsonl_files)} 个 JSONL 文件")
    
    for jsonl_file in jsonl_files:
        try:
            df = extract_data_from_jsonl(jsonl_file, sample_per_file)
            all_data.append(df)
            logger.info(f"处理完成: {jsonl_file.name} ({len(df)} 条记录)")
        except Exception as e:
            logger.error(f"处理文件 {jsonl_file} 失败: {e}")
    
    if not all_data:
        logger.error("没有找到可用的数据")
        return pd.DataFrame()
    
    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # 重新分配 question_id
    merged_df['question_id'] = range(1, len(merged_df) + 1)
    
    logger.info(f"合并完成: 总共 {len(merged_df)} 条记录")
    logger.info(f"数据来源分布: {merged_df['source_file'].value_counts().to_dict()}")
    
    return merged_df


def save_as_csv(df: pd.DataFrame, output_path: Path):
    """
    保存为 CSV 文件
    
    Args:
        df: 要保存的 DataFrame
        output_path: 输出路径
    """
    try:
        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"数据已保存到 {output_path}，共 {len(df)} 条记录")
        
        # 显示统计信息
        print("\n数据统计:")
        print(f"总记录数: {len(df)}")
        print(f"来源文件分布:")
        for file, count in df['source_file'].value_counts().items():
            print(f"  {file}: {count} 条")
        
        print(f"\n类别分布:")
        for category, count in df['category'].value_counts().items():
            print(f"  {category}: {count} 条")
        
        print(f"\n示例问题 (前3条):")
        for i, row in df.head(3).iterrows():
            core_q = row['core_question']
            preview = core_q[:100] + "..." if len(core_q) > 100 else core_q
            print(f"  {i+1}. {preview}")
            
    except Exception as e:
        logger.error(f"保存 CSV 文件失败: {e}")
        raise


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("sycophancy_database 数据抽取工具")
    logger.info("=" * 60)
    
    try:
        # 设置路径
        data_dir = project_root / "data" / "external" / "sycophancy_database"
        output_dir = project_root / "data" / "raw"
        output_path = output_dir / "sycophancy_extracted.csv"
        
        if not data_dir.exists():
            logger.error(f"数据目录不存在: {data_dir}")
            print(f"请确保 sycophancy_database 目录存在于: {data_dir.parent}")
            return 1
        
        # 合并数据集（每个文件抽取50条样本）
        merged_df = merge_datasets(data_dir, sample_per_file=50)
        
        if merged_df.empty:
            logger.error("没有成功抽取任何数据")
            return 1
        
        # 保存为 CSV
        save_as_csv(merged_df, output_path)
        
        # 创建配置文件更新
        config_update = f"""
# 自动生成的配置更新
# 数据已从 sycophancy_database 抽取并保存到:
# {output_path}

# 要使用这些数据，请更新 .env 文件中的 RAW_DATA_PATH:
RAW_DATA_PATH=data/raw/sycophancy_extracted.csv

# 或者直接在代码中指定:
# pipeline.run_pipeline(input_path=Path("data/raw/sycophancy_extracted.csv"))
"""
        
        print("\n" + "=" * 60)
        print("数据抽取完成!")
        print("=" * 60)
        print(config_update)
        
        return 0
        
    except Exception as e:
        logger.error(f"数据抽取失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
