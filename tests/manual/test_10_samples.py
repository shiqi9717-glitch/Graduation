#!/usr/bin/env python3
"""
测试10条数据：自动提取核心问题 + API自动扰动
结果以JSON格式保存，保持原数据集结构
"""
import sys
import json
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 允许直接运行此脚本时导入共享测试工具
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tests.manual._shared import (
    get_env_file,
    get_samples_dir,
    get_sycophancy_data_dir,
)

from src.utils import load_questions_from_file, extract_question_core
from src.data_pipeline import DataPerturbationPipeline
from src.logging_config import logger


def extract_name_from_question(question_text: str) -> str:
    """从问题文本中提取姓名"""
    import re
    
    patterns = [
        r"Hello, my name is ([A-Z][a-z]+ [A-Z][a-z]+)",
        r"my name is ([A-Z][a-z]+ [A-Z][a-z]+)",
        r"I'm ([A-Z][a-z]+ [A-Z][a-z]+)",
        r"I am ([A-Z][a-z]+ [A-Z][a-z]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question_text)
        if match:
            return match.group(1)
    
    return "未知"


async def create_chinese_copy(perturbed_questions, source_file_name, use_background_perturbation: bool = True):
    """创建中文副本 - 将英文扰动结果转换为中文描述"""
    output_data_zh = []
    
    for item in perturbed_questions:
        if item['success']:
            # 提取核心问题的中文描述
            core_question = item['core_question']
            # 简化核心问题，只保留问题部分
            core_question_zh = core_question.split('\n\n')[0] if '\n\n' in core_question else core_question
            core_question_zh = core_question_zh[:100] + "..." if len(core_question_zh) > 100 else core_question_zh
            
            # 提取英文扰动结果
            perturbed_en = item['perturbed_question_en']
            
            # 提取姓名信息用于对比
            original_name = extract_name_from_question(item['original_full'])
            perturbed_name = extract_name_from_question(perturbed_en)
            
            # 根据扰动类型生成不同的中文说明
            if use_background_perturbation:
                chinese_note = f"这是完整背景扰动版本：姓名从{original_name}改为{perturbed_name}，职业、地点、兴趣全部更换，但核心立场保持不变。"
                change_analysis = f"背景信息完全更改：姓名({original_name}→{perturbed_name})、职业、地点、兴趣全部更换"
            else:
                chinese_note = f"这是核心问题扰动版本：保持原始背景信息不变，仅对核心问题进行同义替换和语境迁移。"
                change_analysis = f"仅核心问题扰动：背景信息保持不变，问题表述方式改变但测试目标相同"
            
            # 创建中文描述
            output_item = {
                'pair_id': f"pair_{item['index']+1:03d}",
                '原始数据': {
                    '完整问题': item['original_full'][:150] + "..." if len(item['original_full']) > 150 else item['original_full'],
                    '匹配答案': item['answer_matching_behavior'],
                    '不匹配答案': item['answer_not_matching_behavior'],
                    '来源文件': source_file_name,
                    '来源索引': item['source_index']
                },
                '处理数据': {
                    '核心问题': core_question_zh,
                    '英文扰动结果': perturbed_en,
                    '中文说明': chinese_note,
                    '处理时间': datetime.now().isoformat(),
                    '处理成功': True,
                    '扰动类型': "完整背景扰动" if use_background_perturbation else "核心问题扰动"
                },
                '对比分析': {
                    '原始姓名': original_name,
                    '扰动后姓名': perturbed_name,
                    '姓名是否更改': original_name != perturbed_name,
                    '扰动变化': change_analysis,
                    '阿谀奉承诱导': "保持原问题的诱导核心：测试回答者是否倾向于迎合提问者的观点"
                }
            }
            output_data_zh.append(output_item)
    
    return output_data_zh


async def test_10_samples(use_background_perturbation: bool = True):
    """测试10条数据 - 生成英文扰动结果（支持完整背景扰动）"""
    perturbation_type = "完整背景扰动" if use_background_perturbation else "核心问题扰动"
    logger.info("=" * 60)
    logger.info(f"开始测试：10条数据自动提取核心问题 + API扰动 ({perturbation_type})")
    logger.info("=" * 60)
    
    try:
        # 1. 加载数据
        data_dir = get_sycophancy_data_dir()
        jsonl_files = list(data_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            logger.error("未找到 sycophancy_database 数据文件")
            return None
        
        # 使用第一个文件
        source_file = jsonl_files[0]
        logger.info(f"使用数据文件: {source_file.name}")
        
        # 2. 加载10条数据
        df = load_questions_from_file(source_file)
        df = df.head(10)  # 取前10条
        
        logger.info(f"加载 {len(df)} 条数据")
        
        # 3. 提取核心问题（仅用于记录，背景扰动不使用核心问题提取）
        logger.info("提取核心问题...")
        df['core_question'] = df['original_question'].apply(extract_question_core)
        
        # 4. 创建数据扰动流水线
        pipeline = DataPerturbationPipeline()
        
        # 5. 对每条数据进行扰动
        logger.info(f"开始API自动扰动 ({perturbation_type})...")
        perturbed_questions = []
        
        for i, row in df.iterrows():
            logger.info(f"处理第 {i+1}/{len(df)} 条数据...")
            
            original_question = row['original_question']
            core_question = row['core_question']
            
            # 调用API进行扰动（使用背景扰动）
            perturbed = await pipeline.perturb_question(
                original_question,
                is_sycophancy_data=True,
                use_background_perturbation=use_background_perturbation
            )
            
            perturbed_questions.append({
                'index': i,
                'original_full': original_question,
                'core_question': core_question,
                'perturbed_question_en': perturbed,  # 英文扰动结果
                'success': perturbed is not None,
                'answer_matching_behavior': row.get('answer_matching_behavior', ''),
                'answer_not_matching_behavior': row.get('answer_not_matching_behavior', ''),
                'source_file': row.get('source_file', ''),
                'source_index': row.get('source_index', i),
                'perturbation_type': perturbation_type
            })
            
            # 显示进度
            if perturbed:
                if use_background_perturbation:
                    # 对于背景扰动，显示姓名变化
                    original_name = extract_name_from_question(original_question)
                    perturbed_name = extract_name_from_question(perturbed)
                    logger.info(f"  原始姓名: {original_name}")
                    logger.info(f"  扰动姓名: {perturbed_name}")
                    logger.info(f"  扰动 (英文): {perturbed[:80]}...")
                else:
                    logger.info(f"  原始: {core_question[:60]}...")
                    logger.info(f"  扰动 (英文): {perturbed[:60]}...")
            else:
                logger.warning(f"  扰动失败")
            
            # 避免速率限制
            if i < len(df) - 1:
                await asyncio.sleep(1)
        
        # 6. 统计结果
        success_count = sum(1 for item in perturbed_questions if item['success'])
        success_rate = success_count / len(perturbed_questions) * 100
        
        logger.info(f"扰动完成: 成功 {success_count}/{len(perturbed_questions)} ({success_rate:.1f}%)")
        
        # 7. 保存结果为JSON格式（英文版本）
        output_data_en = []
        for item in perturbed_questions:
            if item['success']:
                # 创建类似原数据集的JSON结构
                output_item = {
                    'original_data': {
                        'question': item['original_full'],
                        'answer_matching_behavior': item['answer_matching_behavior'],
                        'answer_not_matching_behavior': item['answer_not_matching_behavior'],
                        'source_file': item['source_file'],
                        'source_index': item['source_index']
                    },
                    'processed_data': {
                        'core_question': item['core_question'],
                        'perturbed_question_en': item['perturbed_question_en'],  # 英文扰动结果
                        'processing_time': datetime.now().isoformat(),
                        'success': item['success'],
                        'perturbation_type': item['perturbation_type']
                    },
                    'pair_id': f"pair_{item['index']+1:03d}"
                }
                output_data_en.append(output_item)
        
        # 8. 保存英文JSON文件
        output_dir = get_samples_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = "_background" if use_background_perturbation else "_core"
        output_file_en = output_dir / f"perturbation_test_10_samples{suffix}.json"
        
        with open(output_file_en, 'w', encoding='utf-8') as f:
            json.dump({
                'test_info': {
                    'test_date': datetime.now().isoformat(),
                    'sample_size': len(df),
                    'success_rate': success_rate,
                    'source_file': source_file.name,
                    'total_processed': len(perturbed_questions),
                    'successful_perturbations': success_count,
                    'language': 'english',
                    'perturbation_type': perturbation_type,
                    'use_background_perturbation': use_background_perturbation
                },
                'test_results': output_data_en
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"英文结果已保存到: {output_file_en}")
        
        # 9. 创建中文副本
        logger.info("创建中文副本...")
        output_data_zh = await create_chinese_copy(perturbed_questions, source_file.name, use_background_perturbation)
        
        output_file_zh = output_dir / f"perturbation_test_10_samples{suffix}_zh.json"
        
        with open(output_file_zh, 'w', encoding='utf-8') as f:
            json.dump({
                'test_info': {
                    'test_date': datetime.now().isoformat(),
                    'sample_size': len(df),
                    'success_rate': success_rate,
                    'source_file': source_file.name,
                    'total_processed': len(perturbed_questions),
                    'successful_perturbations': success_count,
                    'language': 'chinese',
                    'perturbation_type': perturbation_type,
                    'note': '此文件为方便查看的中文副本，实际处理使用英文版本'
                },
                'test_results': output_data_zh
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"中文副本已保存到: {output_file_zh}")
        
        # 10. 同时保存为CSV便于查看
        csv_data = []
        for item in perturbed_questions:
            csv_data.append({
                'pair_id': f"pair_{item['index']+1:03d}",
                'original_question': item['original_full'][:200] + "..." if len(item['original_full']) > 200 else item['original_full'],
                'core_question': item['core_question'],
                'perturbed_question_en': item['perturbed_question_en'] if item['success'] else "[扰动失败]",
                'success': item['success'],
                'answer_matching': item['answer_matching_behavior'],
                'answer_not_matching': item['answer_not_matching_behavior'],
                'perturbation_type': item['perturbation_type']
            })
        
        csv_file = output_dir / f"perturbation_test_10_samples{suffix}.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"CSV格式结果已保存到: {csv_file}")
        
        return {
            'output_file_en': output_file_en,
            'output_file_zh': output_file_zh,
            'csv_file': csv_file,
            'results': perturbed_questions,
            'success_rate': success_rate,
            'perturbation_type': perturbation_type
        }
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_results_summary(results):
    """打印结果摘要"""
    if not results:
        print("测试失败，无结果")
        return
    
    perturbation_type = results.get('perturbation_type', '未知类型')
    
    print("\n" + "=" * 60)
    print(f"测试完成！ ({perturbation_type})")
    print("=" * 60)
    
    print(f"\n结果文件位置:")
    print(f"1. 英文JSON格式: {results['output_file_en']}")
    print(f"2. 中文副本JSON: {results['output_file_zh']}")
    print(f"3. CSV格式: {results['csv_file']}")
    
    print(f"\n测试统计:")
    print(f"  处理数据量: {len(results['results'])} 条")
    print(f"  成功扰动: {sum(1 for r in results['results'] if r['success'])} 条")
    print(f"  成功率: {results['success_rate']:.1f}%")
    print(f"  扰动类型: {perturbation_type}")
    
    print(f"\n示例对比 (前3条成功扰动的数据):")
    success_count = 0
    for item in results['results']:
        if item['success'] and success_count < 3:
            success_count += 1
            print(f"\n[示例 {success_count}]")
            
            # 提取姓名信息
            original_name = extract_name_from_question(item['original_full'])
            perturbed_name = extract_name_from_question(item['perturbed_question_en'])
            
            if perturbation_type == "完整背景扰动":
                print(f"  原始姓名: {original_name}")
                print(f"  扰动姓名: {perturbed_name}")
                print(f"  姓名是否更改: {'是' if original_name != perturbed_name else '否'}")
            
            print(f"  原始问题核心: {item['core_question'][:80]}...")
            print(f"  扰动后问题 (英文): {item['perturbed_question_en'][:80]}...")
            print(f"  答案匹配: {item['answer_matching_behavior']}")
            print(f"  答案不匹配: {item['answer_not_matching_behavior']}")
    
    print(f"\n" + "=" * 60)
    print("如何查看完整结果:")
    print("=" * 60)
    print(f"1. 查看英文JSON文件:")
    print(f"   cat {results['output_file_en']} | head -100")
    print(f"\n2. 查看中文副本:")
    print(f"   cat {results['output_file_zh']} | head -100")
    print(f"\n3. 查看CSV文件:")
    print(f"   cat {results['csv_file']}")
    print(f"\n4. 使用Python查看:")
    print(f"   import pandas as pd")
    print(f"   df = pd.read_csv('{results['csv_file']}')")
    print(f"   print(df[['pair_id', 'core_question', 'perturbed_question_en']].head())")


async def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试数据扰动流水线')
    parser.add_argument('--background', action='store_true',
                       help='使用完整背景扰动（默认）')
    parser.add_argument('--core-only', action='store_true',
                       help='仅使用核心问题扰动（不更换背景信息）')
    args = parser.parse_args()
    
    # 确定扰动类型
    use_background_perturbation = True  # 默认使用背景扰动
    if args.core_only:
        use_background_perturbation = False
    elif args.background:
        use_background_perturbation = True
    
    perturbation_type = "完整背景扰动" if use_background_perturbation else "核心问题扰动"
    
    print("=" * 60)
    print(f"10条数据测试：自动提取核心问题 + API自动扰动 ({perturbation_type})")
    print("=" * 60)
    print("注意：")
    print("1. 需要配置 DeepSeek API Key 在 .env 文件中")
    print("2. 需要安装依赖: pip install -r requirements.txt")
    print("3. 测试约需1-2分钟")
    print("=" * 60)
    print(f"扰动类型: {perturbation_type}")
    print(f"   - 完整背景扰动: 更换姓名、职业、地点、兴趣等所有背景信息")
    print(f"   - 核心问题扰动: 仅对核心问题进行同义替换，保持背景信息不变")
    print("=" * 60)
    
    # 检查环境配置
    env_file = get_env_file()
    if not env_file.exists():
        print(f"\n警告: 未找到 .env 文件")
        print(f"请先复制 config/.env.example 为 config/.env 并配置 API Key")
        print(f"命令: cp config/.env.example config/.env")
        return 1
    
    # 运行测试
    results = await test_10_samples(use_background_perturbation=use_background_perturbation)
    
    if results:
        print_results_summary(results)
        return 0
    else:
        print("\n测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    # 运行异步主函数
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
