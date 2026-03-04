"""
数据扰动流水线
处理阿谀奉承测试题，生成实验所需的对照组数据
"""
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from config.settings import settings
from config.prompts import PromptTemplates
from .api_client import APIClient
from .utils import load_questions_from_file, save_to_csv, create_sample_data, extract_question_core
from .logging_config import logger, get_progress_logger


class DataPerturbationPipeline:
    """数据扰动流水线类"""
    
    def __init__(self):
        """初始化流水线"""
        self.settings = settings
        self.prompts = PromptTemplates
        
    async def perturb_question(self, original_question: str, is_sycophancy_data: bool = False, use_background_perturbation: bool = False) -> Optional[str]:
        """
        对单个问题进行扰动
        
        Args:
            original_question: 原始问题
            is_sycophancy_data: 是否为 sycophancy 数据库数据（包含个人简介）
            use_background_perturbation: 是否使用完整的背景扰动（包含姓名、职业、地点等）
            
        Returns:
            扰动后的问题，如果失败则返回 None
        """
        try:
            # 如果是 sycophancy 数据且使用背景扰动，直接使用完整问题
            if is_sycophancy_data and use_background_perturbation:
                # 使用完整的背景扰动提示词
                prompt = self.prompts.get_background_perturbation_prompt(original_question)
                logger.debug(f"使用完整背景扰动，输入长度: {len(original_question)} 字符")
            else:
                # 如果是 sycophancy 数据，提取核心问题部分
                if is_sycophancy_data:
                    core_question = extract_question_core(original_question)
                    logger.debug(f"提取核心问题: {core_question[:100]}...")
                else:
                    core_question = original_question
                
                # 构建扰动提示词
                prompt = self.prompts.get_perturbation_prompt(core_question)
            
            # 调用 API 生成扰动题
            async with APIClient() as client:
                perturbed_question = await client.generate_text(
                    prompt=prompt,
                    temperature=0.7,  # 较高的温度以获得更多样化的扰动
                    system_prompt="你是一位专业的语言学家和心理学家，擅长设计心理测试题。"
                )
            
            # 清理响应
            perturbed_question = perturbed_question.strip()
            
            # 验证扰动结果
            if not perturbed_question or len(perturbed_question) < 10:
                logger.warning(f"扰动结果太短或为空: {perturbed_question}")
                return None
            
            if perturbed_question == original_question:
                logger.warning("扰动结果与原始问题相同")
                return None
            
            # 如果是 sycophancy 数据且使用背景扰动，直接返回完整扰动结果
            if is_sycophancy_data and use_background_perturbation:
                final_question = perturbed_question
            else:
                # 如果是 sycophancy 数据，将扰动后的核心问题重新包装
                if is_sycophancy_data:
                    # 这里可以添加逻辑将扰动后的核心问题重新包装成类似格式
                    # 但为了简化，我们直接使用扰动后的核心问题
                    final_question = perturbed_question
                else:
                    final_question = perturbed_question
            
            logger.debug(f"成功扰动问题: {original_question[:50]}... -> {final_question[:50]}...")
            return final_question
            
        except Exception as e:
            logger.error(f"扰动问题失败: {e}")
            return None
    
    async def process_batch(self, questions: List[str], is_sycophancy_data: bool = False, use_background_perturbation: bool = False) -> List[Optional[str]]:
        """
        批量处理问题扰动
        
        Args:
            questions: 原始问题列表
            is_sycophancy_data: 是否为 sycophancy 数据库数据
            use_background_perturbation: 是否使用完整的背景扰动
            
        Returns:
            扰动后的问题列表（失败则为 None）
        """
        tasks = [self.perturb_question(q, is_sycophancy_data, use_background_perturbation) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"第 {i+1} 个问题扰动失败: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def create_control_group_dataframe(
        self,
        original_df: pd.DataFrame,
        perturbed_questions: List[Optional[str]]
    ) -> pd.DataFrame:
        """
        创建对照组数据 DataFrame
        
        Args:
            original_df: 原始数据 DataFrame
            perturbed_questions: 扰动后的问题列表
            
        Returns:
            包含原始题和扰动题的 DataFrame
        """
        control_data = []
        
        for idx, row in original_df.iterrows():
            original_question = row['original_question']
            perturbed_question = perturbed_questions[idx] if idx < len(perturbed_questions) else None
            
            # 基础信息
            base_info = {
                'question_id': row.get('question_id', idx + 1),
                'pair_id': f"pair_{idx + 1:03d}",
                'source_file': row.get('source_file', 'unknown'),
                'line_number': row.get('line_number', idx + 1),
                'perturbed_success': perturbed_question is not None,
                'timestamp': datetime.now().isoformat()
            }
            
            # 如果是 sycophancy 数据，添加额外信息
            if 'answer_matching_behavior' in row:
                base_info.update({
                    'answer_matching_behavior': row['answer_matching_behavior'],
                    'answer_not_matching_behavior': row['answer_not_matching_behavior']
                })
            
            # 原始题记录
            original_record = base_info.copy()
            original_record.update({
                'question_type': 'original',
                'question': original_question,
                'category': row.get('category', 'general'),
                'difficulty': row.get('difficulty', 'medium'),
            })
            
            # 如果有核心问题列，也保存
            if 'core_question' in row:
                original_record['core_question'] = row['core_question']
            
            control_data.append(original_record)
            
            # 扰动题记录（如果成功生成）
            if perturbed_question:
                perturbed_record = base_info.copy()
                perturbed_record.update({
                    'question_type': 'perturbed',
                    'question': perturbed_question,
                    'category': row.get('category', 'general'),
                    'difficulty': row.get('difficulty', 'medium'),
                })
                control_data.append(perturbed_record)
        
        return pd.DataFrame(control_data)
    
    async def run_pipeline(
        self, 
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        运行完整的数据扰动流水线
        
        Args:
            input_path: 输入文件路径，如果为 None 则使用默认路径或创建示例数据
            output_path: 输出文件路径，如果为 None 则使用默认路径
            sample_size: 样本大小，如果指定则只处理前 N 个问题
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("开始数据扰动流水线")
        logger.info("=" * 60)
        
        # 设置路径
        if input_path is None:
            input_path = self.settings.RAW_DATA_PATH
        
        if output_path is None:
            output_path = self.settings.PROCESSED_DATA_PATH
        
        # 加载数据
        logger.info(f"加载数据从: {input_path}")
        
        if input_path.exists():
            df = load_questions_from_file(input_path)
        else:
            logger.warning(f"输入文件不存在: {input_path}")
            logger.info("创建示例数据...")
            df = create_sample_data(sample_size or 10)
            save_to_csv(df, input_path)
        
        # 限制样本大小
        if sample_size and len(df) > sample_size:
            logger.info(f"限制处理前 {sample_size} 个样本")
            df = df.head(sample_size)
        
        total_questions = len(df)
        logger.info(f"共加载 {total_questions} 个问题")
        
        # 检查是否为 sycophancy 数据
        is_sycophancy_data = 'answer_matching_behavior' in df.columns
        
        if is_sycophancy_data:
            logger.info("检测到 sycophancy 数据库格式数据")
            # 提取核心问题并保存到新列
            df['core_question'] = df['original_question'].apply(extract_question_core)
            original_questions = df['core_question'].tolist()
        else:
            original_questions = df['original_question'].tolist()
        
        # 进度日志
        log_progress = get_progress_logger(total_questions, "数据扰动")
        log_progress(0)
        
        # 批量处理扰动
        perturbed_questions = []
        batch_size = self.settings.BATCH_SIZE
        
        for i in range(0, total_questions, batch_size):
            batch = original_questions[i:i + batch_size]
            batch_results = await self.process_batch(batch, is_sycophancy_data)
            perturbed_questions.extend(batch_results)
            
            # 更新进度
            processed_count = min(i + batch_size, total_questions)
            log_progress(processed_count)
            
            # 避免速率限制
            if i + batch_size < total_questions:
                await asyncio.sleep(2)
        
        # 统计扰动成功率
        success_count = sum(1 for q in perturbed_questions if q is not None)
        success_rate = success_count / total_questions * 100
        logger.info(f"扰动完成: 成功 {success_count}/{total_questions} ({success_rate:.1f}%)")
        
        # 创建对照组数据
        logger.info("创建对照组数据...")
        control_df = self.create_control_group_dataframe(df, perturbed_questions)
        
        # 保存结果
        logger.info(f"保存结果到: {output_path}")
        save_to_csv(control_df, output_path)
        
        # 输出统计信息
        logger.info("=" * 60)
        logger.info("数据扰动流水线完成")
        logger.info(f"原始问题数: {total_questions}")
        logger.info(f"成功扰动数: {success_count}")
        logger.info(f"总记录数: {len(control_df)}")
        logger.info(f"输出文件: {output_path}")
        logger.info("=" * 60)
        
        return control_df


async def main():
    """主函数 - 用于直接运行"""
    pipeline = DataPerturbationPipeline()
    
    try:
        # 验证配置
        settings.validate()
        
        # 运行流水线
        result_df = await pipeline.run_pipeline()
        
        # 显示前几行
        print("\n处理结果预览:")
        print(result_df[['pair_id', 'question_type', 'question']].head(10))
        
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        raise


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())