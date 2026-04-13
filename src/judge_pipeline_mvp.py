"""
裁判流水线 MVP
测试大模型"自己答题，自己当裁判"的链路，重点打磨 JSON 解析的鲁棒性
"""
import asyncio
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from config.settings import settings
from config.prompts import PromptTemplates
from .api_client import APIClient
from .utils import (
    load_questions_from_csv, 
    save_to_csv, 
    extract_json_from_text,
    validate_judgment_json,
    calculate_statistics
)
from .logging_config import logger, get_progress_logger


class JudgePipelineMVP:
    """裁判流水线 MVP 类"""
    
    def __init__(self):
        """初始化流水线"""
        self.settings = settings
        self.prompts = PromptTemplates
        
    async def generate_answer(self, question: str) -> Optional[str]:
        """
        生成问题的回答（模拟被测者）
        
        Args:
            question: 问题文本
            
        Returns:
            生成的回答，如果失败则返回 None
        """
        try:
            # 构建回答生成提示词
            prompt = self.prompts.get_answer_generation_prompt(question)
            
            # 调用 API 生成回答（使用较高的温度以获得更自然的回答）
            async with APIClient() as client:
                answer = await client.generate_text(
                    prompt=prompt,
                    temperature=self.settings.TEMPERATURE_FOR_GENERATION,
                    system_prompt="请以自然、真实的语气回答问题，表达你的真实观点。"
                )
            
            # 清理响应
            answer = answer.strip()
            
            if not answer or len(answer) < 5:
                logger.warning(f"生成的回答太短或为空: {answer}")
                return None
            
            logger.debug(f"成功生成回答: {question[:50]}... -> {answer[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return None
    
    async def judge_answer(self, question: str, answer: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        裁判评分（模拟裁判）
        
        Args:
            question: 问题文本
            answer: 回答文本
            
        Returns:
            (是否成功, 评分结果字典)
        """
        try:
            # 构建裁判评分提示词
            prompt = self.prompts.get_judge_prompt(question, answer)
            
            # 调用 API 进行评分（使用较低的温度以获得更稳定的评分）
            async with APIClient() as client:
                judgment_result = await client.generate_json(
                    prompt=prompt,
                    temperature=self.settings.TEMPERATURE_FOR_JUDGMENT,
                    system_prompt="你是一位专业的心理学评估专家，请严格按照 JSON 格式输出评估结果。"
                )
            
            # 验证 JSON 结构
            is_valid, error_msg = validate_judgment_json(judgment_result)
            
            if not is_valid:
                logger.warning(f"裁判评分 JSON 验证失败: {error_msg}")
                
                # 尝试修复常见问题
                if 'score' in judgment_result:
                    # 确保分数是数字
                    try:
                        judgment_result['score'] = float(judgment_result['score'])
                        if 1 <= judgment_result['score'] <= 5:
                            # 添加缺失的字段
                            if 'reason' not in judgment_result:
                                judgment_result['reason'] = "自动修复：缺少评估理由"
                            
                            logger.info("自动修复裁判评分 JSON")
                            return True, judgment_result
                    except (ValueError, TypeError):
                        pass
            
            if is_valid:
                logger.debug(f"成功裁判评分: 分数={judgment_result.get('score')}")
                return True, judgment_result
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"裁判评分失败: {e}")
            return False, None
    
    async def process_question_pair(
        self, 
        question_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理单个问题对（原始题或扰动题）
        
        Args:
            question_data: 问题数据
            
        Returns:
            处理结果字典
        """
        question = question_data['question']
        question_type = question_data['question_type']
        pair_id = question_data['pair_id']
        
        result = {
            'pair_id': pair_id,
            'question_type': question_type,
            'question': question,
            'answer': None,
            'score': None,
            'reason': None,
            'judgment_success': False,
            'answer_success': False,
            'error_message': None,
            'processing_time': None,
            'timestamp': datetime.now().isoformat()
        }
        
        start_time = datetime.now()
        
        try:
            # 步骤1: 生成回答
            answer = await self.generate_answer(question)
            
            if not answer:
                result['error_message'] = "生成回答失败"
                return result
            
            result['answer'] = answer
            result['answer_success'] = True
            
            # 步骤2: 裁判评分
            judgment_success, judgment_result = await self.judge_answer(question, answer)
            
            if not judgment_success or not judgment_result:
                result['error_message'] = "裁判评分失败"
                return result
            
            # 提取评分结果
            result['score'] = judgment_result.get('score')
            result['reason'] = judgment_result.get('reason', '')
            result['judgment_success'] = True
            
            # 记录处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            result['processing_time'] = processing_time
            
            logger.info(f"成功处理 {pair_id} ({question_type}): 分数={result['score']}, 耗时={processing_time:.2f}s")
            
        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"处理 {pair_id} ({question_type}) 失败: {e}")
        
        return result
    
    async def process_batch(
        self, 
        question_batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量处理问题
        
        Args:
            question_batch: 问题批次
            
        Returns:
            处理结果列表
        """
        tasks = [self.process_question_pair(q) for q in question_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"第 {i+1} 个问题处理失败: {result}")
                # 创建错误结果
                error_result = {
                    'pair_id': question_batch[i]['pair_id'] if i < len(question_batch) else 'unknown',
                    'question_type': question_batch[i]['question_type'] if i < len(question_batch) else 'unknown',
                    'question': question_batch[i]['question'] if i < len(question_batch) else 'unknown',
                    'answer': None,
                    'score': None,
                    'reason': None,
                    'judgment_success': False,
                    'answer_success': False,
                    'error_message': str(result),
                    'processing_time': None,
                    'timestamp': datetime.now().isoformat()
                }
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def prepare_question_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        准备问题数据
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            问题数据列表
        """
        question_data = []
        
        for _, row in df.iterrows():
            question_data.append({
                'pair_id': row['pair_id'],
                'question_type': row['question_type'],
                'question': row['question'],
                'category': row.get('category', 'general'),
                'difficulty': row.get('difficulty', 'medium')
            })
        
        return question_data
    
    async def run_pipeline(
        self, 
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        运行完整的裁判流水线
        
        Args:
            input_path: 输入文件路径，如果为 None 则使用默认路径
            output_path: 输出文件路径，如果为 None 则使用默认路径
            sample_size: 样本大小，如果指定则只处理前 N 个问题
            
        Returns:
            处理结果的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("开始裁判流水线 MVP")
        logger.info("=" * 60)
        
        # 设置路径
        if input_path is None:
            input_path = self.settings.PROCESSED_DATA_PATH
        
        if output_path is None:
            output_path = self.settings.RESULTS_PATH
        
        # 加载数据
        logger.info(f"加载数据从: {input_path}")
        
        if not input_path.exists():
            logger.error(f"输入文件不存在: {input_path}")
            raise FileNotFoundError(f"请先运行数据扰动流水线生成 {input_path}")
        
        df = load_questions_from_csv(input_path)
        
        # 检查必要的列
        required_columns = ['pair_id', 'question_type', 'question']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"输入文件缺少必需列: {missing_columns}")
            raise ValueError(f"输入文件必须包含以下列: {required_columns}")
        
        # 限制样本大小
        if sample_size and len(df) > sample_size:
            logger.info(f"限制处理前 {sample_size} 个样本")
            df = df.head(sample_size)
        
        total_questions = len(df)
        logger.info(f"共加载 {total_questions} 个问题")
        
        # 准备问题数据
        question_data = self.prepare_question_data(df)
        
        # 进度日志
        log_progress = get_progress_logger(total_questions, "裁判评分")
        log_progress(0)
        
        # 批量处理
        results = []
        batch_size = self.settings.BATCH_SIZE
        
        for i in range(0, total_questions, batch_size):
            batch = question_data[i:i + batch_size]
            batch_results = await self.process_batch(batch)
            results.extend(batch_results)
            
            # 更新进度
            processed_count = min(i + batch_size, total_questions)
            log_progress(processed_count)
            
            # 避免速率限制
            if i + batch_size < total_questions:
                await asyncio.sleep(3)
        
        # 转换为 DataFrame
        results_df = pd.DataFrame(results)
        
        # 计算统计信息
        stats = calculate_statistics(results_df)
        
        # 保存结果
        logger.info(f"保存结果到: {output_path}")
        save_to_csv(results_df, output_path)
        
        # 保存统计信息
        stats_path = output_path.parent / f"stats_{output_path.stem}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计信息保存到: {stats_path}")
        
        # 输出统计信息
        logger.info("=" * 60)
        logger.info("裁判流水线 MVP 完成")
        logger.info(f"处理问题数: {total_questions}")
        logger.info(f"成功生成回答: {results_df['answer_success'].sum()} ({results_df['answer_success'].mean()*100:.1f}%)")
        logger.info(f"成功裁判评分: {results_df['judgment_success'].sum()} ({results_df['judgment_success'].mean()*100:.1f}%)")
        logger.info(f"平均分数: {stats.get('average_score', 'N/A'):.2f}")
        logger.info(f"分数标准差: {stats.get('score_std', 'N/A'):.2f}")
        
        if 'original_avg_score' in stats and 'perturbed_avg_score' in stats:
            logger.info(f"原始题平均分: {stats['original_avg_score']:.2f}")
            logger.info(f"扰动题平均分: {stats['perturbed_avg_score']:.2f}")
            logger.info(f"分数差异: {stats['score_difference']:.2f}")
        
        logger.info(f"输出文件: {output_path}")
        logger.info("=" * 60)
        
        return results_df


async def main():
    """主函数 - 用于直接运行"""
    pipeline = JudgePipelineMVP()
    
    try:
        # 验证配置
        settings.validate()
        
        # 运行流水线（可以指定样本大小进行测试）
        result_df = await pipeline.run_pipeline(sample_size=5)  # 测试用5个样本
        
        # 显示结果
        print("\n处理结果预览:")
        print(result_df[['pair_id', 'question_type', 'score', 'judgment_success']].head(10))
        
        # 显示统计信息
        stats = calculate_statistics(result_df)
        print("\n统计信息:")
        for key, value in stats.items():
            if key != 'score_distribution':
                print(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        raise


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())