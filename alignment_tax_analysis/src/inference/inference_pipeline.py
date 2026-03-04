"""
推理流水线
管理整个推理过程，包括数据加载、批量处理、结果保存等
"""

import asyncio
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import pandas as pd

from .schemas import (
    QuestionData, InferenceRequest, InferenceResponse, ModelConfig,
    BatchInferenceRequest, BatchInferenceResult, InferenceStatistics,
    QuestionType, ModelProvider
)
from .model_client import ModelClient
from config.inference_settings import InferenceSettings
from src.utils import load_questions_from_file, save_to_csv


logger = logging.getLogger(__name__)


class InferencePipeline:
    """推理流水线"""
    
    def __init__(self, model_config: ModelConfig):
        """
        初始化推理流水线
        
        Args:
            model_config: 模型配置
        """
        self.model_config = model_config
        self.model_client: Optional[ModelClient] = None
        self.results: List[InferenceResponse] = []
        self.statistics: Optional[InferenceStatistics] = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
    
    async def initialize(self):
        """初始化流水线"""
        self.model_client = ModelClient(self.model_config)
        await self.model_client.connect()
        logger.info(f"Inference pipeline initialized with model: {self.model_config.model_name}")
    
    async def cleanup(self):
        """清理资源"""
        if self.model_client:
            await self.model_client.close()
            self.model_client = None
        logger.info("Inference pipeline cleaned up")
    
    def load_questions_from_jsonl(self, file_path: Path, 
                                 question_type: QuestionType = QuestionType.SYCOPHANCY,
                                 limit: Optional[int] = None) -> List[QuestionData]:
        """
        从JSONL文件加载问题
        
        Args:
            file_path: JSONL文件路径
            question_type: 问题类型
            limit: 限制加载数量
            
        Returns:
            List[QuestionData]: 问题数据列表
        """
        questions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit is not None and i >= limit:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # 提取问题文本
                        question_text = data.get("question", "")
                        if not question_text:
                            # 尝试其他可能的字段
                            question_text = data.get("text", data.get("content", ""))
                        
                        if question_text:
                            question = QuestionData(
                                question_id=data.get("id", str(uuid.uuid4())),
                                question_text=question_text,
                                question_type=question_type,
                                metadata=data
                            )
                            questions.append(question)
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error on line {i}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error loading questions from {file_path}: {e}")
            raise
        
        logger.info(f"Loaded {len(questions)} questions from {file_path}")
        return questions
    
    def load_questions_from_csv(self, file_path: Path,
                               question_column: str = "question",
                               question_type: QuestionType = QuestionType.SYCOPHANCY,
                               limit: Optional[int] = None) -> List[QuestionData]:
        """
        从CSV文件加载问题
        
        Args:
            file_path: CSV文件路径
            question_column: 问题列名
            question_type: 问题类型
            limit: 限制加载数量
            
        Returns:
            List[QuestionData]: 问题数据列表
        """
        try:
            df = pd.read_csv(file_path)
            
            if question_column not in df.columns:
                raise ValueError(f"Column '{question_column}' not found in CSV")
            
            questions = []
            for i, row in df.iterrows():
                if limit is not None and i >= limit:
                    break
                
                question = QuestionData(
                    question_id=str(uuid.uuid4()),
                    question_text=str(row[question_column]),
                    question_type=question_type,
                    metadata=row.to_dict()
                )
                questions.append(question)
                
            logger.info(f"Loaded {len(questions)} questions from {file_path}")
            return questions
            
        except Exception as e:
            logger.error(f"Error loading questions from CSV {file_path}: {e}")
            raise
    
    def create_inference_requests(self, questions: List[QuestionData],
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7,
                                 max_tokens: int = 2048) -> List[InferenceRequest]:
        """
        创建推理请求
        
        Args:
            questions: 问题数据列表
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            List[InferenceRequest]: 推理请求列表
        """
        requests = []
        
        for question in questions:
            request = InferenceRequest(
                request_id=str(uuid.uuid4()),
                question_data=question,
                model_name=self.model_config.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
            requests.append(request)
        
        logger.info(f"Created {len(requests)} inference requests")
        return requests
    
    async def run_batch(self, requests: List[InferenceRequest],
                       concurrency_limit: int = None) -> BatchInferenceResult:
        """
        运行批量推理
        
        Args:
            requests: 推理请求列表
            concurrency_limit: 并发限制
            
        Returns:
            BatchInferenceResult: 批量推理结果
        """
        if self.model_client is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        if concurrency_limit is None:
            concurrency_limit = InferenceSettings.CONCURRENT_REQUESTS
        
        logger.info(f"Running batch inference with {len(requests)} requests, "
                   f"concurrency: {concurrency_limit}")
        
        start_time = datetime.now()
        
        # 执行批量推理
        responses = await self.model_client.batch_infer(requests, concurrency_limit)
        
        # 计算统计信息
        total_latency = sum(r.latency_ms for r in responses if r.success)
        successful = sum(1 for r in responses if r.success)
        failed = len(responses) - successful
        avg_latency = total_latency / successful if successful > 0 else 0
        
        # 创建批量结果
        batch_result = BatchInferenceResult(
            batch_id=str(uuid.uuid4()),
            responses=responses,
            total_requests=len(requests),
            successful_requests=successful,
            failed_requests=failed,
            total_latency_ms=total_latency,
            avg_latency_ms=avg_latency
        )
        
        # 更新流水线状态
        self.results.extend(responses)
        self._update_statistics(batch_result)
        
        logger.info(f"Batch inference completed: {successful} successful, "
                   f"{failed} failed, avg latency: {avg_latency:.2f}ms")
        
        return batch_result
    
    def _update_statistics(self, batch_result: BatchInferenceResult):
        """更新统计信息"""
        if self.statistics is None:
            # 初始化统计信息
            latencies = [r.latency_ms for r in batch_result.responses if r.success]
            self.statistics = InferenceStatistics(
                model_name=self.model_config.model_name,
                total_requests=batch_result.total_requests,
                successful_requests=batch_result.successful_requests,
                failed_requests=batch_result.failed_requests,
                avg_latency_ms=batch_result.avg_latency_ms,
                min_latency_ms=min(latencies) if latencies else 0,
                max_latency_ms=max(latencies) if latencies else 0,
                success_rate=batch_result.successful_requests / batch_result.total_requests 
                if batch_result.total_requests > 0 else 0
            )
        else:
            # 更新现有统计信息
            all_responses = self.results
            successful_responses = [r for r in all_responses if r.success]
            latencies = [r.latency_ms for r in successful_responses]
            
            self.statistics.total_requests = len(all_responses)
            self.statistics.successful_requests = len(successful_responses)
            self.statistics.failed_requests = len(all_responses) - len(successful_responses)
            self.statistics.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
            self.statistics.min_latency_ms = min(latencies) if latencies else 0
            self.statistics.max_latency_ms = max(latencies) if latencies else 0
            self.statistics.success_rate = len(successful_responses) / len(all_responses) \
                if len(all_responses) > 0 else 0
    
    def save_results(self, output_dir: Optional[Path] = None,
                    format: str = "jsonl") -> Dict[str, Path]:
        """
        保存推理结果
        
        Args:
            output_dir: 输出目录
            format: 输出格式 (jsonl, csv, json)
            
        Returns:
            Dict[str, Path]: 保存的文件路径字典
        """
        if not self.results:
            logger.warning("No results to save")
            return {}
        
        # 确定输出目录
        if output_dir is None:
            output_dir = Path(InferenceSettings.get_output_path())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # 保存为JSONL格式
        if format in ["jsonl", "all"]:
            jsonl_path = output_dir / f"inference_results_{timestamp}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for response in self.results:
                    f.write(json.dumps(response.to_dict(), ensure_ascii=False) + "\n")
            saved_files["jsonl"] = jsonl_path
            logger.info(f"Saved {len(self.results)} results to {jsonl_path}")
        
        # 保存为CSV格式
        if format in ["csv", "all"]:
            csv_path = output_dir / f"inference_results_{timestamp}.csv"
            rows = []
            for response in self.results:
                row = {
                    "request_id": response.request_id,
                    "question_id": response.question_data.question_id,
                    "question_text": response.question_data.question_text[:100] + "..." 
                    if len(response.question_data.question_text) > 100 
                    else response.question_data.question_text,
                    "question_type": response.question_data.question_type.value,
                    "model_name": response.model_name,
                    "response_text": response.response_text[:200] + "..." 
                    if len(response.response_text) > 200 
                    else response.response_text,
                    "latency_ms": response.latency_ms,
                    "success": response.success,
                    "error_message": response.error_message or "",
                    "timestamp": response.timestamp.isoformat()
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            saved_files["csv"] = csv_path
            logger.info(f"Saved CSV results to {csv_path}")
        
        # 保存为JSON格式（完整）
        if format in ["json", "all"]:
            json_path = output_dir / f"inference_results_full_{timestamp}.json"
            results_dict = {
                "metadata": {
                    "model_name": self.model_config.model_name,
                    "total_requests": len(self.results),
                    "successful_requests": sum(1 for r in self.results if r.success),
                    "failed_requests": sum(1 for r in self.results if not r.success),
                    "timestamp": timestamp
                },
                "results": [r.to_dict() for r in self.results]
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
            saved_files["json"] = json_path
            logger.info(f"Saved full JSON results to {json_path}")
        
        # 保存统计信息
        if self.statistics:
            stats_path = output_dir / f"inference_statistics_{timestamp}.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.statistics.to_dict(), f, ensure_ascii=False, indent=2)
            saved_files["statistics"] = stats_path
            logger.info(f"Saved statistics to {stats_path}")
        
        return saved_files
    
    def get_statistics(self) -> Optional[InferenceStatistics]:
        """获取统计信息"""
        return self.statistics
    
    def print_summary(self):
        """打印摘要信息"""
        if not self.statistics:
            print("No statistics available")
            return
        
        print("\n" + "="*60)
        print("INFERENCE PIPELINE SUMMARY")
        print("="*60)
        print(f"Model: {self.statistics.model_name}")
        print(f"Total Requests: {self.statistics.total_requests}")
        print(f"Successful: {self.statistics.successful_requests}")
        print(f"Failed: {self.statistics.failed_requests}")
        print(f"Success Rate: {self.statistics.success_rate:.2%}")
        print(f"Average Latency: {self.statistics.avg_latency_ms:.2f} ms")
        print(f"Min Latency: {self.statistics.min_latency_ms:.2f} ms")
        print(f"Max Latency: {self.statistics.max_latency_ms:.2f} ms")
        
        if self.statistics.error_types:
            print("\nError Types:")
            for error_type, count in self.statistics.error_types.items():
                print(f"  {error_type}: {count}")
        
        print("="*60)