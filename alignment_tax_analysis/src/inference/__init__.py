"""
推理模块包初始化文件
包含被测模型推理相关的核心组件
"""

from .schemas import (
    InferenceRequest, InferenceResponse, ModelConfig,
    QuestionData, QuestionType, ModelProvider,
    BatchInferenceRequest, BatchInferenceResult, InferenceStatistics
)
from .model_client import ModelClient, ModelClientError, RateLimitError, AuthenticationError
from .inference_pipeline import InferencePipeline

__all__ = [
    # 数据模型
    "InferenceRequest",
    "InferenceResponse",
    "ModelConfig",
    "QuestionData",
    "QuestionType",
    "ModelProvider",
    "BatchInferenceRequest",
    "BatchInferenceResult",
    "InferenceStatistics",
    
    # 客户端
    "ModelClient",
    "ModelClientError",
    "RateLimitError",
    "AuthenticationError",
    
    # 流水线
    "InferencePipeline"
]