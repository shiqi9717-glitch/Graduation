"""
数据模型定义
包含推理请求、响应、配置等数据类
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import uuid


class QuestionType(Enum):
    """问题类型枚举"""
    SYCOPHANCY = "sycophancy"  # 阿谀奉承问题
    CONTROL = "control"  # 控制组问题
    PERTURBED = "perturbed"  # 扰动后问题


class ModelProvider(Enum):
    """模型提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    CUSTOM = "custom"


@dataclass
class QuestionData:
    """问题数据类"""
    question_id: str
    question_text: str
    question_type: QuestionType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "question_type": self.question_type.value,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionData":
        """从字典创建"""
        return cls(
            question_id=data.get("question_id", str(uuid.uuid4())),
            question_text=data["question_text"],
            question_type=QuestionType(data.get("question_type", "sycophancy")),
            metadata=data.get("metadata", {})
        )


@dataclass
class InferenceRequest:
    """推理请求类"""
    request_id: str
    question_data: QuestionData
    model_name: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    system_prompt: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "question_data": self.question_data.to_dict(),
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "additional_params": self.additional_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceRequest":
        """从字典创建"""
        return cls(
            request_id=data.get("request_id", str(uuid.uuid4())),
            question_data=QuestionData.from_dict(data["question_data"]),
            model_name=data["model_name"],
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            max_tokens=data.get("max_tokens", 2048),
            system_prompt=data.get("system_prompt"),
            additional_params=data.get("additional_params", {})
        )


@dataclass
class InferenceResponse:
    """推理响应类"""
    request_id: str
    question_data: QuestionData
    model_name: str
    response_text: str
    raw_response: Dict[str, Any]
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "question_data": self.question_data.to_dict(),
            "model_name": self.model_name,
            "response_text": self.response_text,
            "raw_response": self.raw_response,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResponse":
        """从字典创建"""
        timestamp_str = data.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        
        return cls(
            request_id=data["request_id"],
            question_data=QuestionData.from_dict(data["question_data"]),
            model_name=data["model_name"],
            response_text=data["response_text"],
            raw_response=data["raw_response"],
            latency_ms=data["latency_ms"],
            timestamp=timestamp,
            success=data.get("success", True),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {})
        )


@dataclass
class ModelConfig:
    """模型配置类"""
    provider: ModelProvider
    model_name: str
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """从字典创建"""
        return cls(
            provider=ModelProvider(data["provider"]),
            model_name=data["model_name"],
            api_key=data.get("api_key", ""),
            api_base=data.get("api_base", ""),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            max_tokens=data.get("max_tokens", 2048),
            timeout=data.get("timeout", 30),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0)
        )


@dataclass
class BatchInferenceRequest:
    """批量推理请求类"""
    batch_id: str
    requests: List[InferenceRequest]
    model_config: ModelConfig
    concurrency_limit: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "batch_id": self.batch_id,
            "requests": [req.to_dict() for req in self.requests],
            "model_config": self.model_config.to_dict(),
            "concurrency_limit": self.concurrency_limit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchInferenceRequest":
        """从字典创建"""
        return cls(
            batch_id=data.get("batch_id", str(uuid.uuid4())),
            requests=[InferenceRequest.from_dict(req) for req in data["requests"]],
            model_config=ModelConfig.from_dict(data["model_config"]),
            concurrency_limit=data.get("concurrency_limit", 5)
        )


@dataclass
class BatchInferenceResult:
    """批量推理结果类"""
    batch_id: str
    responses: List[InferenceResponse]
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_latency_ms: float
    avg_latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "batch_id": self.batch_id,
            "responses": [resp.to_dict() for resp in self.responses],
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchInferenceResult":
        """从字典创建"""
        timestamp_str = data.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        
        return cls(
            batch_id=data["batch_id"],
            responses=[InferenceResponse.from_dict(resp) for resp in data["responses"]],
            total_requests=data["total_requests"],
            successful_requests=data["successful_requests"],
            failed_requests=data["failed_requests"],
            total_latency_ms=data["total_latency_ms"],
            avg_latency_ms=data["avg_latency_ms"],
            timestamp=timestamp
        )


@dataclass
class InferenceStatistics:
    """推理统计信息类"""
    model_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    success_rate: float
    error_types: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "success_rate": self.success_rate,
            "error_types": self.error_types
        }
