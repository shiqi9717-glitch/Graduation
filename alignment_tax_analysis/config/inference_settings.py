"""
推理模块配置文件
包含模型API配置、超时设置、重试策略等
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class ModelProvider(Enum):
    """模型提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """模型配置类"""
    provider: ModelProvider
    model_name: str
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.7
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
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay
        }


def _get_global_api_key() -> str:
    """Read one global API key, fallback to provider-specific env vars."""
    return os.getenv("GLOBAL_API_KEY", "")


class InferenceSettings:
    """推理设置类"""
    
    # 默认模型配置
    DEFAULT_MODELS: Dict[str, ModelConfig] = {
        "gpt-4": ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=2048,
            timeout=30,
            max_retries=3
        ),
        "gpt-3.5-turbo": ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2048,
            timeout=30,
            max_retries=3
        ),
        "claude-3-opus": ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-opus",
            temperature=0.7,
            max_tokens=2048,
            timeout=60,
            max_retries=3
        ),
        "deepseek-chat": ModelConfig(
            provider=ModelProvider.DEEPSEEK,
            model_name="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
            timeout=30,
            max_retries=3
        ),
        "qwen-max": ModelConfig(
            provider=ModelProvider.QWEN,
            model_name="qwen-max",
            temperature=0.7,
            max_tokens=2048,
            timeout=30,
            max_retries=3
        )
    }
    
    # 推理流水线配置
    BATCH_SIZE: int = 10  # 批量处理大小
    CONCURRENT_REQUESTS: int = 5  # 并发请求数
    RATE_LIMIT_DELAY: float = 0.1  # 速率限制延迟（秒）
    
    # 输出配置
    OUTPUT_DIR: str = "data/results/inference"
    OUTPUT_FORMAT: str = "jsonl"  # jsonl, csv, json
    SAVE_INTERMEDIATE: bool = True  # 是否保存中间结果
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/inference.log"
    
    # 验证配置
    VALIDATE_RESPONSES: bool = True  # 是否验证响应
    MIN_RESPONSE_LENGTH: int = 10  # 最小响应长度
    MAX_RESPONSE_LENGTH: int = 5000  # 最大响应长度
    
    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """获取模型配置"""
        if model_name in cls.DEFAULT_MODELS:
            config = cls.DEFAULT_MODELS[model_name]
            # 从环境变量加载API密钥
            if config.provider == ModelProvider.OPENAI:
                config.api_key = _get_global_api_key()
            elif config.provider == ModelProvider.ANTHROPIC:
                config.api_key = _get_global_api_key()
            elif config.provider == ModelProvider.DEEPSEEK:
                config.api_key = _get_global_api_key()
            elif config.provider == ModelProvider.QWEN:
                config.api_key = _get_global_api_key()
            
            # 设置API基础URL
            if config.provider == ModelProvider.DEEPSEEK:
                config.api_base = "https://api.deepseek.com"
            elif config.provider == ModelProvider.QWEN:
                config.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            
            return config
        else:
            # 自定义模型配置
            return ModelConfig(
                provider=ModelProvider.CUSTOM,
                model_name=model_name,
                api_key=_get_global_api_key(),
                api_base=os.getenv("CUSTOM_API_BASE", ""),
                temperature=0.7,
                max_tokens=2048,
                timeout=30,
                max_retries=3
            )
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """获取可用模型列表"""
        return list(cls.DEFAULT_MODELS.keys())
    
    @classmethod
    def get_output_path(cls, timestamp: str = None) -> str:
        """获取输出路径"""
        import datetime
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = os.path.join(cls.OUTPUT_DIR, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir