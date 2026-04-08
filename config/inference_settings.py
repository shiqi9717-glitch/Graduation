"""
推理模块配置文件
包含模型API配置、超时设置、重试策略等
"""

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List


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


def _get_global_api_key() -> str:
    """Read one global API key, fallback to provider-specific env vars."""
    return os.getenv("GLOBAL_API_KEY", "")


class InferenceSettings:
    """推理设置类"""

    MODELS_CONFIG_FILE: Path = Path(__file__).resolve().parent / "models_config.json"
    MODEL_COMPARISON_GROUPS_FILE: Path = Path(__file__).resolve().parent / "model_comparison_groups.json"
    MODEL_METADATA_FIELDS = (
        "model_family",
        "model_variant",
        "reasoning_mode",
        "release_channel",
        "is_preview",
        "comparison_group",
    )
    
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
    OUTPUT_DIR: str = "outputs/experiments/inference"
    OUTPUT_FORMAT: str = "jsonl"  # jsonl, csv, json
    SAVE_INTERMEDIATE: bool = True  # 是否保存中间结果
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "outputs/logs/inference.log"
    
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
                top_p=0.9,
                max_tokens=2048,
                timeout=30,
                max_retries=3
            )

    @classmethod
    def _default_model_registry(cls) -> Dict[str, Dict[str, Any]]:
        env_map = {
            ModelProvider.OPENAI: "OPENAI_API_KEY",
            ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            ModelProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
            ModelProvider.QWEN: "QWEN_API_KEY",
            ModelProvider.CUSTOM: "GLOBAL_API_KEY",
        }
        base_map = {
            ModelProvider.OPENAI: "https://api.openai.com/v1",
            ModelProvider.ANTHROPIC: "https://api.anthropic.com/v1",
            ModelProvider.DEEPSEEK: "https://api.deepseek.com/v1",
            ModelProvider.QWEN: "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ModelProvider.CUSTOM: "",
        }
        registry: Dict[str, Dict[str, Any]] = {}
        for alias, config in cls.DEFAULT_MODELS.items():
            registry[alias] = {
                "provider": config.provider.value,
                "model_name": config.model_name,
                "api_base": base_map.get(config.provider, ""),
                "api_key_env": env_map.get(config.provider, "GLOBAL_API_KEY"),
                "model_family": "unknown",
                "model_variant": alias,
                "reasoning_mode": "unknown",
                "release_channel": "unknown",
                "is_preview": False,
                "comparison_group": "unknown",
                "timeout": config.timeout,
                "max_retries": config.max_retries,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_tokens,
                "retry_delay": config.retry_delay,
            }
        return registry

    @classmethod
    def load_model_registry(cls, config_path: str = "") -> Dict[str, Dict[str, Any]]:
        registry = cls._default_model_registry()
        path = Path(config_path).expanduser() if config_path else cls.MODELS_CONFIG_FILE
        if not path.exists():
            return registry

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError(f"models config must be a JSON object: {path}")

        for alias, raw_config in payload.items():
            if not isinstance(raw_config, dict):
                raise ValueError(f"models config entry must be an object: {alias}")
            merged = deepcopy(registry.get(alias, {}))
            merged.update(raw_config)
            merged.setdefault("model_name", alias)
            registry[alias] = merged

        return registry

    @classmethod
    def resolve_model_profile(cls, model_name: str, config_path: str = "") -> Dict[str, Any]:
        registry = cls.load_model_registry(config_path=config_path)
        if model_name in registry:
            profile = deepcopy(registry[model_name])
            profile.setdefault("model_name", model_name)
            profile.setdefault("provider", ModelProvider.CUSTOM.value)
            profile.setdefault("model_variant", profile.get("model_name", model_name))
            for field in cls.MODEL_METADATA_FIELDS:
                if field == "model_variant":
                    continue
                profile.setdefault(field, "unknown" if field != "is_preview" else False)
            return profile
        fallback = {
            "provider": ModelProvider.CUSTOM.value,
            "model_name": model_name,
            "model_family": "unknown",
            "model_variant": model_name,
            "reasoning_mode": "unknown",
            "release_channel": "unknown",
            "is_preview": False,
            "comparison_group": "unknown",
            "api_base": os.getenv("CUSTOM_API_BASE", ""),
            "api_key_env": "GLOBAL_API_KEY",
            "timeout": 30,
            "max_retries": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
            "retry_delay": 1.0,
        }
        return fallback

    @classmethod
    def extract_model_metadata(cls, profile: Dict[str, Any], alias: str = "") -> Dict[str, Any]:
        alias_clean = str(alias or profile.get("model_name") or "unknown").strip()
        model_name = str(profile.get("model_name") or alias_clean or "unknown").strip()
        return {
            "model_name": model_name,
            "model_alias": alias_clean or model_name,
            "model_family": str(profile.get("model_family") or "unknown").strip() or "unknown",
            "model_variant": str(profile.get("model_variant") or model_name).strip() or model_name,
            "reasoning_mode": str(profile.get("reasoning_mode") or "unknown").strip() or "unknown",
            "release_channel": str(profile.get("release_channel") or "unknown").strip() or "unknown",
            "is_preview": bool(profile.get("is_preview", False)),
            "comparison_group": str(profile.get("comparison_group") or "unknown").strip() or "unknown",
        }

    @classmethod
    def load_model_comparison_groups(cls, config_path: str = "") -> Dict[str, Any]:
        path = Path(config_path).expanduser() if config_path else cls.MODEL_COMPARISON_GROUPS_FILE
        if not path.exists():
            return {"family_pairs": []}
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"model comparison config must be a JSON object: {path}")
        family_pairs = payload.get("family_pairs", [])
        if not isinstance(family_pairs, list):
            raise ValueError(f"family_pairs must be a list: {path}")
        return {"family_pairs": [dict(item) for item in family_pairs if isinstance(item, dict)]}
    
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
