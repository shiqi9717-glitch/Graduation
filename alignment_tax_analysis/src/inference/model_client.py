"""
模型API客户端
支持多种大模型提供商的API调用
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import logging

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .schemas import ModelConfig, InferenceRequest, InferenceResponse, ModelProvider
from config.inference_settings import InferenceSettings


logger = logging.getLogger(__name__)


class ModelClientError(Exception):
    """模型客户端错误"""
    pass


class RateLimitError(ModelClientError):
    """速率限制错误"""
    pass


class AuthenticationError(ModelClientError):
    """认证错误"""
    pass


class ModelClient:
    """模型API客户端"""
    
    def __init__(self, model_config: ModelConfig):
        """
        初始化模型客户端
        
        Args:
            model_config: 模型配置
        """
        self.model_config = model_config
        self.session: Optional[aiohttp.ClientSession] = None
        self._setup_provider_specific_config()
        
    def _setup_provider_specific_config(self):
        """设置提供商特定的配置"""
        self.provider = self.model_config.provider
        
        # 设置API端点
        if self.provider == ModelProvider.OPENAI:
            self.api_base = self.model_config.api_base or "https://api.openai.com/v1"
            self.endpoint = f"{self.api_base}/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json"
            }
        elif self.provider == ModelProvider.ANTHROPIC:
            self.api_base = self.model_config.api_base or "https://api.anthropic.com/v1"
            self.endpoint = f"{self.api_base}/messages"
            self.headers = {
                "x-api-key": self.model_config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
        elif self.provider == ModelProvider.DEEPSEEK:
            self.api_base = self.model_config.api_base or "https://api.deepseek.com/v1"
            self.endpoint = f"{self.api_base}/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json"
            }
        elif self.provider == ModelProvider.QWEN:
            self.api_base = self.model_config.api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.endpoint = f"{self.api_base}/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json"
            }
        else:
            # 自定义提供商
            self.api_base = self.model_config.api_base
            self.endpoint = f"{self.api_base}/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json"
            }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def connect(self):
        """连接API（创建会话）"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.model_config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout, headers=self.headers)
    
    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _build_openai_request(self, request: InferenceRequest) -> Dict[str, Any]:
        """构建OpenAI格式的请求"""
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.question_data.question_text})
        
        return {
            "model": self.model_config.model_name,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            **request.additional_params
        }
    
    def _build_anthropic_request(self, request: InferenceRequest) -> Dict[str, Any]:
        """构建Anthropic格式的请求"""
        system_prompt = request.system_prompt or ""
        
        return {
            "model": self.model_config.model_name,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": system_prompt,
            "messages": [{
                "role": "user",
                "content": request.question_data.question_text
            }],
            **request.additional_params
        }
    
    def _build_request_payload(self, request: InferenceRequest) -> Dict[str, Any]:
        """根据提供商构建请求负载"""
        if self.provider == ModelProvider.OPENAI:
            return self._build_openai_request(request)
        elif self.provider == ModelProvider.ANTHROPIC:
            return self._build_anthropic_request(request)
        elif self.provider == ModelProvider.DEEPSEEK:
            return self._build_openai_request(request)  # DeepSeek使用OpenAI格式
        elif self.provider == ModelProvider.QWEN:
            return self._build_openai_request(request)  # Qwen使用OpenAI格式
        else:
            # 自定义提供商，默认使用OpenAI格式
            return self._build_openai_request(request)
    
    def _parse_openai_response(self, response_data: Dict[str, Any]) -> str:
        """解析OpenAI格式的响应"""
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        raise ModelClientError(f"Invalid OpenAI response: {response_data}")
    
    def _parse_anthropic_response(self, response_data: Dict[str, Any]) -> str:
        """解析Anthropic格式的响应"""
        if "content" in response_data and len(response_data["content"]) > 0:
            return response_data["content"][0]["text"]
        raise ModelClientError(f"Invalid Anthropic response: {response_data}")
    
    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """根据提供商解析响应"""
        if self.provider == ModelProvider.OPENAI:
            return self._parse_openai_response(response_data)
        elif self.provider == ModelProvider.ANTHROPIC:
            return self._parse_anthropic_response(response_data)
        elif self.provider == ModelProvider.DEEPSEEK:
            return self._parse_openai_response(response_data)
        elif self.provider == ModelProvider.QWEN:
            return self._parse_openai_response(response_data)
        else:
            # 自定义提供商，尝试OpenAI格式
            try:
                return self._parse_openai_response(response_data)
            except:
                # 如果失败，返回第一个文本字段
                for key in ["text", "content", "response", "answer"]:
                    if key in response_data:
                        return str(response_data[key])
                raise ModelClientError(f"Cannot parse response: {response_data}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, aiohttp.ClientError))
    )
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        执行推理请求
        
        Args:
            request: 推理请求
            
        Returns:
            InferenceResponse: 推理响应
        """
        start_time = time.time()
        
        try:
            # 确保已连接
            if self.session is None:
                await self.connect()
            
            # 构建请求负载
            payload = self._build_request_payload(request)
            
            # 发送请求
            async with self.session.post(self.endpoint, json=payload) as response:
                response_text = await response.text()
                
                # 检查HTTP状态
                if response.status == 429:
                    raise RateLimitError(f"Rate limit exceeded: {response_text}")
                elif response.status == 401:
                    raise AuthenticationError(f"Authentication failed: {response_text}")
                elif response.status != 200:
                    raise ModelClientError(f"API error {response.status}: {response_text}")
                
                # 解析响应
                response_data = json.loads(response_text)
                response_content = self._parse_response(response_data)
                
                # 计算延迟
                latency_ms = (time.time() - start_time) * 1000
                
                # 构建响应对象
                return InferenceResponse(
                    request_id=request.request_id,
                    question_data=request.question_data,
                    model_name=self.model_config.model_name,
                    response_text=response_content,
                    raw_response=response_data,
                    latency_ms=latency_ms,
                    success=True
                )
                
        except json.JSONDecodeError as e:
            latency_ms = (time.time() - start_time) * 1000
            raise ModelClientError(f"JSON decode error: {str(e)}") from e
        except aiohttp.ClientError as e:
            latency_ms = (time.time() - start_time) * 1000
            raise ModelClientError(f"HTTP client error: {str(e)}") from e
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            raise ModelClientError(f"Unexpected error: {str(e)}") from e
    
    async def batch_infer(self, requests: List[InferenceRequest], 
                         concurrency_limit: int = 5) -> List[InferenceResponse]:
        """
        批量执行推理请求
        
        Args:
            requests: 推理请求列表
            concurrency_limit: 并发限制
            
        Returns:
            List[InferenceResponse]: 推理响应列表
        """
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def infer_with_semaphore(request: InferenceRequest) -> InferenceResponse:
            async with semaphore:
                try:
                    return await self.infer(request)
                except Exception as e:
                    # 创建错误响应
                    return InferenceResponse(
                        request_id=request.request_id,
                        question_data=request.question_data,
                        model_name=self.model_config.model_name,
                        response_text="",
                        raw_response={},
                        latency_ms=0,
                        success=False,
                        error_message=str(e)
                    )
        
        # 并发执行所有请求
        tasks = [infer_with_semaphore(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        return results
    
    def validate_response(self, response: InferenceResponse) -> bool:
        """
        验证响应是否有效
        
        Args:
            response: 推理响应
            
        Returns:
            bool: 是否有效
        """
        if not response.success:
            return False
        
        # 检查响应长度
        if len(response.response_text.strip()) < InferenceSettings.MIN_RESPONSE_LENGTH:
            logger.warning(f"Response too short: {len(response.response_text)} chars")
            return False
        
        if len(response.response_text.strip()) > InferenceSettings.MAX_RESPONSE_LENGTH:
            logger.warning(f"Response too long: {len(response.response_text)} chars")
            return False
        
        # 检查是否包含错误关键词
        error_keywords = ["error", "抱歉", "无法", "不支持", "invalid", "failed"]
        response_lower = response.response_text.lower()
        for keyword in error_keywords:
            if keyword in response_lower:
                logger.warning(f"Response contains error keyword: {keyword}")
                return False
        
        return True