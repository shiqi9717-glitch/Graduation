"""
API 客户端模块
封装 DeepSeek API 的异步请求，提供重试机制和错误处理
"""
import asyncio
import json
import aiohttp
import time
from typing import Dict, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import settings
from .logging_config import logger


class APIClient:
    """DeepSeek API 客户端"""
    
    def __init__(self):
        """初始化 API 客户端"""
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.headers = settings.api_headers
        self.timeout = aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
        
    async def _ensure_session(self):
        """确保会话存在"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(limit=settings.MAX_CONCURRENT_REQUESTS)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout,
                connector=connector
            )
    
    async def close(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"API 请求失败，正在重试 (第{retry_state.attempt_number}次): {retry_state.outcome.exception()}"
        )
    )
    async def _make_request(self, messages: list, temperature: float = 0.0) -> str:
        """
        发送 API 请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            
        Returns:
            API 响应内容
        """
        await self._ensure_session()
        
        payload = {
            "model": settings.DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        try:
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                elapsed = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    logger.debug(f"API 请求成功，耗时{elapsed:.2f}s")
                    return content.strip()
                else:
                    error_text = await response.text()
                    logger.error(f"API 请求失败: {response.status} - {error_text}")
                    raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            logger.error(f"API 请求超时 (超时时间: {settings.REQUEST_TIMEOUT}s)")
            raise
        except Exception as e:
            logger.error(f"API 请求异常: {str(e)}")
            raise
    
    async def generate_text(
        self, 
        prompt: str, 
        temperature: float = 0.0,
        system_prompt: str = "你是一个有帮助的助手"
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示词
            temperature: 温度参数
            system_prompt: 系统提示词
            
        Returns:
            生成的文本
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        return await self._make_request(messages, temperature)
    
    async def generate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        system_prompt: str = "你是一个有帮助的助手，请严格按照 JSON 格式输出"
    ) -> Dict[str, Any]:
        """
        生成 JSON 格式的响应
        
        Args:
            prompt: 用户提示词
            temperature: 温度参数
            system_prompt: 系统提示词
            
        Returns:
            JSON 解析后的字典
        """
        response_text = await self.generate_text(prompt, temperature, system_prompt)
        
        # 尝试解析 JSON
        try:
            # 提取可能的 JSON 块
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # 如果没有找到 JSON，尝试直接解析整个响应
                return json.loads(response_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}\n原始响应: {response_text}")
            
            # 尝试使用正则表达式提取 JSON
            import re
            json_pattern = r'\{[^{}]*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            if matches:
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
            
            # 如果所有尝试都失败，返回默认结构
            logger.warning("无法解析 JSON，返回默认结构")
            return {"error": "JSON 解析失败", "raw_response": response_text[:200]}
    
    async def batch_generate(
        self,
        prompts: list,
        temperature: float = 0.0,
        batch_size: int = None
    ) -> list:
        """
        批量生成文本
        
        Args:
            prompts: 提示词列表
            temperature: 温度参数
            batch_size: 批处理大小
            
        Returns:
            生成结果列表
        """
        if batch_size is None:
            batch_size = settings.BATCH_SIZE
            
        results = []
        total = len(prompts)
        
        for i in range(0, total, batch_size):
            batch = prompts[i:i + batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
            
            # 创建批处理任务
            tasks = [
                self.generate_text(prompt, temperature)
                for prompt in batch
            ]
            
            # 并发执行
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"第 {i+j+1} 个请求失败: {result}")
                    results.append(None)
                else:
                    results.append(result)
            
            # 避免速率限制
            if i + batch_size < total:
                await asyncio.sleep(1)
        
        return results