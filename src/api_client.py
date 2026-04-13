"""Async API client with proxy-aware fallback and robust JSON parsing."""

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, Optional

import httpx

from config.settings import settings
from .common.env_utils import is_placeholder_key, load_project_env, to_bool
from .common.net_utils import build_async_httpx_client, extract_error_code
from .logging_config import logger


class APIClient:
    """DeepSeek API client used by generation/judge pipelines."""

    def __init__(self):
        load_project_env()

        self.base_url = os.getenv("DEEPSEEK_BASE_URL", settings.DEEPSEEK_BASE_URL).rstrip("/")
        self.model = os.getenv("DEEPSEEK_MODEL", settings.DEEPSEEK_MODEL)
        self.api_key = (
            os.getenv("DEEPSEEK_API_KEY", "").strip()
            or os.getenv("GLOBAL_API_KEY", "").strip()
            or settings.GLOBAL_API_KEY
        )
        self.proxy_url = os.getenv("PROXY_URL", "").strip() or None
        self.ssl_verify = to_bool(os.getenv("SSL_VERIFY", "false"), default=False)
        self.timeout = 30.0
        self.max_retries = int(os.getenv("MAX_RETRIES", str(settings.MAX_RETRIES)))

        if is_placeholder_key(self.api_key):
            raise ValueError(
                "Invalid API key: please set GLOBAL_API_KEY/DEEPSEEK_API_KEY via environment or .env, not placeholder."
            )

        if not self.ssl_verify:
            logger.warning("当前处于非安全 SSL 模式 (verify=False)")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_client(self, proxy_url: Optional[str]) -> httpx.AsyncClient:
        return build_async_httpx_client(
            timeout=self.timeout,
            verify_ssl=self.ssl_verify,
            headers=self.headers,
            proxy_url=proxy_url,
        )

    async def _request_once(self, payload: Dict[str, Any]) -> str:
        url = f"{self.base_url}/chat/completions"
        modes = [self.proxy_url] if self.proxy_url else [None]
        if self.proxy_url:
            modes.append(None)

        last_exc: Optional[Exception] = None
        for mode in modes:
            if mode:
                logger.info("正在通过代理 [%s] 访问...", mode)
            else:
                logger.info("正在尝试直连访问...")

            async with self._build_client(mode) as client:
                try:
                    start_time = time.time()
                    response = await client.post(url, json=payload)
                    elapsed = time.time() - start_time

                    if response.status_code == 200:
                        result = response.json()
                        content = result["choices"][0]["message"]["content"]
                        logger.debug("API request success, elapsed=%.2fs", elapsed)
                        return content.strip()

                    body = response.text[:500]
                    if response.status_code in (429, 500, 502, 503, 504):
                        raise RuntimeError(f"HTTP {response.status_code}: {body}")
                    raise RuntimeError(f"Non-retriable HTTP {response.status_code}: {body}")
                except httpx.ConnectError as exc:
                    err_code = extract_error_code(exc)
                    logger.error("ConnectError code=%s detail=%s", err_code, exc)
                    last_exc = exc
                    continue
                except httpx.RequestError as exc:
                    logger.error("RequestError detail=%s", exc)
                    last_exc = exc
                    continue
                except Exception as exc:
                    last_exc = exc
                    break

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unknown network error")

    async def _make_request(self, messages: list, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2000,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._request_once(payload)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                backoff = min(10, 2 ** attempt)
                logger.warning(
                    "API 请求失败，正在重试 (第%s次): %s",
                    attempt,
                    exc,
                )
                await asyncio.sleep(backoff)

        if last_exc is None:
            raise RuntimeError("API request failed with unknown error")
        raise last_exc

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.0,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._make_request(messages, temperature)

    async def generate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        system_prompt: str = "You are a helpful assistant. Return strict JSON.",
    ) -> Dict[str, Any]:
        response_text = await self.generate_text(prompt, temperature, system_prompt)

        # Strong fallback parsing: regex extract then json.loads.
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse failed: %s; raw=%s", exc, response_text[:300])
            return {"error": "json_parse_failed", "raw_response": response_text[:300]}

    async def batch_generate(
        self,
        prompts: list,
        temperature: float = 0.0,
        batch_size: int = None,
    ) -> list:
        if batch_size is None:
            batch_size = settings.BATCH_SIZE

        results = []
        total = len(prompts)

        for i in range(0, total, batch_size):
            batch = prompts[i : i + batch_size]
            logger.info("Processing batch %s/%s", i // batch_size + 1, (total + batch_size - 1) // batch_size)

            tasks = [self.generate_text(prompt, temperature) for prompt in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error("Batch request failed: %s", result)
                    results.append(None)
                else:
                    results.append(result)

            if i + batch_size < total:
                await asyncio.sleep(1)

        return results
