"""Model API client with proxy-aware fallback networking."""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from config.inference_settings import InferenceSettings
from src.common.env_utils import is_placeholder_key, load_project_env, to_bool
from src.common.net_utils import build_async_httpx_client, extract_error_code
from .schemas import InferenceRequest, InferenceResponse, ModelConfig, ModelProvider

logger = logging.getLogger(__name__)


class ModelClientError(Exception):
    """Model client error."""


class RateLimitError(ModelClientError):
    """Rate limit error."""


class AuthenticationError(ModelClientError):
    """Authentication error."""


class ModelClient:
    """Provider-agnostic async model client."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.client: Optional[httpx.AsyncClient] = None
        self.current_proxy_mode: Optional[str] = None
        self._rate_limit_lock = asyncio.Lock()
        self._next_request_ts = 0.0
        self._load_network_env()
        self._setup_provider_specific_config()

    def _load_network_env(self) -> None:
        load_project_env()

        self.proxy_url = os.getenv("PROXY_URL", "").strip() or None
        self.ssl_verify = to_bool(os.getenv("SSL_VERIFY", "false"), default=False)
        self.request_timeout = float(self.model_config.timeout or 30.0)

        # Security notice when SSL verification is disabled.
        if not self.ssl_verify:
            logger.warning("当前处于非安全 SSL 模式 (verify=False)")

        # Allow DEEPSEEK_API_KEY from .env to override missing DeepSeek api_key.
        if (
            self.model_config.provider == ModelProvider.DEEPSEEK
            and not str(self.model_config.api_key or "").strip()
        ):
            env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            if env_key:
                self.model_config.api_key = env_key
        if self.model_config.provider == ModelProvider.DEEPSEEK and is_placeholder_key(
            self.model_config.api_key
        ):
            raise ValueError(
                "Invalid DeepSeek API key: set GLOBAL_API_KEY/DEEPSEEK_API_KEY to a real value."
            )

    def _setup_provider_specific_config(self) -> None:
        self.provider = self.model_config.provider

        if self.provider == ModelProvider.OPENAI:
            self.api_base = self.model_config.api_base or "https://api.openai.com/v1"
            self.endpoint = f"{self.api_base}/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json",
            }
        elif self.provider == ModelProvider.ANTHROPIC:
            self.api_base = self.model_config.api_base or "https://api.anthropic.com/v1"
            self.endpoint = f"{self.api_base}/messages"
            self.headers = {
                "x-api-key": self.model_config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        elif self.provider == ModelProvider.DEEPSEEK:
            self.api_base = self.model_config.api_base or "https://api.deepseek.com/v1"
            self.endpoint = f"{self.api_base}/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json",
            }
        elif self.provider == ModelProvider.QWEN:
            self.api_base = self.model_config.api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.endpoint = f"{self.api_base}/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json",
            }
        else:
            self.api_base = self.model_config.api_base
            self.endpoint = f"{self.api_base}/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json",
            }

    def _build_client(self, proxy_url: Optional[str]) -> httpx.AsyncClient:
        return build_async_httpx_client(
            timeout=self.request_timeout,
            verify_ssl=self.ssl_verify,
            headers=self.headers,
            proxy_url=proxy_url,
        )

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        if self.client is None:
            self.current_proxy_mode = self.proxy_url
            self.client = self._build_client(self.current_proxy_mode)

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None
            self.current_proxy_mode = None

    async def _switch_client_mode(self, proxy_url: Optional[str]) -> None:
        if self.client is not None:
            await self.client.aclose()
        self.current_proxy_mode = proxy_url
        self.client = self._build_client(proxy_url)

    def _build_openai_request(self, request: InferenceRequest) -> Dict[str, Any]:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.question_data.question_text})
        payload = {
            "model": self.model_config.model_name,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            **request.additional_params,
        }
        # DeepSeek reasoner follows a slightly different API contract:
        # max_tokens covers reasoning + final answer, and sampling params are unsupported.
        if str(self.model_config.model_name or "").strip().lower() == "deepseek-reasoner":
            payload.pop("temperature", None)
            payload.pop("top_p", None)
        return payload

    def _build_anthropic_request(self, request: InferenceRequest) -> Dict[str, Any]:
        return {
            "model": self.model_config.model_name,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "system": request.system_prompt or "",
            "messages": [{"role": "user", "content": request.question_data.question_text}],
            **request.additional_params,
        }

    def _build_request_payload(self, request: InferenceRequest) -> Dict[str, Any]:
        if self.provider == ModelProvider.ANTHROPIC:
            return self._build_anthropic_request(request)
        return self._build_openai_request(request)

    def _provider_rate_limit_policy(self) -> Dict[str, float]:
        model_name = str(self.model_config.model_name or "").strip().lower()
        if self.provider == ModelProvider.QWEN or "qwen" in model_name:
            return {
                "request_gap_seconds": 1.0,
                "rate_limit_cooldown_seconds": 8.0,
                "backoff_multiplier": 1.5,
            }
        if "kimi" in model_name:
            return {
                "request_gap_seconds": 3.2,
                "rate_limit_cooldown_seconds": 8.0,
                "backoff_multiplier": 2.0,
            }
        return {
            "request_gap_seconds": 0.0,
            "rate_limit_cooldown_seconds": 3.0,
            "backoff_multiplier": 1.0,
        }

    async def _wait_for_request_slot(self) -> None:
        policy = self._provider_rate_limit_policy()
        gap = float(policy["request_gap_seconds"])
        async with self._rate_limit_lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_request_ts - now)
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = time.monotonic()
            self._next_request_ts = now + gap

    async def _extend_rate_limit_cooldown(self, seconds: float) -> None:
        async with self._rate_limit_lock:
            self._next_request_ts = max(self._next_request_ts, time.monotonic() + seconds)

    def _parse_openai_response(self, response_data: Dict[str, Any]) -> str:
        if "choices" in response_data and response_data["choices"]:
            return response_data["choices"][0]["message"]["content"]
        raise ModelClientError(f"Invalid OpenAI response: {response_data}")

    def _parse_anthropic_response(self, response_data: Dict[str, Any]) -> str:
        if "content" in response_data and response_data["content"]:
            return response_data["content"][0]["text"]
        raise ModelClientError(f"Invalid Anthropic response: {response_data}")

    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        if self.provider == ModelProvider.ANTHROPIC:
            return self._parse_anthropic_response(response_data)
        return self._parse_openai_response(response_data)

    async def _post_with_fallback(self, payload: Dict[str, Any]) -> httpx.Response:
        if self.client is None:
            await self.connect()

        modes: List[Optional[str]] = [self.proxy_url] if self.proxy_url else [None]
        if self.proxy_url:
            modes.append(None)

        last_exc: Optional[Exception] = None
        for mode in modes:
            if self.current_proxy_mode != mode or self.client is None:
                await self._switch_client_mode(mode)

            if mode:
                logger.info("正在通过代理 [%s] 访问...", mode)
            else:
                logger.info("正在尝试直连访问...")

            try:
                assert self.client is not None
                return await self.client.post(self.endpoint, json=payload)
            except httpx.ConnectError as exc:
                err_code = extract_error_code(exc)
                logger.error("ConnectError code=%s detail=%s", err_code, exc)
                last_exc = exc
                continue
            except httpx.RequestError as exc:
                logger.error("RequestError detail=%s", exc)
                last_exc = exc
                continue

        if last_exc is not None:
            raise ModelClientError(f"Network failure after proxy/direct fallback: {last_exc}") from last_exc
        raise ModelClientError("Network failure after proxy/direct fallback.")

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        max_attempts = max(1, int(self.model_config.max_retries or 1))
        policy = self._provider_rate_limit_policy()
        cooldown = float(policy["rate_limit_cooldown_seconds"])
        backoff_multiplier = float(policy["backoff_multiplier"])
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            start_time = time.time()
            request_label = (
                str(request.question_data.metadata.get("task_id") or request.question_data.question_id)
                if isinstance(request.question_data.metadata, dict)
                else request.question_data.question_id
            )
            try:
                logger.debug(
                    "Infer request start model=%s request=%s attempt=%s/%s timeout=%.1fs",
                    self.model_config.model_name,
                    request_label,
                    attempt,
                    max_attempts,
                    self.request_timeout,
                )
                await self._wait_for_request_slot()
                payload = self._build_request_payload(request)
                response = await self._post_with_fallback(payload)
                response_text = response.text

                if response.status_code == 429:
                    raise RateLimitError(f"Rate limit exceeded: {response_text}")
                if response.status_code == 401:
                    raise AuthenticationError(f"Authentication failed: {response_text}")
                if response.status_code != 200:
                    raise ModelClientError(f"API error {response.status_code}: {response_text}")

                response_data = json.loads(response_text)
                response_content = self._parse_response(response_data)
                latency_ms = (time.time() - start_time) * 1000

                return InferenceResponse(
                    request_id=request.request_id,
                    question_data=request.question_data,
                    model_name=self.model_config.model_name,
                    response_text=response_content,
                    raw_response=response_data,
                    latency_ms=latency_ms,
                    success=True,
                )
            except json.JSONDecodeError as exc:
                last_error = ModelClientError(f"JSON decode error: {exc}")
                break
            except AuthenticationError as exc:
                last_error = exc
                break
            except RateLimitError as exc:
                last_error = exc
                await self._extend_rate_limit_cooldown(cooldown * attempt)
                if attempt >= max_attempts:
                    break
                sleep_seconds = max(
                    float(self.model_config.retry_delay or 1.0),
                    cooldown * backoff_multiplier * attempt,
                )
                logger.warning(
                    "Rate limit for %s request=%s on attempt %s/%s, sleeping %.1fs before retry",
                    self.model_config.model_name,
                    request_label,
                    attempt,
                    max_attempts,
                    sleep_seconds,
                )
                await asyncio.sleep(sleep_seconds)
            except httpx.TimeoutException as exc:
                last_error = ModelClientError(f"Timeout after {self.request_timeout:.1f}s: {exc}")
                if attempt >= max_attempts:
                    logger.error(
                        "Timeout for %s request=%s on final attempt %s/%s after %.1fs",
                        self.model_config.model_name,
                        request_label,
                        attempt,
                        max_attempts,
                        self.request_timeout,
                    )
                    break
                sleep_seconds = float(self.model_config.retry_delay or 1.0) * attempt
                logger.warning(
                    "Timeout for %s request=%s on attempt %s/%s after %.1fs; retrying in %.1fs",
                    self.model_config.model_name,
                    request_label,
                    attempt,
                    max_attempts,
                    self.request_timeout,
                    sleep_seconds,
                )
                await asyncio.sleep(sleep_seconds)
            except httpx.ConnectError as exc:
                err_code = extract_error_code(exc)
                last_error = ModelClientError(f"ConnectError code={err_code}: {exc}")
                if attempt >= max_attempts:
                    break
                sleep_seconds = float(self.model_config.retry_delay or 1.0) * attempt
                logger.warning(
                    "ConnectError for %s request=%s on attempt %s/%s: %s; retrying in %.1fs",
                    self.model_config.model_name,
                    request_label,
                    attempt,
                    max_attempts,
                    err_code,
                    sleep_seconds,
                )
                await asyncio.sleep(sleep_seconds)
            except httpx.RequestError as exc:
                last_error = ModelClientError(f"HTTP request error: {exc}")
                if attempt >= max_attempts:
                    break
                sleep_seconds = float(self.model_config.retry_delay or 1.0) * attempt
                logger.warning(
                    "RequestError for %s request=%s on attempt %s/%s: %s; retrying in %.1fs",
                    self.model_config.model_name,
                    request_label,
                    attempt,
                    max_attempts,
                    exc,
                    sleep_seconds,
                )
                await asyncio.sleep(sleep_seconds)
            except Exception as exc:
                last_error = ModelClientError(f"Unexpected error: {exc}")
                break

        assert last_error is not None
        raise last_error

    async def batch_infer(
        self,
        requests: List[InferenceRequest],
        concurrency_limit: int = 5,
    ) -> List[InferenceResponse]:
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def infer_with_semaphore(req: InferenceRequest) -> InferenceResponse:
            async with semaphore:
                try:
                    return await self.infer(req)
                except Exception as exc:
                    return InferenceResponse(
                        request_id=req.request_id,
                        question_data=req.question_data,
                        model_name=self.model_config.model_name,
                        response_text="",
                        raw_response={},
                        latency_ms=0,
                        success=False,
                        error_message=str(exc),
                    )

        tasks = [infer_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def validate_response(self, response: InferenceResponse) -> bool:
        if not response.success:
            return False
        text = response.response_text.strip()
        if len(text) < InferenceSettings.MIN_RESPONSE_LENGTH:
            logger.warning("Response too short: %s chars", len(text))
            return False
        if len(text) > InferenceSettings.MAX_RESPONSE_LENGTH:
            logger.warning("Response too long: %s chars", len(text))
            return False
        error_keywords = ["error", "invalid", "failed", "cannot", "unable"]
        lower = text.lower()
        for keyword in error_keywords:
            if keyword in lower:
                logger.warning("Response contains error keyword: %s", keyword)
                return False
        return True
