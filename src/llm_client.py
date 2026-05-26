"""
高可用内网通用 LLM 客户端。

基于 requests 实现，支持：
- 硬编码 routing-strategy 头
- 标准请求体格式（model, temperature, max_tokens, messages）
- 可选扩展请求体格式（top_p, top_k, stream, chat_template_kwargs）
- 指数退避重试装饰器（最多 3 次）
- 网络抖动 / HTTP 5xx 自动重试
- 所有日志中严禁泄露明文 Token
"""

from __future__ import annotations

import functools
import logging
import re
import time
from typing import Any, Dict, List, Optional

import requests

from src.config import settings, _sanitize_for_logging

logger = logging.getLogger(__name__)


class LlmApiError(Exception):
    """LLM API 调用失败时抛出的异常，携带状态码和响应摘要。"""

    def __init__(self, message: str, status_code: Optional[int] = None, body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:
        text = super().__str__()
        if self.body:
            return f"{text}; response_body={_sanitize_response_body(self.body)}"
        return text


def _sanitize_response_body(body: str) -> str:
    """对 LLM 响应摘要脱敏，避免日志泄露 token/password 等敏感信息。"""
    text = body.replace("\n", " ")[:1000]
    # 保留错误语义里的 token/tokens 字样，只遮常见的密钥字段值。
    patterns = [
        r'("?(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)"?\s*[:=]\s*)"[^"]+"',
        r"((?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\s*[:=]\s*)\S+",
        r"(Bearer\s+)[A-Za-z0-9._~+/=-]+",
    ]
    for pattern in patterns:
        text = re.sub(pattern, r"\1<REDACTED>", text, flags=re.IGNORECASE)
    return text


def _retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    指数退避重试装饰器。
    遇到 requests.ConnectionError、requests.Timeout 或 HTTP 5xx 状态码时自动重试。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.ConnectionError, requests.Timeout) as exc:
                    last_exc = exc
                    logger.warning(
                        "LLM 请求网络异常（第 %d/%d 次）: %s",
                        attempt, max_retries, exc,
                    )
                except LlmApiError as exc:
                    last_exc = exc
                    if exc.status_code is not None and 500 <= exc.status_code < 600:
                        logger.warning(
                            "LLM 服务端 5xx 错误（第 %d/%d 次）: HTTP %s",
                            attempt, max_retries, exc.status_code,
                        )
                    else:
                        # 4xx 等客户端错误不重试，直接抛出
                        raise
                if attempt < max_retries:
                    delay = base_delay * (backoff_factor ** (attempt - 1))
                    logger.info("等待 %.1f 秒后重试...", delay)
                    time.sleep(delay)
            raise LlmApiError(
                f"LLM 请求在 {max_retries} 次重试后仍然失败: {last_exc}"
            ) from last_exc
        return wrapper
    return decorator


class LlmClient:
    """
    高可用 LLM 客户端。

    封装与内网大模型推理服务的交互，支持重试、超时和安全的认证管理。
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_url = api_url or settings.LLM_API_URL
        self.model = model or settings.LLM_MODEL
        self.timeout = timeout or settings.LLM_TIMEOUT_SECONDS

        self._headers: Dict[str, str] = {
            "routing-strategy": "least-request",
            "Content-Type": "application/json",
        }
        if auth_token or settings.LLM_AUTH_TOKEN:
            token = auth_token or settings.LLM_AUTH_TOKEN
            # 安全：headers 中正常携带，但日志中脱敏
            self._headers["Authorization"] = f"Bearer {token}"

        logger.info("LlmClient 初始化: model=%s, url=%s", self.model, _sanitize_for_logging(self.api_url))

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
    ) -> str:
        """
        调用 LLM 生成回复。

        Args:
            messages: 消息列表，格式 [{"role": "system/user/assistant", "content": "..."}]
            temperature: 采样温度
            top_p: 核采样概率
            top_k: Top-K 采样
            max_tokens: 最大生成 token 数
            enable_thinking: 是否启用思考模式

        Returns:
            模型生成的纯文本回复

        Raises:
            LlmApiError: API 调用失败
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if not settings.LLM_STRICT_OPENAI_COMPAT:
            payload["stream"] = False
            payload["top_p"] = top_p
            payload["top_k"] = top_k
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        return self._do_request(payload)

    @_retry_with_backoff(max_retries=3, base_delay=1.0, backoff_factor=2.0)
    def _do_request(self, payload: Dict[str, Any]) -> str:
        """
        执行单次 HTTP POST 请求（受重试装饰器保护）。
        """
        input_chars = sum(len(str(msg.get("content", ""))) for msg in payload["messages"])
        logger.debug(
            "LLM 请求体: model=%s, messages_count=%d, input_chars=%d, max_tokens=%s",
            payload["model"],
            len(payload["messages"]),
            input_chars,
            payload.get("max_tokens"),
        )

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self._headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LlmApiError(f"请求发送失败: {exc}") from exc

        # 日志中只记录状态码和响应长度，严禁记录 Authorization 响应头
        logger.debug("LLM 响应: status=%d, content_length=%d", response.status_code, len(response.text))

        if response.status_code == 200:
            try:
                data = response.json()
            except (ValueError, TypeError) as exc:
                raise LlmApiError(
                    f"响应 JSON 解析失败: {exc}",
                    status_code=response.status_code,
                    body=response.text[:500],
                ) from exc

            # 兼容多种响应格式
            content = self._extract_content(data)
            if content is None:
                raise LlmApiError(
                    "无法从 LLM 响应中提取文本内容",
                    status_code=response.status_code,
                    body=str(data)[:500],
                )
            return content

        # 非 200 状态码
        status_code = response.status_code
        body_snippet = response.text[:1000]
        logger.error("LLM API 非 200 响应正文摘要: %s", _sanitize_response_body(body_snippet))
        raise LlmApiError(
            f"LLM API 返回非 200 状态码: HTTP {status_code}",
            status_code=status_code,
            body=body_snippet,
        )

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> Optional[str]:
        """
        从 LLM 响应 JSON 中提取文本内容。
        兼容 OpenAI 格式和部分自定义格式。
        """
        # OpenAI 标准格式: {"choices": [{"message": {"content": "..."}}]}
        if "choices" in data and isinstance(data["choices"], list):
            choices = data["choices"]
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                if content:
                    return content.strip()
        # 备选格式: {"output": {"text": "..."}} 
        if "output" in data and isinstance(data["output"], dict):
            text = data["output"].get("text", "")
            if text:
                return text.strip()
        # 备选格式: {"response": "..."} 
        if "response" in data and isinstance(data["response"], str):
            return data["response"].strip()
        # 备选格式: {"result": "..."}
        if "result" in data and isinstance(data["result"], str):
            return data["result"].strip()
        return None
