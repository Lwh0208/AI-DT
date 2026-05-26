"""
全局配置管理模块。

解析并管理所有运行时配置，支持从 .env 文件增量覆盖环境变量。
安全防线：启动时校验关键配置项，严禁在任何日志和报错信息中泄露明文 Token。
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 项目根目录推断
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 默认配置值
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, str] = {
    "LLM_API_URL": "http://10.127.23.252:43379/v1/chat/completions",
    "LLM_MODEL": "qwen3-coder",
    "LLM_TIMEOUT_SECONDS": "120",
    "LLM_STRICT_OPENAI_COMPAT": "true",
    "DEFAULT_YEAR": "2026",
    "CONTEXT_WINDOW_SIZE": "20",
    "PROFILE_INIT_MAX_INPUT_CHARS": "60000",
    "PROFILE_INIT_MAX_TOKENS": "16384",
    "DREAM_MAX_TOKENS": "16384",
}

# 张照西的识别别名列表（用于 @提及判定）
ZHANG_ZHAOXI_ALIASES: list[str] = ["张照西", "照西", "西哥"]

# 隐私敏感字段关键词（用于日志脱敏）
_SENSITIVE_KEYWORDS: list[str] = [
    "authorization",
    "token",
    "password",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
]


def _load_dotenv(dotenv_path: Optional[Path] = None) -> None:
    """
    极简 .env 文件解析器，不依赖 python-dotenv 第三方库。
    仅支持 KEY=VALUE 格式，忽略注释行和空行。
    如果对应环境变量已存在，则不覆盖（环境变量优先级更高）。
    """
    if dotenv_path is None:
        dotenv_path = _PROJECT_ROOT / ".env"
    if not dotenv_path.is_file():
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    logger.warning(".env 第 %d 行格式异常，已跳过: %s", line_no, line[:40])
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as exc:
        logger.warning("读取 .env 文件失败: %s", exc)


def _sanitize_for_logging(text: str) -> str:
    """对日志输出进行脱敏，防止泄露明文密钥/Token。"""
    lower = text.lower()
    for kw in _SENSITIVE_KEYWORDS:
        if kw in lower:
            return "<REDACTED>"
    return text


class Settings:
    """
    全局配置单例。
    
    使用方式：
        from src.config import settings
        url = settings.LLM_API_URL
    """

    def __init__(self) -> None:
        # 先加载 .env，再从环境变量读取（环境变量可以覆盖 .env）
        _load_dotenv()

        # --- LLM 配置 ---
        self.LLM_API_URL: str = os.environ.get(
            "LLM_API_URL", _DEFAULTS["LLM_API_URL"]
        )
        self.LLM_MODEL: str = os.environ.get(
            "LLM_MODEL", _DEFAULTS["LLM_MODEL"]
        )
        self.LLM_TIMEOUT_SECONDS: int = int(
            os.environ.get("LLM_TIMEOUT_SECONDS", _DEFAULTS["LLM_TIMEOUT_SECONDS"])
        )
        self.LLM_AUTH_TOKEN: Optional[str] = os.environ.get("LLM_AUTH_TOKEN")
        self.LLM_STRICT_OPENAI_COMPAT: bool = os.environ.get(
            "LLM_STRICT_OPENAI_COMPAT",
            _DEFAULTS["LLM_STRICT_OPENAI_COMPAT"],
        ).lower() in {"1", "true", "yes", "on"}

        # --- 编排配置 ---
        self.DEFAULT_YEAR: int = int(
            os.environ.get("DEFAULT_YEAR", _DEFAULTS["DEFAULT_YEAR"])
        )
        self.CONTEXT_WINDOW_SIZE: int = int(
            os.environ.get("CONTEXT_WINDOW_SIZE", _DEFAULTS["CONTEXT_WINDOW_SIZE"])
        )
        self.PROFILE_INIT_MAX_INPUT_CHARS: int = int(
            os.environ.get(
                "PROFILE_INIT_MAX_INPUT_CHARS",
                _DEFAULTS["PROFILE_INIT_MAX_INPUT_CHARS"],
            )
        )
        self.PROFILE_INIT_MAX_TOKENS: int = int(
            os.environ.get(
                "PROFILE_INIT_MAX_TOKENS",
                _DEFAULTS["PROFILE_INIT_MAX_TOKENS"],
            )
        )
        self.DREAM_MAX_TOKENS: int = int(
            os.environ.get(
                "DREAM_MAX_TOKENS",
                _DEFAULTS["DREAM_MAX_TOKENS"],
            )
        )

        # --- 路径配置 ---
        self.PROJECT_ROOT: Path = _PROJECT_ROOT
        self.DATA_DIR: Path = _PROJECT_ROOT / "data"
        self.HISTORY_DIR: Path = self.DATA_DIR / "history"   # 初始化画像用的历史聊天数据
        self.TEST_DIR: Path = self.DATA_DIR / "test"         # 模拟运行用的测试聊天数据
        self.PROFILES_DIR: Path = _PROJECT_ROOT / "profiles"
        self.RUNTIME_DIR: Path = _PROJECT_ROOT / "runtime"
        self.DIALOGUE_LOG_PATH: Path = self.RUNTIME_DIR / "dialogue_log.jsonl"

        self._validate()

    def _validate(self) -> None:
        """启动时安全防线校验。"""
        if not self.LLM_API_URL:
            raise ValueError(
                "LLM_API_URL 配置为空，系统无法启动。"
                "请在 .env 文件或环境变量中设置 LLM_API_URL。"
            )
        logger.info(
            "配置校验通过: LLM_API_URL=%s, LLM_MODEL=%s",
            _sanitize_for_logging(self.LLM_API_URL),
            self.LLM_MODEL,
        )

    def __repr__(self) -> str:
        safe_token = "<SET>" if self.LLM_AUTH_TOKEN else "<NOT_SET>"
        return (
            f"Settings("
            f"LLM_API_URL={_sanitize_for_logging(self.LLM_API_URL)}, "
            f"LLM_MODEL={self.LLM_MODEL}, "
            f"LLM_AUTH_TOKEN={safe_token}, "
            f"LLM_STRICT_OPENAI_COMPAT={self.LLM_STRICT_OPENAI_COMPAT}, "
            f"DEFAULT_YEAR={self.DEFAULT_YEAR}, "
            f"CONTEXT_WINDOW_SIZE={self.CONTEXT_WINDOW_SIZE}, "
            f"PROFILE_INIT_MAX_INPUT_CHARS={self.PROFILE_INIT_MAX_INPUT_CHARS})"
        )


# 模块级单例
settings = Settings()
