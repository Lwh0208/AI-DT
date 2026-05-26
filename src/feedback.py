"""
回复预测模块。

负责：
1. 即时模拟回复生成：组装 Prompt，调用 LLM 生成张照西的模拟回复
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from src.config import settings
from src.llm_client import LlmClient
from src.profile_manager import ProfileManager
from src.session_store import SessionStore
from src.prompts import (
    SYSTEM_PROMPT_REPLY_GENERATION,
    USER_PROMPT_REPLY_GENERATION,
)

logger = logging.getLogger(__name__)


class FeedbackEngine:
    """
    反馈引擎：模拟回复生成。
    """

    def __init__(
        self,
        llm_client: Optional[LlmClient] = None,
        profile_manager: Optional[ProfileManager] = None,
        session_store: Optional[SessionStore] = None,
    ) -> None:
        self.llm = llm_client or LlmClient()
        self.pm = profile_manager or ProfileManager(self.llm)
        self.ss = session_store or SessionStore()

    # ------------------------------------------------------------------
    # 1. 即时模拟回复生成
    # ------------------------------------------------------------------

    def generate_simulated_reply(
        self,
        current_sender: str,
        new_message: str,
        chat_type: str,
        is_group: bool = False,
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        生成张照西的模拟回复。

        Args:
            current_sender: 当前消息发送者姓名
            new_message: 收到的新消息内容
            chat_type: "群聊" 或 "单聊"
            is_group: 是否是群聊（快捷标记）
            conversation_id: 当前聊天窗口 ID，用于隔离最近上下文。

        Returns:
            模拟回复纯文本
        """
        if is_group:
            chat_type = "群聊"

        # 加载张照西的画像和记忆
        zhang_profile = self.pm.get_profile("张照西")
        zhang_memory = self.pm.get_memory("张照西")

        # 加载发送者的画像和记忆
        sender_profile = self.pm.get_profile(current_sender)
        if not sender_profile:
            sender_profile = "（该对话者尚无画像记录）"

        # 获取当前聊天窗口内最近 20 条真实上下文；模拟回复只用于评估，不回灌预测。
        recent_messages = self.ss.get_recent_context(
            limit=settings.CONTEXT_WINDOW_SIZE,
            conversation_id=conversation_id,
            roles={"context", "real"},
        )
        recent_20_text = self._format_recent_messages(recent_messages)

        system_msg = SYSTEM_PROMPT_REPLY_GENERATION
        user_msg = USER_PROMPT_REPLY_GENERATION.format(
            zhang_profile=zhang_profile,
            zhang_memory=zhang_memory,
            sender_profile=sender_profile,
            recent_20_messages=recent_20_text,
            chat_type=chat_type,
            current_sender=current_sender,
            new_message=new_message,
        )

        try:
            reply = self.llm.chat(messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])
            logger.info("模拟回复已生成 (发送者: %s, 消息: %s...)", current_sender, new_message[:30])
            return reply
        except Exception as exc:
            logger.error("模拟回复生成失败: %s", exc)
            return "我待会儿回你"  # 安全降级

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _format_recent_messages(messages: List[Dict]) -> str:
        """将消息列表格式化为 Prompt 可用的文本。"""
        if not messages:
            return "（暂无上下文记录）"

        lines: list[str] = []
        for msg in messages[-20:]:
            ts = msg.get("timestamp", "")
            sender = msg.get("sender", "未知")
            content = msg.get("content", "")
            role_tag = msg.get("role", "")
            if isinstance(ts, datetime):
                ts = ts.strftime("%Y-%m-%d %H:%M:%S")
            tag = f"[{role_tag}]" if role_tag else ""
            lines.append(f"[{ts}] {sender}{tag}: {content}")

        return "\n".join(lines)
