"""
回复预测、即时反思与经验沉淀模块。

负责：
1. 即时模拟回复生成：组装 Prompt，调用 LLM 生成张照西的模拟回复
2. 即时反思与经验沉淀：对比模拟回复与真实回复，将反思日志增量追加到经验.md
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
    SYSTEM_PROMPT_REFLECT_LOG,
    USER_PROMPT_REFLECT_LOG,
)

logger = logging.getLogger(__name__)


class FeedbackEngine:
    """
    反馈引擎：模拟回复生成 + 即时反思 + 经验沉淀。
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
        self._ensure_experience_file()

    def _ensure_experience_file(self) -> None:
        """确保经验.md文件存在。"""
        settings.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        if not settings.EXPERIENCE_PATH.is_file():
            with open(settings.EXPERIENCE_PATH, "w", encoding="utf-8") as fh:
                fh.write("# 回复预测反思经验库\n\n")
            logger.info("经验库文件已创建: %s", settings.EXPERIENCE_PATH)

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

        # 读取历史经验库
        historical_experience = self._read_experience()

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
            historical_experience=historical_experience,
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
    # 2. 即时反思与经验沉淀
    # ------------------------------------------------------------------

    def reflect_and_learn(
        self,
        new_message: str,
        assistant_reply: str,
        real_reply: str,
        current_time: datetime,
    ) -> Optional[str]:
        """
        对比一组连续新消息、对应模拟回复与随后连续真实回复，
        生成反思日志并增量追加到经验.md。

        Args:
            new_message: 触发回复的原始新消息或连续新消息段
            assistant_reply: 助手生成的模拟回复或连续模拟回复段
            real_reply: 张照西本人的真实回复或连续真实回复段
            current_time: 当前对话时间

        Returns:
            反思日志文本，或 None（如果无偏差）
        """
        if not real_reply or not real_reply.strip():
            logger.debug("真实回复为空，跳过反思")
            return None

        if not assistant_reply or not assistant_reply.strip():
            logger.debug("模拟回复为空，跳过反思")
            return None

        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        system_msg = SYSTEM_PROMPT_REFLECT_LOG.format(current_time=time_str, new_message=new_message)
        user_msg = USER_PROMPT_REFLECT_LOG.format(
            new_message=new_message,
            assistant_reply=assistant_reply,
            real_reply=real_reply,
        )

        try:
            reflection = self.llm.chat(messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])
        except Exception as exc:
            logger.error("反思 LLM 调用失败: %s", exc)
            return None

        reflection = reflection.strip()

        if reflection == "NO_VARIANCE":
            logger.info("模拟回复与真实回复极度接近，无需沉淀经验")
            return None

        # 增量追加到经验.md
        self._append_to_experience(reflection)
        logger.info("反思经验已沉淀")
        return reflection

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _read_experience() -> str:
        """读取经验库全文。"""
        path = settings.EXPERIENCE_PATH
        if not path.is_file():
            return "（尚无历史经验）"
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
            return content if content else "（尚无历史经验）"

    @staticmethod
    def _append_to_experience(text: str) -> None:
        """增量追加反思日志到经验.md。"""
        path = settings.EXPERIENCE_PATH
        settings.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n\n" + text)

        logger.debug("经验已追加写入: %s", path)

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
