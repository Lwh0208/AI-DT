"""
自动化批处理流水线核心入口。

系统运行分两个阶段，数据源明确分离：

阶段一：画像初始化（离线）
    数据来源：data/history/ 目录下的 .txt 文件
    流程：按聊天窗口加载历史聊天 → 为张照西及所有角色构建画像与记忆
    特点：只在首次运行或需要重建画像时执行；已有画像的角色自动跳过

阶段二：模拟运行（在线）
    数据来源：data/test/ 目录下的 .txt 文件
    流程：按聊天窗口逐文件遍历测试消息 → 模拟回复 → 即时反思
    特点：每个聊天窗口独立维护最近上下文；日期变更时触发 Dream 模式状态演进

使用方式：
    python -m src.app                      # 默认路径（data/history/ + data/test/）
    python -m src.app --history-dir /path   # 指定历史数据目录
    python -m src.app --test-dir /path      # 指定测试数据目录
    python -m src.app --init-only           # 仅执行画像初始化，不运行模拟
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import settings, ZHANG_ZHAOXI_ALIASES
from src.data_loader import discover_raw_data_by_file
from src.llm_client import LlmClient
from src.profile_manager import ProfileManager
from src.session_store import SessionStore
from src.feedback import FeedbackEngine
from src.report_generator import ReportGenerator
from src.prompts import SYSTEM_PROMPT_DREAM_SYNC, USER_PROMPT_DREAM_SYNC

logger = logging.getLogger(__name__)

# 目标角色
TARGET_CHARACTER = "张照西"


class PipelineInitializationError(RuntimeError):
    """角色画像初始化失败，流水线必须中止。"""


class Pipeline:
    """
    智能回复助手自动化批处理流水线。
    
    严格区分历史数据（画像初始化）和测试数据（模拟运行）。
    """

    def __init__(
        self,
        history_dir: Optional[Path] = None,
        test_dir: Optional[Path] = None,
        llm_client: Optional[LlmClient] = None,
    ) -> None:
        self.history_dir = history_dir or settings.HISTORY_DIR
        self.test_dir = test_dir or settings.TEST_DIR
        self._startup_profiles_existed = settings.PROFILES_DIR.exists()
        self._startup_runtime_existed = settings.RUNTIME_DIR.exists()
        self._startup_profiles_snapshot = self._snapshot_tree(settings.PROFILES_DIR)
        self._startup_runtime_snapshot = self._snapshot_tree(settings.RUNTIME_DIR)

        # 核心组件
        self.llm = llm_client or LlmClient()
        self.pm = ProfileManager(self.llm)
        self.ss = SessionStore()
        self.fb = FeedbackEngine(self.llm, self.pm, self.ss)

        # 运行时状态追踪
        self._last_date: Optional[datetime] = None
        self._pending_predictions: List[Dict[str, Any]] = []
        self._pending_real_replies: List[Dict[str, Any]] = []
        self._daily_trace_buffer: List[Dict[str, Any]] = []
        self._known_characters: Set[str] = set()

        logger.info(
            "Pipeline 初始化完成: history_dir=%s, test_dir=%s",
            self.history_dir, self.test_dir,
        )

    # ==================================================================
    # 公共入口
    # ==================================================================

    def run(self, init_only: bool = False) -> None:
        """
        执行完整的批处理流水线。

        Args:
            init_only: 若为 True，仅执行画像初始化阶段后退出。
        """
        logger.info("=" * 60)
        logger.info("智能回复助手流水线启动")
        logger.info("  历史数据目录: %s", self.history_dir)
        logger.info("  测试数据目录: %s", self.test_dir)
        logger.info("=" * 60)

        # ============================================================
        # 阶段一：画像初始化（基于历史数据）
        # ============================================================
        history_conversations = discover_raw_data_by_file(self.history_dir)
        if history_conversations:
            total_history = sum(len(messages) for messages in history_conversations.values())
            logger.info(
                "从历史数据加载 %d 个聊天窗口、%d 条聊天记录，开始画像初始化...",
                len(history_conversations),
                total_history,
            )
            try:
                self._initialize_profiles(history_conversations)
            except Exception as exc:
                logger.error("画像初始化失败，开始清理本次运行残留并停止程序: %s", exc)
                self._cleanup_after_initialization_failure()
                raise PipelineInitializationError("角色画像初始化失败，已回滚本次运行残留") from exc
        else:
            logger.warning("历史数据目录 %s 中无聊天记录，跳过画像初始化", self.history_dir)

        if init_only:
            logger.info("仅初始化模式，流水线结束")
            return

        # ============================================================
        # 阶段二：模拟运行（基于测试数据）
        # ============================================================
        test_conversations = discover_raw_data_by_file(self.test_dir)
        if test_conversations:
            total_test = sum(len(messages) for messages in test_conversations.values())
            logger.info(
                "从测试数据加载 %d 个聊天窗口、%d 条聊天记录，开始按窗口模拟运行...",
                len(test_conversations),
                total_test,
            )
            self._process_conversations(test_conversations)
        else:
            logger.warning("测试数据目录 %s 中无聊天记录，跳过模拟运行", self.test_dir)

        # 生成运行汇总报告
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_gen = ReportGenerator(
            session_store=self.ss,
            profile_manager=self.pm,
        )
        report_path = report_gen.generate(run_id=run_id)
        logger.info("运行报告已生成: %s", report_path)
        transcript_path = report_gen.generate_transcript_txt(run_id=run_id)
        logger.info("测试对照文本已生成: %s", transcript_path)

        logger.info("=" * 60)
        logger.info("智能回复助手流水线执行完毕")
        logger.info("=" * 60)

    # ==================================================================
    # 阶段一：角色画像初始化（历史数据）
    # ==================================================================

    def _initialize_profiles(self, conversations: Dict[str, List[Dict]]) -> None:
        """
        分析历史聊天记录，为张照西和所有已识别的其他角色构建画像。
        已有画像的角色自动跳过。
        """
        all_senders: Set[str] = set()
        for messages in conversations.values():
            for msg in messages:
                all_senders.add(msg["sender"])
        self._known_characters = all_senders
        raw_text = self._serialize_conversations_for_prompt(conversations)

        # 为张照西构建画像
        if not self.pm.profile_exists(TARGET_CHARACTER):
            try:
                self.pm.build_initial_profiles(TARGET_CHARACTER, raw_text)
                logger.info("张照西画像与记忆初始化完成")
            except Exception as exc:
                logger.error("张照西画像初始化失败: %s", exc)
                raise
        else:
            logger.info("张照西画像已存在，跳过初始化")

        # 为其他角色构建画像
        other_senders = all_senders - {TARGET_CHARACTER}
        for sender in other_senders:
            if not self.pm.profile_exists(sender):
                sender_conversations = self._filter_conversations_for_character(
                    conversations,
                    sender,
                )
                sender_raw_text = self._serialize_conversations_for_prompt(sender_conversations)
                try:
                    self.pm.build_initial_profiles(sender, sender_raw_text)
                    logger.info("角色 [%s] 画像与记忆初始化完成", sender)
                except Exception as exc:
                    logger.error("角色 [%s] 画像初始化失败: %s", sender, exc)
                    raise

    # ==================================================================
    # 阶段二：消息流处理（测试数据）
    # ==================================================================

    def _process_conversations(self, conversations: Dict[str, List[Dict]]) -> None:
        """按聊天窗口逐个处理测试数据。"""
        for conversation_id, messages in conversations.items():
            if not messages:
                continue

            logger.info(
                "开始处理聊天窗口 [%s]，共 %d 条消息",
                conversation_id,
                len(messages),
            )
            self._reset_window_state()
            self._process_message_stream(messages)

            # 每个窗口独立完成最后一天的 Dream 同步，避免跨窗口轨迹混杂。
            if self._daily_trace_buffer:
                self._flush_pending_reflection()
                last_date = self._last_date or self._daily_trace_buffer[-1].get("timestamp")
                if last_date:
                    self._trigger_dream_mode(last_date)
                self._daily_trace_buffer.clear()

    def _process_message_stream(self, messages: List[Dict]) -> None:
        """
        逐条遍历测试数据消息流，执行完整的模拟运行逻辑。
        """
        for idx, msg in enumerate(messages):
            ts: datetime = msg["timestamp"]
            sender: str = msg["sender"]
            content: str = msg["content"]

            logger.debug(
                "[%d/%d] [%s] %s: %s",
                idx + 1, len(messages), ts.strftime("%Y-%m-%d %H:%M:%S"), sender, content[:50],
            )

            # --- 日期变更检测 → Dream 模式 ---
            current_date = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            if self._last_date is not None and current_date > self._last_date:
                logger.info(
                    "检测到日期变更: %s → %s，触发 Dream 模式",
                    self._last_date.strftime("%Y-%m-%d"),
                    current_date.strftime("%Y-%m-%d"),
                )
                self._flush_pending_reflection()
                self._trigger_dream_mode(self._last_date)
                self._daily_trace_buffer.clear()

            self._last_date = current_date

            # --- 消息分类处理 ---
            if sender == TARGET_CHARACTER:
                self._handle_real_reply(msg, ts)
            else:
                self._handle_incoming_message(msg, sender, content, ts)

    def _handle_incoming_message(
        self,
        msg: Dict,
        sender: str,
        content: str,
        ts: datetime,
    ) -> None:
        """处理非张照西发送的新消息：按窗口类型决定是否生成模拟回复。"""
        self._flush_pending_reflection()

        chat_type = self._get_chat_type(msg)
        is_group = chat_type == "群聊"

        if not is_group:
            need_reply = True
            logger.debug("单聊新消息直接触发回复")
        else:
            mentions_zhang = any(alias in content for alias in ZHANG_ZHAOXI_ALIASES)
            need_reply = mentions_zhang
            if need_reply:
                logger.debug("群聊消息明确提及张照西或其别名，触发回复")
            else:
                logger.debug("群聊消息未提及张照西或其别名，跳过模拟回复")

        self._record_message(msg, role="context")
        self._daily_trace_buffer.append({
            "timestamp": ts,
            "type": "incoming_message",
            "sender": sender,
            "content": content,
            "conversation_id": msg.get("conversation_id", msg.get("source_file", "")),
        })

        if need_reply:
            simulated = self.fb.generate_simulated_reply(
                current_sender=sender,
                new_message=content,
                chat_type=chat_type,
                is_group=is_group,
                conversation_id=msg.get("conversation_id", msg.get("source_file", "")),
            )

            simulated_record = {
                "timestamp": ts,
                "sender": TARGET_CHARACTER,
                "content": simulated,
                "role": "simulated",
                "trigger_message": content,
                "source_file": msg.get("source_file", ""),
                "conversation_id": msg.get("conversation_id", msg.get("source_file", "")),
                "chat_type": chat_type,
            }
            self.ss.append(simulated_record)

            self._pending_predictions.append({
                "timestamp": ts,
                "sender": sender,
                "message": content,
                "assistant_reply": simulated,
                "conversation_id": msg.get("conversation_id", msg.get("source_file", "")),
            })

            self._daily_trace_buffer.append({
                "timestamp": ts,
                "type": "simulated_reply",
                "content": simulated,
                "conversation_id": msg.get("conversation_id", msg.get("source_file", "")),
            })

    def _handle_real_reply(self, msg: Dict, ts: datetime) -> None:
        """处理张照西的真实回复：记录并触发即时反思。"""
        content = msg["content"]
        self._record_message(msg, role="real")

        self._daily_trace_buffer.append({
            "timestamp": ts,
            "type": "real_reply",
            "content": content,
            "conversation_id": msg.get("conversation_id", msg.get("source_file", "")),
        })

        if self._pending_predictions:
            self._pending_real_replies.append({
                "timestamp": ts,
                "sender": TARGET_CHARACTER,
                "content": content,
                "conversation_id": msg.get("conversation_id", msg.get("source_file", "")),
            })

    def _flush_pending_reflection(self) -> None:
        """将一段 pending 预测与随后连续真实回复整体反思。"""
        if not self._pending_predictions or not self._pending_real_replies:
            return

        reflection_time = self._pending_real_replies[-1]["timestamp"]
        conversation_id = self._pending_real_replies[-1].get("conversation_id", "")
        reflection = self.fb.reflect_and_learn(
            new_message=self._format_pending_messages(self._pending_predictions),
            assistant_reply=self._format_pending_predictions(self._pending_predictions),
            real_reply=self._format_real_replies(self._pending_real_replies),
            current_time=reflection_time,
        )
        if reflection:
            self._daily_trace_buffer.append({
                "timestamp": reflection_time,
                "type": "reflection",
                "content": reflection,
                "conversation_id": conversation_id,
            })

        self._pending_predictions.clear()
        self._pending_real_replies.clear()

    # ==================================================================
    # Dream 模式
    # ==================================================================

    def _trigger_dream_mode(self, target_date: datetime) -> None:
        """触发 Dream 模式状态同步引擎。"""
        date_str = target_date.strftime("%Y-%m-%d")
        logger.info("===== Dream 模式启动 (%s) =====", date_str)

        if not self._daily_trace_buffer:
            logger.info("本日无对话踪迹，跳过 Dream 模式")
            return

        current_profile = self.pm.get_profile(TARGET_CHARACTER)
        current_memory = self.pm.get_memory(TARGET_CHARACTER)

        traces_text = self._format_daily_traces(self._daily_trace_buffer)
        current_time_str = target_date.strftime("%Y-%m-%d 23:59:59")

        system_msg = SYSTEM_PROMPT_DREAM_SYNC.format(current_time=current_time_str)
        user_msg = USER_PROMPT_DREAM_SYNC.format(
            current_time=current_time_str,
            current_profile=current_profile,
            current_memory=current_memory,
            daily_conversation_traces=traces_text,
        )

        try:
            raw_output = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=settings.DREAM_MAX_TOKENS,
            )
            success = self.pm.dream_update(
                character_name=TARGET_CHARACTER,
                raw_output=raw_output,
                update_time=target_date,
            )
            if success:
                logger.info("Dream 模式更新成功 (%s)", date_str)
            else:
                logger.warning("Dream 模式更新失败，输出已安全兜底 (%s)", date_str)
        except Exception as exc:
            logger.error("Dream 模式 LLM 调用失败: %s", exc)

        logger.info("===== Dream 模式结束 (%s) =====", date_str)

    # ==================================================================
    # 工具方法
    # ==================================================================

    def _record_message(self, msg: Dict, role: str) -> None:
        """将消息记录到会话存储。"""
        record = {
            "timestamp": msg["timestamp"],
            "sender": msg["sender"],
            "content": msg["content"],
            "role": role,
            "source_file": msg.get("source_file", ""),
            "conversation_id": msg.get("conversation_id", msg.get("source_file", "")),
            "chat_type": msg.get("chat_type", "单聊"),
        }
        self.ss.append(record)

    @staticmethod
    def _format_pending_messages(predictions: List[Dict[str, Any]]) -> str:
        """格式化一组触发预测的连续新消息。"""
        lines: list[str] = []
        for idx, item in enumerate(predictions, start=1):
            ts = item.get("timestamp", "")
            if isinstance(ts, datetime):
                ts = ts.strftime("%Y-%m-%d %H:%M:%S")
            sender = item.get("sender", "未知")
            message = item.get("message", "")
            lines.append(f"{idx}. [{ts}] {sender}: {message}")
        return "\n".join(lines)

    @staticmethod
    def _format_pending_predictions(predictions: List[Dict[str, Any]]) -> str:
        """格式化每条新消息对应的模拟回复。"""
        lines: list[str] = []
        for idx, item in enumerate(predictions, start=1):
            message = item.get("message", "")
            reply = item.get("assistant_reply", "")
            lines.append(f"{idx}. 触发消息：{message}\n   模拟回复：{reply}")
        return "\n".join(lines)

    @staticmethod
    def _format_real_replies(real_replies: List[Dict[str, Any]]) -> str:
        """格式化随后连续出现的张照西真实回复。"""
        lines: list[str] = []
        for idx, item in enumerate(real_replies, start=1):
            ts = item.get("timestamp", "")
            if isinstance(ts, datetime):
                ts = ts.strftime("%Y-%m-%d %H:%M:%S")
            content = item.get("content", "")
            lines.append(f"{idx}. [{ts}] {TARGET_CHARACTER}: {content}")
        return "\n".join(lines)

    def _reset_window_state(self) -> None:
        """重置单个聊天窗口内的运行状态。"""
        self._last_date = None
        self._pending_predictions = []
        self._pending_real_replies = []
        self._daily_trace_buffer.clear()

    def _cleanup_after_initialization_failure(self) -> None:
        """清理本次启动后新增的 profiles/runtime 文件，恢复到启动前状态。"""
        self._restore_tree(
            settings.PROFILES_DIR,
            self._startup_profiles_snapshot,
            self._startup_profiles_existed,
        )
        self._restore_tree(
            settings.RUNTIME_DIR,
            self._startup_runtime_snapshot,
            self._startup_runtime_existed,
        )

    @staticmethod
    def _snapshot_tree(root: Path) -> Set[Tuple[str, bool]]:
        """记录目录启动前的相对路径快照，元素为 (relative_path, is_dir)。"""
        if not root.exists():
            return set()

        snapshot: Set[Tuple[str, bool]] = set()
        for path in root.rglob("*"):
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            snapshot.add((rel, path.is_dir()))
        return snapshot

    @staticmethod
    def _restore_tree(root: Path, snapshot: Set[Tuple[str, bool]], root_existed: bool = True) -> None:
        """删除启动后新增的文件和目录，保留启动前已有内容。"""
        if not root.exists():
            return

        snapshot_paths = {rel for rel, _ in snapshot}
        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            if rel in snapshot_paths:
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                logger.debug("已清理初始化失败残留: %s", path)
            except OSError as exc:
                logger.warning("清理初始化失败残留失败: %s (%s)", path, exc)

        if not root_existed:
            try:
                root.rmdir()
                logger.debug("已清理初始化失败新增目录: %s", root)
            except OSError:
                pass

    @staticmethod
    def _get_chat_type(msg: Dict) -> str:
        """读取数据加载阶段根据文件夹标记出的窗口类型。"""
        return "群聊" if msg.get("chat_type") == "群聊" else "单聊"

    @staticmethod
    def _serialize_messages_for_prompt(messages: List[Dict]) -> str:
        """将消息列表序列化为 Prompt 文本。"""
        lines: list[str] = []
        for msg in messages:
            ts = msg["timestamp"]
            sender = msg["sender"]
            content = msg["content"]
            if isinstance(ts, datetime):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = str(ts)
            lines.append(f"[{ts_str}] {sender}: {content}")
        return "\n".join(lines)

    @classmethod
    def _serialize_conversations_for_prompt(cls, conversations: Dict[str, List[Dict]]) -> str:
        """将多个聊天窗口分块序列化，避免把私聊窗口混成同一条时间线。"""
        blocks: list[str] = []
        for conversation_id, messages in conversations.items():
            blocks.append(f"## 聊天窗口：{conversation_id}")
            blocks.append(cls._serialize_messages_for_prompt(messages))
        return cls._limit_profile_init_prompt("\n\n".join(blocks))

    @staticmethod
    def _limit_profile_init_prompt(text: str) -> str:
        """限制初始化画像 Prompt 的历史输入长度，避免超出 LLM 上下文。"""
        max_chars = settings.PROFILE_INIT_MAX_INPUT_CHARS
        if max_chars <= 0 or len(text) <= max_chars:
            return text

        marker = (
            f"【系统提示：原始历史聊天文本共 {len(text)} 字符，"
            f"为适配模型上下文限制，仅保留最近 {max_chars} 字符。】\n\n"
        )
        keep_chars = max(max_chars - len(marker), 0)
        return marker + text[-keep_chars:]

    @staticmethod
    def _filter_conversations_for_character(
        conversations: Dict[str, List[Dict]],
        character_name: str,
    ) -> Dict[str, List[Dict]]:
        """只保留指定角色实际参与过的聊天窗口。"""
        return {
            conversation_id: messages
            for conversation_id, messages in conversations.items()
            if any(msg.get("sender") == character_name for msg in messages)
        }

    @staticmethod
    def _format_daily_traces(traces: List[Dict[str, Any]]) -> str:
        """将本日审计踪迹格式化为文本。"""
        lines: list[str] = []
        for trace in traces:
            ts = trace.get("timestamp", "")
            trace_type = trace.get("type", "unknown")
            content = trace.get("content", "")
            sender = trace.get("sender", "")
            conversation_id = trace.get("conversation_id", "")
            if isinstance(ts, datetime):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = str(ts)
            window = f"[{conversation_id}] " if conversation_id else ""
            lines.append(f"{window}[{ts_str}] [{trace_type}] {sender}: {content}")
        return "\n".join(lines)


def main() -> None:
    """CLI 入口函数。"""
    parser = argparse.ArgumentParser(
        description="智能回复助手 - 张照西数字孪生回复引擎",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help="历史聊天数据目录（用于画像初始化，默认: data/history/）",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="测试聊天数据目录（用于模拟运行，默认: data/test/）",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="仅执行画像初始化，不运行模拟",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志输出",
    )
    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pipeline = Pipeline(
        history_dir=args.history_dir,
        test_dir=args.test_dir,
    )
    try:
        pipeline.run(init_only=args.init_only)
    except PipelineInitializationError as exc:
        logger.critical("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
