"""
滑动窗口上下文与持久化层。

负责：
- 对话日志的 JSONL 持久化存储
- 滑动窗口上下文管理（默认最近 20 条）
- 文件锁安全并发写入
- 按时间戳升序返回上下文窗口
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Set

from src.config import settings

logger = logging.getLogger(__name__)

if os.name == "nt":
    import msvcrt
else:
    import fcntl


@contextmanager
def _file_lock(fh: IO[str], exclusive: bool = True):
    """
    跨平台文件锁。

    Unix/macOS 使用 fcntl.flock；Windows 使用 msvcrt.locking。
    Windows 的 msvcrt 是字节范围锁，这里统一锁住文件起始处 1 个字节，
    用作整个 JSONL 文件的进程间互斥哨兵。
    """
    if os.name == "nt":
        fh.seek(0)
        mode = msvcrt.LK_LOCK
        msvcrt.locking(fh.fileno(), mode, 1)
        try:
            yield
        finally:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        lock_mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fh, lock_mode)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class SessionStore:
    """
    滑动窗口会话上下文管理器。

    所有对话记录以 JSONL 格式追加写入 dialogue_log.jsonl，
    读取时自动按时间戳排序并返回最近 N 条上下文。
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        window_size: Optional[int] = None,
    ) -> None:
        self.log_path = log_path or settings.DIALOGUE_LOG_PATH
        self.window_size = window_size or settings.CONTEXT_WINDOW_SIZE
        self._ensure_dir()
        logger.info(
            "SessionStore 初始化: log=%s, window_size=%d",
            self.log_path, self.window_size,
        )

    def _ensure_dir(self) -> None:
        """确保日志文件所在目录存在。"""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        """
        追加一条对话记录到 JSONL 文件。

        Args:
            record: 对话记录字典，至少包含 timestamp, sender, content 字段。
                    可额外包含 role(simulated/real/context), is_group 等元数据。
        """
        # 确保时间戳序列化
        record_to_write = dict(record)
        if "timestamp" in record_to_write:
            ts = record_to_write["timestamp"]
            if isinstance(ts, datetime):
                record_to_write["timestamp"] = ts.isoformat()
            record_to_write["_ts_order"] = ts.isoformat() if isinstance(ts, datetime) else str(ts)

        line = json.dumps(record_to_write, ensure_ascii=False)

        try:
            with open(self.log_path, "a+", encoding="utf-8") as fh:
                # 跨平台文件锁，防止多进程并发写入交叉。
                with _file_lock(fh, exclusive=True):
                    fh.seek(0, os.SEEK_END)
                    fh.write(line + "\n")
                    fh.flush()
        except OSError as exc:
            logger.error("写入对话日志失败: %s", exc)
            raise

    def get_recent_context(
        self,
        limit: Optional[int] = None,
        conversation_id: Optional[str] = None,
        roles: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取最近 N 条上下文对话记录（按时间戳升序排列）。

        Args:
            limit: 返回条数上限，默认使用初始化时的 window_size。
            conversation_id: 指定聊天窗口 ID 时，仅返回该窗口内的上下文。
            roles: 指定角色类型集合时，仅返回这些 role 的记录。

        Returns:
            按时间升序排列的记录列表。
            每条记录中 timestamp 字段已被还原为 datetime 对象。
        """
        limit = limit if limit is not None else self.window_size
        if not self.log_path.is_file():
            return []

        all_records: List[Dict[str, Any]] = []
        try:
            with open(self.log_path, "r+", encoding="utf-8") as fh:
                with _file_lock(fh, exclusive=False):
                    fh.seek(0)
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            # 还原时间戳
                            if "timestamp" in record:
                                raw_ts = record["timestamp"]
                                if isinstance(raw_ts, str):
                                    record["timestamp"] = datetime.fromisoformat(raw_ts)
                            if (
                                conversation_id is not None
                                and record.get("conversation_id") != conversation_id
                            ):
                                continue
                            if roles is not None and record.get("role") not in roles:
                                continue
                            all_records.append(record)
                        except (json.JSONDecodeError, ValueError) as exc:
                            logger.warning("跳过无法解析的日志行: %s", exc)
                            continue
        except OSError as exc:
            logger.error("读取对话日志失败: %s", exc)
            return []

        # 按时间戳排序后取最后 limit 条
        all_records.sort(key=lambda r: r.get("timestamp", datetime.min))
        return all_records[-limit:]

    def get_all_records(self) -> List[Dict[str, Any]]:
        """
        获取全部对话记录（按时间戳升序排列）。
        主要用于 Dream 模式中的全量审计。
        """
        return self.get_recent_context(limit=self._count_lines() if self.log_path.is_file() else 0)

    def _count_lines(self) -> int:
        """统计日志文件行数。"""
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                return sum(1 for _ in fh if _.strip())
        except OSError:
            return 0

    def get_records_by_date(self, target_date: datetime) -> List[Dict[str, Any]]:
        """
        获取指定日期的全部对话记录。

        Args:
            target_date: 目标日期（仅比较年-月-日）

        Returns:
            该日期内的所有记录列表
        """
        all_records = self.get_all_records()
        target_str = target_date.strftime("%Y-%m-%d")
        return [
            r for r in all_records
            if isinstance(r.get("timestamp"), datetime)
            and r["timestamp"].strftime("%Y-%m-%d") == target_str
        ]

    def clear(self) -> None:
        """清空对话日志文件。仅用于测试环境。"""
        if self.log_path.is_file():
            try:
                with open(self.log_path, "w", encoding="utf-8") as fh:
                    pass
                logger.info("对话日志已清空: %s", self.log_path)
            except OSError as exc:
                logger.error("清空对话日志失败: %s", exc)
                raise
