"""
滑动窗口与持久化单元测试。

测试用例1: 连续追加 25 条带时间戳的对话记录
测试用例2: 调用 get_recent_context(limit=20) 断言返回列表长度恰好 20，
           且内容按时间戳升序排列，正好是最后注入的 20 条记录
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from src.session_store import SessionStore


@pytest.fixture
def temp_session_store(tmp_path):
    """创建一个使用临时目录的 SessionStore 实例。"""
    log_path = tmp_path / "test_dialogue_log.jsonl"
    return SessionStore(log_path=log_path, window_size=20)


def _make_record(index: int, base_time: datetime) -> dict:
    """生成一条测试用对话记录。"""
    ts = base_time + timedelta(minutes=index)
    return {
        "timestamp": ts,
        "sender": f"用户{index % 5}",
        "content": f"这是第 {index + 1} 条测试消息",
        "source_file": "test_data.txt",
    }


# ====================================================================
# 测试用例 1: 连续追加 25 条记录
# ====================================================================

class TestAppendRecords:
    """验证连续追加对话记录到 JSONL 文件。"""

    def test_append_25_records(self, temp_session_store, tmp_path):
        """连续追加 25 条带时间戳的对话记录，验证文件存在且行数正确。"""
        base_time = datetime(2026, 1, 15, 9, 0, 0)
        for i in range(25):
            record = _make_record(i, base_time)
            temp_session_store.append(record)

        log_path = tmp_path / "test_dialogue_log.jsonl"
        assert log_path.is_file(), "JSONL 日志文件应该已创建"

        # 计算行数
        with open(log_path, "r", encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        assert len(lines) == 25, f"应有 25 行记录，实际 {len(lines)} 行"

    def test_append_preserves_all_fields(self, temp_session_store):
        """验证追加的记录各字段完整保存。"""
        record = {
            "timestamp": datetime(2026, 3, 20, 14, 30, 0),
            "sender": "张照西",
            "content": "我待会回工位看下再弄",
            "role": "real",
            "source_file": "chat.txt",
        }
        temp_session_store.append(record)

        recent = temp_session_store.get_recent_context(limit=1)
        assert len(recent) == 1
        result = recent[0]
        assert result["sender"] == "张照西"
        assert result["content"] == "我待会回工位看下再弄"
        assert result["role"] == "real"
        # 时间戳应被还原为 datetime
        assert isinstance(result["timestamp"], datetime)

    def test_append_to_empty_store(self, temp_session_store):
        """首次追加记录后应能正确读取。"""
        record = _make_record(0, datetime(2026, 1, 1, 8, 0, 0))
        temp_session_store.append(record)

        recent = temp_session_store.get_recent_context(limit=10)
        assert len(recent) == 1


# ====================================================================
# 测试用例 2: 滑动窗口滑出测试
# ====================================================================

class TestSlidingWindow:
    """验证滑动窗口行为：取最近 N 条，时间升序，旧记录被滑出。"""

    def test_recent_context_returns_exactly_20(self, temp_session_store):
        """
        追加 25 条记录后，调用 get_recent_context(limit=20)，
        断言返回列表长度精确等于 20。
        """
        base_time = datetime(2026, 1, 15, 9, 0, 0)
        for i in range(25):
            record = _make_record(i, base_time)
            temp_session_store.append(record)

        recent = temp_session_store.get_recent_context(limit=20)
        assert len(recent) == 20, f"预期 20 条，实际 {len(recent)} 条"

    def test_recent_context_returns_last_20(self, temp_session_store):
        """
        断言返回的内容正好是最后注入的 20 条记录（索引 5~24）。
        """
        base_time = datetime(2026, 1, 15, 9, 0, 0)
        for i in range(25):
            record = _make_record(i, base_time)
            temp_session_store.append(record)

        recent = temp_session_store.get_recent_context(limit=20)

        # 验证内容是最后 20 条（消息编号从 "第 6 条" 到 "第 25 条"）
        contents = [r["content"] for r in recent]
        assert contents[0] == "这是第 6 条测试消息"
        assert contents[-1] == "这是第 25 条测试消息"

    def test_recent_context_is_chronologically_ordered(self, temp_session_store):
        """
        断言返回的 20 条记录按时间戳升序排列。
        """
        base_time = datetime(2026, 1, 15, 9, 0, 0)
        for i in range(25):
            record = _make_record(i, base_time)
            temp_session_store.append(record)

        recent = temp_session_store.get_recent_context(limit=20)

        timestamps = [r["timestamp"] for r in recent]
        for j in range(len(timestamps) - 1):
            assert timestamps[j] <= timestamps[j + 1], (
                f"时间戳未升序排列: {timestamps[j]} > {timestamps[j + 1]}"
            )

    def test_recent_context_fewer_than_limit(self, temp_session_store):
        """当总记录数少于 limit 时，返回全部记录。"""
        base_time = datetime(2026, 5, 1, 10, 0, 0)
        for i in range(5):
            record = _make_record(i, base_time)
            temp_session_store.append(record)

        recent = temp_session_store.get_recent_context(limit=20)
        assert len(recent) == 5

    def test_recent_context_empty_store(self, temp_session_store):
        """空存储应返回空列表。"""
        recent = temp_session_store.get_recent_context(limit=20)
        assert recent == []

    def test_recent_context_filters_by_conversation_id(self, temp_session_store):
        """指定 conversation_id 时，只返回该聊天窗口内的最近上下文。"""
        base_time = datetime(2026, 1, 15, 9, 0, 0)
        for i in range(6):
            record = _make_record(i, base_time)
            record["conversation_id"] = "liwenhao.txt" if i % 2 == 0 else "zhaoyu.txt"
            temp_session_store.append(record)

        recent = temp_session_store.get_recent_context(
            limit=20,
            conversation_id="liwenhao.txt",
        )

        assert len(recent) == 3
        assert all(r["conversation_id"] == "liwenhao.txt" for r in recent)
        assert [r["content"] for r in recent] == [
            "这是第 1 条测试消息",
            "这是第 3 条测试消息",
            "这是第 5 条测试消息",
        ]

    def test_recent_context_filters_by_roles(self, temp_session_store):
        """指定 roles 时，模拟回复不会进入真实上下文窗口。"""
        base_time = datetime(2026, 1, 15, 9, 0, 0)
        records = [
            {
                "timestamp": base_time,
                "sender": "李文浩",
                "content": "真实新消息",
                "role": "context",
                "conversation_id": "liwenhao.txt",
            },
            {
                "timestamp": base_time + timedelta(seconds=1),
                "sender": "张照西",
                "content": "助手模拟回复",
                "role": "simulated",
                "conversation_id": "liwenhao.txt",
            },
            {
                "timestamp": base_time + timedelta(seconds=2),
                "sender": "张照西",
                "content": "真实回复",
                "role": "real",
                "conversation_id": "liwenhao.txt",
            },
        ]
        for record in records:
            temp_session_store.append(record)

        recent = temp_session_store.get_recent_context(
            limit=20,
            conversation_id="liwenhao.txt",
            roles={"context", "real"},
        )

        assert [r["content"] for r in recent] == ["真实新消息", "真实回复"]


# ====================================================================
# 额外测试: 持久化与清空
# ====================================================================

class TestPersistenceAndClear:
    """验证持久化行为与清空操作。"""

    def test_clear_empties_log(self, temp_session_store):
        """清空操作应将日志文件重置为空。"""
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        for i in range(3):
            temp_session_store.append(_make_record(i, base_time))

        temp_session_store.clear()

        recent = temp_session_store.get_recent_context(limit=20)
        assert recent == []

    def test_records_persist_across_sessions(self, tmp_path):
        """验证记录在不同 SessionStore 实例间持久化。"""
        log_path = tmp_path / "persistent_log.jsonl"

        # 第一次写入
        store1 = SessionStore(log_path=log_path, window_size=20)
        store1.append({
            "timestamp": datetime(2026, 1, 1, 12, 0, 0),
            "sender": "张照西",
            "content": "持久化测试消息",
        })

        # 第二次读取（新建实例）
        store2 = SessionStore(log_path=log_path, window_size=20)
        recent = store2.get_recent_context(limit=20)
        assert len(recent) == 1
        assert recent[0]["content"] == "持久化测试消息"

    def test_get_records_by_date(self, temp_session_store):
        """验证按日期筛选记录。"""
        base_time = datetime(2026, 4, 10, 9, 0, 0)
        for i in range(10):
            record = _make_record(i, base_time)
            temp_session_store.append(record)

        # 再追加另一天的记录
        next_day = datetime(2026, 4, 11, 9, 0, 0)
        for i in range(5):
            record = _make_record(i, next_day)
            temp_session_store.append(record)

        target_date = datetime(2026, 4, 10)
        day_records = temp_session_store.get_records_by_date(target_date)
        assert len(day_records) == 10
        for rec in day_records:
            assert rec["timestamp"].strftime("%Y-%m-%d") == "2026-04-10"
