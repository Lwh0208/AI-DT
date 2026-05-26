"""
测试数据回复触发规则。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.app import Pipeline
from src.session_store import SessionStore


class FakeFeedbackEngine:
    def __init__(self) -> None:
        self.generated: list[tuple[str, str, str]] = []
        self.reflections: list[tuple[str, str, str]] = []

    def generate_simulated_reply(
        self,
        current_sender: str,
        new_message: str,
        chat_type: str,
        is_group: bool = False,
        conversation_id: Optional[str] = None,
    ) -> str:
        self.generated.append((current_sender, new_message, chat_type))
        return f"模拟回复{len(self.generated)}"

    def reflect_and_learn(
        self,
        new_message: str,
        assistant_reply: str,
        real_reply: str,
        current_time: datetime,
    ) -> str:
        self.reflections.append((new_message, assistant_reply, real_reply))
        return "反思经验"


def _make_pipeline(tmp_path) -> Pipeline:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.ss = SessionStore(log_path=tmp_path / "dialogue_log.jsonl", window_size=20)
    pipeline.fb = FakeFeedbackEngine()
    pipeline._pending_predictions = []
    pipeline._pending_real_replies = []
    pipeline._daily_trace_buffer = []
    return pipeline


def _make_msg(
    chat_type: str,
    content: str,
    sender: str = "李文浩",
    ts: Optional[datetime] = None,
) -> dict:
    return {
        "timestamp": ts or datetime(2026, 4, 1, 9, 0, 0),
        "sender": sender,
        "content": content,
        "source_file": f"{chat_type}/liwenhao.txt",
        "conversation_id": f"{chat_type}/liwenhao.txt",
        "chat_type": chat_type,
    }


def test_single_chat_non_target_message_always_triggers_prediction(tmp_path):
    pipeline = _make_pipeline(tmp_path)

    pipeline._handle_incoming_message(
        _make_msg("单聊", "这个问题帮我看下"),
        sender="李文浩",
        content="这个问题帮我看下",
        ts=datetime(2026, 4, 1, 9, 0, 0),
    )

    assert pipeline.fb.generated == [("李文浩", "这个问题帮我看下", "单聊")]
    records = pipeline.ss.get_recent_context(limit=20, conversation_id="单聊/liwenhao.txt")
    assert [record["role"] for record in records] == ["context", "simulated"]


def test_group_chat_without_target_mention_does_not_trigger_prediction(tmp_path):
    pipeline = _make_pipeline(tmp_path)

    pipeline._handle_incoming_message(
        _make_msg("群聊", "这个问题谁看一下"),
        sender="李文浩",
        content="这个问题谁看一下",
        ts=datetime(2026, 4, 1, 9, 0, 0),
    )

    assert pipeline.fb.generated == []
    records = pipeline.ss.get_recent_context(limit=20, conversation_id="群聊/liwenhao.txt")
    assert [record["role"] for record in records] == ["context"]


def test_group_chat_with_target_mention_triggers_prediction(tmp_path):
    pipeline = _make_pipeline(tmp_path)

    pipeline._handle_incoming_message(
        _make_msg("群聊", "西哥这个问题帮忙看一下"),
        sender="李文浩",
        content="西哥这个问题帮忙看一下",
        ts=datetime(2026, 4, 1, 9, 0, 0),
    )

    assert pipeline.fb.generated == [("李文浩", "西哥这个问题帮忙看一下", "群聊")]


def test_consecutive_predictions_reflect_against_consecutive_real_replies(tmp_path):
    pipeline = _make_pipeline(tmp_path)

    pipeline._handle_incoming_message(
        _make_msg("单聊", "第一个问题", ts=datetime(2026, 4, 1, 9, 0, 0)),
        sender="李文浩",
        content="第一个问题",
        ts=datetime(2026, 4, 1, 9, 0, 0),
    )
    pipeline._handle_incoming_message(
        _make_msg("单聊", "第二个补充", ts=datetime(2026, 4, 1, 9, 1, 0)),
        sender="李文浩",
        content="第二个补充",
        ts=datetime(2026, 4, 1, 9, 1, 0),
    )

    pipeline._handle_real_reply(
        _make_msg("单聊", "真实回复一", sender="张照西", ts=datetime(2026, 4, 1, 9, 2, 0)),
        ts=datetime(2026, 4, 1, 9, 2, 0),
    )
    pipeline._handle_real_reply(
        _make_msg("单聊", "真实回复二", sender="张照西", ts=datetime(2026, 4, 1, 9, 3, 0)),
        ts=datetime(2026, 4, 1, 9, 3, 0),
    )

    assert pipeline.fb.reflections == []

    pipeline._handle_incoming_message(
        _make_msg("单聊", "下一轮问题", ts=datetime(2026, 4, 1, 9, 4, 0)),
        sender="李文浩",
        content="下一轮问题",
        ts=datetime(2026, 4, 1, 9, 4, 0),
    )

    assert len(pipeline.fb.reflections) == 1
    new_messages, simulated_replies, real_replies = pipeline.fb.reflections[0]
    assert "第一个问题" in new_messages
    assert "第二个补充" in new_messages
    assert "模拟回复1" in simulated_replies
    assert "模拟回复2" in simulated_replies
    assert "真实回复一" in real_replies
    assert "真实回复二" in real_replies
    assert len(pipeline._pending_predictions) == 1


def test_initialization_failure_rolls_back_new_profiles_and_runtime(tmp_path, monkeypatch):
    import src.app as app_module

    profiles_dir = tmp_path / "profiles"
    runtime_dir = tmp_path / "runtime"
    profiles_dir.mkdir()
    runtime_dir.mkdir()
    (profiles_dir / ".gitkeep").write_text("", encoding="utf-8")
    (runtime_dir / ".gitkeep").write_text("", encoding="utf-8")

    monkeypatch.setattr(app_module.settings, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(app_module.settings, "RUNTIME_DIR", runtime_dir)

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._startup_profiles_existed = True
    pipeline._startup_runtime_existed = True
    pipeline._startup_profiles_snapshot = Pipeline._snapshot_tree(profiles_dir)
    pipeline._startup_runtime_snapshot = Pipeline._snapshot_tree(runtime_dir)
    pipeline.pm = type(
        "FakeProfileManager",
        (),
        {
            "profile_exists": lambda self, name: False,
            "build_initial_profiles": lambda self, name, text: (
                (profiles_dir / name).mkdir(parents=True, exist_ok=True),
                (profiles_dir / name / "profile.md").write_text("partial", encoding="utf-8"),
                (_ for _ in ()).throw(RuntimeError("boom")),
            ),
        },
    )()
    pipeline._known_characters = set()

    conversations = {
        "单聊/liwenhao.txt": [
            {
                "timestamp": datetime(2026, 4, 1, 9, 0, 0),
                "sender": "李文浩",
                "content": "测试消息",
            }
        ]
    }

    try:
        pipeline._initialize_profiles(conversations)
    except RuntimeError:
        pipeline._cleanup_after_initialization_failure()
    else:
        raise AssertionError("初始化失败应抛出异常")

    assert sorted(p.name for p in profiles_dir.iterdir()) == [".gitkeep"]
    assert sorted(p.name for p in runtime_dir.iterdir()) == [".gitkeep"]
