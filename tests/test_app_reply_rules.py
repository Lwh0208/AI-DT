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


class FakeDreamLlm:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], max_tokens: int = 1024) -> str:
        self.calls.append(messages)
        return (
            "---PROFILE_UPDATED_START---\n"
            "# updated profile\n"
            "---PROFILE_UPDATED_END---\n"
            "---MEMORY_UPDATED_START---\n"
            "# updated memory\n"
            "---MEMORY_UPDATED_END---"
        )


class FakeProfileManager:
    def __init__(self) -> None:
        self.updated: list[str] = []

    def get_profile(self, character_name: str) -> str:
        return f"# {character_name} profile"

    def get_memory(self, character_name: str) -> str:
        return f"# {character_name} memory"

    def dream_update(self, character_name: str, raw_output: str, update_time: datetime) -> bool:
        self.updated.append(character_name)
        return True


def _make_pipeline(tmp_path) -> Pipeline:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.ss = SessionStore(log_path=tmp_path / "dialogue_log.jsonl", window_size=20)
    pipeline.fb = FakeFeedbackEngine()
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


def test_consecutive_messages_are_tracked_for_dream_without_reflection(tmp_path):
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

    trace_types = [trace["type"] for trace in pipeline._daily_trace_buffer]
    assert trace_types == [
        "incoming_message",
        "simulated_reply",
        "incoming_message",
        "simulated_reply",
        "real_reply",
        "real_reply",
    ]
    assert Pipeline._get_dream_target_characters(pipeline._daily_trace_buffer) == [
        "张照西",
        "李文浩",
    ]


def test_dream_updates_all_characters_in_daily_traces(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    pipeline.llm = FakeDreamLlm()
    pipeline.pm = FakeProfileManager()
    pipeline._daily_trace_buffer = [
        {
            "timestamp": datetime(2026, 4, 1, 9, 0, 0),
            "type": "incoming_message",
            "sender": "李文浩",
            "content": "帮忙看下",
            "conversation_id": "单聊/liwenhao.txt",
        },
        {
            "timestamp": datetime(2026, 4, 1, 9, 0, 1),
            "type": "simulated_reply",
            "sender": "张照西",
            "content": "我看看",
            "conversation_id": "单聊/liwenhao.txt",
        },
        {
            "timestamp": datetime(2026, 4, 1, 9, 1, 0),
            "type": "incoming_message",
            "sender": "赵宇",
            "content": "我补充一下",
            "conversation_id": "单聊/zhaoyu.txt",
        },
    ]

    pipeline._trigger_dream_mode(datetime(2026, 4, 1))

    assert pipeline.pm.updated == ["张照西", "李文浩", "赵宇"]
    assert len(pipeline.llm.calls) == 3
    user_prompts = [call[1]["content"] for call in pipeline.llm.calls]
    assert any('角色"张照西"' in prompt for prompt in user_prompts)
    assert any('角色"李文浩"' in prompt for prompt in user_prompts)
    assert any('角色"赵宇"' in prompt for prompt in user_prompts)


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


def test_rollback_restores_existing_file_contents(tmp_path):
    profiles_dir = tmp_path / "profiles"
    runtime_dir = tmp_path / "runtime"
    char_dir = profiles_dir / "张照西"
    char_dir.mkdir(parents=True)
    runtime_dir.mkdir()
    profile_path = char_dir / "profile.md"
    log_path = runtime_dir / "dialogue_log.jsonl"
    profile_path.write_text("original profile", encoding="utf-8")
    log_path.write_text("original log", encoding="utf-8")

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._startup_profiles_existed = True
    pipeline._startup_runtime_existed = True
    pipeline._startup_profiles_snapshot = Pipeline._snapshot_tree(profiles_dir)
    pipeline._startup_runtime_snapshot = Pipeline._snapshot_tree(runtime_dir)

    profile_path.write_text("changed profile", encoding="utf-8")
    log_path.write_text("changed log", encoding="utf-8")
    (runtime_dir / "new_report.md").write_text("new", encoding="utf-8")

    import src.app as app_module

    original_profiles_dir = app_module.settings.PROFILES_DIR
    original_runtime_dir = app_module.settings.RUNTIME_DIR
    app_module.settings.PROFILES_DIR = profiles_dir
    app_module.settings.RUNTIME_DIR = runtime_dir
    try:
        pipeline._rollback_to_startup_state()
    finally:
        app_module.settings.PROFILES_DIR = original_profiles_dir
        app_module.settings.RUNTIME_DIR = original_runtime_dir

    assert profile_path.read_text(encoding="utf-8") == "original profile"
    assert log_path.read_text(encoding="utf-8") == "original log"
    assert not (runtime_dir / "new_report.md").exists()
