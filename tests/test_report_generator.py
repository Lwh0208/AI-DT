"""
运行报告生成器测试。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.report_generator import ReportGenerator
from src.session_store import SessionStore


def test_generate_transcript_txt_contains_context_simulated_and_real(tmp_path):
    store = SessionStore(log_path=tmp_path / "dialogue_log.jsonl", window_size=20)
    base_time = datetime(2026, 4, 1, 9, 0, 0)
    store.append({
        "timestamp": base_time,
        "sender": "李文浩",
        "content": "西哥帮忙看下",
        "role": "context",
        "conversation_id": "单聊/liwenhao.txt",
    })
    store.append({
        "timestamp": base_time + timedelta(seconds=1),
        "sender": "张照西",
        "content": "我看看",
        "role": "simulated",
        "trigger_message": "西哥帮忙看下",
        "conversation_id": "单聊/liwenhao.txt",
    })
    store.append({
        "timestamp": base_time + timedelta(seconds=2),
        "sender": "张照西",
        "content": "我待会儿看下",
        "role": "real",
        "conversation_id": "单聊/liwenhao.txt",
    })

    generator = ReportGenerator(session_store=store, output_dir=tmp_path)
    path = generator.generate_transcript_txt(run_id="test_run")
    content = path.read_text(encoding="utf-8")

    assert "===== 聊天窗口: 单聊/liwenhao.txt =====" in content
    assert "[测试消息] 李文浩: 西哥帮忙看下" in content
    assert "[预测回复] 张照西: 我看看" in content
    assert "触发消息: 西哥帮忙看下" in content
    assert "[真实回复] 张照西: 我待会儿看下" in content
