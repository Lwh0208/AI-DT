"""
聊天记录数据加载测试。
"""

from __future__ import annotations

from src.data_loader import discover_raw_data_by_file


def test_discover_raw_data_by_file_marks_chat_type_from_folders(tmp_path):
    """根据一级目录识别群聊/单聊，并保留相对路径作为窗口 ID。"""
    single_dir = tmp_path / "单聊"
    group_dir = tmp_path / "群聊"
    single_dir.mkdir()
    group_dir.mkdir()

    (single_dir / "liwenhao.txt").write_text(
        "李文浩 4/1 09:00:00\n西哥，帮忙看下\n",
        encoding="utf-8",
    )
    (group_dir / "project.txt").write_text(
        "赵宇 4/1 10:00:00\n这个问题谁处理\n",
        encoding="utf-8",
    )

    conversations = discover_raw_data_by_file(tmp_path)

    assert set(conversations) == {"单聊/liwenhao.txt", "群聊/project.txt"}
    assert conversations["单聊/liwenhao.txt"][0]["chat_type"] == "单聊"
    assert conversations["单聊/liwenhao.txt"][0]["conversation_id"] == "单聊/liwenhao.txt"
    assert conversations["群聊/project.txt"][0]["chat_type"] == "群聊"
    assert conversations["群聊/project.txt"][0]["conversation_id"] == "群聊/project.txt"


def test_discover_raw_data_by_file_defaults_root_txt_to_single_chat(tmp_path):
    """兼容旧结构：直接放在数据目录下的 txt 默认按单聊处理。"""
    (tmp_path / "zhaoyu.txt").write_text(
        "赵宇 4/1 10:00:00\n今天看下这个\n",
        encoding="utf-8",
    )

    conversations = discover_raw_data_by_file(tmp_path)

    assert set(conversations) == {"zhaoyu.txt"}
    assert conversations["zhaoyu.txt"][0]["chat_type"] == "单聊"


def test_metadata_date_line_is_message_content_not_sender(tmp_path):
    """会议卡片中的“日期:”字段不能被识别为角色。"""
    (tmp_path / "liwenhao.txt").write_text(
        "\n".join([
            "张照西 2/13 19:43:49",
            "录屏:",
            "",
            "会议录制: 张照西的快速会议",
            "日期: 2026-2-13 18:42:35",
            "录制文件: https://meeting.tencent.com/example",
            "",
            "张照西 2/13 19:45:36",
            "[文件: db2messages_v2.py]",
        ]),
        encoding="utf-8",
    )

    conversations = discover_raw_data_by_file(tmp_path)
    messages = conversations["liwenhao.txt"]

    assert [msg["sender"] for msg in messages] == ["张照西", "张照西"]
    assert "日期: 2026-2-13 18:42:35" in messages[0]["content"]
