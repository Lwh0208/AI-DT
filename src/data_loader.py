"""
聊天记录时序解析与多窗口解析流。

负责：
- 动态扫描 data/ 目录下所有 .txt 文件
- 编码自动探测 (utf-8 → utf-8-sig → gbk)
- 解析多种聊天记录格式为按聊天窗口隔离的结构化时序列表
- 缺失年份自动补全为配置的默认年份（2026）

支持的聊天格式：
  格式A（微信/QQ导出 - 最常见）:
      发送者 M/DD HH:MM:SS
      消息内容（换行）
      发送者 M/DD HH:MM:SS
      消息内容

  格式B（带年份的微信导出）:
      发送者 2026/4/22 HH:MM:SS
      消息内容

  格式C（中划线日期）:
      [2026-01-15 10:30:22] 发送者: 消息内容

  格式D（无括号中划线）:
      2026-01-15 10:30:22 发送者: 消息内容
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 不可见 Unicode 字符清洗（AIGC 水印零宽字符等）
# ---------------------------------------------------------------------------
_INVISIBLE_CHARS_RE = re.compile(
    "["
    "\u200b-\u200f"  # 零宽空格/非连接符/连接符/左右标记
    "\u202a-\u202e"  # 双向文本控制
    "\u2060-\u2064"  # 词连接符等
    "\ufeff"         # BOM
    "\u180e"         # 蒙古语元音分隔符
    "]",
)


def _strip_invisible(text: str) -> str:
    """移除字符串中的不可见 Unicode 字符（零宽空格、水印字符等）。"""
    return _INVISIBLE_CHARS_RE.sub("", text)


# ---------------------------------------------------------------------------
# 时间戳解析格式（用于 strptime）
# ---------------------------------------------------------------------------
_DATETIME_FORMATS: List[str] = [
    # 完整年份 + 各种分隔符
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    # 缺失年份（跨月/日用斜杠分隔） — 微信导出最常见格式
    "%m/%d %H:%M:%S",
    "%m/%d %H:%M",
    # 缺失年份（跨月/日用中划线分隔）
    "%m-%d %H:%M:%S",
    "%m-%d %H:%M",
    # 仅时间
    "%H:%M:%S",
    "%H:%M",
]


def _parse_timestamp(
    raw_ts: str,
    default_year: int,
) -> Optional[datetime]:
    """
    尝试用多种格式解析时间戳字符串。
    若格式中不含年份，自动补全为 default_year。
    """
    raw_ts = raw_ts.strip()
    for fmt in _DATETIME_FORMATS:
        try:
            dt = datetime.strptime(raw_ts, fmt)
            # strptime 解析不含年份的格式时，year 默认为 1900
            if dt.year == 1900:
                dt = dt.replace(year=default_year)
            return dt
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# 行级格式检测正则
# ---------------------------------------------------------------------------

# 格式A/B: "发送者 日期时间" — 内容在下一行
#   如: 张照西 4/22 18:12:10     或   张照西 2026/4/22 18:12:10
#   发送者可以是中文/英文名，后跟空格和日期时间
_RE_SENDER_DATETIME = re.compile(
    r"^(\S+(?:\s+\S+){0,2}?)\s+"       # 发送者（1~3个词，兼容「张照西」「A组长」等）
    r"("                                 # === 时间戳开始 ===
    r"\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?"   # 2026/4/22 18:12:10
    r"|\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?"        # 4/22 18:12:10
    r"|\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?"  # 2026-4-22 18:12:10
    r"|\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?"        # 4-22 18:12:10
    r")$"
)

# 格式C: [2026-01-15 10:30:22] 发送者: 消息内容
_RE_BRACKET_TS_CONTENT = re.compile(
    r"^\[(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+?)[:：]\s*(.*)",
    re.DOTALL,
)

# 格式C变体: [4/22 18:12:10] 发送者: 消息内容（缺年份+斜杠+方括号）
_RE_BRACKET_TS_PARTIAL = re.compile(
    r"^\[(\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+?)[:：]\s*(.*)",
    re.DOTALL,
)

# 格式D: 2026-01-15 10:30:22 发送者: 消息内容（无方括号）
_RE_NOBRACKET_TS_CONTENT = re.compile(
    r"^(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\s+(.+?)[:：]\s*(.*)",
    re.DOTALL,
)

_RE_ONLY_DATE_TIME_CHARS = re.compile(r"^[\d/\-:\s]+$")


def _is_valid_sender_name(sender: str) -> bool:
    """判断消息头中的发送者字段是否像真实角色名，而不是正文标签。"""
    sender = sender.strip()
    if not sender:
        return False
    if sender.endswith((":","：")):
        return False
    if _RE_ONLY_DATE_TIME_CHARS.match(sender):
        return False
    return True


def _detect_file_encoding(file_path: Path) -> str:
    """
    编码自动探测：依次尝试 utf-8 → utf-8-sig → gbk。
    返回第一个能成功解码的编码名称。
    """
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(file_path, "r", encoding=enc) as fh:
                fh.read(4096)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise UnicodeDecodeError(
        "multi", b"", 0, 1,
        f"无法以 utf-8/utf-8-sig/gbk 中的任何编码读取文件: {file_path}",
    )


def _try_parse_header_line(line: str) -> Optional[tuple]:
    """
    尝试将一行文本解析为消息头（提取 sender + timestamp）。

    优先匹配「发送者 日期时间」（格式A/B），然后匹配方括号/无括号的同行格式。

    Returns:
        (sender, raw_timestamp) 元组，或 None 表示该行不是消息头。
    """
    # --- 优先: 格式A/B — 发送者在前，时间戳在后，内容另起一行 ---
    m = _RE_SENDER_DATETIME.match(line)
    if m:
        sender = m.group(1).strip()
        raw_ts = m.group(2).strip()
        if _is_valid_sender_name(sender):
            return (sender, raw_ts)

    # --- 格式C: [时间戳] 发送者: 内容（同行） ---
    m = _RE_BRACKET_TS_CONTENT.match(line)
    if m:
        sender = m.group(2).strip()
        if _is_valid_sender_name(sender):
            return (sender, m.group(1))

    m = _RE_BRACKET_TS_PARTIAL.match(line)
    if m:
        sender = m.group(2).strip()
        if _is_valid_sender_name(sender):
            return (sender, m.group(1))

    # --- 格式D: 时间戳 发送者: 内容（同行） ---
    m = _RE_NOBRACKET_TS_CONTENT.match(line)
    if m:
        sender = m.group(2).strip()
        if _is_valid_sender_name(sender):
            return (sender, m.group(1))

    return None


def _parse_single_file(file_path: Path) -> List[Dict]:
    """
    解析单个聊天记录文件，返回结构化的消息列表。

    核心逻辑：
    1. 逐行读取，对每行先尝试识别为「消息头」（sender + timestamp）
    2. 如果是消息头，开始一条新消息
    3. 后续非空行（直到下一条消息头出现）都作为当前消息的内容追加
    4. 空行在不同格式下的处理：
       - 格式A/B（内容换行）：空行是消息间的分隔
       - 格式C/D（同行格式）：空行被忽略

    返回格式:
    [{"timestamp": datetime, "sender": str, "content": str, "source_file": str}]
    """
    encoding = _detect_file_encoding(file_path)
    logger.info("解析文件 %s (编码: %s)", file_path.name, encoding)

    messages: List[Dict] = []
    current_sender: Optional[str] = None
    current_ts: Optional[datetime] = None
    current_content_lines: List[str] = []
    is_newline_format: Optional[bool] = None  # None=未判定, True=换行格式, False=同行格式

    def _flush_current_message() -> None:
        """将当前正在收集的消息刷入消息列表。"""
        nonlocal current_sender, current_ts, current_content_lines
        if current_sender is not None and current_ts is not None:
            content = "\n".join(current_content_lines).strip()
            # 清洗不可见字符（零宽空格、AIGC水印等）
            content = _strip_invisible(content)
            if content:  # 跳过空内容
                messages.append({
                    "timestamp": current_ts,
                    "sender": _strip_invisible(current_sender),
                    "content": content,
                    "source_file": file_path.name,
                })
        current_sender = None
        current_ts = None
        current_content_lines = []

    with open(file_path, "r", encoding=encoding) as fh:
        raw_lines = fh.readlines()

    for raw_line in raw_lines:
        line = raw_line.rstrip("\n\r")

        # 尝试识别为消息头
        header = _try_parse_header_line(line)

        if header is not None:
            # --- 新消息头 → 先刷入上一条消息 ---
            _flush_current_message()

            sender, raw_ts = header
            sender = _strip_invisible(sender)
            dt = _parse_timestamp(raw_ts, settings.DEFAULT_YEAR)
            if dt is None:
                logger.warning(
                    "时间戳解析失败（文件: %s，行: %s），已跳过",
                    file_path.name, line[:80],
                )
                continue

            current_sender = _strip_invisible(sender)
            current_ts = dt
            current_content_lines = []
            is_newline_format = True  # 检测到换行格式

        else:
            # --- 非消息头 ---
            # 1. 先检查是否为同行格式（格式C/D: [ts] sender: content）
            inline_content = _try_parse_inline_line(line)
            if inline_content is not None:
                _flush_current_message()
                sender, raw_ts, content = inline_content
                dt = _parse_timestamp(raw_ts, settings.DEFAULT_YEAR)
                if dt is None:
                    logger.warning(
                        "时间戳解析失败（文件: %s，行: %s），已跳过",
                        file_path.name, line[:80],
                    )
                    continue
                current_sender = _strip_invisible(sender)
                current_ts = dt
                current_content_lines = [_strip_invisible(content)]
                is_newline_format = False
                _flush_current_message()  # 同行格式，内容已经完整，直接刷入
                continue

            # 2. 换行格式下的内容行
            if is_newline_format and current_sender is not None:
                # 空行在换行格式中：标记消息间分隔
                if not line.strip():
                    # 空行作为消息分隔，但不立即 flush（因为可能只是段落间距）
                    # 只在遇到下一条消息头时才 flush
                    continue
                current_content_lines.append(line)
            elif current_sender is not None:
                # 同行格式下的无关行（不应该出现，做兜底）
                if line.strip():
                    current_content_lines.append(line)
            else:
                # 文件开头的非消息行
                if line.strip():
                    logger.debug(
                        "文件 %s 首条消息前的无关行，已跳过: %s",
                        file_path.name, line[:60],
                    )

    # 文件末尾：刷入最后一条消息
    _flush_current_message()

    logger.info("文件 %s 解析完成，共 %d 条消息", file_path.name, len(messages))
    return messages


def _try_parse_inline_line(line: str) -> Optional[tuple]:
    """
    尝试解析「同行格式」的消息行（内容与时间戳在同一行）。

    Returns:
        (sender, raw_timestamp, content) 元组，或 None。
    """
    # 格式C: [时间戳] 发送者: 内容
    m = _RE_BRACKET_TS_CONTENT.match(line)
    if m and _is_valid_sender_name(m.group(2).strip()):
        return (m.group(2).strip(), m.group(1), m.group(3).strip())

    m = _RE_BRACKET_TS_PARTIAL.match(line)
    if m and _is_valid_sender_name(m.group(2).strip()):
        return (m.group(2).strip(), m.group(1), m.group(3).strip())

    # 格式D: 时间戳 发送者: 内容
    m = _RE_NOBRACKET_TS_CONTENT.match(line)
    if m and _is_valid_sender_name(m.group(2).strip()):
        return (m.group(2).strip(), m.group(1), m.group(3).strip())

    return None


def discover_raw_data_by_file(data_dir: Path) -> Dict[str, List[Dict]]:
    """
    动态扫描 data/ 目录下所有 .txt 文件，按聊天窗口分别解析。

    每个 .txt 文件被视为一个独立聊天窗口，返回值按文件名升序排列；
    单个文件内部仍按时间戳升序排列。

    Returns:
        {
            "liwenhao.txt": [
                {
                    "timestamp": datetime,
                    "sender": str,
                    "content": str,
                    "source_file": str,
                    "conversation_id": str,
                }
            ]
        }
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    txt_files = sorted(
        data_dir.rglob("*.txt"),
        key=lambda p: p.relative_to(data_dir).as_posix(),
    )
    if not txt_files:
        logger.warning("数据目录 %s 下未发现任何 .txt 文件", data_dir)
        return {}

    logger.info("在 %s 中发现 %d 个 .txt 文件，按聊天窗口解析...", data_dir, len(txt_files))

    conversations: Dict[str, List[Dict]] = {}
    for txt_file in txt_files:
        relative_name = txt_file.relative_to(data_dir).as_posix()
        chat_type = _infer_chat_type_from_relative_path(txt_file.relative_to(data_dir))
        try:
            file_messages = _parse_single_file(txt_file)
        except Exception as exc:
            logger.error("解析文件 %s 失败: %s，已跳过", relative_name, exc)
            continue

        for msg in file_messages:
            msg["source_file"] = relative_name
            msg["conversation_id"] = relative_name
            msg["chat_type"] = chat_type
        file_messages.sort(key=lambda m: m["timestamp"])
        conversations[relative_name] = file_messages

    total = sum(len(messages) for messages in conversations.values())
    logger.info("全部聊天窗口解析完成，共 %d 个窗口、%d 条消息", len(conversations), total)
    return conversations


def _infer_chat_type_from_relative_path(relative_path: Path) -> str:
    """
    根据数据目录下的一级文件夹推断窗口类型。

    推荐测试数据结构：
        data/test/单聊/liwenhao.txt
        data/test/群聊/project_group.txt

    为兼容旧数据，直接放在 data/test/ 下的 .txt 默认按单聊处理。
    """
    parts = relative_path.parts
    if len(parts) > 1:
        first_dir = parts[0].lower()
        if first_dir in {"群聊", "group", "groups"}:
            return "群聊"
        if first_dir in {"单聊", "private", "single", "dm", "direct"}:
            return "单聊"
    return "单聊"
