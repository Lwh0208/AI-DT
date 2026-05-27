"""
画像与记忆的初始化及 Dream 增量写入器。

负责：
- 解析 LLM 输出中的硬性隔离符（---PROFILE_START--- / ---PROFILE_END--- 等）
- 角色画像和记忆的初始化写入
- Dream 模式输出的安全解析与增量写入
- 更新审计追踪标记自动注入
- 损坏输出的安全兜底（dump 到 failed_update_*.txt）
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

from src.config import settings
from src.llm_client import LlmClient

logger = logging.getLogger(__name__)


class CorruptedLlmOutputError(Exception):
    """LLM 输出损坏（缺失必要隔离符）时抛出的异常。"""

    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


class ParsedInitialOutput(NamedTuple):
    """初始化构建输出的解析结果。"""
    profile: str
    memory: str


class ParsedDreamOutput(NamedTuple):
    """Dream 模式输出的解析结果。"""
    profile: str
    memory: str


# ---------------------------------------------------------------------------
# 隔离符正则
# ---------------------------------------------------------------------------
_RE_PROFILE_START = re.compile(r"---PROFILE_START---")
_RE_PROFILE_END = re.compile(r"---PROFILE_END---")
_RE_MEMORY_START = re.compile(r"---MEMORY_START---")
_RE_MEMORY_END = re.compile(r"---MEMORY_END---")

_RE_PROFILE_UPDATED_START = re.compile(r"---PROFILE_UPDATED_START---")
_RE_PROFILE_UPDATED_END = re.compile(r"---PROFILE_UPDATED_END---")
_RE_MEMORY_UPDATED_START = re.compile(r"---MEMORY_UPDATED_START---")
_RE_MEMORY_UPDATED_END = re.compile(r"---MEMORY_UPDATED_END---")


def parse_initial_build_output(raw_output: str) -> ParsedInitialOutput:
    """
    解析初始化构建 LLM 输出，提取 Profile 和 Memory 文本。

    Args:
        raw_output: LLM 原始输出字符串

    Returns:
        ParsedInitialOutput(profile=..., memory=...)

    Raises:
        CorruptedLlmOutputError: 隔离符缺失或损坏
    """
    profile = _extract_between(raw_output, _RE_PROFILE_START, _RE_PROFILE_END, "PROFILE")
    memory = _extract_between(raw_output, _RE_MEMORY_START, _RE_MEMORY_END, "MEMORY")
    return ParsedInitialOutput(profile=profile, memory=memory)


def parse_dream_output(raw_output: str) -> ParsedDreamOutput:
    """
    解析 Dream 模式 LLM 输出，提取更新后的 Profile 和 Memory 文本。

    Args:
        raw_output: LLM 原始输出字符串

    Returns:
        ParsedDreamOutput(profile=..., memory=...)

    Raises:
        CorruptedLlmOutputError: 隔离符缺失或损坏
    """
    profile = _extract_between(
        raw_output, _RE_PROFILE_UPDATED_START, _RE_PROFILE_UPDATED_END, "PROFILE_UPDATED"
    )
    memory = _extract_between(
        raw_output, _RE_MEMORY_UPDATED_START, _RE_MEMORY_UPDATED_END, "MEMORY_UPDATED"
    )
    return ParsedDreamOutput(profile=profile, memory=memory)


def _extract_between(
    text: str,
    start_pattern: re.Pattern,
    end_pattern: re.Pattern,
    section_name: str,
) -> str:
    """
    从文本中提取两个隔离符之间的内容。

    Args:
        text: 完整文本
        start_pattern: 起始隔离符正则
        end_pattern: 结束隔离符正则
        section_name: 段落名称（用于错误信息）

    Returns:
        隔离符之间的内容（已去除首尾空白）

    Raises:
        CorruptedLlmOutputError: 缺失起始或结束隔离符
    """
    start_match = start_pattern.search(text)
    if not start_match:
        raise CorruptedLlmOutputError(
            f"LLM 输出缺少 {section_name} 起始隔离符", raw_output=text
        )

    end_match = end_pattern.search(text, start_match.end())
    if not end_match:
        raise CorruptedLlmOutputError(
            f"LLM 输出缺少 {section_name} 结束隔离符", raw_output=text
        )

    content = text[start_match.end():end_match.start()].strip()
    return content


def _dump_failed_update(raw_output: str, timestamp: datetime) -> Path:
    """
    将损坏的 LLM 输出 dump 到 runtime/failed_update_[时间戳].txt。

    Args:
        raw_output: 损坏的原始输出
        timestamp: 时间戳

    Returns:
        dump 文件路径
    """
    runtime_dir = settings.RUNTIME_DIR
    runtime_dir.mkdir(parents=True, exist_ok=True)
    dump_path = runtime_dir / f"failed_update_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(dump_path, "w", encoding="utf-8") as fh:
            fh.write(raw_output)
        logger.warning("损坏的 LLM 输出已 dump 至: %s", dump_path)
    except OSError as exc:
        logger.error("dump 损坏输出失败: %s", exc)
    return dump_path


class ProfileManager:
    """
    角色画像与记忆管理器。

    负责：
    - 初始化角色画像与记忆（从历史聊天记录中抽取）
    - 首次运行时的自动构建
    - Dream 模式下的安全增量写入
    """

    def __init__(self, llm_client: Optional[LlmClient] = None) -> None:
        self.llm = llm_client or LlmClient()
        self.profiles_dir = settings.PROFILES_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 读取操作
    # ------------------------------------------------------------------

    def get_profile(self, character_name: str) -> str:
        """读取角色画像全文。若文件不存在，返回空字符串。"""
        path = self._latest_version_file(character_name, "profile.md")
        if path is None:
            return ""
        if not path.is_file():
            return ""
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    def get_profile_as_of(self, character_name: str, as_of: datetime) -> str:
        """读取指定时间点可用的最近角色画像。若文件不存在，返回空字符串。"""
        path = self._version_file_as_of(character_name, "profile.md", as_of)
        if not path.is_file():
            return ""
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    def get_memory(self, character_name: str) -> str:
        """读取角色记忆全文。若文件不存在，返回空字符串。"""
        path = self._latest_version_file(character_name, "memory.md")
        if path is None:
            return ""
        if not path.is_file():
            return ""
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    def get_memory_as_of(self, character_name: str, as_of: datetime) -> str:
        """读取指定时间点可用的最近角色记忆。若文件不存在，返回空字符串。"""
        path = self._version_file_as_of(character_name, "memory.md", as_of)
        if not path.is_file():
            return ""
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    def profile_exists(self, character_name: str) -> bool:
        """判断角色的画像文件是否存在。"""
        path = self._latest_version_file(character_name, "profile.md")
        return path is not None and path.is_file()

    # ------------------------------------------------------------------
    # 初始化构建
    # ------------------------------------------------------------------

    def build_initial_profiles(
        self,
        target_character: str,
        raw_chat_text: str,
    ) -> ParsedInitialOutput:
        """
        从历史聊天记录中为指定角色构建初始化画像与记忆。

        Args:
            target_character: 目标角色名
            raw_chat_text: 原始聊天记录文本

        Returns:
            解析后的画像和记忆文本
        """
        from src.prompts import SYSTEM_PROMPT_INITIAL_BUILD, USER_PROMPT_INITIAL_BUILD

        system_msg = SYSTEM_PROMPT_INITIAL_BUILD.format(target_character=target_character)
        user_msg = USER_PROMPT_INITIAL_BUILD.format(
            target_character=target_character, raw_chat_text=raw_chat_text
        )

        logger.info("开始为角色 [%s] 构建初始化画像与记忆...", target_character)
        raw_output = self.llm.chat(messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ], max_tokens=settings.PROFILE_INIT_MAX_TOKENS)

        parsed = parse_initial_build_output(raw_output)
        self._write_profile_and_memory(
            target_character,
            parsed.profile,
            parsed.memory,
            version_time=settings.PROFILE_INIT_CUTOFF,
        )
        logger.info("角色 [%s] 画像与记忆初始化完成", target_character)
        return parsed

    # ------------------------------------------------------------------
    # Dream 模式增量更新
    # ------------------------------------------------------------------

    def dream_update(
        self,
        character_name: str,
        raw_output: str,
        update_time: datetime,
    ) -> bool:
        """
        安全地执行 Dream 模式增量更新。

        解析 LLM 输出，校验隔离符完整性，成功则写入；失败则 dump 损坏输出，
        保持原有文件不被覆盖。

        Args:
            character_name: 角色名
            raw_output: Dream 模式 LLM 原始输出
            update_time: 更新时间戳（使用对话时间而非真实时间）

        Returns:
            True 表示更新成功，False 表示解析失败（已兜底 dump）
        """
        try:
            parsed = parse_dream_output(raw_output)
        except CorruptedLlmOutputError as exc:
            logger.error(
                "Dream 输出解析失败: %s，将执行安全兜底", exc,
            )
            _dump_failed_update(exc.raw_output, update_time)
            return False

        # 强制审计追踪：在二级或三级标题变更处注入 HTML 注释
        profile_with_audit = self._inject_audit_trail(
            parsed.profile, self.get_profile_as_of(character_name, update_time), update_time
        )
        memory_with_audit = self._inject_audit_trail(
            parsed.memory, self.get_memory_as_of(character_name, update_time), update_time
        )

        self._write_profile_and_memory(
            character_name,
            profile_with_audit,
            memory_with_audit,
            version_time=update_time,
        )
        logger.info("角色 [%s] Dream 增量更新成功 (时间: %s)", character_name, update_time)
        return True

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _write_profile_and_memory(
        self,
        character_name: str,
        profile_text: str,
        memory_text: str,
        version_time: Optional[datetime] = None,
    ) -> None:
        """将画像和记忆文本写入对应日期版本目录（目录并发创建安全）。"""
        version_dir = self._version_dir(character_name, version_time or datetime.now())
        char_dir = self.profiles_dir / character_name
        char_dir.mkdir(parents=True, exist_ok=True)
        version_dir.mkdir(parents=True, exist_ok=True)

        profile_path = version_dir / "profile.md"
        memory_path = version_dir / "memory.md"

        with open(profile_path, "w", encoding="utf-8") as fh:
            fh.write(profile_text)
        with open(memory_path, "w", encoding="utf-8") as fh:
            fh.write(memory_text)

        logger.debug("写入角色 [%s] 画像(%d字符) 和记忆(%d字符)", character_name, len(profile_text), len(memory_text))

    def _version_dir(self, character_name: str, version_time: datetime) -> Path:
        """角色在指定日期的画像版本目录。"""
        return self.profiles_dir / character_name / version_time.strftime("%Y-%m-%d")

    def _latest_version_file(self, character_name: str, file_name: str) -> Optional[Path]:
        """读取角色最新日期版本文件。"""
        char_dir = self.profiles_dir / character_name
        if not char_dir.is_dir():
            return None

        candidates: list[Tuple[str, Path]] = []
        legacy_path = char_dir / file_name
        if legacy_path.is_file():
            candidates.append(("0000-00-00", legacy_path))
        for child in char_dir.iterdir():
            if child.is_dir() and self._parse_version_date(child.name) is not None:
                version_file = child / file_name
                if version_file.is_file():
                    candidates.append((child.name, version_file))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0])[-1][1]

    def _version_file_as_of(
        self,
        character_name: str,
        file_name: str,
        as_of: datetime,
    ) -> Path:
        """读取角色在指定时间可用的最近日期版本文件。"""
        char_dir = self.profiles_dir / character_name
        if not char_dir.is_dir():
            return char_dir / file_name

        as_of_date = as_of.date()
        candidates: list[Tuple[datetime, Path]] = []
        legacy_path = char_dir / file_name
        if legacy_path.is_file():
            candidates.append((datetime.min, legacy_path))
        for child in char_dir.iterdir():
            version_date = self._parse_version_date(child.name)
            if version_date is None or version_date.date() > as_of_date:
                continue
            version_file = child / file_name
            if version_file.is_file():
                candidates.append((version_date, version_file))
        if not candidates:
            return char_dir / file_name
        return sorted(candidates, key=lambda item: item[0])[-1][1]

    @staticmethod
    def _parse_version_date(name: str) -> Optional[datetime]:
        """解析 YYYY-MM-DD 版本目录名。"""
        try:
            return datetime.strptime(name, "%Y-%m-%d")
        except ValueError:
            return None

    def _inject_audit_trail(
        self,
        new_content: str,
        old_content: str,
        update_time: datetime,
    ) -> str:
        """
        对比新旧内容，在有变更的二级/三级标题下方自动注入审计追踪 HTML 注释。

        审计标记格式: <!-- 最后更新时间：YYYY-MM-DD HH:mm:ss；更改内容：具体变化说明 -->
        """
        time_str = update_time.strftime("%Y-%m-%d %H:%M:%S")
        new_lines = new_content.split("\n")
        old_lines = old_content.split("\n") if old_content else []

        # 构建旧内容中各标题下的内容映射
        old_sections: Dict[str, str] = self._extract_sections(old_lines)
        new_sections: Dict[str, str] = self._extract_sections(new_lines)

        result_lines: list[str] = []
        i = 0
        while i < len(new_lines):
            line = new_lines[i]
            result_lines.append(line)

            # 检测是否是二级或三级标题
            if line.startswith("## ") or line.startswith("### "):
                heading = line.strip()
                heading_key = heading.lstrip("#").strip()

                old_body = old_sections.get(heading_key, "")
                # 收集该标题下的新内容（直到下一个同级或更高级标题）
                new_body_parts: list[str] = []
                j = i + 1
                while j < len(new_lines):
                    next_line = new_lines[j]
                    if next_line.startswith("## ") or next_line.startswith("### "):
                        break
                    new_body_parts.append(next_line)
                    j += 1
                new_body = "\n".join(new_body_parts)

                # 比较新旧内容
                if old_body.strip() != new_body.strip() and new_body.strip():
                    change_desc = self._summarize_change(old_body, new_body)
                    audit_comment = f"<!-- 最后更新时间：{time_str}；更改内容：{change_desc} -->"
                    # 如果下一行不是空行，先确保标题与注释之间有一个空行
                    if i + 1 < len(new_lines) and new_lines[i + 1].strip():
                        result_lines.append("")
                    result_lines.append(audit_comment)

            i += 1

        return "\n".join(result_lines)

    @staticmethod
    def _extract_sections(lines: list[str]) -> Dict[str, str]:
        """从 Markdown 文本行中提取各标题下的内容。"""
        sections: Dict[str, str] = {}
        current_heading: Optional[str] = None
        current_body: list[str] = []

        for line in lines:
            if line.startswith("## ") or line.startswith("### "):
                if current_heading is not None:
                    sections[current_heading] = "\n".join(current_body)
                current_heading = line.lstrip("#").strip()
                current_body = []
            else:
                current_body.append(line)
        if current_heading is not None:
            sections[current_heading] = "\n".join(current_body)

        return sections

    @staticmethod
    def _summarize_change(old_body: str, new_body: str) -> str:
        """生成变更摘要（简化版）。"""
        old_stripped = old_body.strip()
        new_stripped = new_body.strip()

        if not old_stripped:
            return "新增内容"
        if not new_stripped:
            return "删除内容"

        # 简易差异估算
        old_set = set(old_stripped.split("\n"))
        new_set = set(new_stripped.split("\n"))
        added = new_set - old_set
        removed = old_set - new_set

        parts: list[str] = []
        if added:
            parts.append(f"新增{len(added)}行")
        if removed:
            parts.append(f"移除{len(removed)}行")
        if not parts:
            parts.append("内容微调")

        return "；".join(parts)
