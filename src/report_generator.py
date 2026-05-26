"""
运行报告生成器。

在流水线结束后，从 runtime/ 中的 dialogue_log.jsonl
提取数据，生成一份按时间线排列的人类可读汇总报告。

报告内容：
1. 运行概览：数据量、角色统计、时间跨度
2. 逐条消息处理明细：触发消息、模拟回复、真实回复
3. Dream 模式更新摘要
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.config import settings
from src.session_store import SessionStore
from src.profile_manager import ProfileManager

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    汇总报告生成器。
    
    从运行时数据中提取信息，生成 Markdown 格式的可读报告。
    """

    def __init__(
        self,
        session_store: Optional[SessionStore] = None,
        profile_manager: Optional[ProfileManager] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.ss = session_store or SessionStore()
        self.pm = profile_manager or ProfileManager()
        self.output_dir = output_dir or settings.RUNTIME_DIR

    def generate(self, run_id: Optional[str] = None) -> Path:
        """
        生成完整的运行汇总报告。

        Args:
            run_id: 运行标识（默认使用当前时间戳）

        Returns:
            报告文件路径
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_path = self.output_dir / f"report_{run_id}.md"

        sections: List[str] = []
        sections.append(self._build_header(run_id))
        sections.append(self._build_overview())
        sections.append(self._build_message_timeline())
        sections.append(self._build_dream_summary())

        report_content = "\n\n".join(sections)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report_content)

        logger.info("运行报告已生成: %s", report_path)
        return report_path

    def generate_transcript_txt(self, run_id: Optional[str] = None) -> Path:
        """
        生成测试消息、预测回复、真实回复的纯文本对照记录。

        Args:
            run_id: 运行标识（默认使用当前时间戳）

        Returns:
            文本文件路径
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        transcript_path = self.output_dir / f"transcript_{run_id}.txt"
        all_records = self.ss.get_all_records()

        grouped: Dict[str, List[dict]] = {}
        for rec in all_records:
            conversation_id = rec.get("conversation_id") or rec.get("source_file") or "unknown"
            grouped.setdefault(str(conversation_id), []).append(rec)

        lines: List[str] = [
            "智能回复助手测试对照记录",
            f"运行标识: {run_id}",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        if not grouped:
            lines.append("本次运行无测试消息记录。")
        else:
            for conversation_id in sorted(grouped):
                lines.append(f"===== 聊天窗口: {conversation_id} =====")
                lines.append("")

                records = sorted(
                    grouped[conversation_id],
                    key=lambda r: r.get("timestamp", datetime.min),
                )
                for rec in records:
                    lines.append(self._format_transcript_record(rec))
                    trigger = rec.get("trigger_message", "")
                    if rec.get("role") == "simulated" and trigger:
                        lines.append(f"    触发消息: {trigger}")
                    lines.append("")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines).rstrip() + "\n")

        logger.info("测试对照文本已生成: %s", transcript_path)
        return transcript_path

    # ==================================================================
    # 报告各部分构建
    # ==================================================================

    def _build_header(self, run_id: str) -> str:
        """报告头部。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"# 智能回复助手运行报告\n"
            f"\n"
            f"- **运行标识**: {run_id}\n"
            f"- **生成时间**: {now}\n"
            f"- **目标角色**: 张照西"
        )

    @staticmethod
    def _format_transcript_record(rec: dict) -> str:
        """格式化单条测试对照记录。"""
        ts = rec.get("timestamp", "")
        if isinstance(ts, datetime):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts)

        role_label = {
            "context": "测试消息",
            "simulated": "预测回复",
            "real": "真实回复",
        }.get(rec.get("role", "unknown"), rec.get("role", "unknown"))

        sender = rec.get("sender", "未知")
        content = rec.get("content", "")
        return f"[{ts_str}] [{role_label}] {sender}: {content}"

    def _build_overview(self) -> str:
        """运行概览统计。"""
        all_records = self.ss.get_all_records()

        if not all_records:
            return (
                "## 运行概览\n"
                "\n"
                "*本次运行无对话记录。*"
            )

        # 统计
        total = len(all_records)
        roles: Dict[str, int] = {}
        senders: Dict[str, int] = {}
        date_range: List[str] = []

        for rec in all_records:
            role = rec.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
            sender = rec.get("sender", "未知")
            senders[sender] = senders.get(sender, 0) + 1

            ts = rec.get("timestamp")
            if isinstance(ts, datetime):
                date_str = ts.strftime("%Y-%m-%d")
                if date_str not in date_range:
                    date_range.append(date_str)

        # 画像状态
        profile_status: List[str] = []
        for sender_name in sorted(senders.keys()):
            exists = self.pm.profile_exists(sender_name)
            status = "已初始化" if exists else "未初始化"
            profile_status.append(f"  - {sender_name}: {status}")

        lines = [
            "## 运行概览",
            "",
            f"- **消息总数**: {total}",
            f"- **时间跨度**: {date_range[0] if date_range else 'N/A'} ~ {date_range[-1] if date_range else 'N/A'}",
            "",
            "### 消息角色分布",
            "",
        ]
        for role_name, count in sorted(roles.items()):
            role_label = {
                "real": "真实回复",
                "simulated": "模拟回复",
                "context": "上下文记录",
            }.get(role_name, role_name)
            lines.append(f"- {role_label}: {count} 条")

        lines.extend([
            "",
            "### 参与角色与画像状态",
            "",
        ])
        lines.extend(profile_status)

        return "\n".join(lines)

    def _build_message_timeline(self) -> str:
        """逐条消息处理明细。"""
        all_records = self.ss.get_all_records()

        if not all_records:
            return "## 消息处理时间线\n\n*无记录。*"

        lines: List[str] = ["## 消息处理时间线", ""]

        for idx, rec in enumerate(all_records, 1):
            ts = rec.get("timestamp", "")
            if isinstance(ts, datetime):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = str(ts)

            sender = rec.get("sender", "未知")
            content = rec.get("content", "")
            role = rec.get("role", "unknown")
            trigger = rec.get("trigger_message", "")

            role_label = {
                "real": "真实回复",
                "simulated": "模拟回复",
                "context": "上下文",
            }.get(role, role)

            lines.append(f"### [{idx}] {ts_str} — {sender}（{role_label}）")
            lines.append("")

            # 消息内容（多行缩进引用）
            content_display = content[:300] if content else "（空）"
            for cline in content_display.split("\n"):
                lines.append(f"> {cline}")
            lines.append("")

            # 如果是模拟回复，标注触发消息
            if role == "simulated" and trigger:
                lines.append(f"- **触发消息**: {trigger[:100]}")

        return "\n".join(lines)

    def _build_dream_summary(self) -> str:
        """Dream 模式更新摘要。"""
        # 检查 profiles 目录下的画像文件中是否包含审计追踪标记
        profiles_dir = settings.PROFILES_DIR
        if not profiles_dir.is_dir():
            return "## Dream 模式更新摘要\n\n*尚无 Dream 更新记录。*"

        audit_entries: List[str] = []
        for char_dir in sorted(profiles_dir.iterdir()):
            if not char_dir.is_dir():
                continue
            char_name = char_dir.name
            for md_file in ("profile.md", "memory.md"):
                md_path = char_dir / md_file
                if not md_path.is_file():
                    continue
                with open(md_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip().startswith("<!-- 最后更新时间："):
                            # 提取审计信息
                            audit_line = line.strip()
                            # 去掉 <!--  --> 包裹
                            inner = audit_line.lstrip("<!-- ").rstrip(" -->")
                            audit_entries.append(f"- **{char_name}** ({md_file}): {inner}")

        lines = [
            "## Dream 模式更新摘要",
            "",
        ]

        if audit_entries:
            lines.append(f"共 {len(audit_entries)} 处画像/记忆变更：")
            lines.append("")
            lines.extend(audit_entries)
        else:
            lines.append("*尚无 Dream 更新记录。*")

        return "\n".join(lines)
