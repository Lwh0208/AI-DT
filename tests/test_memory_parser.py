"""
边界隔离符解析单元测试。

测试用例1: 完美隔离符字符串 → 成功提取 Profile 与 Memory 文本
测试用例2: 损坏字符串（缺失结束标签）→ 抛出 CorruptedLlmOutputError，
           同时验证 dump 到 failed_update_*.txt 且原角色文件未被损毁
"""

from __future__ import annotations

import pytest
from datetime import datetime
from pathlib import Path

from src.profile_manager import (
    parse_dream_output,
    parse_initial_build_output,
    CorruptedLlmOutputError,
    ParsedDreamOutput,
    ParsedInitialOutput,
)


# ====================================================================
# 测试用例 1: 完美隔离符 → 成功提取
# ====================================================================

class TestParseDreamOutputPerfect:
    """构造完美的 Dream 输出，验证能够成功提取完整的 Profile 与 Memory。"""

    PERFECT_DREAM_OUTPUT = """---PROFILE_UPDATED_START---
# 张照西 用户画像

## 基本身份与基本信息
- 姓名：张照西
- 性别：男
- 年龄：约30岁

## 工作角色与组织网络
- 职位：云计算工程师
- 部门：IT基础设施部

## 性格特征与沟通风格
- 务实直接，不绕弯子
- 技术讨论时偏好简洁表达

## 常用语与特征句式
- "我待会儿看下"
- "先让他把材料发过来"

## 核心回复策略
- 优先确认事实再回复
- 对不确定的事保守表达

## 规避的回复方式
- 不做不可逆承诺
- 不透露他人隐私

## 证据链与不确定项
- 年龄为推断值，证据不足，暂不确定
---PROFILE_UPDATED_END---
---MEMORY_UPDATED_START---
# 张照西 个人记忆库

### 云桌面项目
- 总体摘要：负责公司云桌面基础设施的部署与维护
- 首次涉及时间：2026-01-10
- 关联核心角色：李文浩、王磊
- 当前已知进度：测试环境已部署完成
- 关键上下文信息：使用Citrix方案
- 遗留待办/下一步动作：等待生产环境审批
- 审计日志：初始化建立。2026-01-15 新增生产环境审批进度。
---MEMORY_UPDATED_END---"""

    def test_parse_dream_output_extracts_profile(self):
        """验证能正确提取 Profile 文本。"""
        result = parse_dream_output(self.PERFECT_DREAM_OUTPUT)
        assert isinstance(result, ParsedDreamOutput)
        assert "# 张照西 用户画像" in result.profile
        assert "## 基本身份与基本信息" in result.profile
        assert "云计算工程师" in result.profile

    def test_parse_dream_output_extracts_memory(self):
        """验证能正确提取 Memory 文本。"""
        result = parse_dream_output(self.PERFECT_DREAM_OUTPUT)
        assert "# 张照西 个人记忆库" in result.memory
        assert "### 云桌面项目" in result.memory
        assert "Citrix" in result.memory

    def test_parse_dream_output_no_extra_content(self):
        """验证提取结果不包含隔离符本身。"""
        result = parse_dream_output(self.PERFECT_DREAM_OUTPUT)
        assert "---PROFILE_UPDATED_START---" not in result.profile
        assert "---PROFILE_UPDATED_END---" not in result.profile
        assert "---MEMORY_UPDATED_START---" not in result.memory
        assert "---MEMORY_UPDATED_END---" not in result.memory


# ====================================================================
# 测试用例 2: 损坏输出 → 抛出异常 + 安全兜底
# ====================================================================

class TestParseDreamOutputCorrupted:
    """构造缺失结束标签的损坏字符串，验证异常与安全兜底行为。"""

    CORRUPTED_DREAM_OUTPUT = """---PROFILE_UPDATED_START---
# 张照西 用户画像

## 基本身份与基本信息
- 姓名：张照西
- 这段内容故意缺少结束标签

---MEMORY_UPDATED_START---
# 张照西 个人记忆库

### 云桌面项目
- 总体摘要：测试内容
---MEMORY_UPDATED_END---"""

    def test_missing_end_tag_raises_corrupted_error(self):
        """缺失 PROFILE_UPDATED_END 标签时必须抛出 CorruptedLlmOutputError。"""
        with pytest.raises(CorruptedLlmOutputError) as exc_info:
            parse_dream_output(self.CORRUPTED_DREAM_OUTPUT)
        assert "PROFILE_UPDATED" in str(exc_info.value)
        assert "结束隔离符" in str(exc_info.value)

    def test_corrupted_output_has_raw_output_attribute(self):
        """异常对象必须携带 raw_output 属性。"""
        with pytest.raises(CorruptedLlmOutputError) as exc_info:
            parse_dream_output(self.CORRUPTED_DREAM_OUTPUT)
        assert exc_info.value.raw_output == self.CORRUPTED_DREAM_OUTPUT

    def test_dump_file_created_on_corruption(self, tmp_path):
        """损坏输出被 dump 到 runtime/failed_update_[时间戳].txt。"""
        from src.profile_manager import _dump_failed_update

        # 使用 tmp_path 模拟 runtime 目录
        # 通过 monkey-patch 临时改变 settings.RUNTIME_DIR
        import src.config as cfg
        original_runtime_dir = cfg.settings.RUNTIME_DIR
        try:
            cfg.settings.RUNTIME_DIR = tmp_path
            timestamp = datetime(2026, 1, 15, 23, 59, 59)
            dump_path = _dump_failed_update(self.CORRUPTED_DREAM_OUTPUT, timestamp)

            assert dump_path.exists()
            assert "failed_update_" in dump_path.name
            with open(dump_path, "r", encoding="utf-8") as fh:
                assert fh.read() == self.CORRUPTED_DREAM_OUTPUT
        finally:
            cfg.settings.RUNTIME_DIR = original_runtime_dir

    def test_original_files_not_corrupted_on_failure(self, tmp_path):
        """解析失败时，原有角色的 Markdown 文件必须保持原样未被覆盖损毁。"""
        import src.config as cfg
        from src.profile_manager import ProfileManager

        original_runtime_dir = cfg.settings.RUNTIME_DIR
        original_profiles_dir = cfg.settings.PROFILES_DIR
        try:
            cfg.settings.PROFILES_DIR = tmp_path / "profiles"
            cfg.settings.RUNTIME_DIR = tmp_path / "runtime"

            pm = ProfileManager()
            # 先写入一份原始画像
            char_dir = tmp_path / "profiles" / "张照西"
            char_dir.mkdir(parents=True, exist_ok=True)
            original_profile = "# 原始画像\n\n这是原始内容，不应被覆盖。"
            original_memory = "# 原始记忆\n\n这是原始记忆。"
            with open(char_dir / "profile.md", "w", encoding="utf-8") as fh:
                fh.write(original_profile)
            with open(char_dir / "memory.md", "w", encoding="utf-8") as fh:
                fh.write(original_memory)

            # 尝试用损坏输出执行 dream_update
            update_time = datetime(2026, 1, 15, 23, 59, 59)
            success = pm.dream_update("张照西", self.CORRUPTED_DREAM_OUTPUT, update_time)

            # 断言更新失败
            assert success is False

            # 断言原始文件未被修改
            with open(char_dir / "profile.md", "r", encoding="utf-8") as fh:
                assert fh.read() == original_profile
            with open(char_dir / "memory.md", "r", encoding="utf-8") as fh:
                assert fh.read() == original_memory

            # 断言 dump 文件已生成
            runtime_dir = tmp_path / "runtime"
            dump_files = list(runtime_dir.glob("failed_update_*.txt"))
            assert len(dump_files) >= 1

        finally:
            cfg.settings.PROFILES_DIR = original_profiles_dir
            cfg.settings.RUNTIME_DIR = original_runtime_dir


# ====================================================================
# 额外测试: 初始构建隔离符解析
# ====================================================================

class TestParseInitialBuildOutput:
    """验证初始化构建输出的隔离符解析。"""

    PERFECT_INITIAL_OUTPUT = """---PROFILE_START---
# 张照西 用户画像
## 基本身份与基本信息
- 姓名：张照西
---PROFILE_END---
---MEMORY_START---
# 张照西 个人记忆库
### 项目A
- 总体摘要：测试项目
---MEMORY_END---"""

    def test_parse_initial_build_extracts_both(self):
        """验证初始化构建输出能完整提取 Profile 和 Memory。"""
        result = parse_initial_build_output(self.PERFECT_INITIAL_OUTPUT)
        assert isinstance(result, ParsedInitialOutput)
        assert "张照西 用户画像" in result.profile
        assert "张照西 个人记忆库" in result.memory

    def test_parse_initial_build_missing_profile_end(self):
        """缺失 PROFILE_END 标签时抛出异常。"""
        corrupted = "---PROFILE_START---\n内容缺失结束标签\n---MEMORY_START---\n记忆内容\n---MEMORY_END---"
        with pytest.raises(CorruptedLlmOutputError):
            parse_initial_build_output(corrupted)

    def test_parse_initial_build_missing_memory_end(self):
        """缺失 MEMORY_END 标签时抛出异常。"""
        corrupted = "---PROFILE_START---\n画像内容\n---PROFILE_END---\n---MEMORY_START---\n记忆缺失结束标签"
        with pytest.raises(CorruptedLlmOutputError):
            parse_initial_build_output(corrupted)


def test_profile_manager_reads_latest_version_as_of(tmp_path):
    """画像按日期版本读取：预测 3.2 时应读取 3.1 更新后的完整版本。"""
    import src.config as cfg
    from src.profile_manager import ProfileManager

    original_profiles_dir = cfg.settings.PROFILES_DIR
    try:
        cfg.settings.PROFILES_DIR = tmp_path / "profiles"
        pm = ProfileManager()
        char_dir = cfg.settings.PROFILES_DIR / "张照西"
        (char_dir / "2026-03-01").mkdir(parents=True)
        (char_dir / "2026-03-02").mkdir(parents=True)
        (char_dir / "2026-03-01" / "profile.md").write_text("profile 0301", encoding="utf-8")
        (char_dir / "2026-03-01" / "memory.md").write_text("memory 0301", encoding="utf-8")
        (char_dir / "2026-03-02" / "profile.md").write_text("profile 0302", encoding="utf-8")
        (char_dir / "2026-03-02" / "memory.md").write_text("memory 0302", encoding="utf-8")

        assert pm.get_profile_as_of("张照西", datetime(2026, 3, 1, 12, 0, 0)) == "profile 0301"
        assert pm.get_memory_as_of("张照西", datetime(2026, 3, 2, 12, 0, 0)) == "memory 0302"
        assert pm.get_profile("张照西") == "profile 0302"
    finally:
        cfg.settings.PROFILES_DIR = original_profiles_dir
