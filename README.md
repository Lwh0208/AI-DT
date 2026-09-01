# smart_reply_assistant

高仿真数字孪生回复助手系统 — 为固定使用者"zzx"构建的生产级本地智能回复预测引擎。

## 系统概述

本系统通过分析历史聊天数据，自动沉淀多角色的原生画像与主题记忆；在运行阶段结合多方画像、当前聊天窗口最近 20 条上下文消息及历史经验，高精度预测zzx的回复；并在模拟一天结束时进入 Dream 模式完成状态自我反思与数据的增量演进，使助手的回复行为无限接近本人。

## 后续计划
扩展至整个团队的回复助手，代替团队个人进行知识性、领域性回复而不只是闲聊与个性化问答。
Multi-agent拥有团队所有人的画像与领域知识，建立团队知识图谱并进行画像隔离，实现点到点精准预测。

## 两阶段数据分离设计

系统严格区分两种数据源：

| 数据类型 | 目录 | 用途 | 何时使用 |
|---------|------|------|---------|
| 历史数据 | `data/history/` | 初始化画像与记忆 | 首次运行或重建画像 |
| 测试数据 | `data/test/` | 模拟运行与反思 | 每次模拟测试 |

- **历史数据**只在画像初始化阶段读取，用于让 LLM 从充足的历史行为中抽取角色特征；多个文件会按聊天窗口分块注入，避免不同私聊被混成同一条会话
- **测试数据**在模拟运行阶段按聊天窗口逐文件处理，触发回复预测、即时反思和 Dream 模式；当前窗口的最近上下文会独立记录
- 两种数据互不干扰，可以独立更新

## 项目结构

```
smart_reply_assistant/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── history/              # 历史聊天数据（画像初始化用）
│   │   └── history_chat.txt
│   └── test/                 # 测试聊天数据（模拟运行用）
│       └── test_chat.txt
├── profiles/                 # 动态生成的角色画像
│   └── 张照西/
│       ├── profile.md        # 用户画像
│       └── memory.md         # 个人记忆库
├── runtime/                  # 运行时数据
│   ├── dialogue_log.jsonl    # 会话流水日志
│   ├── 经验.md               # 预测反思经验库
│   ├── report_*.md           # Markdown 运行报告
│   └── transcript_*.txt      # 测试消息/预测回复/真实回复对照文本
├── src/                      # 核心源码
│   ├── __init__.py
│   ├── app.py                # 流水线核心入口
│   ├── config.py             # 全局配置管理
│   ├── llm_client.py         # LLM 客户端
│   ├── prompts.py            # Prompt 定义库
│   ├── data_loader.py        # 聊天记录解析器
│   ├── profile_manager.py    # 画像与记忆管理
│   ├── session_store.py      # 滑动窗口上下文
│   └── feedback.py           # 回复预测与反思
└── tests/                    # 自动化测试
    ├── test_memory_parser.py
    └── test_session_store.py
```

## 快速开始

### 1. 安装依赖

```bash
cd smart_reply_assistant
pip install -r requirements.txt
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑 .env 填入实际 LLM API 地址和 Token
```

### 3. 准备数据

**历史数据**（`data/history/`）：放入较多量的历史聊天记录，用于初始化画像。格式示例：

```
zzx 2025/9/1 12:01:56
赵博，啥情况

zy 2025/9/1 14:10:05
毕然他们好像要投资源，本地部署GLM5
```

**测试数据**（`data/test/`）：放入需要模拟运行的对话记录。推荐按窗口类型放到 `单聊/` 和 `群聊/` 子目录；缺失年份会自动补 2026：

```
data/test/
├── 单聊/
│   ├── lwh.txt
│   └── zy.txt
└── 群聊/
    └── project_group.txt
```

直接放在 `data/test/` 根目录下的 `.txt` 会按单聊兼容处理。单聊中所有非zzx消息都会触发模拟回复；群聊中只有提及zzx或其别名的消息才触发模拟回复。

```
lwh 4/10 09:15:00
@zzx 昨天的测试环境问题解决了吗？
```

支持的完整格式列表：
- `发送者 2025/9/1 12:01:23` → 保留原始年份 2025
- `发送者 4/22 18:12:10` → 自动补全为 2026 年
- `[2026-01-15 10:30:22] 发送者: 消息内容` → 同行格式
- `[图片]`、`[文件]` 等特殊内容正常识别

### 4. 运行系统

```bash
# 完整运行（初始化 + 模拟）
PYTHONPATH=. python -m src.app --verbose

# 仅初始化画像（不运行模拟）
PYTHONPATH=. python -m src.app --init-only

# 指定自定义数据目录
PYTHONPATH=. python -m src.app --history-dir /path/to/history --test-dir /path/to/test
```

### 5. 运行测试

```bash
PYTHONPATH=. pytest tests/ -v
```

## 关键设计约束

- 所有结构化 Markdown 严禁使用表格语法，统一使用多级标题和无序列表
- 任一角色画像初始化失败时，程序会停止，并清理本次运行新增的 profiles/runtime 残留
- 单聊中所有非张照西消息都会触发回复预测
- 群聊中只有提及zzx/zx/西哥的消息才触发回复预测
- zzx主动发起的消息和真实回复会进入当前窗口上下文，但不会触发预测
- 完整运行结束后会生成 `runtime/transcript_*.txt`，用于查看测试消息、预测回复、真实回复对照
- 日期变更时强制触发 Dream 模式
- 更新时间为对话时间而非真实时间

