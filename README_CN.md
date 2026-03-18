# SRT 字幕翻译器

通过后端 API 翻译字幕文件，保留时间轴和行结构。5 步流水线：ASR 错误修正 → 翻译 → 质量标记 → 校对。自动按系列分组文件，确保角色名称一致。

## 安装

```bash
# 创建虚拟环境
python -m venv .venv

# 激活（Linux/Mac）
source .venv/bin/activate
# 激活（Windows）
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 完整流水线

```bash
python cli.py run <输入> --endpoint <API地址> [选项]
```

输入可以是文件、目录或通配符模式，支持混合使用：

```bash
# 单个文件
python cli.py run movie.srt --endpoint http://127.0.0.1:5000/v1/chat/completions

# 目录（递归查找所有 .srt 文件）
python cli.py run subs/ --endpoint http://127.0.0.1:5000/v1/chat/completions --out-dir out

# 混合文件和目录
python cli.py run subs/ extras/bonus.srt --endpoint http://127.0.0.1:5000/v1/chat/completions

# 其他语言对
python cli.py run subs/ --endpoint http://... --source-lang Korean --target-lang English --suffix .en.srt
```

### 逐步运行

使用 `--debug` 保留 tmp 文件夹，可逐步运行流水线以便调试或在步骤之间手动干预。

```bash
# 步骤 1：输入 — 展开文件、按系列分组、创建 tmp/
python cli.py input subs/ --endpoint http://... --debug

# 步骤 2：预处理 — ASR 错误修正、上下文摘要、术语提取
python cli.py preprocess --endpoint http://... --debug

# 步骤 3：翻译 — 分块翻译，使用词汇表+上下文
python cli.py translate --endpoint http://... --debug

# 步骤 4：后处理 — 标记不合格翻译
python cli.py postprocess --endpoint http://... --debug

# 步骤 5：校对 — 修正标记行、最终审查、复制到 out/
python cli.py proofread --endpoint http://... --out-dir out --debug
```

## 参数说明

### 通用参数（所有子命令）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--endpoint` | http://127.0.0.1:5000/v1/chat/completions | 后端 API 地址 |
| `--source-lang` | Japanese | 源语言名称 |
| `--target-lang` | Simplified Chinese | 目标语言名称 |
| `--timeout` | 300 | HTTP 超时时间（秒） |
| `--retry` | 2 | 失败重试次数 |
| `--retry-sleep` | 1.0 | 重试间隔（秒） |
| `--extra-payload` | "" | API 请求体的额外 JSON 字段 |
| `--no-stream` | false | 禁用流式传输 |
| `--debug` | false | 详细输出 + 保留 tmp 文件夹 |
| `--tmp-dir` | ./tmp | 临时文件夹路径 |

### `run` 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--out-dir` | out | 输出目录 |
| `--suffix` | .zh.srt | 输出文件名后缀 |
| `--no-group` | false | 禁用系列自动分组 |
| `--chunk-size` | 10 | 每块翻译行数（越小越稳定） |
| `--repetition-penalty` | 1.3 | 重复惩罚（1.0 禁用） |
| `--vocab` | vocab.txt | 外部词汇表文件 |

### `translate` 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--chunk-size` | 10 | 每块翻译行数 |
| `--repetition-penalty` | 1.3 | 重复惩罚 |
| `--vocab` | vocab.txt | 外部词汇表文件 |

### `proofread` 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--out-dir` | out | 输出目录 |
| `--suffix` | .zh.srt | 输出文件名后缀 |
| `--context-radius` | 50 | 标记行周围的上下文行数 |

## 工作原理

### 流水线总览

```
输入 → 预处理 → 翻译 → 后处理 → 校对 → 输出
      (ASR修正)  (分块)  (标记问题)  (修正+审查)
```

### 步骤 1：输入
展开文件/目录/通配符输入，过滤非 SRT 文件，使用 LLM 根据文件名按系列分组，创建 tmp 文件夹结构，写入 manifest.json。

### 步骤 2：预处理（新功能）
按文件依次执行 5 个步骤：(1) 生成上下文摘要（语气、说话人、情节），先理解场景；(2) 头脑风暴该场景的常见词汇（专业术语、角色名、常用短语）；(3) 结合上下文和预期词汇标记疑似 ASR 错误；(4) 修正标记的行；(5) 仅提取高置信度的专有名词。按系列：逐条对照上下文审核累积的词汇表，积极删除不确定的条目——误导性的词条比缺失更糟。写入 `context.md` 和 `vocab.md`。

### 步骤 3：翻译
将字幕按小块（`--chunk-size`，默认 10 行）逐块翻译。每块独立翻译，使用直译提示（关闭思考模式）。使用 `[N] 文本` 编号格式精确追踪每行。缺失行有一次修补机会。合并系列 vocab.md 与外部 `--vocab` 文件。在系列内跨集累积滚动术语表。

### 步骤 4：后处理（新功能）
比较源文本+翻译对，结合上下文和词汇表标记问题行（词汇不匹配、意思反转、信息缺失、`??` 标记）。写入 `flags.json`。

### 步骤 5：校对
第一轮：逐行修正每个标记行（每行一次 LLM 调用，±50 行上下文窗口）。第二轮：完整文件最终审查。修正失败和遗留问题写入 `confused.md` 供人工审阅。最终输出复制到 `out/`。

### 系列分组
输入多个文件时，LLM 根据文件名自动分组。同一系列的文件共享 context.md 和 vocab.md，确保跨集命名一致。使用 `--no-group` 禁用。

### 溢出检测
SSE 流式传输中有三层保护：
1. **精确重复** — 如果某个模式重复 10 次以上，立即截断，仅保留 1 次
2. **推理废话** — 如果模型开始"自言自语"（密集的犹豫词如"Wait,"、"Actually,"、"however"），自动剥离废话部分
3. **长度倍数** — 如果输出超过预期长度的 N 倍，关闭连接并截断

每个流水线步骤有各自的重试/降级策略。使用 `--no-stream` 禁用。

## 推荐设置

- **温度 0.2–0.3**：足够低以保持稳定，足够高以避免直译错误内容。通过 `--extra-payload '{"temperature":0.3}'` 设置。
- **重复惩罚 1.3+**：防止模型循环。如仍有循环可增大。
- **块大小 10**：默认值。如模型在复杂内容上仍循环，可降至 5。

## 常见问题

**输出中有缺失行**：每块有一次自动修补机会

**模型循环/卡死**：增大 `--repetition-penalty`，减小 `--chunk-size`，或降低温度

**后端错误**：确认 endpoint 地址正确且后端服务已启动

**超时错误**：增大 `--timeout` 值

**流水线中途失败**：使用 `--debug` 保留 tmp/，然后重新运行单个步骤

## 依赖

- requests

## 许可证

[MIT](LICENSE)
