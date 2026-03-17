# SRT 字幕翻译器

通过后端 API 翻译字幕文件，保留时间轴和行结构。利用上下文修正 ASR/Whisper 语音识别错误。自动按系列分组文件，确保角色名称一致。

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

```bash
python main.py <输入> --endpoint <API地址> [选项]
```

输入可以是文件、目录或通配符模式，支持混合使用：

```bash
# 单个文件
python main.py movie.srt --endpoint http://127.0.0.1:5000/v1/chat/completions

# 目录（递归查找所有 .srt 文件）
python main.py subs/ --endpoint http://127.0.0.1:5000/v1/chat/completions --out-dir out

# 混合文件和目录
python main.py subs/ extras/bonus.srt --endpoint http://127.0.0.1:5000/v1/chat/completions

# 通配符模式
python main.py "subs/*.srt" --endpoint http://127.0.0.1:5000/v1/chat/completions

# 其他语言对
python main.py subs/ --endpoint http://... --source-lang Korean --target-lang English --suffix .en.srt
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--endpoint` | http://127.0.0.1:5000 | 后端 API 地址 |
| `--out-dir` | translated_out | 输出目录 |
| `--suffix` | .zh.srt | 输出文件名后缀 |
| `--source-lang` | Japanese | 源语言名称 |
| `--target-lang` | Simplified Chinese | 目标语言名称 |
| `--timeout` | 300 | HTTP 超时时间（秒） |
| `--retry` | 2 | 失败重试次数 |
| `--retry-sleep` | 1.0 | 重试间隔（秒） |
| `--system-prompt` | （内置） | 系统提示词（支持 `{source_lang}` / `{target_lang}` 占位符） |
| `--user-prefix` | （内置） | 用户消息前缀（支持占位符） |
| `--extra-payload` | "" | API 请求体的额外 JSON 字段 |
| `--chunk-size` | 10 | 每块翻译行数（越小越稳定） |
| `--repetition-penalty` | 1.3 | 所有 LLM 调用的重复惩罚（1.0 禁用） |
| `--no-group` | false | 禁用系列自动分组 |
| `--no-stream` | false | 禁用流式传输 |
| `--verbose` / `-v` | false | 显示详细进度并将 LLM 流式回复输出到 stdout |

## 工作原理

### 分块翻译
1. **分块**：将字幕文本按小块（`--chunk-size`，默认 10 行）逐块翻译，小块避免模型陷入循环。
2. **直译模式**：每块独立翻译，关闭模型思考功能。不确定的词汇用 `??` 标记。
3. **编号 I/O**：使用 `[N] 文本` 格式精确追踪每一行。缺失行每块有一次修补机会。

### 系列分组
输入多个文件时，工具会让 LLM 根据文件名自动识别并分组（如按动漫/剧集系列）。同一系列的文件按集数顺序翻译，并共享**术语表**（角色名、地名等关键词），确保跨集命名一致。切换到不同系列时术语表自动重置。使用 `--no-group` 禁用。

### 溢出检测
使用 SSE 流式传输监控输出长度。当内容输出超过预期长度 3 倍时，关闭连接并在重复模式处截断已收集的输出（保留前 2 次重复）。使用部分结果继续翻译下一块。使用 `--no-stream` 禁用流式传输。

## 推荐设置

- **温度 0.2–0.3**：足够低以保持稳定，足够高以避免直译错误内容。通过 `--extra-payload '{"temperature":0.3}'` 设置。
- **重复惩罚 1.3+**：防止模型循环。如仍有循环可增大。通过 `--repetition-penalty` 设置。
- **块大小 10**：默认值。如模型在复杂内容上仍循环，可降至 5。

## 常见问题

**输出中有缺失行**：每块有一次自动修补机会

**模型循环/卡死**：增大 `--repetition-penalty`，减小 `--chunk-size`，或降低温度

**后端错误**：确认 endpoint 地址正确且后端服务已启动

**超时错误**：增大 `--timeout` 值

## 依赖

- requests

## 许可证

[MIT](LICENSE)
