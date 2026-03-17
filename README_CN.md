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
| `--no-group` | false | 禁用系列自动分组 |
| `--no-stream` | false | 禁用流式传输（关闭循环检测） |

## 工作原理

### 翻译流程
1. **行编号**：将所有字幕文本格式化为 `[1] 文本`、`[2] 文本`...
2. **一次性翻译**：将整个文件通过单次请求发送给后端
3. **修正语音识别错误**：模型利用完整上下文修正 Whisper 的转录错误
4. **解析编号输出**：通过匹配 `[N] 翻译文本` 模式提取翻译结果
5. **自动修补缺失**：对遗漏行发送针对性修补请求（最多 3 轮）

### 系列分组
输入多个文件时，工具会让 LLM 根据文件名自动识别并分组（如按动漫/剧集系列）。同一系列的文件按集数顺序翻译，并共享**术语表**（角色名、地名等关键词），确保跨集命名一致。切换到不同系列时术语表自动重置。使用 `--no-group` 禁用。

### 循环检测
使用 SSE 流式传输实时监测模型输出。当检测到模型陷入无限循环（重复相同模式）时，立即断开连接并重试请求。使用 `--no-stream` 禁用。

## 推荐温度设置

建议使用 **temperature 0.3**，通过 `--extra-payload '{"temperature":0.3}'` 设置。

- **不建议 0**：模型需要一定灵活度来推断和修正 Whisper 转录错误。完全确定性输出容易原样翻译错误的转录内容。
- **不建议 0.6+**：温度过高会增加幻觉风险、角色名不一致、格式偏差（触发更多修补轮次），也更容易触发无限循环。
- 如果模型仍有循环问题，可降至 **0.2**；如果 ASR 纠错效果过于保守，可升至 **0.4**。

## 常见问题

**输出中有缺失行**：自动修补，最多 3 轮

**模型循环/卡死**：通过流式传输自动检测。如使用 `--no-stream`，可增大 `--retry`

**后端错误**：确认 endpoint 地址正确且后端服务已启动

**超时错误**：增大 `--timeout` 值

## 依赖

- requests

## 许可证

[MIT](LICENSE)
