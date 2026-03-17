# SRT 字幕翻译器

通过后端 API 翻译字幕文件，保留时间轴和行结构。同时利用上下文修正 ASR/Whisper 语音识别错误。

## 安装

```bash
# 创建虚拟环境
python -m venv .venv

# 激活（Windows）
.\.venv\Scripts\Activate.ps1

# 激活（Linux/Mac）
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py <输入文件> --endpoint <API地址> [选项]
```

### 示例

翻译单个文件：
```bash
python main.py subs/movie.srt --endpoint http://127.0.0.1:5000/v1/chat/completions --out-dir out
```

批量翻译并指定模型参数：
```bash
python main.py "subs/*.srt" \
  --endpoint http://127.0.0.1:5000/v1/chat/completions \
  --extra-payload '{"model":"gpt-4","temperature":0}' \
  --out-dir out
```

翻译韩语到英语：
```bash
python main.py subs/drama.srt \
  --endpoint http://127.0.0.1:5000/v1/chat/completions \
  --source-lang Korean --target-lang English --suffix .en.srt
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
| `--retry` | 1 | 失败重试次数 |
| `--retry-sleep` | 1.0 | 重试间隔（秒） |
| `--system-prompt` | （内置） | 系统提示词（支持 `{source_lang}` / `{target_lang}` 占位符） |
| `--user-prefix` | （内置） | 用户消息前缀（支持占位符） |
| `--extra-payload` | "" | API 请求体的额外 JSON 字段 |

## 工作原理

1. **行编号**：将所有字幕文本格式化为 `[1] 文本`、`[2] 文本`... 以精确追踪每一行
2. **一次性翻译**：将整个文件通过单次请求发送给后端
3. **修正语音识别错误**：模型在翻译的同时，利用完整上下文修正 Whisper 的转录错误（错字、同音词、断句等）
4. **解析编号输出**：通过匹配 `[N] 翻译文本` 模式提取翻译结果
5. **自动修补缺失**：如有遗漏行，自动发送修补请求（最多 3 轮）

## 常见问题

**输出中有缺失行**：工具会自动修补，最多尝试 3 轮

**后端错误**：确认 endpoint 地址正确且后端服务已启动

**超时错误**：增大 `--timeout` 值

## 依赖

- requests

## 许可证

[MIT](LICENSE)
