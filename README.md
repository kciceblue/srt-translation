# SRT Translator

[中文文档](README_CN.md)

Translate subtitle files via backend API while preserving timestamps and line structure. Fixes ASR/Whisper transcription errors using context. Auto-groups files by series for consistent character naming.

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate
# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py <inputs> --endpoint <api_url> [options]
```

Inputs can be files, directories, or glob patterns — any mix:

```bash
# Single file
python main.py movie.srt --endpoint http://127.0.0.1:5000/v1/chat/completions

# Directory (recursively finds all .srt files)
python main.py subs/ --endpoint http://127.0.0.1:5000/v1/chat/completions --out-dir out

# Mix of files and directories
python main.py subs/ extras/bonus.srt --endpoint http://127.0.0.1:5000/v1/chat/completions

# Glob pattern
python main.py "subs/*.srt" --endpoint http://127.0.0.1:5000/v1/chat/completions

# Different language pair
python main.py subs/ --endpoint http://... --source-lang Korean --target-lang English --suffix .en.srt
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | http://127.0.0.1:5000 | Backend API URL |
| `--out-dir` | translated_out | Output directory |
| `--suffix` | .zh.srt | Output filename suffix |
| `--source-lang` | Japanese | Source language name |
| `--target-lang` | Simplified Chinese | Target language name |
| `--timeout` | 300 | HTTP timeout (seconds) |
| `--retry` | 2 | Retries on failure |
| `--retry-sleep` | 1.0 | Sleep between retries (seconds) |
| `--system-prompt` | (built-in) | System prompt (`{source_lang}` / `{target_lang}` placeholders) |
| `--user-prefix` | (built-in) | User message prefix (supports placeholders) |
| `--extra-payload` | "" | Extra JSON fields for API body |
| `--no-group` | false | Disable series grouping |
| `--no-stream` | false | Disable streaming (no loop detection) |

## How It Works

### Translation
1. **Number lines**: Formats all subtitle text as `[1] text`, `[2] text`, ...
2. **Translate in one shot**: Sends the entire file to the backend in a single request
3. **Fix ASR errors**: Model corrects Whisper transcription errors using full context
4. **Parse numbered output**: Extracts translations by `[N] translated_text` pattern
5. **Auto-repair gaps**: Sends targeted repair requests for missing lines (up to 3 rounds)

### Series Grouping
When given multiple files, the tool asks the LLM to group them by series (based on filenames). Files in the same series are translated in episode order with a shared **glossary** of character names and key terms, ensuring consistent naming across episodes. The glossary resets between different series. Use `--no-group` to disable.

### Loop Detection
Uses SSE streaming to detect when the model gets stuck in an infinite loop (repeating the same pattern). On detection, the connection is closed immediately and the request is retried. Use `--no-stream` to disable.

## Troubleshooting

**Missing lines in output**: Auto-repaired in up to 3 rounds

**Model loops / hangs**: Detected automatically via streaming. If `--no-stream`, increase `--retry`

**Backend errors**: Ensure endpoint URL is correct and backend is running

**Timeout errors**: Increase `--timeout` value

## Dependencies

- requests

## License

[MIT](LICENSE)
