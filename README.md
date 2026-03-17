# SRT Translator

Translate subtitle files via backend API while preserving timestamps and line structure. Fixes ASR/Whisper transcription errors using context.

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py <input_files> --endpoint <api_url> [options]
```

### Examples

Translate single file:
```bash
python main.py subs/movie.srt --endpoint http://127.0.0.1:5000/v1/chat/completions --out-dir out
```

Translate all SRTs with custom parameters:
```bash
python main.py "subs/*.srt" \
  --endpoint http://127.0.0.1:5000/v1/chat/completions \
  --extra-payload '{"model":"gpt-4","temperature":0}' \
  --out-dir out
```

Translate Korean to English:
```bash
python main.py subs/drama.srt \
  --endpoint http://127.0.0.1:5000/v1/chat/completions \
  --source-lang Korean --target-lang English --suffix .en.srt
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
| `--retry` | 1 | Retries on failure |
| `--retry-sleep` | 1.0 | Sleep between retries (seconds) |
| `--system-prompt` | (built-in) | System prompt (supports `{source_lang}` / `{target_lang}` placeholders) |
| `--user-prefix` | (built-in) | User message prefix (supports placeholders) |
| `--extra-payload` | "" | Extra JSON fields for API body |

## How It Works

1. **Number lines**: Formats all subtitle text as `[1] text`, `[2] text`, ... for precise line tracking
2. **Translate in one shot**: Sends the entire file to the backend in a single request
3. **Fix ASR errors**: Model translates and corrects Whisper transcription errors using full context
4. **Parse numbered output**: Extracts translations by matching `[N] translated_text` pattern
5. **Auto-repair gaps**: If some lines are missing, sends targeted repair requests (up to 3 rounds)

## Troubleshooting

**Missing lines in output**: The tool auto-repairs gaps in up to 3 rounds

**Backend errors**: Ensure endpoint URL is correct and backend is running

**Timeout errors**: Increase `--timeout` value

## Dependencies

- requests
