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
| `--chunk-size` | 10 | Lines per translation chunk (smaller = more stable) |
| `--repetition-penalty` | 1.3 | Repetition penalty for all LLM calls (1.0 to disable) |
| `--no-group` | false | Disable series grouping |
| `--no-stream` | false | Disable streaming |
| `--verbose` / `-v` | false | Show detailed progress and stream LLM responses to stdout |
| `--proofread` | false | Proofread mode: skip translation, proofread existing translated files using vocab |
| `--vocab` | vocab.txt | Vocabulary file — loaded at startup, updated with learnt terms after each run. Set to `''` to disable. |

## How It Works

### Chunked Translation
1. **Split into chunks**: Divides subtitle lines into small chunks (`--chunk-size`, default 10). Small chunks prevent the model from looping.
2. **Literal translation**: Each chunk is translated independently with thinking disabled. Uncertain words are flagged with `??` markers.
3. **Numbered I/O**: Uses `[N] text` format for precise line tracking. Missing lines get one repair attempt per chunk.

### Series Grouping
When given multiple files, the tool asks the LLM to group them by series (based on filenames). Files in the same series are translated in episode order with a shared **glossary** of character names and key terms, ensuring consistent naming across episodes. The glossary resets between different series. Use `--no-group` to disable.

### Persistent Vocabulary
A vocabulary file (`vocab.txt` by default) is loaded at startup and updated with learnt terms after each translation run. Entries support two formats:
- `SourceTerm → TranslatedTerm` — source language to target language mapping
- `DraftTranslation → Corrected` — fix a specific draft translation

Change the path with `--vocab`, or set `--vocab ''` to disable. Vocabulary is fed into the translation glossary and proofread prompt for consistent naming.

### Two-Step Workflow (Translate → Review → Proofread)

Proofread runs as a **separate invocation** so you can review and curate the auto-generated vocabulary between steps:

```bash
# Step 1: Translate (saves learnt vocab to vocab.txt)
python main.py subs/ --endpoint http://... --out-dir out

# Step 2: Review/edit vocab.txt — remove bad entries, fix translations, add custom terms

# Step 3: Proofread (reads existing translations from out-dir, thinking enabled)
python main.py subs/ --proofread --endpoint http://... --out-dir out
```

When `--proofread` is specified, the tool **does not translate**. It finds each input file's corresponding translated file in `--out-dir` (using `stem + suffix`), reads both source and translated SRT, proofreads with vocabulary context and thinking enabled, then overwrites the translated file. If a translated file is not found, it warns and skips. If proofread fails for a file, the original translation is kept.

### Runaway Detection
Uses SSE streaming to monitor output length. If content output exceeds 3x the expected length, the stream is closed and the collected output is truncated at the first repeating pattern (keeping 2 occurrences). The partial result is used and translation continues to the next chunk. Use `--no-stream` to disable streaming.

## Recommended Settings

- **Temperature 0.2–0.3**: Low enough to stay stable, high enough to not translate garbled words literally. Set via `--extra-payload '{"temperature":0.3}'`.
- **Repetition penalty 1.3+**: Prevents the model from looping. Increase if loops persist. Set via `--repetition-penalty`.
- **Chunk size 10**: Default. Reduce to 5 if the model still loops on complex content.

## Troubleshooting

**Missing lines in output**: Each chunk gets one auto-repair attempt

**Model loops / hangs**: Increase `--repetition-penalty`, decrease `--chunk-size`, or lower temperature

**Backend errors**: Ensure endpoint URL is correct and backend is running

**Timeout errors**: Increase `--timeout` value

## Dependencies

- requests

## License

[MIT](LICENSE)
