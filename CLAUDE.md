# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SRT subtitle translator — a single-file Python CLI tool (`main.py`) that translates `.srt` subtitle files via a backend API (OpenAI-compatible format), preserving timestamps and line structure. Also corrects ASR/Whisper transcription errors using surrounding context.

Default configuration translates Japanese to Simplified Chinese. Language pair is configurable via `--source-lang` / `--target-lang`.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Run
python main.py <input_files> --endpoint <api_url> [options]
python main.py subs/*.srt --endpoint http://127.0.0.1:5000/v1/chat/completions --out-dir out

# Build standalone executable (PyInstaller)
pyinstaller srt-translate.spec
```

There are no tests or linting configured.

## Architecture

Everything lives in `main.py` with four logical sections:

1. **SRT parsing/writing** — `parse_srt()` / `write_srt()` / `SrtBlock` dataclass. Handles encoding detection (UTF-8 BOM, UTF-16, CP1252 fallback) and tolerant parsing of malformed SRT files.

2. **Backend interaction** — `post_messages()` sends OpenAI-format `{"messages": [...]}` POSTs. `extract_text_from_response()` handles multiple response formats (OpenAI chat completions, simple JSON, plain text).

3. **Numbered I/O helpers** — `format_numbered_input()` formats lines as `[N] text`, `parse_numbered_output()` extracts translations by matching `[N] translated_text` patterns. This replaces brute-force line-count checking with explicit line anchors.

4. **Translation workflow** — `translate_lines_via_backend()` sends the entire file in one request (backend has 262K context). Uses a 2-tier strategy:
   - **Tier 1 (numbered)**: Send all lines as numbered input, parse numbered output — the normal happy path.
   - **Tier 2 (repair)**: If some lines are missing, send targeted repair requests for just those lines (up to 3 rounds).

5. **CLI** — `main()` with argparse. Key flags: `--endpoint`, `--source-lang`/`--target-lang`, `--extra-payload` (JSON), `--suffix` (default `.zh.srt`), `--out-dir`, `--retry`. System prompt and user prefix support `{source_lang}`/`{target_lang}` template placeholders.

## Key Design Decisions

- **Single-request translation**: The entire SRT file is sent in one request. The backend (Qwen3.5-35B, 262K context) handles any realistic subtitle file in one shot. No chunking needed.
- **Numbered line protocol**: Input sent as `[1] text\n[2] text\n...`, output parsed by regex `^\[(\d+)\]`. Explicit anchors make missing/extra lines detectable by number rather than position.
- **ASR error correction**: The system prompt instructs the model to fix Whisper transcription errors (misspelled words, homophones, broken phrases) using full file context.
- **`subs/`** is the conventional input directory; **`out/`** is the conventional output directory.
