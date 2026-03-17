# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SRT subtitle translator — a single-file Python CLI tool (`main.py`) that translates `.srt` subtitle files via an OpenAI-compatible backend API. Corrects ASR/Whisper transcription errors, auto-groups files by series for consistent terminology, and detects model infinite loops via streaming.

Default configuration translates Japanese to Simplified Chinese. Language pair configurable via `--source-lang` / `--target-lang`.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run (files, directories, or globs)
python main.py subs/ --endpoint http://127.0.0.1:5000/v1/chat/completions --out-dir out
python main.py movie.srt subs/ --endpoint http://... --out-dir out

# Quick run with local config
./run.sh
```

There are no tests or linting configured.

## Architecture

Everything lives in `main.py` with six logical sections:

1. **SRT parsing/writing** — `parse_srt()` / `write_srt()` / `SrtBlock` dataclass. Handles encoding detection (UTF-8 BOM, UTF-16, CP1252 fallback) and tolerant parsing of malformed SRT files.

2. **Backend interaction** — `post_messages()` with dual-mode: non-streaming (blocking) and SSE streaming with loop detection. `extract_text_from_response()` handles multiple response formats. `_stream_with_loop_detection()` reads SSE chunks and uses `_detect_repetition()` to catch infinite loops early. Auto-falls back to non-streaming if the backend doesn't support it.

3. **Numbered I/O helpers** — `format_numbered_input()` / `parse_numbered_output()`. Explicit `[N] text` anchors for robust line mapping.

4. **Translation workflow** — `translate_lines_via_backend()` sends the entire file in one request. Tier 1: parse numbered output. Tier 2: repair missing lines (up to 3 rounds). Streaming is used by default to detect model loops.

5. **Series grouping** — `group_files_by_series()` sends filenames to the LLM to group by series and sort by episode order. `extract_glossary()` builds a rolling glossary of character names/terms from each translated episode. Glossary resets between series.

6. **CLI** — `main()` with argparse. Accepts files, directories, and globs. Key flags: `--endpoint`, `--source-lang`/`--target-lang`, `--no-group`, `--no-stream`, `--extra-payload`, `--suffix`, `--out-dir`, `--retry`. System prompt and user prefix support `{source_lang}`/`{target_lang}` template placeholders.

## Key Design Decisions

- **Single-request translation**: Entire SRT file sent in one request. Backend (Qwen3.5-35B, 262K context) handles any realistic subtitle file.
- **Numbered line protocol**: `[N] text` format with regex parsing. Missing lines detected by number, not position.
- **Series context sharing**: LLM groups files by series from filenames. Glossary of names/terms accumulated per episode, appended to system prompt for next episode, reset between series.
- **Streaming loop detection**: SSE streaming + sliding-window repetition check. Catches model loops in seconds rather than waiting for 300s timeout.
- **`subs/`** is the conventional input directory; **`out/`** is the conventional output directory.
