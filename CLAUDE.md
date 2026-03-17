# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SRT subtitle translator — a single-file Python CLI tool (`main.py`) that translates `.srt` subtitle files via an OpenAI-compatible backend API. Translates in small chunks to avoid model loops, auto-groups files by series for consistent terminology, and monitors streaming output for runaway detection.

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

Everything lives in `main.py` with seven logical sections:

1. **SRT parsing/writing** — `parse_srt()` / `write_srt()` / `SrtBlock` dataclass. Handles encoding detection (UTF-8 BOM, UTF-16, CP1252 fallback) and tolerant parsing of malformed SRT files.

2. **Backend interaction** — `post_messages()` with dual-mode: non-streaming (blocking) and SSE streaming. `_stream_with_loop_detection()` monitors content length and truncates at repetition on runaway. `_truncate_at_repetition()` cleans up looped output keeping first 2 occurrences. Thinking is disabled globally (`enable_thinking: false`). Auto-falls back to non-streaming if backend doesn't support it.

3. **Numbered I/O helpers** — `format_numbered_input()` / `parse_numbered_output()`. Explicit `[N] text` anchors for robust line mapping.

4. **Translation workflow** — `translate_lines_via_backend()` translates in small chunks (`--chunk-size`, default 10). Each chunk is translated independently with a literal translation prompt (no context reasoning). Uncertain words flagged with `??` markers. One repair attempt per chunk for missing lines. Repetition penalty applied globally.

5. **Series grouping** — `group_files_by_series()` sends filenames to the LLM to group by series and sort by episode order. `extract_glossary()` builds a rolling glossary of character names/terms from each translated episode. Glossary resets between series.

6. **Proofread pass** — `proofread_file()` reviews source+translation pairs with vocabulary context. Sends side-by-side `[N] source → translation` pairs to the LLM. Fixes `??` markers, inconsistent names, and mistranslations. Falls back to original translation on failure. Opt-in via `--proofread`. User vocabulary loaded from `--vocab` file, merged with series glossary.

7. **CLI** — `main()` with argparse. Accepts files, directories, and globs. Key flags: `--endpoint`, `--source-lang`/`--target-lang`, `--chunk-size`, `--repetition-penalty`, `--no-group`, `--no-stream`, `--verbose`, `--extra-payload`, `--suffix`, `--out-dir`, `--retry`, `--proofread`, `--vocab`.

## Key Design Decisions

- **Chunked translation**: Small chunks (default 10 lines) prevent the model from looping. Each chunk is translated independently with thinking disabled and high repetition penalty.
- **Literal first-pass prompt**: Pass 1 translates mechanically without reasoning about context. Uncertain words are flagged with `??` for manual review.
- **Numbered line protocol**: `[N] text` format with regex parsing. Missing lines detected by number, not position.
- **Runaway handling**: If content output exceeds 3x expected, stream is closed and output is truncated at the first repeating pattern (keeping 2 occurrences). No retries — partial result is used and next chunk proceeds.
- **Series context sharing**: LLM groups files by series from filenames. Glossary accumulated per episode, appended to system prompt for next episode, reset between series.
- **Proofread pass**: Opt-in second pass (`--proofread`) runs after all files in a series are translated. Reviews full source+translation pairs with merged vocabulary (user `--vocab` + learnt glossary). Sends entire file in one request. Falls back to original translation on failure.
- **`subs/`** is the conventional input directory; **`out/`** is the conventional output directory.
