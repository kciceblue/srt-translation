# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SRT subtitle translator — two Python CLI tools that translate and proofread `.srt` subtitle files via an OpenAI-compatible backend API. `main.py` handles translation (chunked, with series grouping and runaway detection). `proofread.py` is a standalone multi-step agent that reviews translations without thinking mode.

Default configuration translates Japanese to Simplified Chinese. Language pair configurable via `--source-lang` / `--target-lang`.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Translate (files, directories, or globs)
python main.py subs/ --endpoint http://127.0.0.1:5000/v1/chat/completions --out-dir out
python main.py movie.srt subs/ --endpoint http://... --out-dir out

# Proofread (separate step, after translation)
python proofread.py subs/ --endpoint http://127.0.0.1:5000/v1/chat/completions --out-dir out

# Quick run with local config
./run.sh
```

There are no tests or linting configured.

## Architecture

### main.py — Translation

Shared infrastructure and translation logic in six sections:

1. **SRT parsing/writing** — `parse_srt()` / `write_srt()` / `SrtBlock` dataclass. Handles encoding detection (UTF-8 BOM, UTF-16, CP1252 fallback) and tolerant parsing of malformed SRT files.

2. **Backend interaction** — `post_messages()` with dual-mode: non-streaming (blocking) and SSE streaming. `_stream_with_loop_detection()` monitors content length and truncates at repetition on runaway. `_truncate_at_repetition()` cleans up looped output keeping first 2 occurrences. Auto-falls back to non-streaming if backend doesn't support it.

3. **Numbered I/O helpers** — `format_numbered_input()` / `parse_numbered_output()`. Explicit `[N] text` anchors for robust line mapping.

4. **Translation workflow** — `translate_lines_via_backend()` translates in small chunks (`--chunk-size`, default 10). Each chunk is translated independently with a literal translation prompt (no context reasoning). Uncertain words flagged with `??` markers. One repair attempt per chunk for missing lines. Repetition penalty applied globally.

5. **Series grouping** — `group_files_by_series()` sends filenames to the LLM to group by series and sort by episode order. `extract_glossary()` builds a rolling glossary of character names/terms from each translated episode. Glossary resets between series.

6. **Persistent vocabulary** — `parse_vocab()` / `format_vocab()` / `save_vocab()`. Public functions also imported by `proofread.py`.

7. **CLI** — `main()` with argparse. Translation only. Key flags: `--endpoint`, `--source-lang`/`--target-lang`, `--chunk-size`, `--repetition-penalty`, `--no-group`, `--no-stream`, `--verbose`, `--extra-payload`, `--suffix`, `--out-dir`, `--retry`, `--vocab`.

### proofread.py — Multi-Step Agent Proofreader

Standalone CLI that reviews translated SRT files using a multi-step pipeline (all with thinking disabled):

- **Step 0** (per-file): Context Building — summarises scene, characters, tone from a sample of the file.
- **Step 1** (per-chunk): Confidence Scoring — rates each line HIGH/MEDIUM/LOW with adaptive 2x window. All-HIGH chunks skip correction.
- **Step 2** (per-chunk): Issue Analysis — entire chunk shown with ⚠/❌ flags on MEDIUM/LOW lines. Free-form reasoning output.
- **Step 3** (per-chunk): Apply Corrections — fed step 2's analysis, outputs clean `[N] corrected_text`. Repair attempt for missing lines.
- **Step 4** (per-chunk): Vocab Extraction — learns new character names/terms, merges into running vocab dict.

Imports shared infrastructure from `main.py` (post_messages, SRT parsing, numbered I/O, vocab functions). Key flags: `--endpoint`, `--out-dir`, `--chunk-size` (default 20), `--context-window`, `--vocab`, `--no-vocab-update`.

## Key Design Decisions

- **Chunked translation**: Small chunks (default 10 lines) prevent the model from looping. Each chunk is translated independently with thinking disabled and high repetition penalty.
- **Literal first-pass prompt**: Pass 1 translates mechanically without reasoning about context. Uncertain words are flagged with `??` for manual review.
- **Numbered line protocol**: `[N] text` format with regex parsing. Missing lines detected by number, not position.
- **Runaway handling**: If content output exceeds 3x expected, stream is closed and output is truncated at the first repeating pattern (keeping 2 occurrences). No retries — partial result is used and next chunk proceeds.
- **Series context sharing**: LLM groups files by series from filenames. Glossary accumulated per episode, appended to system prompt for next episode, reset between series.
- **Agent-based proofread**: `proofread.py` is a separate CLI tool (not part of `main.py`). User translates first, optionally edits `vocab.txt`, then runs proofread. Proofread uses a multi-step agent pipeline (5 steps) to externalise reasoning without thinking mode. Vocab is actively updated during proofread unless `--no-vocab-update` is set.
- **`subs/`** is the conventional input directory; **`out/`** is the conventional output directory.
