# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SRT subtitle translator — two Python CLI tools that translate and proofread `.srt` subtitle files via an OpenAI-compatible backend API. `main.py` handles translation (chunked, with series grouping and runaway detection). `proofread.py` is a standalone 2-pass proofreader that reviews translations without thinking mode.

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

Shared infrastructure and translation logic:

1. **SRT parsing/writing** — `parse_srt()` / `write_srt()` / `SrtBlock` dataclass. Handles encoding detection (UTF-8 BOM, UTF-16, CP1252 fallback) and tolerant parsing of malformed SRT files.

2. **Backend interaction** — `post_messages()` with dual-mode: non-streaming (blocking) and SSE streaming. `_stream_with_loop_detection()` monitors content length and truncates at repetition on runaway (`runaway_multiplier` param, default 3x). `raise_on_runaway=True` makes it raise instead of returning truncated output (used by proofread steps with retry logic). Auto-falls back to non-streaming if backend doesn't support it.

3. **Numbered I/O helpers** — `format_numbered_input()` / `parse_numbered_output()`. Explicit `[N] text` anchors for robust line mapping.

4. **Translation workflow** — `translate_lines_via_backend()` translates in small chunks (`--chunk-size`, default 10). Each chunk is translated independently with a literal translation prompt (no context reasoning). Uncertain words flagged with `??` markers. One repair attempt per chunk for missing lines. Repetition penalty applied globally.

5. **Series grouping** — `group_files_by_series()` sends filenames to the LLM to group by series and sort by episode order. `extract_glossary()` builds a rolling glossary of character names/terms from each translated episode. Glossary resets between series.

6. **Persistent vocabulary** — `parse_vocab()` / `format_vocab()` / `save_vocab()`. Public functions also imported by `proofread.py`.

7. **CLI** — `main()` with argparse. Translation only. Key flags: `--endpoint`, `--source-lang`/`--target-lang`, `--chunk-size`, `--repetition-penalty`, `--no-group`, `--no-stream`, `--verbose`, `--extra-payload`, `--suffix`, `--out-dir`, `--retry`, `--vocab`.

### proofread.py — 2-Pass Proofreader

Standalone CLI that reviews translated SRT files using a 2-pass pipeline (all with `enable_thinking=False`):

**Step 0** (per-file): Context Building — scene/character summary from sampled lines.

**Pass 1 — Vocab Replace & Flag** (mechanical, per-chunk):
- **Step 1.1**: Strict Vocab Replacement — LLM replaces terms exactly per vocab sheet, nothing else.
- **Step 1.2**: Confidence Scoring — rates each chunk line HIGH/MEDIUM/LOW on vocab-replaced text. Context from previous chunks passed separately for reference.

**Pass 2 — Per-Line Correction** (only MEDIUM/LOW lines from Pass 1):
- **Step 2**: Analyze + Correct — one LLM call per flagged line with ±50 line context window (source+translated pairs). Hard output limit (`expected=300`, `runaway_multiplier=5`, `raise_on_runaway=True`). On runaway or failure, retries up to `--retry` times, then keeps Pass 1 output.
- **Step 2.3**: Generate Vocab Sheet — one call per file, proper nouns only (names, places, fictional terms). Hard limit with `raise_on_runaway=True` and retry. Output validated: entries >30 chars rejected, capped at 30 entries. Saved to separate file (e.g., `vocab_proofread.txt`).
- **Step 2.4**: Final Vocab Replacement — apply new vocab sheet strictly across ALL lines.

HIGH lines skip Pass 2 entirely. `--no-vocab-update` skips Steps 2.3-2.4.

Imports shared infrastructure from `main.py` (post_messages, SRT parsing, numbered I/O, vocab functions). Key flags: `--endpoint`, `--out-dir`, `--chunk-size` (default 20), `--context-window`, `--vocab`, `--proofread-vocab`, `--no-vocab-update`.

## Key Design Decisions

- **Chunked translation**: Small chunks (default 10 lines) prevent the model from looping. Each chunk is translated independently with thinking disabled and high repetition penalty.
- **Literal first-pass prompt**: Pass 1 translates mechanically without reasoning about context. Uncertain words are flagged with `??` for manual review.
- **Numbered line protocol**: `[N] text` format with regex parsing. Missing lines detected by number, not position.
- **Runaway handling**: Configurable via `runaway_multiplier` (default 3x) and `raise_on_runaway` (default False). When multiplier > 0 and output exceeds threshold, stream is closed. Default: truncate at repetition pattern and return partial result. With `raise_on_runaway=True`: raise RuntimeError so caller can retry (used by proofread per-line correction and vocab generation).
- **Series context sharing**: LLM groups files by series from filenames. Glossary accumulated per episode, appended to system prompt for next episode, reset between series.
- **2-pass proofread**: `proofread.py` is a separate CLI tool. User translates first, optionally edits `vocab.txt`, then runs proofread. Pass 1 is mechanical (vocab replacement first, then confidence scoring on the improved text). Pass 2 corrects each flagged line individually (one LLM call per line, ±50 line context window, hard output limit with raise-on-runaway + retry). Vocab generation produces proper nouns only, validated and capped, saved to separate file. `--no-vocab-update` skips vocab generation and final replacement.
- **`subs/`** is the conventional input directory; **`out/`** is the conventional output directory.
