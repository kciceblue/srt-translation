# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SRT subtitle translator — a 5-step pipeline that translates `.srt` subtitle files via an OpenAI-compatible backend API. Single CLI entry point (`cli.py`) with subcommands for each step, plus a `run` subcommand for the full pipeline. Per-series tmp folder state tracks progress across steps.

Default configuration translates Japanese to Simplified Chinese. Language pair configurable via `--source-lang` / `--target-lang`.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline (input → preprocess → translate → postprocess → proofread)
python cli.py run subs/ --endpoint http://127.0.0.1:5000/v1/chat/completions --out-dir out

# Individual steps (each reads/writes tmp/ state)
python cli.py input subs/ --endpoint http://... --debug
python cli.py preprocess --endpoint http://... --debug
python cli.py translate --endpoint http://... --debug
python cli.py postprocess --endpoint http://... --debug
python cli.py proofread --endpoint http://... --out-dir out --debug

# Quick run with local config
./run.sh
```

There are no tests or linting configured.

## Architecture

### File Structure

```
cli.py          # Entry point with argparse subcommands
core.py         # Shared: SRT parsing, backend/runaway, numbered I/O, vocab, utils
input_step.py   # Step 1: Expand inputs, warn non-SRT, series grouping, tmp setup
preprocess.py   # Step 2: ASR flag/fix, context.md, vocab.md
translate.py    # Step 3: Chunked translation with vocab+context
postprocess.py  # Step 4: Flag unfit translations → flags.json
proofread.py    # Step 5: Fix flagged lines, final review, confused.md
```

### core.py — Shared Infrastructure

1. **SRT parsing/writing** — `SrtBlock`, `parse_srt()`, `write_srt()`, `read_text_file()`. Handles encoding detection (UTF-8 BOM, UTF-16, CP1252 fallback) and tolerant parsing of malformed SRT files.

2. **Backend interaction** — `post_messages()` with dual-mode: non-streaming and SSE streaming. `_stream_with_loop_detection()` has three layers of protection: (a) exact repetition detection (pattern ≥10x → keep 1 occurrence), (b) reasoning babble detection (hedge words like "Wait,", "Actually," ≥6 in 600 chars → strip babble), (c) multiplier-based runaway (output > N× expected → truncate). Auto-falls back to non-streaming if backend doesn't support it.

3. **LLM wrappers** — `call_llm()` (thin wrapper, builds messages list, thinking always disabled) and `call_llm_with_retry()` (adds retry logic with sleep).

4. **Numbered I/O** — `format_numbered_input()` / `parse_numbered_output()`. Explicit `[N] text` anchors for robust line mapping.

5. **Line mapping** — `build_line_mapping()` / `apply_translations()` for SRT block ↔ flat line conversion.

6. **Vocabulary** — `parse_vocab()` / `format_vocab()` / `save_vocab()`. Supports `Source → Target` and `Source -> Target` formats.

7. **Input expansion** — `expand_inputs()` handles files, directories, and glob patterns.

8. **Manifest/tmp management** — `load_manifest()` / `save_manifest()` / `TmpPaths` helper class for computing paths within per-series tmp folders.

### Pipeline Steps

**Step 1 — Input** (`input_step.py`): Expand inputs → warn non-SRT → LLM groups files by series → create tmp folder structure → write manifest.json.

**Step 2 — Preprocess** (`preprocess.py`): Per file, 5 passes: (1) context summary — understand the scene first, (2) brainstorm expected words for the scenario, (3) ASR error flagging informed by context + expected words, (4) ASR error fixing, (5) term extraction (high-confidence proper nouns only). Per series: (6) vocab audit — line-by-line verification against context, deletes uncertain entries. Writes context.md and vocab.md.

**Step 3 — Translate** (`translate.py`): Chunked translation with numbered-line protocol. Merges series vocab.md with external `--vocab` file. Accumulates rolling glossary across episodes within a series.

**Step 4 — Postprocess** (`postprocess.py`): Compares source+translated pairs against context and vocab, flags unfit lines. Writes flags.json per series.

**Step 5 — Proofread** (`proofread.py`): Pass 1 corrects flagged lines (from flags.json) with ±50 line context. Pass 2 does final audit review. Failed corrections and remaining issues go to confused.md for human review. Copies final output to out/.

### Tmp Folder Structure

```
./tmp/
├── manifest.json              # Series grouping + pipeline state
├── SeriesA/
│   ├── ep01.srt               # Copied source (overwritten after ASR fix)
│   ├── ep01.translated.srt    # Step 3 output
│   ├── context.md             # Step 2: tone, speakers, plot summary
│   ├── vocab.md               # Step 2: extracted terms
│   ├── flags.json             # Step 4: {stem: {line_num: reason}}
│   └── confused.md            # Step 5: lines needing human help
```

### cli.py — Entry Point

Subcommands: `run`, `input`, `preprocess`, `translate`, `postprocess`, `proofread`.

**`run`** executes all 5 steps sequentially, cleans tmp after (unless `--debug`).

Individual subcommands check prerequisites via manifest `*_done` flags but proceed with a warning if not met. `--debug` enables verbose output and preserves tmp folder.

## Key Design Decisions

- **5-step pipeline**: INPUT → Preprocess → Translate → Postprocess → Proofread. Each step reads/writes state in tmp folder. Steps can be run individually for debugging.
- **ASR pre-processing**: Step 2 builds context first, brainstorms expected words for the scenario, then flags/fixes ASR errors with that knowledge. Context-first approach catches more errors than blind flagging.
- **Vocab accuracy over coverage**: Term extraction and vocab cleanup prioritize accuracy — a misleading entry is worse than a missing one. Cleanup pass audits line-by-line against context and aggressively deletes uncertain entries.
- **Chunked translation**: Small chunks (default 10 lines) prevent the model from looping. Literal first-pass prompt with no context reasoning.
- **Numbered line protocol**: `[N] text` format with regex parsing. Missing lines detected by number, not position.
- **Runaway handling**: Three layers — exact repetition (≥10x same pattern), reasoning babble (hedge-word density), and multiplier-based length limit. Each step has its own retry/fallback strategy. All flag/reason prompts use structured format: `"wrong_word" → candidate1, candidate2; short reason` to prevent model from reasoning in output.
- **Series context sharing**: LLM groups files by series from filenames. Context.md + vocab.md shared within series. Rolling glossary across episodes.
- **flags.json → confused.md**: Postprocess flags problems, proofread attempts fixes. Lines that can't be fixed go to confused.md for human review.
- **`subs/`** is the conventional input directory; **`out/`** is the conventional output directory; **`tmp/`** is the pipeline state directory.

## Legacy Files

- `main.py` — Original monolithic translator (kept for reference, not used by pipeline)
- `proofread_legacy.py` — Original monolithic proofreader (kept for reference, not used by pipeline)
