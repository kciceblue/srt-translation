# SRT Translator

[中文文档](README_CN.md)

Translate subtitle files via backend API while preserving timestamps and line structure. 5-step pipeline: ASR error fixing → translation → quality flagging → proofreading. Auto-groups files by series for consistent character naming.

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

### Full Pipeline

```bash
python cli.py run <inputs> --endpoint <api_url> [options]
```

Inputs can be files, directories, or glob patterns — any mix:

```bash
# Single file
python cli.py run movie.srt --endpoint http://127.0.0.1:5000/v1/chat/completions

# Directory (recursively finds all .srt files)
python cli.py run subs/ --endpoint http://127.0.0.1:5000/v1/chat/completions --out-dir out

# Mix of files and directories
python cli.py run subs/ extras/bonus.srt --endpoint http://127.0.0.1:5000/v1/chat/completions

# Different language pair
python cli.py run subs/ --endpoint http://... --source-lang Korean --target-lang English --suffix .en.srt
```

### Step-by-Step

Run individual pipeline steps for debugging or manual intervention between steps. Use `--debug` to preserve the tmp folder.

```bash
# Step 1: Input — expand files, group by series, create tmp/
python cli.py input subs/ --endpoint http://... --debug

# Step 2: Preprocess — ASR error fix, context summary, term extraction
python cli.py preprocess --endpoint http://... --debug

# Step 3: Translate — chunked translation with vocab+context
python cli.py translate --endpoint http://... --debug

# Step 4: Postprocess — flag unfit translations
python cli.py postprocess --endpoint http://... --debug

# Step 5: Proofread — fix flagged lines, final review, copy to out/
python cli.py proofread --endpoint http://... --out-dir out --debug
```

## Options

### Common Options (all subcommands)

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | http://127.0.0.1:5000/v1/chat/completions | Backend API URL |
| `--source-lang` | Japanese | Source language name |
| `--target-lang` | Simplified Chinese | Target language name |
| `--timeout` | 300 | HTTP timeout (seconds) |
| `--retry` | 2 | Retries on failure |
| `--retry-sleep` | 1.0 | Sleep between retries (seconds) |
| `--extra-payload` | "" | Extra JSON fields for API body |
| `--no-stream` | false | Disable streaming |
| `--debug` | false | Verbose output + preserve tmp folder |
| `--tmp-dir` | ./tmp | Tmp folder path |

### `run` Options

| Option | Default | Description |
|--------|---------|-------------|
| `--out-dir` | out | Output directory |
| `--suffix` | .zh.srt | Output filename suffix |
| `--no-group` | false | Disable series grouping |
| `--chunk-size` | 10 | Lines per translation chunk (smaller = more stable) |
| `--repetition-penalty` | 1.3 | Repetition penalty (1.0 to disable) |
| `--vocab` | vocab.txt | External vocabulary file |

### `translate` Options

| Option | Default | Description |
|--------|---------|-------------|
| `--chunk-size` | 10 | Lines per translation chunk |
| `--repetition-penalty` | 1.3 | Repetition penalty |
| `--vocab` | vocab.txt | External vocabulary file |

### `proofread` Options

| Option | Default | Description |
|--------|---------|-------------|
| `--out-dir` | out | Output directory |
| `--suffix` | .zh.srt | Output filename suffix |
| `--context-radius` | 50 | Context lines around flagged line |

## How It Works

### Pipeline Overview

```
INPUT → Preprocess → Translate → Postprocess → Proofread → OUTPUT
         (ASR fix)   (chunked)    (flag bad)    (fix+review)
```

### Step 1: Input
Expands file/directory/glob inputs, warns about non-SRT files, uses the LLM to group files by series (based on filenames), creates tmp folder structure, and writes manifest.json.

### Step 2: Preprocess (NEW)
Per file, 5 passes in order: (1) builds a context summary (tone, speakers, plot) to understand the scene first, (2) brainstorms expected words for the scenario (domain vocab, character names, common phrases), (3) flags suspected ASR errors using context + expected words, (4) fixes flagged lines, (5) extracts high-confidence proper nouns only. Per series: audits the accumulated vocabulary line-by-line against context, aggressively removing uncertain entries — a misleading vocab entry is worse than a missing one. Writes `context.md` and `vocab.md` to tmp.

### Step 3: Translate
Divides subtitle lines into small chunks (`--chunk-size`, default 10). Each chunk is translated independently with a literal translation prompt (thinking disabled). Uses `[N] text` numbered format for precise line tracking. Missing lines get one repair attempt. Merges series vocab.md with external `--vocab` file. Accumulates a rolling glossary across episodes within a series.

### Step 4: Postprocess (NEW)
Compares source+translated pairs against context and vocabulary. Flags lines with problems (vocab mismatch, reversed meaning, missing info, `??` markers). Writes `flags.json` per series.

### Step 5: Proofread
Pass 1: Corrects each flagged line individually (one LLM call per line, ±50 line context window). Pass 2: Final audit review of the complete file. Failed corrections and remaining issues go to `confused.md` for human review. Copies final output to `out/`.

### Series Grouping
When given multiple files, the LLM groups them by series based on filenames. Files in the same series share context.md and vocab.md, ensuring consistent naming across episodes. Use `--no-group` to disable.

### Runaway Detection
Three layers of protection during SSE streaming:
1. **Exact repetition** — if a pattern repeats 10+ times, the stream is cut and only 1 occurrence is kept
2. **Reasoning babble** — if the model starts "thinking out loud" (dense hedge words like "Wait,", "Actually,", "however"), the babble is stripped
3. **Length multiplier** — if output exceeds N× the expected length, the stream is closed and truncated

Each pipeline step has its own retry/fallback strategy. Use `--no-stream` to disable streaming.

## Recommended Settings

- **Temperature 0.2–0.3**: Low enough to stay stable, high enough to not translate garbled words literally. Set via `--extra-payload '{"temperature":0.3}'`.
- **Repetition penalty 1.3+**: Prevents the model from looping. Increase if loops persist.
- **Chunk size 10**: Default. Reduce to 5 if the model still loops on complex content.

## Troubleshooting

**Missing lines in output**: Each chunk gets one auto-repair attempt

**Model loops / hangs**: Increase `--repetition-penalty`, decrease `--chunk-size`, or lower temperature

**Backend errors**: Ensure endpoint URL is correct and backend is running

**Timeout errors**: Increase `--timeout` value

**Step failed mid-pipeline**: Use `--debug` to preserve tmp/, then re-run individual steps

## Dependencies

- requests

## License

[MIT](LICENSE)
