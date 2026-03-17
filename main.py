#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
srt_translate_client.py

What it does
1) Read multiple .srt files, parse blocks, extract text lines only.
2) Send text lines to a backend API in an OpenAI "messages" format,
   using a numbered-line protocol ([N] text) for robust line mapping.
3) The model translates and also fixes potential ASR/Whisper transcription
   errors using surrounding context.
4) Rebuild .srt with original timestamps and translated lines.

Backend expectations (flexible)
- Accepts POST JSON: {"messages": [...]} (plus optional extra fields)
- Returns either:
  A) OpenAI-like JSON: {"choices":[{"message":{"content":"..."} }]}
  B) Simple JSON: {"text":"..."}  (or similar)
  C) Plain text containing: text='...'
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import re
import sys
import time
from typing import List, Tuple, Optional, Dict, Any

try:
    import requests
except ImportError:
    print("ERROR: missing dependency 'requests'. Install with: pip install requests", file=sys.stderr)
    sys.exit(2)


# ---------------------------
# SRT parsing / writing
# ---------------------------

_TS_LINE_RE = re.compile(
    r"^\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})(?:\s+.*)?$"
)


@dataclasses.dataclass
class SrtBlock:
    index: int
    ts_line: str
    text_lines: List[str]


def _read_text_file(path: str) -> str:
    # SRT often uses UTF-8 with BOM; sometimes UTF-16; occasionally legacy encodings.
    encodings = ["utf-8-sig", "utf-8", "utf-16", "cp1252"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception as e:
            last_err = e
    # Last resort: lossy read
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_srt(content: str) -> List[SrtBlock]:
    # Normalize newlines
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not content.strip():
        return []

    chunks = re.split(r"\n\s*\n", content)
    blocks: List[SrtBlock] = []

    for chunk_i, chunk in enumerate(chunks, start=1):
        lines = [ln.rstrip("\n") for ln in chunk.split("\n")]
        lines = [ln.rstrip() for ln in lines]

        if len(lines) < 2:
            continue

        # index line
        idx_line = lines[0].strip()
        try:
            idx = int(idx_line)
        except ValueError:
            # Try to recover: sometimes missing index; treat as sequential
            idx = len(blocks) + 1

        # timestamp line
        ts_line = lines[1].strip()
        if not _TS_LINE_RE.match(ts_line):
            # Try to find timestamp line within chunk
            ts_pos = None
            for j, ln in enumerate(lines):
                if _TS_LINE_RE.match(ln.strip()):
                    ts_pos = j
                    break
            if ts_pos is None:
                # Skip unrecognized chunk
                continue
            ts_line = lines[ts_pos].strip()
            text_lines = [ln for ln in lines[ts_pos + 1 :] if ln != ""]
        else:
            text_lines = [ln for ln in lines[2:] if ln != ""]

        blocks.append(SrtBlock(index=idx, ts_line=ts_line, text_lines=text_lines))

    # Re-index sequentially for clean output
    for i, b in enumerate(blocks, start=1):
        b.index = i

    return blocks


def write_srt(blocks: List[SrtBlock]) -> str:
    out_lines: List[str] = []
    for i, b in enumerate(blocks, start=1):
        out_lines.append(str(i))
        out_lines.append(b.ts_line)
        out_lines.extend(b.text_lines if b.text_lines else [""])
        out_lines.append("")  # blank line between blocks
    return "\n".join(out_lines).rstrip("\n") + "\n"


# ---------------------------
# Backend interaction
# ---------------------------

_TEXT_FIELD_RE = re.compile(r"text\s*=\s*(['\"])(.*?)\1", re.DOTALL)


def extract_text_from_response(resp_text: str, resp_json: Optional[Dict[str, Any]]) -> str:
    """
    Try multiple formats:
    - OpenAI-like: choices[0].message.content
    - OpenAI completion-like: choices[0].text
    - Simple JSON: text/content
    - Plain: text='...'
    - Otherwise: raw body
    """
    if resp_json is not None:
        # OpenAI chat completions style
        try:
            choices = resp_json.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                msg = c0.get("message") or {}
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
                if isinstance(c0.get("text"), str):
                    return c0["text"]
        except Exception:
            pass

        # Common alternatives
        for k in ("text", "content", "output", "result"):
            v = resp_json.get(k)
            if isinstance(v, str):
                return v

    m = _TEXT_FIELD_RE.search(resp_text)
    if m:
        return m.group(2)

    return resp_text.strip()


def post_messages(
    endpoint: str,
    messages: List[Dict[str, str]],
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {"messages": messages}
    if extra_payload:
        payload.update(extra_payload)

    r = requests.post(endpoint, json=payload, timeout=timeout_s)
    r.raise_for_status()

    resp_text = r.text
    resp_json: Optional[Dict[str, Any]] = None
    try:
        resp_json = r.json()
    except Exception:
        resp_json = None

    return extract_text_from_response(resp_text, resp_json)


# ---------------------------
# Numbered I/O helpers
# ---------------------------

# Permissive: matches [N] text, [N]: text, [N]. text, etc.
_NUMBERED_LINE_RE = re.compile(r"^\[(\d+)\][.:)\s]*\s*(.*)", re.MULTILINE)


def format_numbered_input(lines: List[str], start_index: int = 1) -> str:
    """Format lines as [N] text for the numbered I/O protocol."""
    parts = []
    for i, line in enumerate(lines, start=start_index):
        parts.append(f"[{i}] {line}")
    return "\n".join(parts)


def parse_numbered_output(
    text: str, expected_count: int, start_index: int = 1
) -> Tuple[List[Optional[str]], List[int]]:
    """
    Parse numbered output like '[1] translated text'.

    Returns:
        (translations, missing_indices)
        - translations: list of length expected_count, None for missing entries
        - missing_indices: list of indices that were not found in the output
    """
    # Normalize line endings before parsing
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    found: Dict[int, str] = {}
    for m in _NUMBERED_LINE_RE.finditer(text):
        num = int(m.group(1))
        found[num] = m.group(2)

    translations: List[Optional[str]] = []
    missing: List[int] = []
    for i in range(start_index, start_index + expected_count):
        if i in found:
            translations.append(found[i])
        else:
            translations.append(None)
            missing.append(i)

    return translations, missing


# ---------------------------
# Translation workflow
# ---------------------------

def translate_lines_via_backend(
    lines: List[str],
    endpoint: str,
    timeout_s: int,
    system_prompt: str,
    user_prefix: str,
    extra_payload: Optional[Dict[str, Any]],
    retry: int,
    retry_sleep_s: float,
) -> List[str]:
    """
    Translate all lines in a single request using numbered I/O.

    The entire file is sent at once (modern LLMs have large enough context).
    Uses a 2-tier strategy:
      Tier 1: Parse numbered output — if all lines present, done.
      Tier 2: If some lines missing, send targeted repair requests until complete.
    """
    if not lines:
        return []

    def _call_backend(messages: List[Dict[str, str]]) -> str:
        """Call backend with retry logic. Raises on total failure."""
        last_err: Optional[Exception] = None
        for attempt in range(1, retry + 2):
            try:
                return post_messages(
                    endpoint=endpoint,
                    messages=messages,
                    timeout_s=timeout_s,
                    extra_payload=extra_payload,
                )
            except Exception as e:
                last_err = e
                sys.stderr.write(
                    f"[RETRY] Request failed (attempt {attempt}/{retry + 1}): {e}\n"
                )
                if attempt <= retry:
                    time.sleep(retry_sleep_s)
        raise RuntimeError(f"Backend failed after {retry + 1} attempts: {last_err}")

    sys.stderr.write(f"[INFO] Translating {len(lines)} lines in one request...\n")

    # Tier 1: send all lines with numbered format
    numbered_input = format_numbered_input(lines, start_index=1)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prefix + numbered_input},
    ]

    out_text = _call_backend(messages)
    translations, missing = parse_numbered_output(out_text, len(lines), start_index=1)

    if not missing:
        sys.stderr.write(f"[OK] All {len(lines)} lines translated.\n")
        return [t or "" for t in translations]

    # Tier 2: repair missing lines (up to 3 rounds)
    max_repair_rounds = 3
    for round_num in range(1, max_repair_rounds + 1):
        sys.stderr.write(
            f"[REPAIR] Round {round_num}: {len(missing)}/{len(lines)} lines missing {missing}. "
            f"Sending repair request...\n"
        )
        repair_lines = [f"[{i}] {lines[i - 1]}" for i in missing]
        repair_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "The following lines were missing from your previous output. "
                    "Translate ONLY these lines, keeping the [N] format:\n"
                    + "\n".join(repair_lines)
                ),
            },
        ]

        try:
            repair_text = _call_backend(repair_messages)
            for m in _NUMBERED_LINE_RE.finditer(repair_text):
                num = int(m.group(1))
                if 1 <= num <= len(lines) and translations[num - 1] is None:
                    translations[num - 1] = m.group(2)
        except Exception as e:
            sys.stderr.write(f"[WARN] Repair round {round_num} failed: {e}\n")

        missing = [i for i in range(1, len(lines) + 1) if translations[i - 1] is None]
        if not missing:
            sys.stderr.write(f"[OK] All {len(lines)} lines translated after {round_num} repair round(s).\n")
            break

    if missing:
        sys.stderr.write(
            f"[WARN] {len(missing)} lines still missing after {max_repair_rounds} repair rounds: "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}. Using empty strings.\n"
        )

    return [t if t is not None else "" for t in translations]


def build_line_mapping(blocks: List[SrtBlock]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Flatten all subtitle text lines into a list for translation,
    and keep a back-reference: (block_index_in_list, line_index_in_block)
    """
    lines: List[str] = []
    refs: List[Tuple[int, int]] = []
    for bi, b in enumerate(blocks):
        if not b.text_lines:
            # keep empty line to preserve structure
            lines.append("")
            refs.append((bi, 0))
        else:
            for li, ln in enumerate(b.text_lines):
                lines.append(ln)
                refs.append((bi, li))
    return lines, refs


def apply_translations(blocks: List[SrtBlock], refs: List[Tuple[int, int]], translated_lines: List[str]) -> None:
    if len(refs) != len(translated_lines):
        raise ValueError(f"Internal error: refs={len(refs)} translated_lines={len(translated_lines)}")

    # Prepare structure if some blocks originally had no lines
    for b in blocks:
        if not b.text_lines:
            b.text_lines = [""]

    for (bi, li), tln in zip(refs, translated_lines):
        b = blocks[bi]
        # Safety: extend if needed
        if li >= len(b.text_lines):
            b.text_lines.extend([""] * (li - len(b.text_lines) + 1))
        b.text_lines[li] = tln


# ---------------------------
# CLI
# ---------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a professional subtitle translator and proofreader.\n"
    "\n"
    "Your tasks:\n"
    "1. Translate each line from {source_lang} to {target_lang}.\n"
    "2. The input comes from automatic speech recognition (Whisper) and may contain "
    "transcription errors — misspelled words, homophones, broken phrases, or garbled text. "
    "Use surrounding context to infer the intended meaning and produce a correct, natural translation.\n"
    "3. Input/output format: each input line is numbered as `[N] text`. "
    "Output exactly `[N] translated_text` for every input line, in order.\n"
    "4. You MUST output every number from the input. No gaps, no merges, no additions, "
    "no commentary outside the `[N] ...` format.\n"
    "5. Preserve the tone and register of spoken dialogue."
)

_DEFAULT_USER_PREFIX = "Translate the following {source_lang} subtitles to {target_lang}:\n"


def _format_template(template: str, source_lang: str, target_lang: str) -> str:
    """Format a prompt template, returning it unchanged if placeholders are absent."""
    try:
        return template.format(source_lang=source_lang, target_lang=target_lang)
    except (KeyError, IndexError):
        return template


def expand_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for it in inputs:
        # Allow glob patterns
        if any(ch in it for ch in "*?[]"):
            paths.extend(glob.glob(it))
        else:
            paths.append(it)
    # De-dup, keep order
    seen = set()
    out = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            seen.add(ap)
            out.append(ap)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Translate .srt subtitle files via a backend API, with ASR error correction."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Input .srt files or glob patterns (e.g. subs/*.srt)",
    )
    ap.add_argument(
        "--endpoint",
        default="http://127.0.0.1:5000",
        help="Backend POST endpoint URL (default: http://127.0.0.1:5000)",
    )
    ap.add_argument(
        "--out-dir",
        default="translated_out",
        help="Output directory (default: translated_out)",
    )
    ap.add_argument(
        "--suffix",
        default=".zh.srt",
        help="Output filename suffix (default: .zh.srt)",
    )
    ap.add_argument(
        "--source-lang",
        default="Japanese",
        help="Source language name for the prompt (default: Japanese)",
    )
    ap.add_argument(
        "--target-lang",
        default="Simplified Chinese",
        help="Target language name for the prompt (default: Simplified Chinese)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP timeout seconds (default: 300)",
    )
    ap.add_argument(
        "--retry",
        type=int,
        default=1,
        help="Retry count per chunk on failure (default: 1)",
    )
    ap.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between retries (default: 1.0)",
    )
    ap.add_argument(
        "--system-prompt",
        default=_DEFAULT_SYSTEM_PROMPT,
        help="System prompt text (supports {source_lang} and {target_lang} placeholders)",
    )
    ap.add_argument(
        "--user-prefix",
        default=_DEFAULT_USER_PREFIX,
        help="User message prefix before text (supports {source_lang} and {target_lang} placeholders)",
    )
    ap.add_argument(
        "--extra-payload",
        default="",
        help="Extra JSON fields to include in POST body (e.g. '{\"model\":\"local\",\"temperature\":0}')",
    )

    args = ap.parse_args()

    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        print("ERROR: No valid input files found.", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    # Format prompt templates with language names
    system_prompt = _format_template(args.system_prompt, args.source_lang, args.target_lang)
    user_prefix = _format_template(args.user_prefix, args.source_lang, args.target_lang)

    extra_payload: Optional[Dict[str, Any]] = None
    if args.extra_payload.strip():
        try:
            extra_payload = json.loads(args.extra_payload)
            if not isinstance(extra_payload, dict):
                raise ValueError("extra payload must be a JSON object")
        except Exception as e:
            print(f"ERROR: --extra-payload is not valid JSON object: {e}", file=sys.stderr)
            return 2

    for path in input_paths:
        base = os.path.basename(path)
        stem, _ext = os.path.splitext(base)
        out_path = os.path.join(args.out_dir, stem + args.suffix)

        print(f"[INFO] Processing: {path}")
        content = _read_text_file(path)
        blocks = parse_srt(content)

        if not blocks:
            print(f"[WARN] No blocks parsed from: {path}. Writing empty output.")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("")
            continue

        flat_lines, refs = build_line_mapping(blocks)

        # Translate
        translated_lines = translate_lines_via_backend(
            lines=flat_lines,
            endpoint=args.endpoint,
            timeout_s=args.timeout,
            system_prompt=system_prompt,
            user_prefix=user_prefix,
            extra_payload=extra_payload,
            retry=args.retry,
            retry_sleep_s=args.retry_sleep,
        )

        apply_translations(blocks, refs, translated_lines)

        out_srt = write_srt(blocks)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_srt)

        print(f"[INFO] Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
