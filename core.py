#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core.py — Shared infrastructure for the SRT translation pipeline.

SRT parsing/writing, backend interaction with runaway detection,
numbered I/O protocol, vocabulary helpers, manifest/tmp management.
"""

from __future__ import annotations

import dataclasses
import glob as glob_mod
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


# ---------------------------------------------------------------------------
# SRT parsing / writing
# ---------------------------------------------------------------------------

_TS_LINE_RE = re.compile(
    r"^\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})(?:\s+.*)?$"
)


@dataclasses.dataclass
class SrtBlock:
    index: int
    ts_line: str
    text_lines: List[str]


def read_text_file(path: str) -> str:
    encodings = ["utf-8-sig", "utf-8", "utf-16", "cp1252"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            pass
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_srt(content: str) -> List[SrtBlock]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not content.strip():
        return []

    chunks = re.split(r"\n\s*\n", content)
    blocks: List[SrtBlock] = []

    for chunk in chunks:
        lines = [ln.rstrip() for ln in chunk.split("\n")]
        if len(lines) < 2:
            continue

        idx_line = lines[0].strip()
        try:
            idx = int(idx_line)
        except ValueError:
            idx = len(blocks) + 1

        ts_line = lines[1].strip()
        if not _TS_LINE_RE.match(ts_line):
            ts_pos = None
            for j, ln in enumerate(lines):
                if _TS_LINE_RE.match(ln.strip()):
                    ts_pos = j
                    break
            if ts_pos is None:
                continue
            ts_line = lines[ts_pos].strip()
            text_lines = [ln for ln in lines[ts_pos + 1:] if ln != ""]
        else:
            text_lines = [ln for ln in lines[2:] if ln != ""]

        blocks.append(SrtBlock(index=idx, ts_line=ts_line, text_lines=text_lines))

    for i, b in enumerate(blocks, start=1):
        b.index = i
    return blocks


def write_srt(blocks: List[SrtBlock]) -> str:
    out_lines: List[str] = []
    for i, b in enumerate(blocks, start=1):
        out_lines.append(str(i))
        out_lines.append(b.ts_line)
        out_lines.extend(b.text_lines if b.text_lines else [""])
        out_lines.append("")
    return "\n".join(out_lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Backend interaction
# ---------------------------------------------------------------------------

_TEXT_FIELD_RE = re.compile(r"text\s*=\s*(['\"])(.*?)\1", re.DOTALL)

# Auto-detected streaming support (None = not yet tested)
_streaming_available: Optional[bool] = None


def extract_text_from_response(resp_text: str, resp_json: Optional[Dict[str, Any]]) -> str:
    if resp_json is not None:
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
        for k in ("text", "content", "output", "result"):
            v = resp_json.get(k)
            if isinstance(v, str):
                return v

    m = _TEXT_FIELD_RE.search(resp_text)
    if m:
        return m.group(2)
    return resp_text.strip()


def _truncate_at_repetition(text: str, min_pattern_len: int = 10, max_pattern_len: int = 80, keep: int = 2) -> str:
    """Find where text starts looping and truncate, keeping `keep` occurrences of the repeated pattern."""
    buf_len = len(text)
    for plen in range(min_pattern_len, min(max_pattern_len + 1, buf_len // 3 + 1)):
        for start in range(buf_len - plen * 3, -1, -plen):
            if start < 0:
                break
            candidate = text[start : start + plen]
            count = 0
            pos = start
            while pos + plen <= buf_len and text[pos : pos + plen] == candidate:
                count += 1
                pos += plen
            if count >= 3:
                # Walk backwards to find the true start of the consecutive run
                run_start = start
                while run_start - plen >= 0 and text[run_start - plen : run_start] == candidate:
                    run_start -= plen
                cut_at = run_start + plen * max(keep, 1)
                return text[:cut_at]
    return text


def _detect_tail_repetition(
    text: str,
    min_pattern_len: int = 10,
    max_pattern_len: int = 80,
    threshold: int = 10,
) -> int:
    """Scan the tail of `text` for a pattern repeating >= `threshold` times.

    Returns the cut position (keeping 1 occurrence) or -1 if no loop found.
    Designed to run periodically during streaming to catch loops early.
    """
    buf_len = len(text)
    for plen in range(min_pattern_len, min(max_pattern_len + 1, buf_len // threshold + 1)):
        candidate = text[buf_len - plen : buf_len]
        count = 0
        pos = buf_len
        while pos - plen >= 0:
            pos -= plen
            if text[pos : pos + plen] == candidate:
                count += 1
            else:
                break
        if count >= threshold:
            # pos is either the mismatch position (broke on content) or
            # the earliest match position (broke on pos-plen < 0).
            # In both cases: first non-repeating content ends where the
            # consecutive run begins.  The run starts at (buf_len - count*plen).
            first_repeat_start = buf_len - count * plen
            return first_repeat_start + plen  # keep 1 occurrence
    return -1


# ---------------------------------------------------------------------------
# Reasoning babble detection
# ---------------------------------------------------------------------------

# Markers that indicate the model is "thinking out loud" instead of producing
# structured output.  Case-insensitive matching.  Covers English hedging words
# that appear when the model gets confused and starts reasoning in the content
# stream where only structured [N] lines should appear.
_BABBLE_MARKERS = re.compile(
    r"\b(?:"
    r"wait|actually|however|but(?:\s+(?:maybe|wait|actually|let))|"
    r"let(?:'|')s\s+(?:re-?evaluate|reconsider|think|look)|"
    r"hmm|on\s+second\s+thought|re-?evaluat|looking\s+(?:at|again)|"
    r"maybe\s+(?:it(?:'|')s|the)|"
    r"I\s+think|that\s+said|in\s+fact|to\s+be\s+(?:fair|honest)"
    r")\b",
    re.IGNORECASE,
)

# Check the last _BABBLE_WINDOW chars.  If >= _BABBLE_THRESHOLD markers appear
# in that window the model is babbling.
_BABBLE_WINDOW = 600
_BABBLE_THRESHOLD = 6


def _detect_reasoning_babble(text: str) -> int:
    """Detect reasoning-chain babble in the tail of `text`.

    Returns the cut position (start of the babble section) or -1 if clean.
    """
    if len(text) < _BABBLE_WINDOW:
        window = text
        window_start = 0
    else:
        window_start = len(text) - _BABBLE_WINDOW
        window = text[window_start:]

    hits = list(_BABBLE_MARKERS.finditer(window))
    if len(hits) < _BABBLE_THRESHOLD:
        return -1

    # Cut at the position of the first marker in the window — everything from
    # that point onward is babble.
    first_hit_offset = hits[0].start()
    cut_pos = window_start + first_hit_offset
    # Walk back to the last newline before the cut so we don't chop mid-line
    newline_pos = text.rfind("\n", 0, cut_pos)
    if newline_pos > 0:
        cut_pos = newline_pos + 1
    return cut_pos


def _stream_with_loop_detection(
    endpoint: str,
    payload: Dict[str, Any],
    timeout_s: int,
    expected_output_len: int,
    verbose: bool = False,
    max_thinking_len: int = 0,
    runaway_multiplier: float = 3.0,
    raise_on_runaway: bool = False,
) -> str:
    """Stream SSE response. On runaway, truncate and return partial result (or raise if raise_on_runaway)."""
    r = requests.post(
        endpoint, json=payload,
        timeout=(10, timeout_s),
        stream=True,
    )
    r.raise_for_status()
    r.encoding = "utf-8"

    collected: List[str] = []
    total_content_len = 0
    total_thinking_len = 0
    had_output = False
    runaway = False
    loop_detected = False
    babble_detected = False
    # Track how much content we've checked for in-stream loop/babble detection
    _last_loop_check_len = 0
    _LOOP_CHECK_INTERVAL = 500  # check every 500 chars of new content

    try:
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data_str = raw_line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            try:
                delta_obj = chunk["choices"][0]["delta"]
            except (KeyError, IndexError, TypeError):
                continue

            # Process thinking/reasoning tokens
            thinking = ""
            for key in ("reasoning_content", "reasoning"):
                t = delta_obj.get(key, "")
                if t:
                    thinking += t
            if thinking:
                had_output = True
                total_thinking_len += len(thinking)
                if verbose:
                    sys.stdout.write(thinking)
                    sys.stdout.flush()

            # Thinking runaway: abort so caller can retry
            if max_thinking_len > 0 and total_thinking_len > max_thinking_len:
                r.close()
                raise RuntimeError(
                    f"Thinking output ({total_thinking_len} chars) exceeded limit "
                    f"({max_thinking_len}). Model may be producing garbage."
                )

            # Process content tokens
            delta = delta_obj.get("content", "")
            if delta:
                had_output = True
                if verbose:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                collected.append(delta)
                total_content_len += len(delta)

            # In-stream loop & babble detection: check every _LOOP_CHECK_INTERVAL chars
            if total_content_len - _last_loop_check_len >= _LOOP_CHECK_INTERVAL:
                _last_loop_check_len = total_content_len
                buf = "".join(collected)
                # Check 1: exact repetition (pattern repeating 10+ times)
                cut = _detect_tail_repetition(buf, threshold=10)
                if cut >= 0:
                    loop_detected = True
                    break
                # Check 2: reasoning babble ("Wait,", "Actually,", etc.)
                babble_cut = _detect_reasoning_babble(buf)
                if babble_cut >= 0:
                    babble_detected = True
                    break

            # Runaway: stop collecting, close connection
            if (runaway_multiplier > 0 and expected_output_len > 0
                    and total_content_len > expected_output_len * runaway_multiplier):
                runaway = True
                break
    finally:
        if verbose and had_output:
            sys.stdout.write("\n")
            sys.stdout.flush()
        r.close()

    raw_output = "".join(collected)

    # In-stream loop: truncate to 1 occurrence, treat as runaway
    if loop_detected:
        cut = _detect_tail_repetition(raw_output, threshold=10)
        cleaned = raw_output[:cut] if cut >= 0 else raw_output
        sys.stderr.write(
            f"[LOOP] Repetition detected during streaming at {total_content_len} chars. "
            f"Kept {len(cleaned)}/{len(raw_output)} chars (1 occurrence).\n"
        )
        if raise_on_runaway:
            raise RuntimeError(
                f"Loop detected: pattern repeating >10x at {total_content_len} chars"
            )
        return cleaned

    # Reasoning babble: model started "thinking out loud" in the content stream
    if babble_detected:
        babble_cut = _detect_reasoning_babble(raw_output)
        cleaned = raw_output[:babble_cut] if babble_cut >= 0 else raw_output
        sys.stderr.write(
            f"[BABBLE] Reasoning chain detected in content at {total_content_len} chars. "
            f"Kept {len(cleaned)}/{len(raw_output)} chars (babble stripped).\n"
        )
        if raise_on_runaway:
            raise RuntimeError(
                f"Babble detected: reasoning chain in content at {total_content_len} chars"
            )
        return cleaned

    if runaway:
        sys.stderr.write(
            f"[RUNAWAY] Content output {total_content_len} chars exceeded {runaway_multiplier}x expected "
            f"(~{expected_output_len}).\n"
        )
        if raise_on_runaway:
            raise RuntimeError(
                f"Runaway detected: {total_content_len} chars exceeded "
                f"{runaway_multiplier}x expected (~{expected_output_len})"
            )
        cleaned = _truncate_at_repetition(raw_output)
        sys.stderr.write(
            f"[RUNAWAY] Kept {len(cleaned)}/{len(raw_output)} chars after cleanup.\n"
        )
        return cleaned

    return raw_output


def post_messages(
    endpoint: str,
    messages: List[Dict[str, str]],
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    expected_output_len: int = 0,
    verbose: bool = False,
    override_params: Optional[Dict[str, Any]] = None,
    enable_thinking: bool = False,
    max_thinking_len: int = 0,
    runaway_multiplier: float = 3.0,
    raise_on_runaway: bool = False,
) -> str:
    global _streaming_available
    payload: Dict[str, Any] = {"messages": messages}
    if extra_payload:
        payload.update(extra_payload)
    payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if override_params:
        payload.update(override_params)

    if verbose:
        msg_lens = ", ".join(f"{m['role']}:{len(m['content'])}ch" for m in messages)
        sys.stderr.write(f"  [POST] {endpoint} stream={stream} ({msg_lens})\n")

    # Streaming path
    if stream and _streaming_available is not False:
        stream_payload = {**payload, "stream": True}
        try:
            return _stream_with_loop_detection(
                endpoint, stream_payload, timeout_s, expected_output_len,
                verbose=verbose,
                max_thinking_len=max_thinking_len,
                runaway_multiplier=runaway_multiplier,
                raise_on_runaway=raise_on_runaway,
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code < 500:
                _streaming_available = False
                sys.stderr.write("[WARN] Streaming not supported by backend, falling back.\n")
            else:
                raise

    # Non-streaming path
    r = requests.post(endpoint, json=payload, timeout=timeout_s)
    r.raise_for_status()

    resp_text = r.text
    resp_json: Optional[Dict[str, Any]] = None
    try:
        resp_json = r.json()
    except Exception:
        resp_json = None

    return extract_text_from_response(resp_text, resp_json)


# ---------------------------------------------------------------------------
# Numbered I/O helpers
# ---------------------------------------------------------------------------

_NUMBERED_LINE_RE = re.compile(r"^\[(\d+)\][.:)\s]*\s*(.*)", re.MULTILINE)


def format_numbered_input(lines: List[str], start_index: int = 1) -> str:
    parts = []
    for i, line in enumerate(lines, start=start_index):
        parts.append(f"[{i}] {line}")
    return "\n".join(parts)


def parse_numbered_output(
    text: str, expected_count: int, start_index: int = 1
) -> Tuple[List[Optional[str]], List[int]]:
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


# ---------------------------------------------------------------------------
# Line mapping
# ---------------------------------------------------------------------------

def build_line_mapping(blocks: List[SrtBlock]) -> Tuple[List[str], List[Tuple[int, int]]]:
    lines: List[str] = []
    refs: List[Tuple[int, int]] = []
    for bi, b in enumerate(blocks):
        if not b.text_lines:
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
    for b in blocks:
        if not b.text_lines:
            b.text_lines = [""]
    for (bi, li), tln in zip(refs, translated_lines):
        b = blocks[bi]
        if li >= len(b.text_lines):
            b.text_lines.extend([""] * (li - len(b.text_lines) + 1))
        b.text_lines[li] = tln


# ---------------------------------------------------------------------------
# Persistent vocabulary
# ---------------------------------------------------------------------------

VOCAB_HEADER = (
    "# Vocabulary file for SRT subtitle translator\n"
    "# Lines starting with # are comments and are ignored\n"
    "#\n"
    "# Supported formats (one entry per line):\n"
    "#   SourceTerm → TranslatedTerm     (source language → target language)\n"
    "#   DraftTranslation → Corrected    (fix a specific draft translation)\n"
    "#\n"
    "# This file is loaded automatically at startup.\n"
    "# Learnt terms from each translation run are merged back here.\n"
    "# Feel free to edit, add, or remove entries between runs.\n"
    "#\n"
)


def parse_vocab(text: str) -> Dict[str, str]:
    """Parse vocab text into {source_term: translated_term} dict."""
    entries: Dict[str, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("→", "->"):
            if sep in line:
                parts = line.split(sep, 1)
                src = parts[0].strip()
                tgt = parts[1].strip()
                if src and tgt:
                    entries[src] = tgt
                break
    return entries


def format_vocab(entries: Dict[str, str]) -> str:
    """Format vocab entries as SourceTerm → TranslatedTerm lines."""
    return "\n".join(f"{src} → {tgt}" for src, tgt in sorted(entries.items()))


def save_vocab(path: str, entries: Dict[str, str]) -> None:
    """Save vocab entries to file with header."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(VOCAB_HEADER)
        f.write(format_vocab(entries))
        f.write("\n")


# ---------------------------------------------------------------------------
# Input expansion
# ---------------------------------------------------------------------------

def expand_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for it in inputs:
        if any(ch in it for ch in "*?[]"):
            paths.extend(glob_mod.glob(it))
        else:
            paths.append(it)

    # Expand directories into .srt files recursively
    expanded: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if f.lower().endswith(".srt"):
                        expanded.append(os.path.join(root, f))
        else:
            expanded.append(p)

    # De-dup, keep order
    seen: set = set()
    out: List[str] = []
    for p in expanded:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            seen.add(ap)
            out.append(ap)
    return out


# ---------------------------------------------------------------------------
# LLM call wrappers (consolidate retry patterns)
# ---------------------------------------------------------------------------

def call_llm(
    system: str,
    user: str,
    endpoint: str,
    timeout_s: int = 300,
    extra_payload: Optional[Dict[str, Any]] = None,
    use_stream: bool = True,
    verbose: bool = False,
    expected_output_len: int = 0,
    runaway_multiplier: float = 3.0,
    raise_on_runaway: bool = False,
) -> str:
    """Thin wrapper around post_messages that builds the messages list. Thinking always disabled."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return post_messages(
        endpoint=endpoint,
        messages=messages,
        timeout_s=timeout_s,
        extra_payload=extra_payload,
        stream=use_stream,
        expected_output_len=expected_output_len,
        verbose=verbose,
        enable_thinking=False,
        max_thinking_len=0,
        runaway_multiplier=runaway_multiplier,
        raise_on_runaway=raise_on_runaway,
    )


def call_llm_with_retry(
    system: str,
    user: str,
    endpoint: str,
    timeout_s: int = 300,
    extra_payload: Optional[Dict[str, Any]] = None,
    use_stream: bool = True,
    verbose: bool = False,
    expected_output_len: int = 0,
    runaway_multiplier: float = 3.0,
    raise_on_runaway: bool = False,
    retry: int = 2,
    retry_sleep_s: float = 1.0,
    override_params: Optional[Dict[str, Any]] = None,
) -> str:
    """call_llm with retry logic. Raises after all attempts exhausted."""
    last_err: Optional[Exception] = None
    for attempt in range(1, retry + 2):
        if verbose:
            sys.stderr.write(f"  [CALL] Attempt {attempt}/{retry + 1}\n")
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return post_messages(
                endpoint=endpoint,
                messages=messages,
                timeout_s=timeout_s,
                extra_payload=extra_payload,
                stream=use_stream,
                expected_output_len=expected_output_len,
                verbose=verbose,
                override_params=override_params,
                enable_thinking=False,
                max_thinking_len=0,
                runaway_multiplier=runaway_multiplier,
                raise_on_runaway=raise_on_runaway,
            )
        except Exception as e:
            last_err = e
            sys.stderr.write(
                f"[RETRY] Request failed (attempt {attempt}/{retry + 1}): {e}\n"
            )
            if attempt <= retry:
                time.sleep(retry_sleep_s)
    raise RuntimeError(f"Backend failed after {retry + 1} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Manifest & tmp folder management
# ---------------------------------------------------------------------------

def load_manifest(tmp_dir: str) -> dict:
    """Load manifest.json from tmp_dir. Returns empty dict if missing."""
    path = os.path.join(tmp_dir, "manifest.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(tmp_dir: str, manifest: dict) -> None:
    """Save manifest.json to tmp_dir."""
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


class TmpPaths:
    """Helper for computing paths within a series tmp folder."""

    def __init__(self, tmp_dir: str, series_name: str):
        # Sanitize series name for filesystem
        safe = re.sub(r'[<>:"/\\|?*]', '_', series_name).strip().strip('.')
        if not safe:
            safe = "default"
        self.series_dir = os.path.join(tmp_dir, safe)
        self.context_md = os.path.join(self.series_dir, "context.md")
        self.vocab_md = os.path.join(self.series_dir, "vocab.md")
        self.flags_json = os.path.join(self.series_dir, "flags.json")
        self.confused_md = os.path.join(self.series_dir, "confused.md")

    def source_srt(self, stem: str) -> str:
        return os.path.join(self.series_dir, f"{stem}.srt")

    def translated_srt(self, stem: str) -> str:
        return os.path.join(self.series_dir, f"{stem}.translated.srt")
