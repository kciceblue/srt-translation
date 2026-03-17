#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
srt_translate_client.py

What it does
1) Read .srt files from files, directories, or glob patterns.
2) Group files by series using the LLM (auto-detect which files belong together).
3) Translate each series with shared context (glossary of character names/terms).
4) Uses numbered-line protocol ([N] text) and streaming with loop detection.
5) Rebuild .srt with original timestamps and translated lines.

Backend expectations (flexible)
- Accepts POST JSON: {"messages": [...]} (plus optional extra fields)
- Supports SSE streaming when {"stream": true} is included
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


# ---------------------------
# Backend interaction
# ---------------------------

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



def _truncate_at_repetition(text: str, min_pattern_len: int = 10, max_pattern_len: int = 80) -> str:
    """Find where text starts looping and truncate, keeping first 2 occurrences of the repeated pattern."""
    buf_len = len(text)
    for plen in range(min_pattern_len, min(max_pattern_len + 1, buf_len // 3 + 1)):
        # Scan from the end backwards to find where repetition starts
        for start in range(buf_len - plen * 3, -1, -plen):
            if start < 0:
                break
            candidate = text[start : start + plen]
            # Count consecutive repeats from this position
            count = 0
            pos = start
            while pos + plen <= buf_len and text[pos : pos + plen] == candidate:
                count += 1
                pos += plen
            if count >= 3:
                # Found repetition: keep up to 2 occurrences
                cut_at = start + plen * 2
                return text[:cut_at]
    # No repetition found, return as-is
    return text


def _stream_with_loop_detection(
    endpoint: str,
    payload: Dict[str, Any],
    timeout_s: int,
    expected_output_len: int,
    verbose: bool = False,
) -> str:
    """Stream SSE response. On runaway, truncate and return partial result."""
    r = requests.post(
        endpoint, json=payload,
        timeout=(10, timeout_s),
        stream=True,
    )
    r.raise_for_status()
    r.encoding = "utf-8"

    collected: List[str] = []
    total_content_len = 0
    had_output = False
    runaway = False

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
                if verbose:
                    sys.stdout.write(thinking)
                    sys.stdout.flush()

            # Process content tokens
            delta = delta_obj.get("content", "")
            if delta:
                had_output = True
                if verbose:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                collected.append(delta)
                total_content_len += len(delta)

            # Runaway: stop collecting, close connection
            if expected_output_len > 0 and total_content_len > expected_output_len * 3:
                runaway = True
                break
    finally:
        if verbose and had_output:
            sys.stdout.write("\n")
            sys.stdout.flush()
        r.close()

    raw_output = "".join(collected)
    if runaway:
        sys.stderr.write(
            f"[RUNAWAY] Content output {total_content_len} chars exceeded 3x expected "
            f"(~{expected_output_len}). Truncating at repetition...\n"
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
) -> str:
    global _streaming_available
    payload: Dict[str, Any] = {"messages": messages}
    if extra_payload:
        payload.update(extra_payload)
    payload["chat_template_kwargs"] = {"enable_thinking": False}
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


# ---------------------------
# Numbered I/O helpers
# ---------------------------

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
    use_stream: bool = True,
    verbose: bool = False,
    chunk_size: int = 10,
) -> List[str]:
    """Translate lines in small chunks (chunk_size each) to avoid model loops."""
    if not lines:
        return []

    def _call_backend(messages: List[Dict[str, str]], hint_len: int = 0,
                      override_params: Optional[Dict[str, Any]] = None) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, retry + 2):
            if verbose:
                sys.stderr.write(f"  [CALL] Attempt {attempt}/{retry + 1}, hint_len={hint_len}\n")
            try:
                return post_messages(
                    endpoint=endpoint,
                    messages=messages,
                    timeout_s=timeout_s,
                    extra_payload=extra_payload,
                    stream=use_stream,
                    expected_output_len=hint_len,
                    verbose=verbose,
                    override_params=override_params,
                )
            except Exception as e:
                last_err = e
                sys.stderr.write(
                    f"[RETRY] Request failed (attempt {attempt}/{retry + 1}): {e}\n"
                )
                if attempt <= retry:
                    time.sleep(retry_sleep_s)
        raise RuntimeError(f"Backend failed after {retry + 1} attempts: {last_err}")

    # Pass 1: direct literal translation only. No reasoning about context.
    # The user_prefix already specifies the language pair.
    pass1_prompt = (
        "You are a literal subtitle translator.\n"
        "\n"
        "RULES:\n"
        "- Translate each line INDEPENDENTLY. Do NOT reason about context between lines.\n"
        "- Do NOT think about what makes sense in the story. Do NOT infer meaning.\n"
        "- Just convert the words directly. Be fast and mechanical.\n"
        "- If a word looks strange or meaningless, translate it literally and wrap the "
        "ENTIRE line with ?? markers. Example: `[5] ??他说了奇怪的话??`\n"
        "- Output format: `[N] translated_text` for every input line, in order.\n"
        "- No commentary, no explanations, no merging lines."
    )
    # Append glossary from system_prompt if present
    if "---" in system_prompt:
        glossary_part = system_prompt.split("---", 1)[1]
        pass1_prompt += "\n\n---" + glossary_part

    def _translate_chunk(chunk: List[str], start_idx: int, chunk_label: str) -> List[str]:
        """Translate a single chunk with numbered I/O, retry on missing lines."""
        numbered = format_numbered_input(chunk, start_index=start_idx)
        msgs = [
            {"role": "system", "content": pass1_prompt},
            {"role": "user", "content": user_prefix + numbered},
        ]

        out_text = _call_backend(msgs, hint_len=len(numbered),
                                 override_params=None)
        translations, missing = parse_numbered_output(out_text, len(chunk), start_index=start_idx)

        # One repair attempt for missing lines
        if missing and len(missing) < len(chunk):
            sys.stderr.write(
                f"[REPAIR] {chunk_label}: {len(missing)} missing lines. Repairing...\n"
            )
            repair_lines = [f"[{i}] {chunk[i - start_idx]}" for i in missing]
            repair_msgs = [
                {"role": "system", "content": pass1_prompt},
                {
                    "role": "user",
                    "content": "Translate ONLY these lines, keeping the [N] format:\n"
                    + "\n".join(repair_lines),
                },
            ]
            try:
                repair_text = _call_backend(repair_msgs, hint_len=len("\n".join(repair_lines)),
                                                                              override_params=None)
                for m in _NUMBERED_LINE_RE.finditer(repair_text):
                    num = int(m.group(1))
                    idx = num - start_idx
                    if 0 <= idx < len(chunk) and translations[idx] is None:
                        translations[idx] = m.group(2)
            except Exception as e:
                sys.stderr.write(f"[WARN] {chunk_label} repair failed: {e}\n")

        return [t if t is not None else "" for t in translations]

    # === Pass 1: Chunked translation ===
    num_chunks = (len(lines) + chunk_size - 1) // chunk_size
    sys.stderr.write(f"[INFO] Translating {len(lines)} lines in {num_chunks} chunks of {chunk_size}...\n")

    if verbose:
        sys.stderr.write(f"  [DETAIL] System prompt: {len(system_prompt)} chars\n")

    translated: List[str] = []
    for ci in range(0, len(lines), chunk_size):
        chunk = lines[ci : ci + chunk_size]
        chunk_num = ci // chunk_size + 1
        sys.stderr.write(f"[CHUNK {chunk_num}/{num_chunks}] Lines {ci + 1}-{ci + len(chunk)}\n")
        result = _translate_chunk(chunk, start_idx=ci + 1, chunk_label=f"chunk_{chunk_num}")
        translated.extend(result)

    sys.stderr.write(f"[OK] All {len(lines)} lines translated.\n")
    return translated


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


# ---------------------------
# Series grouping
# ---------------------------

def group_files_by_series(
    file_paths: List[str],
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    verbose: bool = False,
) -> List[Tuple[str, List[str]]]:
    """Use the LLM to group subtitle files by series and sort by episode order."""
    if len(file_paths) <= 2:
        return [("All", file_paths)]

    # Build basename-to-path lookup (handle duplicate basenames)
    basenames = [os.path.basename(p) for p in file_paths]
    has_dupes = len(set(basenames)) < len(basenames)

    if has_dupes:
        # Use relative paths from common ancestor
        common = os.path.commonpath(file_paths)
        display_names = [os.path.relpath(p, common) for p in file_paths]
    else:
        display_names = basenames

    name_to_path: Dict[str, str] = {}
    for display, full in zip(display_names, file_paths):
        name_to_path[display] = full

    file_list_text = "\n".join(display_names)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a file organizer. Given subtitle filenames, group them "
                "by TV series / movie and sort each group by episode order.\n\n"
                "Return ONLY valid JSON in this exact format, no commentary:\n"
                '{"groups": [\n'
                '  {"series": "Series Name", "files": ["file1.srt", "file2.srt"]},\n'
                '  {"series": "Series Name 2", "files": ["file3.srt"]}\n'
                "]}\n\n"
                "Rules:\n"
                "- Every input filename must appear in exactly one group.\n"
                "- Sort files within each group by episode number.\n"
                '- If you cannot determine the series, put those files in a group named "Unknown".\n'
                "- Output raw JSON only. No markdown fences, no explanation."
            ),
        },
        {"role": "user", "content": f"Filenames:\n{file_list_text}"},
    ]

    try:
        sys.stderr.write(f"[INFO] Grouping {len(file_paths)} files by series...\n")
        if verbose:
            sys.stderr.write(f"  [DETAIL] Files to group:\n")
            for dn in display_names:
                sys.stderr.write(f"    {dn}\n")
        raw = post_messages(
            endpoint=endpoint,
            messages=messages,
            timeout_s=min(timeout_s, 60),
            extra_payload=extra_payload,
            stream=verbose,
            verbose=verbose,
        )
        if verbose:
            sys.stderr.write(f"  [DETAIL] Grouping response: {raw[:500]}\n")

        # Strip markdown fences if present
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in response")
        result = json.loads(json_match.group())

        groups_data = result.get("groups")
        if not isinstance(groups_data, list):
            raise ValueError("'groups' is not a list")

        groups: List[Tuple[str, List[str]]] = []
        matched: set = set()

        for g in groups_data:
            series = g.get("series", "Unknown")
            files = g.get("files", [])
            if not isinstance(files, list):
                continue
            paths_in_group: List[str] = []
            for f in files:
                f_str = str(f)
                if f_str in name_to_path and f_str not in matched:
                    paths_in_group.append(name_to_path[f_str])
                    matched.add(f_str)
            if paths_in_group:
                groups.append((str(series), paths_in_group))

        # Catch-all for unmatched files
        unmatched = [p for d, p in name_to_path.items() if d not in matched]
        if unmatched:
            groups.append(("Unknown", unmatched))

        # Log grouping results
        for series, paths in groups:
            sys.stderr.write(f"[GROUP] \"{series}\" ({len(paths)} files)\n")

        return groups

    except Exception as e:
        sys.stderr.write(f"[WARN] Series grouping failed: {e}. Treating all files as one group.\n")
        return [("All", file_paths)]


def extract_glossary(
    source_lines: List[str],
    translated_lines: List[str],
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    existing_glossary: str,
    verbose: bool = False,
) -> str:
    """Extract character names and key terms from a translated episode."""
    # Sample first 50 + last 50 lines to stay compact
    def _sample(lines: List[str], n: int = 50) -> str:
        if len(lines) <= n * 2:
            return "\n".join(lines)
        return "\n".join(lines[:n] + ["..."] + lines[-n:])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a translation glossary extractor. Given subtitle lines "
                "(source and translation), extract character names and key terms.\n\n"
                "Output format (one per line):\n"
                "SourceTerm → TranslatedTerm\n\n"
                "Rules:\n"
                "- Include character names, place names, and recurring proper nouns only.\n"
                "- Maximum 50 entries.\n"
                "- If a previous glossary is provided, merge with it (keep latest translation for conflicts).\n"
                "- Output ONLY glossary lines, no commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Previous glossary:\n{existing_glossary or '(none)'}\n\n"
                f"Source lines (sample):\n{_sample(source_lines)}\n\n"
                f"Translated lines (sample):\n{_sample(translated_lines)}"
            ),
        },
    ]

    try:
        raw = post_messages(
            endpoint=endpoint,
            messages=messages,
            timeout_s=min(timeout_s, 60),
            extra_payload=extra_payload,
            stream=verbose,
            verbose=verbose,
        )
        # Basic validation: should have at least one → symbol
        if "→" in raw or "->" in raw:
            if verbose:
                sys.stderr.write(f"  [DETAIL] Glossary extracted ({raw.count(chr(10)) + 1} entries)\n")
            return raw.strip()
        return existing_glossary
    except Exception as e:
        sys.stderr.write(f"[WARN] Glossary extraction failed: {e}\n")
        return existing_glossary


# ---------------------------
# Persistent vocabulary
# ---------------------------

_VOCAB_HEADER = (
    "# Vocabulary file for SRT subtitle translator\n"
    "# Format: SourceTerm → TranslatedTerm  (one per line)\n"
    "# Lines starting with # are comments and are ignored\n"
    "#\n"
    "# This file is loaded automatically at startup.\n"
    "# Learnt terms from each translation run are merged back here.\n"
    "# Feel free to edit, add, or remove entries between runs.\n"
    "#\n"
)


def _parse_vocab(text: str) -> Dict[str, str]:
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


def _format_vocab(entries: Dict[str, str]) -> str:
    """Format vocab entries as SourceTerm → TranslatedTerm lines."""
    return "\n".join(f"{src} → {tgt}" for src, tgt in sorted(entries.items()))


def _save_vocab(path: str, entries: Dict[str, str]) -> None:
    """Save vocab entries to file with header."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(_VOCAB_HEADER)
        f.write(_format_vocab(entries))
        f.write("\n")


# ---------------------------
# Proofread pass
# ---------------------------


def proofread_file(
    source_lines: List[str],
    translated_lines: List[str],
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    vocab_table: str,
    use_stream: bool,
    verbose: bool,
    retry: int,
    retry_sleep_s: float,
    source_lang: str,
    target_lang: str,
) -> List[str]:
    """Proofread translated lines using source+translation pairs and vocabulary context."""
    if not source_lines or not translated_lines:
        return translated_lines

    # Build side-by-side numbered pairs
    pairs = []
    for i, (src, tgt) in enumerate(zip(source_lines, translated_lines), start=1):
        pairs.append(f"[{i}] {src} → {tgt}")
    pairs_text = "\n".join(pairs)

    system_content = (
        f"You are a subtitle proofreader reviewing {source_lang} to {target_lang} translations.\n"
        "\n"
        "RULES:\n"
        "- Review each line's translation for accuracy and naturalness.\n"
        "- Fix lines flagged with ?? markers using surrounding context.\n"
        "- Fix inconsistent character names, mistranslations, and broken continuity.\n"
        "- Use the provided vocabulary table for correct term translations.\n"
        "- Output ALL lines as `[N] corrected_text` (the corrected translation only, not the source).\n"
        "- If a line's translation is correct, output it unchanged (but remove any ?? markers).\n"
        "- No commentary, no explanations, no merging lines."
    )

    if vocab_table:
        system_content += f"\n\nVocabulary reference:\n{vocab_table}"

    user_content = f"Review and correct these translations:\n{pairs_text}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    last_err: Optional[Exception] = None
    for attempt in range(1, retry + 2):
        if verbose:
            sys.stderr.write(f"  [PROOFREAD] Attempt {attempt}/{retry + 1}\n")
        try:
            raw = post_messages(
                endpoint=endpoint,
                messages=messages,
                timeout_s=timeout_s,
                extra_payload=extra_payload,
                stream=use_stream,
                expected_output_len=len(pairs_text),
                verbose=verbose,
            )
            results, missing = parse_numbered_output(raw, len(source_lines))

            # Merge: use proofread result where available, keep original for missing
            final = []
            for proofread, original in zip(results, translated_lines):
                if proofread is not None:
                    final.append(proofread)
                else:
                    final.append(original)

            if missing:
                sys.stderr.write(
                    f"[WARN] Proofread missed {len(missing)} lines, kept original translations.\n"
                )

            return final
        except Exception as e:
            last_err = e
            sys.stderr.write(
                f"[RETRY] Proofread failed (attempt {attempt}/{retry + 1}): {e}\n"
            )
            if attempt <= retry:
                time.sleep(retry_sleep_s)

    sys.stderr.write(
        f"[WARN] Proofread failed after {retry + 1} attempts: {last_err}. "
        "Keeping original translation.\n"
    )
    return translated_lines


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
    try:
        return template.format(source_lang=source_lang, target_lang=target_lang)
    except (KeyError, IndexError):
        return template


def expand_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for it in inputs:
        if any(ch in it for ch in "*?[]"):
            paths.extend(glob.glob(it))
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Translate .srt subtitle files via a backend API, with ASR error correction and series grouping."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Input .srt files, directories, or glob patterns (e.g. subs/ subs/*.srt movie.srt)",
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
        default=2,
        help="Retry count on failure (default: 2)",
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
        help="User message prefix (supports {source_lang} and {target_lang} placeholders)",
    )
    ap.add_argument(
        "--extra-payload",
        default="",
        help="Extra JSON fields for POST body (e.g. '{\"model\":\"local\",\"temperature\":0}')",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Lines per translation chunk in pass 1 (default: 10). Smaller = more stable, larger = faster.",
    )
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.3,
        help="Repetition penalty for pass 1 to prevent model loops (default: 1.3). Set 1.0 to disable.",
    )
    ap.add_argument(
        "--no-group",
        action="store_true",
        default=False,
        help="Disable automatic series grouping (treat all files independently)",
    )
    ap.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming mode (no loop detection, use blocking requests)",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show detailed progress and stream LLM responses to stderr",
    )
    ap.add_argument(
        "--proofread",
        action="store_true",
        default=False,
        help="Enable proofread pass after translation to fix ?? markers, inconsistencies, and errors",
    )
    ap.add_argument(
        "--vocab",
        default="vocab.txt",
        help="Path to vocabulary file (default: vocab.txt). Loaded at startup, updated with learnt terms after each run. Set to '' to disable.",
    )

    args = ap.parse_args()

    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        print("ERROR: No valid .srt files found.", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

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

    # Inject repetition_penalty into extra_payload for all LLM calls
    if args.repetition_penalty != 1.0:
        if extra_payload is None:
            extra_payload = {}
        extra_payload.setdefault("repetition_penalty", args.repetition_penalty)

    use_stream = not args.no_stream
    verbose = args.verbose

    # Load user vocabulary file
    user_vocab = ""
    vocab_entries: Dict[str, str] = {}
    if args.vocab:
        vocab_path = args.vocab
        if os.path.isfile(vocab_path):
            raw = _read_text_file(vocab_path).strip()
            if raw:
                vocab_entries = _parse_vocab(raw)
                user_vocab = _format_vocab(vocab_entries)
                sys.stderr.write(f"[INFO] Loaded vocabulary: {len(vocab_entries)} entries from {vocab_path}\n")
        elif vocab_path != "vocab.txt":
            # Only warn for non-default paths; default will be created after first run
            print(f"[WARN] Vocabulary file not found: {vocab_path}", file=sys.stderr)

    # Group files by series
    if args.no_group:
        groups: List[Tuple[str, List[str]]] = [("All", input_paths)]
    else:
        groups = group_files_by_series(
            input_paths, args.endpoint, args.timeout, extra_payload,
            verbose=verbose,
        )

    for group_idx, (series_name, group_paths) in enumerate(groups, 1):
        if len(groups) > 1:
            print(f"\n[SERIES {group_idx}/{len(groups)}] \"{series_name}\" ({len(group_paths)} files)")

        glossary = ""  # Reset per series
        series_results = []  # Accumulate for proofread pass

        for file_idx, path in enumerate(group_paths):
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

            # Build augmented prompt with glossary
            aug_prompt = system_prompt
            if glossary:
                aug_prompt += (
                    "\n\n---\n"
                    "Glossary from previous episodes in this series "
                    "(use these translations consistently):\n"
                    + glossary
                )
                if verbose:
                    sys.stderr.write(f"  [DETAIL] Glossary appended ({glossary.count(chr(10)) + 1} entries, {len(glossary)} chars)\n")

            if verbose:
                sys.stderr.write(f"  [DETAIL] {len(flat_lines)} text lines, {len(blocks)} SRT blocks\n")

            translated_lines = translate_lines_via_backend(
                lines=flat_lines,
                endpoint=args.endpoint,
                timeout_s=args.timeout,
                system_prompt=aug_prompt,
                user_prefix=user_prefix,
                extra_payload=extra_payload,
                retry=args.retry,
                retry_sleep_s=args.retry_sleep,
                use_stream=use_stream,
                verbose=verbose,
                chunk_size=args.chunk_size,
            )

            apply_translations(blocks, refs, translated_lines)

            out_srt = write_srt(blocks)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(out_srt)

            print(f"[INFO] Wrote: {out_path}")

            if args.proofread:
                series_results.append((out_path, flat_lines, translated_lines, blocks, refs))

            # Extract glossary: for next episode, or on last file for vocab persistence / proofread
            is_last = file_idx == len(group_paths) - 1
            need_glossary_for_next = not is_last and len(group_paths) > 1
            need_glossary_for_save = is_last and (args.vocab or args.proofread)
            if need_glossary_for_next or need_glossary_for_save:
                sys.stderr.write("[INFO] Extracting glossary...\n")
                glossary = extract_glossary(
                    flat_lines, translated_lines,
                    args.endpoint, args.timeout,
                    extra_payload, glossary,
                    verbose=verbose,
                )

        # Proofread pass: review each file with full vocabulary context
        if args.proofread and series_results:
            vocab_parts = []
            if user_vocab:
                vocab_parts.append(user_vocab)
            if glossary:
                vocab_parts.append(glossary)
            vocab_table = "\n".join(vocab_parts)

            print(f"\n[PROOFREAD] Reviewing {len(series_results)} file(s) in \"{series_name}\"...")
            for out_path, flat_lines, translated_lines, blocks, refs in series_results:
                sys.stderr.write(f"[PROOFREAD] {os.path.basename(out_path)}\n")
                proofread_result = proofread_file(
                    source_lines=flat_lines,
                    translated_lines=translated_lines,
                    endpoint=args.endpoint,
                    timeout_s=args.timeout,
                    extra_payload=extra_payload,
                    vocab_table=vocab_table,
                    use_stream=use_stream,
                    verbose=verbose,
                    retry=args.retry,
                    retry_sleep_s=args.retry_sleep,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                )
                apply_translations(blocks, refs, proofread_result)
                out_srt = write_srt(blocks)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(out_srt)
                print(f"[PROOFREAD] Updated: {out_path}")

        # Save learnt glossary to vocab file
        if args.vocab and glossary:
            new_entries = _parse_vocab(glossary)
            vocab_entries.update(new_entries)
            user_vocab = _format_vocab(vocab_entries)
            _save_vocab(args.vocab, vocab_entries)
            sys.stderr.write(f"[INFO] Saved {len(vocab_entries)} vocab entries to {args.vocab}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
