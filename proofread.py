#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proofread.py

Lightweight agent-based proofreader for translated SRT subtitles.
Replaces the old single-call thinking-mode proofreader with a multi-step
pipeline that externalises reasoning into focused LLM calls:

  Step 0  (per-file)   Context Building — scene/character summary
  Step 1  (per-chunk)  Confidence Scoring — rate each line HIGH/MEDIUM/LOW
  Step 2  (per-chunk)  Issue Analysis — free-form reasoning about flagged lines
  Step 3  (per-chunk)  Apply Corrections — clean [N] corrected_text output
  Step 4  (per-chunk)  Vocab Extraction — learn new terms

All steps run with enable_thinking=False.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from main import (
    post_messages,
    parse_srt,
    write_srt,
    SrtBlock,
    parse_numbered_output,
    build_line_mapping,
    apply_translations,
    read_text_file,
    parse_vocab,
    format_vocab,
    save_vocab,
    expand_inputs,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceScore:
    level: str        # HIGH, MEDIUM, LOW
    reason: str       # empty for HIGH


_CONFIDENCE_RE = re.compile(
    r"^\[(\d+)\]\s*(HIGH|MEDIUM|LOW)(?:\s*[|:]\s*(.*))?$",
    re.MULTILINE | re.IGNORECASE,
)

_VOCAB_LINE_RE = re.compile(r"^(.+?)\s*(?:→|->)\s*(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_context_prompt(
    source_lines: List[str],
    translated_lines: List[str],
    source_lang: str,
    target_lang: str,
) -> Tuple[str, str]:
    """Return (system, user) messages for Step 0: file-level context building."""
    # Sample: first 50 + last 50 pairs (no overlap)
    n = len(source_lines)
    if n <= 100:
        sample_indices = list(range(n))
    else:
        sample_indices = list(range(50)) + list(range(n - 50, n))

    pairs = []
    for idx in sample_indices:
        i = idx + 1
        pairs.append(f"[{i}] {source_lines[idx]} → {translated_lines[idx]}")
    pairs_text = "\n".join(pairs)

    system = (
        f"You are a subtitle analyst. Given {source_lang} → {target_lang} subtitle pairs, "
        "provide a brief summary to help a proofreader understand the context.\n"
    )

    user = (
        "Summarise these subtitle pairs:\n"
        "1. What is this scene/episode about? (1-2 sentences)\n"
        "2. List all character names mentioned (source name → translated name)\n"
        "3. Overall tone (formal, casual, comedic, dramatic)\n"
        "4. Any recurring terms or phrases worth noting\n"
        "\n"
        "Keep the summary under 200 words.\n"
        "\n"
        f"{pairs_text}"
    )
    return system, user


def _build_confidence_prompt(
    source_lines: List[str],
    translated_lines: List[str],
    start_index: int,
    vocab_table: str,
    file_context: str,
    context_before: List[str],
    source_lang: str,
    target_lang: str,
) -> Tuple[str, str]:
    """Return (system, user) messages for Step 1: confidence scoring."""
    system = (
        f"You are a {source_lang} to {target_lang} subtitle translation reviewer.\n"
        "Rate each translation line's confidence.\n"
        "\n"
        "Output format (one line per input, in order):\n"
        "[N] HIGH/MEDIUM/LOW | reason (if not HIGH)\n"
        "\n"
        "Definitions:\n"
        "- HIGH = translation is correct, natural, and consistent with vocabulary\n"
        "- MEDIUM = understandable but awkward phrasing, or a specific word seems off\n"
        "- LOW = likely wrong, missing meaning, has ?? marker, or contradicts vocabulary\n"
        "\n"
        "Output ONLY the rating lines. No other text."
    )

    parts = []
    if file_context:
        parts.append(f"FILE CONTEXT:\n{file_context}")
    if vocab_table:
        parts.append(f"VOCABULARY:\n{vocab_table}")
    if context_before:
        ctx = "\n".join(f"  {line}" for line in context_before)
        parts.append(f"PREVIOUS (already proofread):\n{ctx}")

    pairs = []
    for i, (src, tgt) in enumerate(zip(source_lines, translated_lines)):
        pairs.append(f"[{start_index + i}] {src} → {tgt}")

    parts.append("Lines to rate:\n" + "\n".join(pairs))
    user = "\n\n".join(parts)
    return system, user


def _build_analysis_prompt(
    source_lines: List[str],
    translated_lines: List[str],
    scores: List[ConfidenceScore],
    start_index: int,
    vocab_table: str,
    file_context: str,
    context_before: List[str],
    context_after: List[str],
    source_lang: str,
    target_lang: str,
) -> Tuple[str, str]:
    """Return (system, user) messages for Step 2: issue analysis."""
    system = (
        f"You are a {source_lang} to {target_lang} subtitle proofreader.\n"
        "Lines marked with ⚠ (MEDIUM) or ❌ (LOW) have potential issues.\n"
        "\n"
        "For EACH flagged line, explain:\n"
        f"1. What the {source_lang} source text means in context\n"
        "2. What is wrong with the current translation\n"
        "3. What the correct translation should be and why\n"
        "\n"
        "Be thorough but concise. Focus on accuracy, naturalness, and consistency."
    )

    parts = []
    if file_context:
        parts.append(f"FILE CONTEXT:\n{file_context}")
    if vocab_table:
        parts.append(f"VOCABULARY:\n{vocab_table}")
    if context_before:
        ctx = "\n".join(f"  {line}" for line in context_before)
        parts.append(f"PREVIOUS (already proofread, for continuity):\n{ctx}")

    # Build lines with flag markers
    lines_block = []
    for i, (src, tgt, score) in enumerate(zip(source_lines, translated_lines, scores)):
        idx = start_index + i
        line = f"[{idx}] {src} → {tgt}"
        if score.level == "LOW":
            line += f"  ❌ LOW: {score.reason}" if score.reason else "  ❌ LOW"
        elif score.level == "MEDIUM":
            line += f"  ⚠ MEDIUM: {score.reason}" if score.reason else "  ⚠ MEDIUM"
        lines_block.append(line)

    parts.append("Lines to review:\n" + "\n".join(lines_block))

    if context_after:
        ctx = "\n".join(f"  {line}" for line in context_after)
        parts.append(f"UPCOMING (for reference):\n{ctx}")

    user = "\n\n".join(parts)
    return system, user


def _build_correction_prompt(
    translated_lines: List[str],
    analysis: str,
    start_index: int,
) -> Tuple[str, str]:
    """Return (system, user) messages for Step 3: apply corrections."""
    system = (
        "Apply the corrections described in the analysis below.\n"
        "Output ALL lines as [N] corrected_text.\n"
        "For lines with no issues, output the original translation unchanged.\n"
        "Remove any ?? markers when you can determine the correct translation.\n"
        "\n"
        "Output ONLY [N] corrected_text lines. No other text."
    )

    lines = []
    for i, tgt in enumerate(translated_lines):
        lines.append(f"[{start_index + i}] {tgt}")
    originals_text = "\n".join(lines)

    user = (
        f"ANALYSIS OF ISSUES:\n{analysis}\n"
        "\n"
        f"Original translations:\n{originals_text}"
    )
    return system, user


def _build_vocab_prompt(
    source_lines: List[str],
    corrected_lines: List[str],
    start_index: int,
    vocab_table: str,
    source_lang: str,
    target_lang: str,
) -> Tuple[str, str]:
    """Return (system, user) messages for Step 4: vocab extraction."""
    system = (
        f"Extract new character names, place names, and key recurring terms "
        f"from these {source_lang} → {target_lang} subtitle pairs.\n"
        "\n"
        f"Output format (one entry per line):\n"
        f"{source_lang}Term → {target_lang}Term\n"
        "\n"
        "Rules:\n"
        "- Only proper nouns, character names, place names, and important recurring terms\n"
        "- Maximum 10 entries\n"
        "- Do NOT repeat entries already in the existing vocabulary\n"
        "- Output ONLY vocabulary lines. No commentary."
    )

    parts = []
    if vocab_table:
        parts.append(f"Existing vocabulary (do NOT repeat these):\n{vocab_table}")

    pairs = []
    for i, (src, cor) in enumerate(zip(source_lines, corrected_lines)):
        pairs.append(f"[{start_index + i}] {src} → {cor}")
    parts.append("Pairs:\n" + "\n".join(pairs))

    user = "\n\n".join(parts)
    return system, user


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def _parse_confidence_scores(
    raw: str,
    expected_count: int,
    start_index: int = 1,
) -> List[ConfidenceScore]:
    """Parse Step 1 output into ConfidenceScore list. Missing lines default to MEDIUM."""
    found: Dict[int, ConfidenceScore] = {}
    for m in _CONFIDENCE_RE.finditer(raw):
        num = int(m.group(1))
        level = m.group(2).upper()
        reason = (m.group(3) or "").strip()
        found[num] = ConfidenceScore(level=level, reason=reason)

    result = []
    for i in range(expected_count):
        idx = start_index + i
        if idx in found:
            result.append(found[idx])
        else:
            # Default to MEDIUM so we don't skip unscored lines
            result.append(ConfidenceScore(level="MEDIUM", reason="not scored"))
    return result


def _parse_vocab_entries(raw: str) -> Dict[str, str]:
    """Parse Step 4 output into {source_term: translated_term} dict."""
    entries: Dict[str, str] = {}
    for m in _VOCAB_LINE_RE.finditer(raw):
        src = m.group(1).strip()
        tgt = m.group(2).strip()
        if src and tgt and not src.startswith("#"):
            entries[src] = tgt
    return entries


# ---------------------------------------------------------------------------
# Step executors
# ---------------------------------------------------------------------------

def _call_llm(
    system: str,
    user: str,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    expected_output_len: int = 0,
) -> str:
    """Thin wrapper around post_messages with thinking disabled."""
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
    )


def _build_file_context(
    source_lines: List[str],
    translated_lines: List[str],
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    source_lang: str,
    target_lang: str,
) -> str:
    """Step 0: Build file-level context summary."""
    if not source_lines:
        return ""

    system, user = _build_context_prompt(
        source_lines, translated_lines, source_lang, target_lang,
    )
    try:
        raw = _call_llm(system, user, endpoint, timeout_s, extra_payload,
                         use_stream, verbose, expected_output_len=800)
        return raw.strip()
    except Exception as e:
        sys.stderr.write(f"[WARN] Step 0 (context building) failed: {e}. Continuing without context.\n")
        return ""


def _score_confidence(
    source_lines: List[str],
    translated_lines: List[str],
    start_index: int,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    vocab_table: str,
    file_context: str,
    context_before: List[str],
    source_lang: str,
    target_lang: str,
) -> List[ConfidenceScore]:
    """Step 1: Confidence scoring for a (possibly wider) window of lines."""
    system, user = _build_confidence_prompt(
        source_lines, translated_lines, start_index,
        vocab_table, file_context, context_before,
        source_lang, target_lang,
    )
    try:
        raw = _call_llm(system, user, endpoint, timeout_s, extra_payload,
                         use_stream, verbose,
                         expected_output_len=len(source_lines) * 30)
        return _parse_confidence_scores(raw, len(source_lines), start_index)
    except Exception as e:
        sys.stderr.write(f"[WARN] Step 1 (confidence scoring) failed: {e}. Treating all as MEDIUM.\n")
        return [ConfidenceScore(level="MEDIUM", reason="scoring failed") for _ in source_lines]


def _analyze_issues(
    source_lines: List[str],
    translated_lines: List[str],
    scores: List[ConfidenceScore],
    start_index: int,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    vocab_table: str,
    file_context: str,
    context_before: List[str],
    context_after: List[str],
    source_lang: str,
    target_lang: str,
) -> str:
    """Step 2: Free-form issue analysis for the chunk. Returns analysis text."""
    system, user = _build_analysis_prompt(
        source_lines, translated_lines, scores, start_index,
        vocab_table, file_context, context_before, context_after,
        source_lang, target_lang,
    )
    try:
        raw = _call_llm(system, user, endpoint, timeout_s, extra_payload,
                         use_stream, verbose,
                         expected_output_len=len(source_lines) * 100)
        return raw.strip()
    except Exception as e:
        sys.stderr.write(f"[WARN] Step 2 (issue analysis) failed: {e}. Skipping analysis.\n")
        return ""


def _apply_corrections(
    translated_lines: List[str],
    analysis: str,
    start_index: int,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    retry: int,
    retry_sleep_s: float,
) -> List[str]:
    """Step 3: Apply corrections using analysis. Returns corrected lines."""
    system, user = _build_correction_prompt(
        translated_lines, analysis, start_index,
    )
    expected_count = len(translated_lines)

    last_err: Optional[Exception] = None
    for attempt in range(1, retry + 2):
        try:
            raw = _call_llm(system, user, endpoint, timeout_s, extra_payload,
                             use_stream, verbose,
                             expected_output_len=sum(len(l) for l in translated_lines) + expected_count * 5)
            results, missing = parse_numbered_output(raw, expected_count, start_index)

            # Merge: use correction where available, keep original for missing
            corrected = []
            for i, (result, original) in enumerate(zip(results, translated_lines)):
                if result is not None:
                    corrected.append(result)
                else:
                    corrected.append(original)

            if missing and attempt <= retry:
                # Repair attempt: ask for just the missing lines
                sys.stderr.write(
                    f"  [REPAIR] {len(missing)} lines missing, retrying...\n"
                )
                repair_system = (
                    "Some line numbers were missing from your previous output. "
                    "Output ONLY the missing lines as [N] corrected_text."
                )
                missing_lines = "\n".join(
                    f"[{idx}] {translated_lines[idx - start_index]}"
                    for idx in missing if start_index <= idx < start_index + len(translated_lines)
                )
                repair_user = (
                    f"ANALYSIS:\n{analysis}\n\n"
                    f"Missing lines (output corrections for these):\n{missing_lines}"
                )
                try:
                    repair_raw = _call_llm(
                        repair_system, repair_user, endpoint, timeout_s,
                        extra_payload, use_stream, verbose,
                        expected_output_len=len(missing) * 50,
                    )
                    repair_results, _ = parse_numbered_output(
                        repair_raw, expected_count, start_index,
                    )
                    for i, r in enumerate(repair_results):
                        if r is not None and corrected[i] == translated_lines[i]:
                            corrected[i] = r
                except Exception:
                    pass  # repair failed, keep what we have

            return corrected

        except Exception as e:
            last_err = e
            sys.stderr.write(
                f"  [RETRY] Step 3 failed (attempt {attempt}/{retry + 1}): {e}\n"
            )
            if attempt <= retry:
                time.sleep(retry_sleep_s)

    sys.stderr.write(
        f"[WARN] Step 3 (corrections) failed after {retry + 1} attempts: {last_err}. "
        "Keeping originals for this chunk.\n"
    )
    return list(translated_lines)


def _extract_vocab(
    source_lines: List[str],
    corrected_lines: List[str],
    start_index: int,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    vocab_table: str,
    source_lang: str,
    target_lang: str,
) -> Dict[str, str]:
    """Step 4: Extract new vocab entries from corrected chunk."""
    system, user = _build_vocab_prompt(
        source_lines, corrected_lines, start_index,
        vocab_table, source_lang, target_lang,
    )
    try:
        raw = _call_llm(system, user, endpoint, timeout_s, extra_payload,
                         use_stream, verbose, expected_output_len=500)
        entries = _parse_vocab_entries(raw)
        if entries and verbose:
            sys.stderr.write(f"  [VOCAB] Learned {len(entries)} new entries\n")
        return entries
    except Exception as e:
        sys.stderr.write(f"[WARN] Step 4 (vocab extraction) failed: {e}. Skipping.\n")
        return {}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def proofread_file(
    source_lines: List[str],
    translated_lines: List[str],
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    vocab_entries: Dict[str, str],
    use_stream: bool,
    verbose: bool,
    retry: int,
    retry_sleep_s: float,
    source_lang: str,
    target_lang: str,
    chunk_size: int = 20,
    context_window: int = 5,
    update_vocab: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Proofread translated lines using a multi-step agent pipeline.

    Returns (corrected_lines, updated_vocab_entries).
    """
    if not source_lines or not translated_lines:
        return list(translated_lines), vocab_entries

    n = min(len(source_lines), len(translated_lines))
    source_lines = source_lines[:n]
    translated_lines = translated_lines[:n]

    # Step 0: Build file context (once per file)
    sys.stderr.write(f"  [Step 0] Building file context...\n")
    file_context = _build_file_context(
        source_lines, translated_lines,
        endpoint, timeout_s, extra_payload, use_stream, verbose,
        source_lang, target_lang,
    )
    if verbose and file_context:
        sys.stderr.write(f"  [Step 0] Context ({len(file_context)} chars): {file_context[:120]}...\n")

    corrected = list(translated_lines)
    vocab_table = format_vocab(vocab_entries)
    total_chunks = (n + chunk_size - 1) // chunk_size

    for chunk_idx, chunk_start in enumerate(range(0, n, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_num = chunk_idx + 1
        sys.stderr.write(f"  [Chunk {chunk_num}/{total_chunks}] Lines {chunk_start + 1}-{chunk_end}\n")

        # Sliding context
        ctx_start = max(0, chunk_start - context_window)
        ctx_before = corrected[ctx_start:chunk_start]
        ctx_after = translated_lines[chunk_end:min(n, chunk_end + 3)]

        # Step 1: Confidence scoring with adaptive 2x window
        score_lookback = chunk_size // 2
        score_lookahead = chunk_size // 2
        score_start = max(0, chunk_start - score_lookback)
        score_end = min(n, chunk_end + score_lookahead)

        # Use corrected lines for lookback (already proofread) + originals for lookahead
        score_translations = corrected[score_start:chunk_start] + translated_lines[chunk_start:score_end]

        sys.stderr.write(f"  [Step 1] Scoring confidence (window: lines {score_start + 1}-{score_end})...\n")
        all_scores = _score_confidence(
            source_lines[score_start:score_end],
            score_translations,
            start_index=score_start + 1,
            endpoint=endpoint,
            timeout_s=timeout_s,
            extra_payload=extra_payload,
            use_stream=use_stream,
            verbose=verbose,
            vocab_table=vocab_table,
            file_context=file_context,
            context_before=ctx_before,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        # Extract scores for just our chunk within the wider scoring window
        offset = chunk_start - score_start
        chunk_scores = all_scores[offset:offset + (chunk_end - chunk_start)]

        # Report scores
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for s in chunk_scores:
            counts[s.level] = counts.get(s.level, 0) + 1
        sys.stderr.write(
            f"  [Step 1] Scores: {counts['HIGH']} HIGH, {counts['MEDIUM']} MEDIUM, {counts['LOW']} LOW\n"
        )

        # Optimisation: skip if all HIGH
        if all(s.level == "HIGH" for s in chunk_scores):
            sys.stderr.write(f"  [Step 1] All HIGH — skipping correction for this chunk.\n")
            # Still do vocab extraction if enabled
            if update_vocab:
                new_vocab = _extract_vocab(
                    source_lines[chunk_start:chunk_end],
                    corrected[chunk_start:chunk_end],
                    start_index=chunk_start + 1,
                    endpoint=endpoint,
                    timeout_s=timeout_s,
                    extra_payload=extra_payload,
                    use_stream=use_stream,
                    verbose=verbose,
                    vocab_table=vocab_table,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                if new_vocab:
                    vocab_entries.update(new_vocab)
                    vocab_table = format_vocab(vocab_entries)
            continue

        # Step 2: Issue analysis
        sys.stderr.write(f"  [Step 2] Analysing issues...\n")
        analysis = _analyze_issues(
            source_lines[chunk_start:chunk_end],
            translated_lines[chunk_start:chunk_end],
            chunk_scores,
            start_index=chunk_start + 1,
            endpoint=endpoint,
            timeout_s=timeout_s,
            extra_payload=extra_payload,
            use_stream=use_stream,
            verbose=verbose,
            vocab_table=vocab_table,
            file_context=file_context,
            context_before=ctx_before,
            context_after=ctx_after,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        # Step 3: Apply corrections
        sys.stderr.write(f"  [Step 3] Applying corrections...\n")
        chunk_corrected = _apply_corrections(
            translated_lines[chunk_start:chunk_end],
            analysis,
            start_index=chunk_start + 1,
            endpoint=endpoint,
            timeout_s=timeout_s,
            extra_payload=extra_payload,
            use_stream=use_stream,
            verbose=verbose,
            retry=retry,
            retry_sleep_s=retry_sleep_s,
        )
        corrected[chunk_start:chunk_end] = chunk_corrected

        # Step 4: Vocab extraction
        if update_vocab:
            sys.stderr.write(f"  [Step 4] Extracting vocabulary...\n")
            new_vocab = _extract_vocab(
                source_lines[chunk_start:chunk_end],
                chunk_corrected,
                start_index=chunk_start + 1,
                endpoint=endpoint,
                timeout_s=timeout_s,
                extra_payload=extra_payload,
                use_stream=use_stream,
                verbose=verbose,
                vocab_table=vocab_table,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            if new_vocab:
                vocab_entries.update(new_vocab)
                vocab_table = format_vocab(vocab_entries)

    return corrected, vocab_entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Proofread translated SRT subtitles using a multi-step agent pipeline."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Source .srt files, directories, or glob patterns",
    )
    ap.add_argument(
        "--endpoint",
        default="http://127.0.0.1:5000/v1/chat/completions",
        help="Backend POST endpoint URL",
    )
    ap.add_argument(
        "--out-dir",
        default="out",
        help="Directory containing translated files to proofread (default: out)",
    )
    ap.add_argument(
        "--suffix",
        default=".zh.srt",
        help="Output filename suffix (default: .zh.srt)",
    )
    ap.add_argument(
        "--source-lang",
        default="Japanese",
        help="Source language name (default: Japanese)",
    )
    ap.add_argument(
        "--target-lang",
        default="Simplified Chinese",
        help="Target language name (default: Simplified Chinese)",
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
        help="Retry count per chunk correction step (default: 2)",
    )
    ap.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="Seconds between retries (default: 1.0)",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Lines per correction chunk (default: 20)",
    )
    ap.add_argument(
        "--context-window",
        type=int,
        default=5,
        help="Context lines from previous chunk (default: 5)",
    )
    ap.add_argument(
        "--vocab",
        default="vocab.txt",
        help="Vocabulary file path (default: vocab.txt)",
    )
    ap.add_argument(
        "--no-vocab-update",
        action="store_true",
        default=False,
        help="Disable vocab learning (read-only mode for vocab)",
    )
    ap.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming mode",
    )
    ap.add_argument(
        "--extra-payload",
        default="",
        help="Extra JSON fields for POST body (e.g. '{\"model\":\"local\",\"temperature\":0}')",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show detailed progress and LLM responses",
    )

    args = ap.parse_args()

    # Resolve input files
    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        print("ERROR: No valid .srt files found.", file=sys.stderr)
        return 2

    # Parse extra payload
    extra_payload: Optional[Dict[str, Any]] = None
    if args.extra_payload.strip():
        try:
            extra_payload = json.loads(args.extra_payload)
            if not isinstance(extra_payload, dict):
                raise ValueError("extra payload must be a JSON object")
        except Exception as e:
            print(f"ERROR: --extra-payload is not valid JSON object: {e}", file=sys.stderr)
            return 2

    use_stream = not args.no_stream
    update_vocab = not args.no_vocab_update

    # Load vocabulary
    vocab_entries: Dict[str, str] = {}
    if args.vocab and os.path.isfile(args.vocab):
        raw = read_text_file(args.vocab).strip()
        if raw:
            vocab_entries = parse_vocab(raw)
            sys.stderr.write(f"[INFO] Loaded vocabulary: {len(vocab_entries)} entries from {args.vocab}\n")

    # Process each file
    for path in input_paths:
        base = os.path.basename(path)
        stem, _ext = os.path.splitext(base)
        out_path = os.path.join(args.out_dir, stem + args.suffix)

        if not os.path.isfile(out_path):
            sys.stderr.write(f"[WARN] Translated file not found, skipping: {out_path}\n")
            continue

        print(f"[PROOFREAD] {path}")

        # Parse source SRT
        source_content = read_text_file(path)
        source_blocks = parse_srt(source_content)
        if not source_blocks:
            sys.stderr.write(f"[WARN] No blocks in source file, skipping: {path}\n")
            continue
        source_lines, _ = build_line_mapping(source_blocks)

        # Parse translated SRT
        trans_content = read_text_file(out_path)
        trans_blocks = parse_srt(trans_content)
        if not trans_blocks:
            sys.stderr.write(f"[WARN] No blocks in translated file, skipping: {out_path}\n")
            continue
        translated_lines, trans_refs = build_line_mapping(trans_blocks)

        # Align line counts
        n = min(len(source_lines), len(translated_lines))
        if len(source_lines) != len(translated_lines):
            sys.stderr.write(
                f"[WARN] Line count mismatch: source={len(source_lines)} "
                f"translated={len(translated_lines)}. Using first {n} lines.\n"
            )

        # Run the agent pipeline (with whole-file error recovery)
        try:
            corrected, vocab_entries = proofread_file(
                source_lines=source_lines[:n],
                translated_lines=translated_lines[:n],
                endpoint=args.endpoint,
                timeout_s=args.timeout,
                extra_payload=extra_payload,
                vocab_entries=vocab_entries,
                use_stream=use_stream,
                verbose=args.verbose,
                retry=args.retry,
                retry_sleep_s=args.retry_sleep,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                chunk_size=args.chunk_size,
                context_window=args.context_window,
                update_vocab=update_vocab,
            )

            # Apply results and overwrite translated file
            apply_translations(trans_blocks, trans_refs[:n], corrected)
            out_srt = write_srt(trans_blocks)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(out_srt)
            print(f"[PROOFREAD] Updated: {out_path}")

            # Save vocab after each file
            if update_vocab and args.vocab and vocab_entries:
                save_vocab(args.vocab, vocab_entries)
                sys.stderr.write(f"[INFO] Saved {len(vocab_entries)} vocab entries to {args.vocab}\n")
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to proofread {path}: {e}. File left untouched.\n")
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
