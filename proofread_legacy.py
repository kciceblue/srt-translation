#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proofread.py

Two-pass proofreader for translated SRT subtitles.

Pass 1 — Vocab Replace & Flag (mechanical, per-chunk):
  Step 1.1  Strict vocab replacement — LLM replaces terms exactly per vocab sheet
  Step 1.2  Confidence scoring — rate each chunk line HIGH/MEDIUM/LOW (on vocab-replaced text)

Pass 2 — Per-Line Correction (only MEDIUM/LOW lines):
  Step 2    Analyze + correct each flagged line individually (one LLM call per line,
            with context window ±50 lines). Hard output limit with retry on runaway.
  Step 2.3  Generate new vocab sheet — one call per file
  Step 2.4  Final vocab replacement — apply new vocab sheet across ALL lines

Step 0 (per-file): Context building — scene/character summary, used in both passes.
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


# Match "[N] LEVEL" with optional reason after any separator (|, :, -, space, tab)
# Note: \s is NOT used in the separator class — it matches \n which eats the next line
_CONFIDENCE_RE = re.compile(
    r"^\[(\d+)\]\s*(HIGH|MEDIUM|LOW)\b[|:\- –—\t]*(.*?)$",
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
    """Return (system, user) messages for Step 1.1: confidence scoring."""
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


def _build_vocab_replace_prompt(
    translated_lines: List[str],
    start_index: int,
    vocab_table: str,
    source_lang: str,
    target_lang: str,
) -> Tuple[str, str]:
    """Return (system, user) messages for Step 1.2 / 2.4: strict vocab replacement."""
    system = (
        f"You are a {target_lang} subtitle editor. You apply a vocabulary sheet to translations.\n"
        "\n"
        "RULES:\n"
        "- Replace terms in the translations EXACTLY according to the vocabulary sheet below.\n"
        "- Change ONLY vocabulary terms. Do NOT rephrase, fix grammar, or improve style.\n"
        "- If a vocabulary entry's left side appears in the translation text, replace it with the right side.\n"
        f"- If a vocabulary entry's left side is in {source_lang}, find the corresponding "
        "translated word in the line and replace it with the vocabulary's right side.\n"
        "- Output ALL lines as [N] text, even if unchanged.\n"
        "\n"
        "Output ONLY [N] text lines. No other text."
    )

    parts = []
    parts.append(f"VOCABULARY SHEET:\n{vocab_table}")

    lines = []
    for i, tgt in enumerate(translated_lines):
        lines.append(f"[{start_index + i}] {tgt}")
    parts.append("Lines to process:\n" + "\n".join(lines))

    user = "\n\n".join(parts)
    return system, user


def _build_line_correction_prompt(
    line_index: int,
    source_line: str,
    translated_line: str,
    score: ConfidenceScore,
    vocab_table: str,
    file_context: str,
    full_source: List[str],
    full_translated: List[str],
    source_lang: str,
    target_lang: str,
) -> Tuple[str, str]:
    """Return (system, user) messages for per-line analysis + correction."""
    system = (
        f"You are a {source_lang} to {target_lang} subtitle proofreader.\n"
        "You will correct ONE specific flagged line. The surrounding context is "
        "provided ONLY for reference — do NOT retranslate or revise any other line.\n"
        "\n"
        "Output format (exactly these 2 lines):\n"
        f"REASON: brief explanation of the issue (1 sentence)\n"
        f"[{line_index}] corrected_text\n"
        "\n"
        "Rules:\n"
        "- Fix ONLY the flagged line. Do not output any other line numbers.\n"
        "- Use the surrounding context to understand the scene and choose the right translation.\n"
        "- Remove ?? markers when you determine the correct translation.\n"
        "- If the current translation is actually correct, output it unchanged.\n"
        "- Output ONLY the 2 lines above. Nothing else."
    )

    parts = []
    if file_context:
        parts.append(f"FILE SUMMARY:\n{file_context}")
    if vocab_table:
        parts.append(f"VOCABULARY:\n{vocab_table}")

    # Context window around the flagged line (±50 lines, or whole file if small)
    ctx_radius = 50
    n = len(full_source)
    line_pos = line_index - 1  # 0-based
    if n <= ctx_radius * 2 + 20:
        ctx_start, ctx_end = 0, n
    else:
        ctx_start = max(0, line_pos - ctx_radius)
        ctx_end = min(n, line_pos + ctx_radius + 1)

    ctx_pairs = []
    for i in range(ctx_start, ctx_end):
        ctx_pairs.append(f"[{i + 1}] {full_source[i]} → {full_translated[i]}")
    parts.append("CONTEXT:\n" + "\n".join(ctx_pairs))

    # The flagged line
    flag = "❌ LOW" if score.level == "LOW" else "⚠ MEDIUM"
    reason = f": {score.reason}" if score.reason else ""
    parts.append(
        f"FLAGGED LINE:\n"
        f"[{line_index}] {source_line} → {translated_line}  {flag}{reason}"
    )

    user = "\n\n".join(parts)
    return system, user


def _build_full_vocab_gen_prompt(
    source_lines: List[str],
    corrected_lines: List[str],
    source_lang: str,
    target_lang: str,
) -> Tuple[str, str]:
    """Return (system, user) messages for Step 2.3: per-file vocab generation."""
    system = (
        f"You extract proper nouns from {source_lang} → {target_lang} subtitles.\n"
        "\n"
        "ONLY output entries that are:\n"
        "- Character names (people, nicknames, honorific+name pairs)\n"
        "- Place names (locations, buildings, countries, fictional places)\n"
        "- Organization names (companies, schools, groups, factions)\n"
        "- Fictional/in-universe terms (made-up words, special abilities, titles)\n"
        "\n"
        "NEVER output:\n"
        "- Common nouns, verbs, adjectives, adverbs\n"
        "- Everyday phrases or expressions\n"
        "- Any word found in a standard dictionary with the same meaning\n"
        "- Sentence fragments or multi-word phrases that aren't proper nouns\n"
        "\n"
        f"Format: one entry per line, exactly: {source_lang}Term → {target_lang}Term\n"
        "Maximum 30 entries. Fewer is better — only include what you are certain about.\n"
        "Output ONLY the vocabulary lines. No commentary, no headers, no numbering, no blank lines."
    )

    # Sample: first 50 + last 50 pairs
    n = len(source_lines)
    if n <= 100:
        sample_indices = list(range(n))
    else:
        sample_indices = list(range(50)) + list(range(n - 50, n))

    pairs = []
    for idx in sample_indices:
        pairs.append(f"{source_lines[idx]} → {corrected_lines[idx]}")

    user = "Subtitle pairs:\n" + "\n".join(pairs)
    return system, user


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def _parse_confidence_scores(
    raw: str,
    expected_count: int,
    start_index: int = 1,
) -> List[ConfidenceScore]:
    """Parse Step 1.1 output into ConfidenceScore list. Missing lines default to MEDIUM."""
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
            result.append(ConfidenceScore(level="MEDIUM", reason="not scored"))
    return result


def _parse_vocab_entries(raw: str) -> Dict[str, str]:
    """Parse vocab output into {source_term: translated_term} dict."""
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
    runaway_multiplier: float = 3.0,
    raise_on_runaway: bool = False,
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
        runaway_multiplier=runaway_multiplier,
        raise_on_runaway=raise_on_runaway,
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
    """Step 1.1: Confidence scoring for a (possibly wider) window of lines."""
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
        sys.stderr.write(f"[WARN] Step 1.1 (confidence scoring) failed: {e}. Treating all as MEDIUM.\n")
        return [ConfidenceScore(level="MEDIUM", reason="scoring failed") for _ in source_lines]


def _apply_vocab_replacement(
    translated_lines: List[str],
    start_index: int,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    vocab_table: str,
    source_lang: str,
    target_lang: str,
) -> List[str]:
    """Step 1.2 / 2.4: Strict LLM-based vocab replacement. Returns replaced lines."""
    if not vocab_table:
        return list(translated_lines)

    system, user = _build_vocab_replace_prompt(
        translated_lines, start_index, vocab_table, source_lang, target_lang,
    )
    try:
        raw = _call_llm(system, user, endpoint, timeout_s, extra_payload,
                         use_stream, verbose,
                         expected_output_len=sum(len(l) for l in translated_lines) + len(translated_lines) * 5)
        results, missing = parse_numbered_output(raw, len(translated_lines), start_index)

        replaced = []
        for i, (result, original) in enumerate(zip(results, translated_lines)):
            replaced.append(result if result is not None else original)
        return replaced
    except Exception as e:
        sys.stderr.write(f"[WARN] Vocab replacement failed: {e}. Keeping originals.\n")
        return list(translated_lines)


# Per-line correction: conservative output limit with hard runaway cutoff
_LINE_EXPECTED_LEN = 300   # ~1 sentence reason + 1 corrected line
_LINE_RUNAWAY_MULT = 5.0   # hard limit at ~1500 chars — triggers retry


def _correct_single_line(
    line_index: int,
    source_line: str,
    translated_line: str,
    score: ConfidenceScore,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    vocab_table: str,
    file_context: str,
    full_source: List[str],
    full_translated: List[str],
    source_lang: str,
    target_lang: str,
    retry: int,
    retry_sleep_s: float,
) -> str:
    """Analyze and correct a single flagged line. Returns corrected text or original on failure."""
    system, user = _build_line_correction_prompt(
        line_index, source_line, translated_line, score,
        vocab_table, file_context, full_source, full_translated,
        source_lang, target_lang,
    )

    for attempt in range(1, retry + 2):
        try:
            raw = _call_llm(
                system, user, endpoint, timeout_s, extra_payload,
                use_stream, verbose,
                expected_output_len=_LINE_EXPECTED_LEN,
                runaway_multiplier=_LINE_RUNAWAY_MULT,
                raise_on_runaway=True,
            )

            results, missing = parse_numbered_output(raw, 1, line_index)
            if results[0] is not None:
                return results[0]

            # No valid [N] found — model may have looped before outputting it
            if attempt <= retry:
                sys.stderr.write(
                    f"    [RETRY] Line [{line_index}]: no valid output, "
                    f"attempt {attempt}/{retry + 1}\n"
                )
                time.sleep(retry_sleep_s)
                continue

        except Exception as e:
            if attempt <= retry:
                sys.stderr.write(
                    f"    [RETRY] Line [{line_index}] failed "
                    f"(attempt {attempt}/{retry + 1}): {e}\n"
                )
                time.sleep(retry_sleep_s)
                continue

    sys.stderr.write(
        f"    [WARN] Line [{line_index}]: all attempts failed, keeping Pass 1 output.\n"
    )
    return translated_line


# Vocab generation: 30 entries × ~30 chars ≈ 900 chars expected
_VOCAB_EXPECTED_LEN = 900
_VOCAB_RUNAWAY_MULT = 5.0   # hard limit at ~4500 chars
_VOCAB_MAX_ENTRIES = 30      # reject output with more than this
_VOCAB_MAX_TERM_LEN = 30    # single term should not exceed this many chars


def _validate_vocab_entries(entries: Dict[str, str]) -> Dict[str, str]:
    """Filter vocab entries: reject overly long terms, cap total count."""
    valid: Dict[str, str] = {}
    for src, tgt in entries.items():
        # Skip entries where either side is too long (likely a sentence, not a term)
        if len(src) > _VOCAB_MAX_TERM_LEN or len(tgt) > _VOCAB_MAX_TERM_LEN:
            continue
        # Skip entries with newlines or multiple sentences
        if "\n" in src or "\n" in tgt:
            continue
        valid[src] = tgt
        if len(valid) >= _VOCAB_MAX_ENTRIES:
            break
    return valid


def _generate_vocab_sheet(
    source_lines: List[str],
    corrected_lines: List[str],
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    source_lang: str,
    target_lang: str,
    retry: int = 2,
    retry_sleep_s: float = 1.0,
) -> Dict[str, str]:
    """Step 2.3: Generate vocab sheet per file. Hard limit with retry on runaway."""
    system, user = _build_full_vocab_gen_prompt(
        source_lines, corrected_lines, source_lang, target_lang,
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, retry + 2):
        try:
            raw = _call_llm(system, user, endpoint, timeout_s, extra_payload,
                             use_stream, verbose,
                             expected_output_len=_VOCAB_EXPECTED_LEN,
                             runaway_multiplier=_VOCAB_RUNAWAY_MULT,
                             raise_on_runaway=True)
            entries = _parse_vocab_entries(raw)
            entries = _validate_vocab_entries(entries)

            if not entries:
                last_err = RuntimeError("no valid entries parsed")
                sys.stderr.write(
                    f"  [RETRY] Step 2.3: no valid entries, "
                    f"attempt {attempt}/{retry + 1}\n"
                )
                if attempt <= retry:
                    time.sleep(retry_sleep_s)
                continue

            if verbose:
                sys.stderr.write(f"  [VOCAB] Generated {len(entries)} vocab entries\n")
            return entries

        except Exception as e:
            last_err = e
            sys.stderr.write(
                f"  [RETRY] Step 2.3 failed (attempt {attempt}/{retry + 1}): {e}\n"
            )
            if attempt <= retry:
                time.sleep(retry_sleep_s)

    sys.stderr.write(
        f"[WARN] Step 2.3 (vocab generation) failed after {retry + 1} attempts: "
        f"{last_err}. Skipping.\n"
    )
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
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Proofread translated lines using a 2-pass pipeline.

    Returns (corrected_lines, updated_vocab_entries, proofread_vocab).
    """
    if not source_lines or not translated_lines:
        return list(translated_lines), vocab_entries, {}

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

    vocab_table = format_vocab(vocab_entries)
    total_chunks = (n + chunk_size - 1) // chunk_size

    # ── PASS 1: Flag & Vocab Replace ──────────────────────────────────
    sys.stderr.write(f"  ── Pass 1: Flag & Vocab Replace ({total_chunks} chunks) ──\n")

    all_scores: List[Optional[ConfidenceScore]] = [None] * n
    pass1_lines = list(translated_lines)

    for chunk_idx, chunk_start in enumerate(range(0, n, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_num = chunk_idx + 1
        sys.stderr.write(f"  [Chunk {chunk_num}/{total_chunks}] Lines {chunk_start + 1}-{chunk_end}\n")

        # Step 1.1: Strict vocab replacement FIRST (may fix issues that would score LOW/MEDIUM)
        if vocab_table:
            sys.stderr.write(f"  [Step 1.1] Applying vocab replacement...\n")
            pass1_lines[chunk_start:chunk_end] = _apply_vocab_replacement(
                translated_lines[chunk_start:chunk_end],
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

        # Step 1.2: Confidence scoring — score exactly the chunk lines (on vocab-replaced text)
        # Context from previous chunks is passed separately to the prompt for reference
        ctx_start = max(0, chunk_start - context_window)
        ctx_before = pass1_lines[ctx_start:chunk_start]

        sys.stderr.write(f"  [Step 1.2] Scoring confidence (lines {chunk_start + 1}-{chunk_end})...\n")
        chunk_scores = _score_confidence(
            source_lines[chunk_start:chunk_end],
            pass1_lines[chunk_start:chunk_end],
            start_index=chunk_start + 1,
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

        for ci, s in enumerate(chunk_scores):
            all_scores[chunk_start + ci] = s

        # Report scores
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for s in chunk_scores:
            counts[s.level] = counts.get(s.level, 0) + 1
        sys.stderr.write(
            f"  [Step 1.2] Scores: {counts['HIGH']} HIGH, {counts['MEDIUM']} MEDIUM, {counts['LOW']} LOW\n"
        )

    # Collect all flagged line indices (MEDIUM or LOW)
    flagged_indices = [
        i for i in range(n)
        if all_scores[i] is not None and all_scores[i].level != "HIGH"
    ]

    sys.stderr.write(
        f"  ── Pass 2: Per-Line Correction ({len(flagged_indices)} flagged lines, "
        f"{n - len(flagged_indices)} skipped) ──\n"
    )

    # ── PASS 2: Per-line correction (flagged lines only) ──────────────
    corrected = list(pass1_lines)

    for fi, idx in enumerate(flagged_indices):
        score = all_scores[idx]
        sys.stderr.write(
            f"  [Line {fi + 1}/{len(flagged_indices)}] [{idx + 1}] {score.level}"
            f"{': ' + score.reason if score.reason else ''}\n"
        )
        corrected[idx] = _correct_single_line(
            line_index=idx + 1,
            source_line=source_lines[idx],
            translated_line=pass1_lines[idx],
            score=score,
            endpoint=endpoint,
            timeout_s=timeout_s,
            extra_payload=extra_payload,
            use_stream=use_stream,
            verbose=verbose,
            vocab_table=vocab_table,
            file_context=file_context,
            full_source=source_lines,
            full_translated=pass1_lines,
            source_lang=source_lang,
            target_lang=target_lang,
            retry=retry,
            retry_sleep_s=retry_sleep_s,
        )

    # Step 2.3: Generate new vocab sheet (once per file)
    proofread_vocab: Dict[str, str] = {}
    if update_vocab:
        sys.stderr.write(f"  [Step 2.3] Generating vocab sheet...\n")
        proofread_vocab = _generate_vocab_sheet(
            source_lines, corrected,
            endpoint=endpoint,
            timeout_s=timeout_s,
            extra_payload=extra_payload,
            use_stream=use_stream,
            verbose=verbose,
            source_lang=source_lang,
            target_lang=target_lang,
            retry=retry,
            retry_sleep_s=retry_sleep_s,
        )

        # Step 2.4: Apply new vocab to ALL lines
        if proofread_vocab:
            proofread_vocab_table = format_vocab(proofread_vocab)
            sys.stderr.write(f"  [Step 2.4] Applying new vocab ({len(proofread_vocab)} entries) to all lines...\n")
            for chunk_start in range(0, n, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n)
                corrected[chunk_start:chunk_end] = _apply_vocab_replacement(
                    corrected[chunk_start:chunk_end],
                    start_index=chunk_start + 1,
                    endpoint=endpoint,
                    timeout_s=timeout_s,
                    extra_payload=extra_payload,
                    use_stream=use_stream,
                    verbose=verbose,
                    vocab_table=proofread_vocab_table,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )

    # Merge proofread vocab back into main vocab entries
    if proofread_vocab:
        vocab_entries.update(proofread_vocab)

    return corrected, vocab_entries, proofread_vocab


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Proofread translated SRT subtitles using a 2-pass pipeline."
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
        "--proofread-vocab",
        default="",
        help="Output path for proofread-generated vocab (default: {vocab_stem}_proofread.txt)",
    )
    ap.add_argument(
        "--no-vocab-update",
        action="store_true",
        default=False,
        help="Disable vocab learning — skips Steps 2.3-2.4 (read-only mode for vocab)",
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

    # Determine proofread vocab output path
    proofread_vocab_path = args.proofread_vocab
    if not proofread_vocab_path and args.vocab:
        stem, ext = os.path.splitext(args.vocab)
        proofread_vocab_path = f"{stem}_proofread{ext}"

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

        # Run the 2-pass pipeline (with whole-file error recovery)
        try:
            corrected, vocab_entries, proofread_vocab = proofread_file(
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

            # Save main vocab after each file
            if update_vocab and args.vocab and vocab_entries:
                save_vocab(args.vocab, vocab_entries)
                sys.stderr.write(f"[INFO] Saved {len(vocab_entries)} vocab entries to {args.vocab}\n")

            # Save proofread vocab to separate file
            if update_vocab and proofread_vocab_path and proofread_vocab:
                save_vocab(proofread_vocab_path, proofread_vocab)
                sys.stderr.write(
                    f"[INFO] Saved {len(proofread_vocab)} proofread vocab entries to {proofread_vocab_path}\n"
                )
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to proofread {path}: {e}. File left untouched.\n")
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
