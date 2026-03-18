#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py — Step 2: Context understanding, ASR error detection/fixing, term extraction.

Six passes per file (all with thinking disabled):
  Pass 1: Context Summary — understand the scene/episode first
  Pass 2: Brainstorm Expected Words — high-frequency words for this scenario
  Pass 3: ASR Error Flagging — informed by context + expected words
  Pass 4: ASR Error Fixing (per flagged line)
  Pass 5: Term Extraction (per file)
Per series:
  Pass 6: Vocab Cleanup (once per series)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import List, Optional, Dict, Any

from core import (
    SrtBlock,
    read_text_file,
    parse_srt,
    write_srt,
    build_line_mapping,
    apply_translations,
    format_numbered_input,
    parse_numbered_output,
    call_llm,
    call_llm_with_retry,
    load_manifest,
    save_manifest,
    TmpPaths,
)


# ---------------------------------------------------------------------------
# Pass 1 — Context Summary (understand the scene first)
# ---------------------------------------------------------------------------

def _build_context_summary(
    source_lines: List[str],
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    source_lang: str,
) -> str:
    """Build a context summary (tone, speakers, plot) from source text."""
    n = len(source_lines)
    if n <= 100:
        sample_indices = list(range(n))
    else:
        sample_indices = list(range(50)) + list(range(n - 50, n))

    sample_text = "\n".join(
        f"[{idx + 1}] {source_lines[idx]}" for idx in sample_indices
    )

    system = (
        f"You are a subtitle analyst. Given {source_lang} subtitle lines, "
        "provide a brief summary to help translators understand the context.\n"
    )
    user = (
        "Summarise these subtitle lines:\n"
        "1. What is this scene/episode about? (1-2 sentences)\n"
        "2. List all character names mentioned\n"
        "3. Overall tone (formal, casual, comedic, dramatic)\n"
        "4. Any recurring terms or phrases worth noting\n"
        "\n"
        "Keep the summary under 200 words.\n"
        "\n"
        f"{sample_text}"
    )

    try:
        return call_llm(
            system=system, user=user,
            endpoint=endpoint, timeout_s=min(timeout_s, 60),
            extra_payload=extra_payload, use_stream=use_stream,
            verbose=verbose, expected_output_len=800,
        ).strip()
    except Exception as e:
        sys.stderr.write(f"[WARN] Context summary failed: {e}\n")
        return ""


# ---------------------------------------------------------------------------
# Pass 2 — Brainstorm Expected Words
# ---------------------------------------------------------------------------

def _brainstorm_expected_words(
    context_summary: str,
    source_lines: List[str],
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    source_lang: str,
) -> str:
    """Given the context, brainstorm high-frequency words expected in this scenario.

    Returns a newline-separated list of words/phrases the ASR might encounter.
    This helps the flagging pass identify misrecognitions by knowing what SHOULD
    appear vs what actually appears.
    """
    if not context_summary:
        return ""

    # Sample a few lines so the model sees the actual content style
    n = len(source_lines)
    if n <= 30:
        sample = "\n".join(source_lines)
    else:
        sample = "\n".join(source_lines[:15] + ["..."] + source_lines[-15:])

    system = (
        f"You are a {source_lang} language expert helping detect ASR errors in subtitles.\n"
        "\n"
        "Given a context summary and sample lines from a subtitle file, brainstorm "
        f"the {source_lang} words and short phrases that are likely to appear frequently "
        "in this scenario. Focus on:\n"
        "- Character names and how they are addressed\n"
        "- Domain-specific vocabulary for the setting (school, workplace, fantasy, etc.)\n"
        "- Common phrases for the tone (formal greetings, slang, battle cries, etc.)\n"
        "- Words that sound similar and are often confused by ASR\n"
        "\n"
        "Output a flat list, one word/phrase per line. No numbering, no explanation.\n"
        "Aim for 20-50 entries. Focus on words that ASR commonly gets wrong."
    )
    user = f"CONTEXT:\n{context_summary}\n\nSAMPLE LINES:\n{sample}"

    try:
        raw = call_llm(
            system=system, user=user,
            endpoint=endpoint, timeout_s=min(timeout_s, 60),
            extra_payload=extra_payload, use_stream=use_stream,
            verbose=verbose, expected_output_len=1000,
            runaway_multiplier=5.0,
        )
        # Basic validation: should have multiple lines, not be a paragraph
        lines = [ln.strip() for ln in raw.strip().split("\n") if ln.strip()]
        if len(lines) < 3:
            sys.stderr.write(f"[WARN] Brainstorm returned too few words ({len(lines)}), discarding.\n")
            return ""
        result = "\n".join(lines)
        if verbose:
            sys.stderr.write(f"  [BRAINSTORM] {len(lines)} expected words\n")
        return result
    except Exception as e:
        sys.stderr.write(f"[WARN] Word brainstorm failed: {e}\n")
        return ""


# ---------------------------------------------------------------------------
# Pass 3 — ASR Error Flagging (with context + expected words)
# ---------------------------------------------------------------------------

_ASR_FLAG_RE = re.compile(r"^\[(\d+)\]\s*(.*)", re.MULTILINE)


def _flag_asr_errors(
    source_lines: List[str],
    context_summary: str,
    expected_words: str,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    source_lang: str,
    chunk_size: int = 200,
) -> List[Dict[str, Any]]:
    """Flag lines with suspected ASR errors. Returns list of {line: N, reason: "..."}."""
    flags: List[Dict[str, Any]] = []

    system = (
        f"You are an ASR error detector for {source_lang} subtitles.\n"
        "\n"
        "You are given the context of the scene and a list of words/phrases "
        "expected to appear in this scenario. Use them to identify lines where "
        "ASR likely misrecognized a word.\n"
        "\n"
        "Flag lines that contain likely ASR errors:\n"
        "- Homophones or near-homophones of expected words\n"
        "- Misspelled words or garbled phrases\n"
        "- Sentences that don't make grammatical sense\n"
        "- Words that seem like misrecognitions of similar-sounding words\n"
        "\n"
        "Output format (one per flagged line):\n"
        "[N] \"wrong_word\" → candidate1, candidate2; short reason\n"
        "\n"
        "Examples:\n"
        "[5] \"家事\" → 菓子, 火事; homophone in kitchen scene\n"
        "[12] \"よろしく\" → broken, garbled phrase\n"
        "[23] \"抑\" → 押さえ, 抑え; wrong kanji for context\n"
        "\n"
        "Rules:\n"
        "- List the wrong word in quotes, then possible correct candidates after →\n"
        "- Keep reason to under 10 words. No long explanations.\n"
        "- Only flag lines with actual problems. Correct lines should NOT appear.\n"
        "- No commentary outside the [N] lines."
    )

    # Build the reference block (context + expected words) once
    ref_parts: List[str] = []
    if context_summary:
        ref_parts.append(f"SCENE CONTEXT:\n{context_summary}")
    if expected_words:
        ref_parts.append(f"EXPECTED WORDS (likely to appear in this scenario):\n{expected_words}")
    reference_block = "\n\n".join(ref_parts)

    # Process in chunks if file is long
    for start in range(0, len(source_lines), chunk_size):
        chunk = source_lines[start:start + chunk_size]
        numbered = format_numbered_input(chunk, start_index=start + 1)

        user_parts = []
        if reference_block:
            user_parts.append(reference_block)
        user_parts.append(f"Review these {source_lang} subtitle lines:\n{numbered}")
        user = "\n\n".join(user_parts)

        try:
            raw = call_llm(
                system=system,
                user=user,
                endpoint=endpoint,
                timeout_s=timeout_s,
                extra_payload=extra_payload,
                use_stream=use_stream,
                verbose=verbose,
                expected_output_len=len(chunk) * 20,
                runaway_multiplier=3.0,
                raise_on_runaway=True,
            )
            for m in _ASR_FLAG_RE.finditer(raw):
                line_num = int(m.group(1))
                reason = m.group(2).strip()
                if 1 <= line_num <= len(source_lines):
                    flags.append({"line": line_num, "reason": reason})
        except RuntimeError:
            # On runaway, retry with half the chunk
            if len(chunk) > 50:
                half = len(chunk) // 2
                try:
                    numbered_half = format_numbered_input(chunk[:half], start_index=start + 1)
                    user_half_parts = []
                    if reference_block:
                        user_half_parts.append(reference_block)
                    user_half_parts.append(f"Review these {source_lang} subtitle lines:\n{numbered_half}")
                    user_half = "\n\n".join(user_half_parts)
                    raw = call_llm(
                        system=system, user=user_half,
                        endpoint=endpoint, timeout_s=timeout_s,
                        extra_payload=extra_payload, use_stream=use_stream,
                        verbose=verbose,
                        expected_output_len=half * 20,
                        runaway_multiplier=3.0,
                    )
                    for m in _ASR_FLAG_RE.finditer(raw):
                        line_num = int(m.group(1))
                        reason = m.group(2).strip()
                        if 1 <= line_num <= len(source_lines):
                            flags.append({"line": line_num, "reason": reason})
                except Exception as e:
                    sys.stderr.write(f"[WARN] ASR flagging retry failed: {e}\n")
        except Exception as e:
            sys.stderr.write(f"[WARN] ASR flagging failed for chunk at {start}: {e}\n")

    return flags


# ---------------------------------------------------------------------------
# Pass 4 — ASR Error Fixing
# ---------------------------------------------------------------------------

def _fix_asr_errors(
    source_lines: List[str],
    flags: List[Dict[str, Any]],
    context_summary: str,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    source_lang: str,
    retry: int,
    retry_sleep_s: float,
    context_radius: int = 50,
) -> List[str]:
    """Fix each flagged ASR error line. Returns full list with fixes applied."""
    fixed = list(source_lines)

    for fi, flag in enumerate(flags):
        line_num = flag["line"]
        reason = flag["reason"]
        idx = line_num - 1
        if idx < 0 or idx >= len(source_lines):
            continue

        sys.stderr.write(
            f"  [ASR Fix {fi + 1}/{len(flags)}] Line [{line_num}]: {reason}\n"
        )

        # Build context window
        n = len(source_lines)
        ctx_start = max(0, idx - context_radius)
        ctx_end = min(n, idx + context_radius + 1)
        context_lines = "\n".join(
            f"[{i + 1}] {source_lines[i]}" for i in range(ctx_start, ctx_end)
        )

        system = (
            f"You are an ASR error corrector for {source_lang} subtitles.\n"
            "Fix the flagged line. Keep the original meaning and style.\n"
            "\n"
            "Output format (exactly 1 line):\n"
            f"[{line_num}] corrected_text\n"
            "\n"
            "Rules:\n"
            "- Fix ONLY the flagged line.\n"
            "- Use surrounding context to infer the correct words.\n"
            "- If the line is actually correct, output it unchanged.\n"
            "- Output ONLY the [N] line. Nothing else."
        )

        parts = []
        if context_summary:
            parts.append(f"CONTEXT SUMMARY:\n{context_summary}")
        parts.append(f"SURROUNDING LINES:\n{context_lines}")
        parts.append(f"FLAGGED LINE [{line_num}]: {source_lines[idx]}\nREASON: {reason}")
        user = "\n\n".join(parts)

        current_radius = context_radius
        for attempt in range(1, retry + 2):
            try:
                raw = call_llm(
                    system=system, user=user,
                    endpoint=endpoint, timeout_s=timeout_s,
                    extra_payload=extra_payload, use_stream=use_stream,
                    verbose=verbose,
                    expected_output_len=300,
                    runaway_multiplier=5.0,
                    raise_on_runaway=True,
                )
                results, missing = parse_numbered_output(raw, 1, line_num)
                if results[0] is not None:
                    fixed[idx] = results[0]
                    break
                if attempt <= retry:
                    sys.stderr.write(f"    [RETRY] No valid output, attempt {attempt}/{retry + 1}\n")
                    time.sleep(retry_sleep_s)
            except RuntimeError:
                # On runaway, shrink context
                if current_radius > 25:
                    current_radius = 25
                    ctx_start = max(0, idx - current_radius)
                    ctx_end = min(n, idx + current_radius + 1)
                    context_lines = "\n".join(
                        f"[{i + 1}] {source_lines[i]}" for i in range(ctx_start, ctx_end)
                    )
                    parts = []
                    if context_summary:
                        parts.append(f"CONTEXT SUMMARY:\n{context_summary}")
                    parts.append(f"SURROUNDING LINES:\n{context_lines}")
                    parts.append(f"FLAGGED LINE [{line_num}]: {source_lines[idx]}\nREASON: {reason}")
                    user = "\n\n".join(parts)
                if attempt <= retry:
                    sys.stderr.write(f"    [RETRY] Runaway, shrinking context, attempt {attempt}/{retry + 1}\n")
                    time.sleep(retry_sleep_s)
            except Exception as e:
                if attempt <= retry:
                    sys.stderr.write(f"    [RETRY] Failed: {e}, attempt {attempt}/{retry + 1}\n")
                    time.sleep(retry_sleep_s)

    return fixed


# ---------------------------------------------------------------------------
# Pass 5 — Term Extraction
# ---------------------------------------------------------------------------

_VOCAB_LINE_RE = re.compile(r"^(.+?)\s*(?:→|->)\s*(.+)$", re.MULTILINE)
_VOCAB_MAX_TERM_LEN = 30
_VOCAB_MAX_ENTRIES = 30


def _extract_terms(
    source_lines: List[str],
    context_summary: str,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    source_lang: str,
    target_lang: str,
) -> Dict[str, str]:
    """Extract terms (names, special words) from source text."""
    n = len(source_lines)
    if n <= 100:
        sample_indices = list(range(n))
    else:
        sample_indices = list(range(50)) + list(range(n - 50, n))

    sample_text = "\n".join(source_lines[idx] for idx in sample_indices)

    system = (
        f"You extract proper nouns and key terms from {source_lang} subtitles "
        f"that will need consistent translation to {target_lang}.\n"
        "\n"
        "ONLY output entries that are:\n"
        "- Character names (people, nicknames, honorific+name pairs)\n"
        "- Place names (locations, buildings, countries, fictional places)\n"
        "- Organization names (companies, schools, groups, factions)\n"
        "- Fictional/in-universe terms (special abilities, titles)\n"
        "\n"
        "ACCURACY IS CRITICAL — this vocab guides all downstream translation.\n"
        "A wrong entry is WORSE than a missing one.\n"
        "\n"
        "Before outputting each entry, verify:\n"
        "1. The source term actually appears in the subtitle lines\n"
        f"2. The {target_lang} translation is correct for this specific context\n"
        "3. You are confident this is a proper noun, not a common word\n"
        "If unsure about ANY of the above, do NOT include the entry.\n"
        "\n"
        f"Format: one entry per line, exactly: {source_lang}Term → {target_lang}Term\n"
        "Maximum 30 entries. 5-10 high-confidence entries is better than 30 guesses.\n"
        "Output ONLY the vocabulary lines. No commentary."
    )

    parts = []
    if context_summary:
        parts.append(f"CONTEXT:\n{context_summary}")
    parts.append(f"SOURCE LINES:\n{sample_text}")
    user = "\n\n".join(parts)

    try:
        raw = call_llm(
            system=system, user=user,
            endpoint=endpoint, timeout_s=min(timeout_s, 60),
            extra_payload=extra_payload, use_stream=use_stream,
            verbose=verbose, expected_output_len=900,
            runaway_multiplier=5.0,
        )

        entries: Dict[str, str] = {}
        for m in _VOCAB_LINE_RE.finditer(raw):
            src = m.group(1).strip()
            tgt = m.group(2).strip()
            if src and tgt and not src.startswith("#"):
                if len(src) <= _VOCAB_MAX_TERM_LEN and len(tgt) <= _VOCAB_MAX_TERM_LEN:
                    if "\n" not in src and "\n" not in tgt:
                        entries[src] = tgt
                        if len(entries) >= _VOCAB_MAX_ENTRIES:
                            break

        if verbose:
            sys.stderr.write(f"  [TERMS] Extracted {len(entries)} terms\n")
        return entries
    except Exception as e:
        sys.stderr.write(f"[WARN] Term extraction failed: {e}\n")
        return {}


# ---------------------------------------------------------------------------
# Pass 6 — Vocab Cleanup
# ---------------------------------------------------------------------------

def _cleanup_vocab(
    context_summary: str,
    vocab_text: str,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    source_lang: str,
    target_lang: str,
) -> str:
    """Deduplicate and clean up the accumulated vocab for a series."""
    if not vocab_text.strip():
        return vocab_text

    system = (
        f"You are a {source_lang} → {target_lang} vocabulary auditor.\n"
        "\n"
        "Review this vocabulary sheet LINE BY LINE against the context.\n"
        "For each entry, check:\n"
        "1. Is this actually a proper noun (name, place, organization, fictional term)?\n"
        "   → DELETE if it is a common word or generic phrase.\n"
        f"2. Does the {target_lang} translation make sense for this specific context?\n"
        "   → DELETE if the translation seems wrong or you are not confident.\n"
        "3. Does this term actually appear in the subtitles based on the context?\n"
        "   → DELETE if it seems fabricated or hallucinated.\n"
        "4. Are there duplicates or conflicts?\n"
        "   → KEEP only the best entry, delete the rest.\n"
        "\n"
        "IMPORTANT: A misleading vocab entry is WORSE than a missing one.\n"
        "When in doubt, DELETE the entry. Be aggressive about removing uncertain entries.\n"
        "It is fine to return fewer entries or even an empty list.\n"
        "\n"
        "Output ONLY the surviving vocabulary lines in SourceTerm → TranslatedTerm format.\n"
        "No commentary, no explanations."
    )

    parts = []
    if context_summary:
        parts.append(f"CONTEXT:\n{context_summary}")
    parts.append(f"VOCABULARY TO CLEAN:\n{vocab_text}")
    user = "\n\n".join(parts)

    try:
        raw = call_llm(
            system=system, user=user,
            endpoint=endpoint, timeout_s=min(timeout_s, 60),
            extra_payload=extra_payload, use_stream=use_stream,
            verbose=verbose, expected_output_len=len(vocab_text),
        )
        # Validate: should contain → or ->
        if "→" in raw or "->" in raw:
            return raw.strip()
        return vocab_text
    except Exception as e:
        sys.stderr.write(f"[WARN] Vocab cleanup failed: {e}. Keeping original.\n")
        return vocab_text


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_preprocess_step(
    manifest: dict,
    endpoint: str,
    timeout_s: int = 300,
    extra_payload: Optional[Dict[str, Any]] = None,
    use_stream: bool = True,
    verbose: bool = False,
    tmp_dir: str = "./tmp",
    source_lang: str = "Japanese",
    target_lang: str = "Simplified Chinese",
    retry: int = 2,
    retry_sleep_s: float = 1.0,
) -> None:
    """Step 2: Context → brainstorm words → ASR flag/fix → term extraction → vocab cleanup."""
    series_list = manifest.get("series", [])
    if not series_list:
        sys.stderr.write("[WARN] No series in manifest. Skipping preprocess.\n")
        return

    for si, series_info in enumerate(series_list):
        series_name = series_info["name"]
        series_dir = series_info["dir"]
        files = series_info["files"]
        paths = TmpPaths(tmp_dir, series_name)

        sys.stderr.write(f"\n[PREPROCESS] Series: \"{series_name}\" ({len(files)} files)\n")

        all_vocab: Dict[str, str] = {}
        all_context_parts: List[str] = []

        for fi, file_info in enumerate(files):
            stem = file_info["stem"]
            source_path = paths.source_srt(stem)

            sys.stderr.write(f"\n[FILE {fi + 1}/{len(files)}] {stem}\n")

            # Read source
            content = read_text_file(source_path)
            blocks = parse_srt(content)
            if not blocks:
                sys.stderr.write(f"[WARN] No blocks in {source_path}, skipping.\n")
                continue
            source_lines, refs = build_line_mapping(blocks)

            # Pass 1: Context Summary — understand the scene first
            sys.stderr.write(f"  [Pass 1] Building context summary...\n")
            context = _build_context_summary(
                source_lines, endpoint, timeout_s, extra_payload,
                use_stream, verbose, source_lang,
            )
            if context:
                all_context_parts.append(f"## {stem}\n{context}")
                if verbose:
                    sys.stderr.write(f"  [Pass 1] Context ({len(context)} chars): {context[:120]}...\n")

            # Pass 2: Brainstorm expected words for this scenario
            sys.stderr.write(f"  [Pass 2] Brainstorming expected words...\n")
            expected_words = _brainstorm_expected_words(
                context, source_lines, endpoint, timeout_s, extra_payload,
                use_stream, verbose, source_lang,
            )
            if expected_words:
                word_count = len(expected_words.strip().split("\n"))
                sys.stderr.write(f"  [Pass 2] {word_count} expected words brainstormed.\n")
            else:
                sys.stderr.write(f"  [Pass 2] No expected words (context may be empty).\n")

            # Pass 3: ASR Error Flagging — now informed by context + expected words
            sys.stderr.write(f"  [Pass 3] Flagging ASR errors...\n")
            asr_flags = _flag_asr_errors(
                source_lines, context, expected_words,
                endpoint, timeout_s, extra_payload,
                use_stream, verbose, source_lang,
            )
            sys.stderr.write(f"  [Pass 3] {len(asr_flags)} lines flagged.\n")

            # Pass 4: ASR Error Fixing
            if asr_flags:
                sys.stderr.write(f"  [Pass 4] Fixing {len(asr_flags)} ASR errors...\n")
                fixed_lines = _fix_asr_errors(
                    source_lines, asr_flags, context,
                    endpoint, timeout_s, extra_payload,
                    use_stream, verbose, source_lang,
                    retry, retry_sleep_s,
                )
                # Overwrite source SRT with fixed version
                apply_translations(blocks, refs, fixed_lines)
                fixed_content = write_srt(blocks)
                with open(source_path, "w", encoding="utf-8") as f:
                    f.write(fixed_content)
                sys.stderr.write(f"  [Pass 4] Source SRT updated with fixes.\n")

                # Re-read for term extraction
                source_lines = fixed_lines

            # Pass 5: Term Extraction
            sys.stderr.write(f"  [Pass 5] Extracting terms...\n")
            terms = _extract_terms(
                source_lines, context,
                endpoint, timeout_s, extra_payload,
                use_stream, verbose, source_lang, target_lang,
            )
            all_vocab.update(terms)
            sys.stderr.write(f"  [Pass 5] {len(terms)} terms extracted (total: {len(all_vocab)})\n")

        # Write context.md (combined for all files in series)
        context_text = "\n\n".join(all_context_parts) if all_context_parts else ""
        if context_text:
            with open(paths.context_md, "w", encoding="utf-8") as f:
                f.write(context_text)
            sys.stderr.write(f"[PREPROCESS] Wrote context.md ({len(context_text)} chars)\n")

        # Write initial vocab.md
        if all_vocab:
            vocab_text = "\n".join(f"{src} → {tgt}" for src, tgt in sorted(all_vocab.items()))

            # Pass 6: Vocab Cleanup (once per series)
            sys.stderr.write(f"[Pass 6] Cleaning up vocab ({len(all_vocab)} entries)...\n")
            cleaned_vocab = _cleanup_vocab(
                context_text, vocab_text,
                endpoint, timeout_s, extra_payload,
                use_stream, verbose, source_lang, target_lang,
            )
            with open(paths.vocab_md, "w", encoding="utf-8") as f:
                f.write(cleaned_vocab)
            sys.stderr.write(f"[PREPROCESS] Wrote vocab.md\n")

    # Mark step done
    manifest["preprocess_done"] = True
    save_manifest(tmp_dir, manifest)
