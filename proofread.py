#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proofread.py (v2) — Step 5: Fix flagged lines, final review, confused.md.

Pass 1: Correct flagged lines (from flags.json) with context
Pass 2: Final review (audit only) → confused.md
Final: Copy *.translated.srt to out/ with suffix rename
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from typing import List, Optional, Dict, Any

from core import (
    read_text_file,
    parse_srt,
    write_srt,
    build_line_mapping,
    apply_translations,
    parse_numbered_output,
    format_vocab,
    parse_vocab,
    call_llm,
    load_manifest,
    save_manifest,
    TmpPaths,
    _NUMBERED_LINE_RE,
)


# ---------------------------------------------------------------------------
# Pass 1 — Correct Flagged Lines
# ---------------------------------------------------------------------------

_LINE_EXPECTED_LEN = 300
_LINE_RUNAWAY_MULT = 5.0


def _correct_flagged_line(
    line_index: int,
    source_line: str,
    translated_line: str,
    flag_reason: str,
    context_text: str,
    vocab_table: str,
    full_source: List[str],
    full_translated: List[str],
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    source_lang: str,
    target_lang: str,
    retry: int,
    retry_sleep_s: float,
    context_radius: int = 50,
) -> Optional[str]:
    """Correct a single flagged line. Returns corrected text, or None if all attempts fail."""
    system = (
        f"You are a {source_lang} to {target_lang} subtitle proofreader.\n"
        "You will correct ONE specific flagged line. The surrounding context is "
        "provided ONLY for reference — do NOT retranslate or revise any other line.\n"
        "\n"
        "Output format (exactly these 2 lines):\n"
        f"REASON: \"wrong_word\" → corrected_word; short reason (under 10 words)\n"
        f"[{line_index}] corrected_text\n"
        "\n"
        "Example:\n"
        "REASON: \"学校\" → 学园; vocab sheet says 学园\n"
        "[42] 他去了学园。\n"
        "\n"
        "Rules:\n"
        "- Fix ONLY the flagged line. Do not output any other line numbers.\n"
        "- Remove ?? markers when you determine the correct translation.\n"
        "- If the current translation is actually correct, output it unchanged.\n"
        "- Output ONLY the 2 lines above. Nothing else."
    )

    n = len(full_source)
    line_pos = line_index - 1  # 0-based

    def _build_user(radius: int) -> str:
        parts = []
        if context_text:
            parts.append(f"FILE SUMMARY:\n{context_text}")
        if vocab_table:
            parts.append(f"VOCABULARY:\n{vocab_table}")

        if n <= radius * 2 + 20:
            ctx_start, ctx_end = 0, n
        else:
            ctx_start = max(0, line_pos - radius)
            ctx_end = min(n, line_pos + radius + 1)

        ctx_pairs = []
        for i in range(ctx_start, ctx_end):
            ctx_pairs.append(f"[{i + 1}] {full_source[i]} → {full_translated[i]}")
        parts.append("CONTEXT:\n" + "\n".join(ctx_pairs))

        parts.append(
            f"FLAGGED LINE:\n"
            f"[{line_index}] {source_line} → {translated_line}\n"
            f"FLAG REASON: {flag_reason}"
        )
        return "\n\n".join(parts)

    current_radius = context_radius
    for attempt in range(1, retry + 2):
        user = _build_user(current_radius)
        try:
            raw = call_llm(
                system=system, user=user,
                endpoint=endpoint, timeout_s=timeout_s,
                extra_payload=extra_payload, use_stream=use_stream,
                verbose=verbose,
                expected_output_len=_LINE_EXPECTED_LEN,
                runaway_multiplier=_LINE_RUNAWAY_MULT,
                raise_on_runaway=True,
            )
            results, missing = parse_numbered_output(raw, 1, line_index)
            if results[0] is not None:
                return results[0]

            if attempt <= retry:
                sys.stderr.write(
                    f"    [RETRY] Line [{line_index}]: no valid output, "
                    f"attempt {attempt}/{retry + 1}\n"
                )
                time.sleep(retry_sleep_s)
        except RuntimeError:
            # Shrink context on runaway
            if current_radius > 25:
                current_radius = 25
            if attempt <= retry:
                sys.stderr.write(
                    f"    [RETRY] Line [{line_index}] runaway, shrinking context, "
                    f"attempt {attempt}/{retry + 1}\n"
                )
                time.sleep(retry_sleep_s)
        except Exception as e:
            if attempt <= retry:
                sys.stderr.write(
                    f"    [RETRY] Line [{line_index}] failed: {e}, "
                    f"attempt {attempt}/{retry + 1}\n"
                )
                time.sleep(retry_sleep_s)

    return None  # All attempts failed


# ---------------------------------------------------------------------------
# Pass 2 — Final Review (audit only)
# ---------------------------------------------------------------------------

_ISSUE_RE = re.compile(r"^\[(\d+)\]\s*ISSUE:\s*(.*)", re.MULTILINE | re.IGNORECASE)


def _final_review(
    source_lines: List[str],
    translated_lines: List[str],
    context_text: str,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    source_lang: str,
    target_lang: str,
) -> List[Dict[str, str]]:
    """Final review pass. Returns list of {line: N, issue: description}. Non-critical."""
    system = (
        f"You are a {source_lang} to {target_lang} subtitle quality auditor.\n"
        "Review the complete translated file and identify any remaining issues.\n"
        "\n"
        "Output format (only for problematic lines):\n"
        "[N] ISSUE: \"wrong_word\" → candidate1, candidate2; short reason\n"
        "\n"
        "Example:\n"
        "[15] ISSUE: \"她走了\" → 他走了; wrong pronoun, speaker is male\n"
        "\n"
        "Only flag significant problems (wrong meaning, missing info, inconsistency).\n"
        "Do NOT flag minor style differences.\n"
        "Keep each reason under 10 words.\n"
        "If everything looks good, output: ALL CLEAR\n"
        "\n"
        "Output ONLY the issue lines or ALL CLEAR. No other text."
    )

    # Sample if file is very long
    n = len(source_lines)
    if n <= 200:
        sample_indices = list(range(n))
    else:
        sample_indices = list(range(100)) + list(range(n - 100, n))

    pairs = []
    for idx in sample_indices:
        pairs.append(f"[{idx + 1}] {source_lines[idx]} → {translated_lines[idx]}")

    parts = []
    if context_text:
        parts.append(f"CONTEXT:\n{context_text}")
    parts.append("TRANSLATED PAIRS:\n" + "\n".join(pairs))
    user = "\n\n".join(parts)

    issues: List[Dict[str, str]] = []
    try:
        raw = call_llm(
            system=system, user=user,
            endpoint=endpoint, timeout_s=timeout_s,
            extra_payload=extra_payload, use_stream=use_stream,
            verbose=verbose,
            expected_output_len=n * 15,
            runaway_multiplier=3.0,
        )
        for m in _ISSUE_RE.finditer(raw):
            issues.append({"line": m.group(1), "issue": m.group(2).strip()})
    except Exception as e:
        sys.stderr.write(f"[WARN] Final review failed: {e}. Skipping (non-critical).\n")

    return issues


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_proofread_step(
    manifest: dict,
    endpoint: str,
    timeout_s: int = 300,
    extra_payload: Optional[Dict[str, Any]] = None,
    use_stream: bool = True,
    verbose: bool = False,
    tmp_dir: str = "./tmp",
    out_dir: str = "out",
    suffix: str = ".zh.srt",
    source_lang: str = "Japanese",
    target_lang: str = "Simplified Chinese",
    context_radius: int = 50,
    retry: int = 2,
    retry_sleep_s: float = 1.0,
) -> None:
    """Step 5: Correct flagged lines, final review, copy to out/."""
    series_list = manifest.get("series", [])
    if not series_list:
        sys.stderr.write("[WARN] No series in manifest. Skipping proofread.\n")
        return

    os.makedirs(out_dir, exist_ok=True)

    for si, series_info in enumerate(series_list):
        series_name = series_info["name"]
        files = series_info["files"]
        paths = TmpPaths(tmp_dir, series_name)

        sys.stderr.write(f"\n[PROOFREAD] Series: \"{series_name}\" ({len(files)} files)\n")

        # Load flags.json
        all_flags: Dict[str, Dict[str, str]] = {}
        if os.path.isfile(paths.flags_json):
            with open(paths.flags_json, "r", encoding="utf-8") as f:
                all_flags = json.load(f)

        # Load context and vocab
        context_text = ""
        if os.path.isfile(paths.context_md):
            context_text = read_text_file(paths.context_md).strip()

        vocab_table = ""
        if os.path.isfile(paths.vocab_md):
            raw = read_text_file(paths.vocab_md).strip()
            if raw:
                vocab_entries = parse_vocab(raw)
                vocab_table = format_vocab(vocab_entries)

        confused_entries: List[str] = []

        for fi, file_info in enumerate(files):
            stem = file_info["stem"]
            source_path = paths.source_srt(stem)
            translated_path = paths.translated_srt(stem)

            sys.stderr.write(f"\n[FILE {fi + 1}/{len(files)}] {stem}\n")

            if not os.path.isfile(translated_path):
                sys.stderr.write(f"[WARN] Translated file not found: {translated_path}, skipping.\n")
                continue

            # Read source
            source_content = read_text_file(source_path)
            source_blocks = parse_srt(source_content)
            if not source_blocks:
                continue
            source_lines, _ = build_line_mapping(source_blocks)

            # Read translated
            trans_content = read_text_file(translated_path)
            trans_blocks = parse_srt(trans_content)
            if not trans_blocks:
                # Copy as-is
                out_path = os.path.join(out_dir, stem + suffix)
                shutil.copy2(translated_path, out_path)
                continue
            translated_lines, trans_refs = build_line_mapping(trans_blocks)

            n = min(len(source_lines), len(translated_lines))
            corrected = list(translated_lines[:n])

            # Pass 1: Correct flagged lines
            file_flags = all_flags.get(stem, {})
            if file_flags:
                sys.stderr.write(f"  [Pass 1] Correcting {len(file_flags)} flagged lines...\n")
                for flag_idx, (line_num_str, reason) in enumerate(file_flags.items()):
                    line_num = int(line_num_str)
                    idx = line_num - 1
                    if idx < 0 or idx >= n:
                        continue

                    sys.stderr.write(
                        f"  [Line {flag_idx + 1}/{len(file_flags)}] [{line_num}] FLAG: {reason}\n"
                    )
                    result = _correct_flagged_line(
                        line_index=line_num,
                        source_line=source_lines[idx],
                        translated_line=corrected[idx],
                        flag_reason=reason,
                        context_text=context_text,
                        vocab_table=vocab_table,
                        full_source=source_lines[:n],
                        full_translated=corrected,
                        endpoint=endpoint,
                        timeout_s=timeout_s,
                        extra_payload=extra_payload,
                        use_stream=use_stream,
                        verbose=verbose,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        retry=retry,
                        retry_sleep_s=retry_sleep_s,
                        context_radius=context_radius,
                    )
                    if result is not None:
                        corrected[idx] = result
                    else:
                        # Failed → add to confused.md
                        confused_entries.append(
                            f"## {stem} line [{line_num}]\n"
                            f"- Source: {source_lines[idx]}\n"
                            f"- Translation: {corrected[idx]}\n"
                            f"- Flag: {reason}\n"
                            f"- Status: All correction attempts failed\n"
                        )
                        sys.stderr.write(f"    [CONFUSED] Line [{line_num}] → confused.md\n")
            else:
                sys.stderr.write(f"  [Pass 1] No flags for {stem}, skipping corrections.\n")

            # Pass 2: Final Review (audit only)
            sys.stderr.write(f"  [Pass 2] Final review...\n")
            issues = _final_review(
                source_lines[:n], corrected,
                context_text, endpoint, timeout_s,
                extra_payload, use_stream, verbose,
                source_lang, target_lang,
            )
            if issues:
                sys.stderr.write(f"  [Pass 2] {len(issues)} issues found → confused.md\n")
                for issue in issues:
                    line_num = int(issue["line"])
                    idx = line_num - 1
                    src = source_lines[idx] if 0 <= idx < n else "?"
                    tgt = corrected[idx] if 0 <= idx < n else "?"
                    confused_entries.append(
                        f"## {stem} line [{issue['line']}]\n"
                        f"- Source: {src}\n"
                        f"- Translation: {tgt}\n"
                        f"- Issue: {issue['issue']}\n"
                    )
            else:
                sys.stderr.write(f"  [Pass 2] No issues found.\n")

            # Apply corrections and write translated SRT
            apply_translations(trans_blocks, trans_refs[:n], corrected)
            corrected_srt = write_srt(trans_blocks)
            with open(translated_path, "w", encoding="utf-8") as f:
                f.write(corrected_srt)

            # Copy to out/
            out_path = os.path.join(out_dir, stem + suffix)
            shutil.copy2(translated_path, out_path)
            sys.stderr.write(f"[OK] Wrote: {out_path}\n")

        # Write confused.md
        if confused_entries:
            with open(paths.confused_md, "w", encoding="utf-8") as f:
                f.write("# Lines needing human review\n\n")
                f.write("\n".join(confused_entries))
            sys.stderr.write(
                f"[PROOFREAD] Wrote confused.md ({len(confused_entries)} entries)\n"
            )

    # Mark step done
    manifest["proofread_done"] = True
    save_manifest(tmp_dir, manifest)
