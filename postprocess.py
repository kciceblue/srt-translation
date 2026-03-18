#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess.py — Step 4: Flag unfit translations.

Compares source + translated pairs against context and vocab,
flags lines that need correction. Writes flags.json per series.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import List, Optional, Dict, Any

from core import (
    read_text_file,
    parse_srt,
    build_line_mapping,
    format_numbered_input,
    parse_vocab,
    format_vocab,
    call_llm,
    load_manifest,
    save_manifest,
    TmpPaths,
)


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------

_FLAG_LINE_RE = re.compile(
    r"^\[(\d+)\]\s*(PASS|FLAG)\b[:\s]*(.*?)$",
    re.MULTILINE | re.IGNORECASE,
)


def _flag_chunk(
    source_chunk: List[str],
    translated_chunk: List[str],
    start_index: int,
    context_text: str,
    vocab_table: str,
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool,
    verbose: bool,
    source_lang: str,
    target_lang: str,
) -> Dict[str, str]:
    """Flag unfit translations in a chunk. Returns {line_num_str: reason}."""
    system = (
        f"You are a {source_lang} to {target_lang} translation quality reviewer.\n"
        "Review each source → translated pair and flag lines with problems.\n"
        "\n"
        "Output format (one line per input, in order):\n"
        "[N] PASS\n"
        "[N] FLAG: \"wrong_word\" → candidate1, candidate2; short reason\n"
        "\n"
        "Examples:\n"
        "[3] PASS\n"
        "[5] FLAG: \"学校\" → 学园; vocab mismatch\n"
        "[8] FLAG: \"??彼は行った??\" → uncertain, has ?? marker\n"
        "[12] FLAG: \"她很高兴\" → 她很伤心; meaning reversed from source\n"
        "\n"
        "Flag a line when:\n"
        "- Translation contradicts vocabulary sheet\n"
        "- Meaning is reversed or significantly wrong\n"
        "- Key information is missing\n"
        "- Line contains ?? markers (uncertain translation)\n"
        "- Translation doesn't match the source at all\n"
        "\n"
        "Do NOT flag minor style differences or slightly awkward phrasing.\n"
        "\n"
        "Rules:\n"
        "- Quote the problematic word, list candidates after →, reason under 10 words.\n"
        "- Output ONLY the [N] lines. No other text."
    )

    parts = []
    if context_text:
        parts.append(f"CONTEXT:\n{context_text}")
    if vocab_table:
        parts.append(f"VOCABULARY:\n{vocab_table}")

    pairs = []
    for i, (src, tgt) in enumerate(zip(source_chunk, translated_chunk)):
        pairs.append(f"[{start_index + i}] {src} → {tgt}")
    parts.append("Lines to review:\n" + "\n".join(pairs))
    user = "\n\n".join(parts)

    flags: Dict[str, str] = {}
    try:
        raw = call_llm(
            system=system, user=user,
            endpoint=endpoint, timeout_s=timeout_s,
            extra_payload=extra_payload, use_stream=use_stream,
            verbose=verbose,
            expected_output_len=len(source_chunk) * 30,
            runaway_multiplier=3.0,
            raise_on_runaway=True,
        )
        for m in _FLAG_LINE_RE.finditer(raw):
            num = m.group(1)
            verdict = m.group(2).upper()
            reason = m.group(3).strip()
            if verdict == "FLAG" and reason:
                flags[num] = reason
    except RuntimeError:
        # On runaway, retry once then mark chunk as unflagged
        try:
            raw = call_llm(
                system=system, user=user,
                endpoint=endpoint, timeout_s=timeout_s,
                extra_payload=extra_payload, use_stream=use_stream,
                verbose=verbose,
                expected_output_len=len(source_chunk) * 30,
                runaway_multiplier=3.0,
            )
            for m in _FLAG_LINE_RE.finditer(raw):
                num = m.group(1)
                verdict = m.group(2).upper()
                reason = m.group(3).strip()
                if verdict == "FLAG" and reason:
                    flags[num] = reason
        except Exception as e:
            sys.stderr.write(f"[WARN] Flagging retry failed: {e}. Marking chunk as unflagged.\n")
    except Exception as e:
        sys.stderr.write(f"[WARN] Flagging failed: {e}. Marking chunk as unflagged.\n")

    return flags


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_postprocess_step(
    manifest: dict,
    endpoint: str,
    timeout_s: int = 300,
    extra_payload: Optional[Dict[str, Any]] = None,
    use_stream: bool = True,
    verbose: bool = False,
    tmp_dir: str = "./tmp",
    source_lang: str = "Japanese",
    target_lang: str = "Simplified Chinese",
    chunk_size: int = 20,
) -> None:
    """Step 4: Flag unfit translations → flags.json per series."""
    series_list = manifest.get("series", [])
    if not series_list:
        sys.stderr.write("[WARN] No series in manifest. Skipping postprocess.\n")
        return

    for si, series_info in enumerate(series_list):
        series_name = series_info["name"]
        files = series_info["files"]
        paths = TmpPaths(tmp_dir, series_name)

        sys.stderr.write(f"\n[POSTPROCESS] Series: \"{series_name}\" ({len(files)} files)\n")

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

        all_flags: Dict[str, Dict[str, str]] = {}

        for fi, file_info in enumerate(files):
            stem = file_info["stem"]
            source_path = paths.source_srt(stem)
            translated_path = paths.translated_srt(stem)

            sys.stderr.write(f"\n[FILE {fi + 1}/{len(files)}] {stem}\n")

            if not os.path.isfile(translated_path):
                sys.stderr.write(f"[WARN] Translated file not found: {translated_path}, skipping.\n")
                continue

            # Read source and translated
            source_content = read_text_file(source_path)
            source_blocks = parse_srt(source_content)
            if not source_blocks:
                continue
            source_lines, _ = build_line_mapping(source_blocks)

            trans_content = read_text_file(translated_path)
            trans_blocks = parse_srt(trans_content)
            if not trans_blocks:
                continue
            translated_lines, _ = build_line_mapping(trans_blocks)

            n = min(len(source_lines), len(translated_lines))
            file_flags: Dict[str, str] = {}

            total_chunks = (n + chunk_size - 1) // chunk_size
            for chunk_start in range(0, n, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n)
                chunk_num = chunk_start // chunk_size + 1
                sys.stderr.write(
                    f"  [Chunk {chunk_num}/{total_chunks}] Lines {chunk_start + 1}-{chunk_end}\n"
                )

                chunk_flags = _flag_chunk(
                    source_lines[chunk_start:chunk_end],
                    translated_lines[chunk_start:chunk_end],
                    start_index=chunk_start + 1,
                    context_text=context_text,
                    vocab_table=vocab_table,
                    endpoint=endpoint,
                    timeout_s=timeout_s,
                    extra_payload=extra_payload,
                    use_stream=use_stream,
                    verbose=verbose,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                file_flags.update(chunk_flags)

            if file_flags:
                all_flags[stem] = file_flags
                sys.stderr.write(f"  [FLAGS] {len(file_flags)} lines flagged for {stem}\n")
            else:
                sys.stderr.write(f"  [FLAGS] No flags for {stem}\n")

        # Write flags.json
        with open(paths.flags_json, "w", encoding="utf-8") as f:
            json.dump(all_flags, f, indent=2, ensure_ascii=False)
        sys.stderr.write(f"[POSTPROCESS] Wrote flags.json ({sum(len(v) for v in all_flags.values())} total flags)\n")

    # Mark step done
    manifest["postprocess_done"] = True
    save_manifest(tmp_dir, manifest)
