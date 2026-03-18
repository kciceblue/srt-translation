#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate.py — Step 3: Chunked translation with vocab+context.

Translates pre-processed source SRTs using numbered-line protocol.
Reads context.md and vocab.md from tmp folder, merges with external vocab.
Accumulates per-series glossary across episodes.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple, Optional, Dict, Any

from core import (
    read_text_file,
    parse_srt,
    write_srt,
    build_line_mapping,
    apply_translations,
    format_numbered_input,
    parse_numbered_output,
    parse_vocab,
    format_vocab,
    post_messages,
    call_llm,
    load_manifest,
    save_manifest,
    TmpPaths,
    _NUMBERED_LINE_RE,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Translation core (adapted from main.py)
# ---------------------------------------------------------------------------

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

    # Literal translation prompt
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
        numbered = format_numbered_input(chunk, start_index=start_idx)
        msgs = [
            {"role": "system", "content": pass1_prompt},
            {"role": "user", "content": user_prefix + numbered},
        ]

        out_text = _call_backend(msgs, hint_len=len(numbered))
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
                repair_text = _call_backend(repair_msgs, hint_len=len("\n".join(repair_lines)))
                for m in _NUMBERED_LINE_RE.finditer(repair_text):
                    num = int(m.group(1))
                    idx = num - start_idx
                    if 0 <= idx < len(chunk) and translations[idx] is None:
                        translations[idx] = m.group(2)
            except Exception as e:
                sys.stderr.write(f"[WARN] {chunk_label} repair failed: {e}\n")

        return [t if t is not None else "" for t in translations]

    num_chunks = (len(lines) + chunk_size - 1) // chunk_size
    sys.stderr.write(f"[INFO] Translating {len(lines)} lines in {num_chunks} chunks of {chunk_size}...\n")

    translated: List[str] = []
    for ci in range(0, len(lines), chunk_size):
        chunk = lines[ci : ci + chunk_size]
        chunk_num = ci // chunk_size + 1
        sys.stderr.write(f"[CHUNK {chunk_num}/{num_chunks}] Lines {ci + 1}-{ci + len(chunk)}\n")
        result = _translate_chunk(chunk, start_idx=ci + 1, chunk_label=f"chunk_{chunk_num}")
        translated.extend(result)

    sys.stderr.write(f"[OK] All {len(lines)} lines translated.\n")
    return translated


# ---------------------------------------------------------------------------
# Glossary extraction (adapted from main.py)
# ---------------------------------------------------------------------------

def extract_glossary(
    source_lines: List[str],
    translated_lines: List[str],
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    existing_glossary: str,
    use_stream: bool = True,
    verbose: bool = False,
) -> str:
    """Extract character names and key terms from a translated episode."""
    def _sample(lines: List[str], n: int = 50) -> str:
        if len(lines) <= n * 2:
            return "\n".join(lines)
        return "\n".join(lines[:n] + ["..."] + lines[-n:])

    system = (
        "You are a translation glossary extractor. Given subtitle lines "
        "(source and translation), extract character names and key terms.\n\n"
        "Output format (one per line):\n"
        "SourceTerm → TranslatedTerm\n\n"
        "Rules:\n"
        "- Include character names, place names, and recurring proper nouns only.\n"
        "- Maximum 50 entries.\n"
        "- If a previous glossary is provided, merge with it (keep latest translation for conflicts).\n"
        "- Output ONLY glossary lines, no commentary."
    )
    user = (
        f"Previous glossary:\n{existing_glossary or '(none)'}\n\n"
        f"Source lines (sample):\n{_sample(source_lines)}\n\n"
        f"Translated lines (sample):\n{_sample(translated_lines)}"
    )

    try:
        raw = call_llm(
            system=system, user=user,
            endpoint=endpoint, timeout_s=min(timeout_s, 60),
            extra_payload=extra_payload, use_stream=use_stream,
            verbose=verbose,
        )
        if "→" in raw or "->" in raw:
            if verbose:
                sys.stderr.write(f"  [DETAIL] Glossary extracted ({raw.count(chr(10)) + 1} entries)\n")
            return raw.strip()
        return existing_glossary
    except Exception as e:
        sys.stderr.write(f"[WARN] Glossary extraction failed: {e}\n")
        return existing_glossary


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_translate_step(
    manifest: dict,
    endpoint: str,
    timeout_s: int = 300,
    extra_payload: Optional[Dict[str, Any]] = None,
    use_stream: bool = True,
    verbose: bool = False,
    tmp_dir: str = "./tmp",
    source_lang: str = "Japanese",
    target_lang: str = "Simplified Chinese",
    chunk_size: int = 10,
    repetition_penalty: float = 1.3,
    vocab_path: str = "",
    retry: int = 2,
    retry_sleep_s: float = 1.0,
) -> None:
    """Step 3: Translate pre-processed SRTs. Reads vocab.md + external vocab, writes *.translated.srt."""
    series_list = manifest.get("series", [])
    if not series_list:
        sys.stderr.write("[WARN] No series in manifest. Skipping translate.\n")
        return

    # Inject repetition penalty
    if repetition_penalty != 1.0:
        if extra_payload is None:
            extra_payload = {}
        extra_payload = dict(extra_payload)
        extra_payload.setdefault("repetition_penalty", repetition_penalty)

    # Load external vocab
    external_vocab: Dict[str, str] = {}
    if vocab_path and os.path.isfile(vocab_path):
        raw = read_text_file(vocab_path).strip()
        if raw:
            external_vocab = parse_vocab(raw)
            sys.stderr.write(f"[INFO] Loaded external vocab: {len(external_vocab)} entries from {vocab_path}\n")

    system_prompt = _format_template(_DEFAULT_SYSTEM_PROMPT, source_lang, target_lang)
    user_prefix = _format_template(_DEFAULT_USER_PREFIX, source_lang, target_lang)

    for si, series_info in enumerate(series_list):
        series_name = series_info["name"]
        files = series_info["files"]
        paths = TmpPaths(tmp_dir, series_name)

        sys.stderr.write(f"\n[TRANSLATE] Series: \"{series_name}\" ({len(files)} files)\n")

        # Load series vocab.md
        series_vocab: Dict[str, str] = {}
        if os.path.isfile(paths.vocab_md):
            raw = read_text_file(paths.vocab_md).strip()
            if raw:
                series_vocab = parse_vocab(raw)
                sys.stderr.write(f"[INFO] Loaded series vocab: {len(series_vocab)} entries\n")

        # Merge: external takes priority
        merged_vocab = dict(series_vocab)
        merged_vocab.update(external_vocab)

        # Load context.md
        context_text = ""
        if os.path.isfile(paths.context_md):
            context_text = read_text_file(paths.context_md).strip()

        glossary = ""  # Rolling glossary for series

        for fi, file_info in enumerate(files):
            stem = file_info["stem"]
            source_path = paths.source_srt(stem)
            out_path = paths.translated_srt(stem)

            sys.stderr.write(f"\n[FILE {fi + 1}/{len(files)}] {stem}\n")

            content = read_text_file(source_path)
            blocks = parse_srt(content)
            if not blocks:
                sys.stderr.write(f"[WARN] No blocks in {source_path}, writing empty output.\n")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("")
                continue

            flat_lines, refs = build_line_mapping(blocks)

            # Build augmented system prompt with vocab + glossary + context
            aug_prompt = system_prompt
            if merged_vocab:
                vocab_text = format_vocab(merged_vocab)
                aug_prompt += (
                    "\n\n---\n"
                    "Vocabulary (use these translations consistently):\n"
                    + vocab_text
                )
            if glossary:
                aug_prompt += (
                    "\n\n---\n"
                    "Glossary from previous episodes in this series "
                    "(use these translations consistently):\n"
                    + glossary
                )
            if context_text:
                aug_prompt += (
                    "\n\n---\n"
                    "Context about this series:\n"
                    + context_text
                )

            translated_lines = translate_lines_via_backend(
                lines=flat_lines,
                endpoint=endpoint,
                timeout_s=timeout_s,
                system_prompt=aug_prompt,
                user_prefix=user_prefix,
                extra_payload=extra_payload,
                retry=retry,
                retry_sleep_s=retry_sleep_s,
                use_stream=use_stream,
                verbose=verbose,
                chunk_size=chunk_size,
            )

            apply_translations(blocks, refs, translated_lines)
            out_srt = write_srt(blocks)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(out_srt)
            sys.stderr.write(f"[OK] Wrote: {out_path}\n")

            # Extract glossary for next episode
            is_last = fi == len(files) - 1
            if not is_last and len(files) > 1:
                sys.stderr.write("[INFO] Extracting glossary...\n")
                glossary = extract_glossary(
                    flat_lines, translated_lines,
                    endpoint, timeout_s, extra_payload, glossary,
                    use_stream=use_stream, verbose=verbose,
                )

    # Mark step done
    manifest["translate_done"] = True
    save_manifest(tmp_dir, manifest)
