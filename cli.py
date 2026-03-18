#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli.py — Single CLI entry point for the SRT translation pipeline.

Subcommands:
  run         Full pipeline (input → preprocess → translate → postprocess → proofread)
  input       Step 1: Expand inputs, series grouping, tmp setup
  preprocess  Step 2: ASR flag/fix, context, term extraction
  translate   Step 3: Chunked translation
  postprocess Step 4: Flag unfit translations
  proofread   Step 5: Fix flagged lines, final review, copy to out/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Optional, Dict, Any

from core import load_manifest, save_manifest
from input_step import run_input_step
from preprocess import run_preprocess_step
from translate import run_translate_step
from postprocess import run_postprocess_step
from proofread import run_proofread_step


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments common to all subcommands."""
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:5000/v1/chat/completions",
        help="Backend POST endpoint URL",
    )
    parser.add_argument(
        "--source-lang",
        default="Japanese",
        help="Source language name (default: Japanese)",
    )
    parser.add_argument(
        "--target-lang",
        default="Simplified Chinese",
        help="Target language name (default: Simplified Chinese)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP timeout seconds (default: 300)",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=2,
        help="Retry count on failure (default: 2)",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="Seconds between retries (default: 1.0)",
    )
    parser.add_argument(
        "--extra-payload",
        default="",
        help="Extra JSON fields for POST body (e.g. '{\"model\":\"local\",\"temperature\":0}')",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming mode",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Verbose output + preserve tmp folder after completion",
    )
    parser.add_argument(
        "--tmp-dir",
        default="./tmp",
        help="Tmp folder path (default: ./tmp)",
    )


def _parse_extra_payload(raw: str) -> Optional[Dict[str, Any]]:
    """Parse --extra-payload JSON string. Returns None if empty."""
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("extra payload must be a JSON object")
        return payload
    except Exception as e:
        print(f"ERROR: --extra-payload is not valid JSON object: {e}", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Subcommand: run (full pipeline)
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    extra_payload = _parse_extra_payload(args.extra_payload)
    use_stream = not args.no_stream
    verbose = args.debug

    # Step 1: Input
    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.write("[PIPELINE] Step 1: Input\n")
    sys.stderr.write("=" * 60 + "\n")
    manifest = run_input_step(
        inputs=args.inputs,
        endpoint=args.endpoint,
        timeout_s=args.timeout,
        extra_payload=extra_payload,
        use_stream=use_stream,
        verbose=verbose,
        tmp_dir=args.tmp_dir,
        no_group=args.no_group,
    )
    if not manifest:
        return 2

    # Step 2: Preprocess
    sys.stderr.write("\n" + "=" * 60 + "\n")
    sys.stderr.write("[PIPELINE] Step 2: Preprocess\n")
    sys.stderr.write("=" * 60 + "\n")
    run_preprocess_step(
        manifest=manifest,
        endpoint=args.endpoint,
        timeout_s=args.timeout,
        extra_payload=extra_payload,
        use_stream=use_stream,
        verbose=verbose,
        tmp_dir=args.tmp_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        retry=args.retry,
        retry_sleep_s=args.retry_sleep,
    )

    # Step 3: Translate
    sys.stderr.write("\n" + "=" * 60 + "\n")
    sys.stderr.write("[PIPELINE] Step 3: Translate\n")
    sys.stderr.write("=" * 60 + "\n")
    run_translate_step(
        manifest=manifest,
        endpoint=args.endpoint,
        timeout_s=args.timeout,
        extra_payload=extra_payload,
        use_stream=use_stream,
        verbose=verbose,
        tmp_dir=args.tmp_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        chunk_size=args.chunk_size,
        repetition_penalty=args.repetition_penalty,
        vocab_path=args.vocab,
        retry=args.retry,
        retry_sleep_s=args.retry_sleep,
    )

    # Step 4: Postprocess
    sys.stderr.write("\n" + "=" * 60 + "\n")
    sys.stderr.write("[PIPELINE] Step 4: Postprocess\n")
    sys.stderr.write("=" * 60 + "\n")
    run_postprocess_step(
        manifest=manifest,
        endpoint=args.endpoint,
        timeout_s=args.timeout,
        extra_payload=extra_payload,
        use_stream=use_stream,
        verbose=verbose,
        tmp_dir=args.tmp_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        chunk_size=args.chunk_size,
    )

    # Step 5: Proofread
    sys.stderr.write("\n" + "=" * 60 + "\n")
    sys.stderr.write("[PIPELINE] Step 5: Proofread\n")
    sys.stderr.write("=" * 60 + "\n")
    run_proofread_step(
        manifest=manifest,
        endpoint=args.endpoint,
        timeout_s=args.timeout,
        extra_payload=extra_payload,
        use_stream=use_stream,
        verbose=verbose,
        tmp_dir=args.tmp_dir,
        out_dir=args.out_dir,
        suffix=args.suffix,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        retry=args.retry,
        retry_sleep_s=args.retry_sleep,
    )

    sys.stderr.write("\n" + "=" * 60 + "\n")
    sys.stderr.write("[PIPELINE] Complete!\n")
    sys.stderr.write("=" * 60 + "\n")

    # Clean tmp unless debug
    if not args.debug:
        shutil.rmtree(args.tmp_dir, ignore_errors=True)
        sys.stderr.write("[CLEANUP] Removed tmp folder.\n")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: input
# ---------------------------------------------------------------------------

def cmd_input(args: argparse.Namespace) -> int:
    extra_payload = _parse_extra_payload(args.extra_payload)
    manifest = run_input_step(
        inputs=args.inputs,
        endpoint=args.endpoint,
        timeout_s=args.timeout,
        extra_payload=extra_payload,
        use_stream=not args.no_stream,
        verbose=args.debug,
        tmp_dir=args.tmp_dir,
        no_group=args.no_group,
    )
    if not manifest:
        return 2
    manifest["input_done"] = True
    save_manifest(args.tmp_dir, manifest)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: preprocess
# ---------------------------------------------------------------------------

def cmd_preprocess(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.tmp_dir)
    if not manifest.get("series"):
        sys.stderr.write("[ERROR] No manifest found. Run 'input' step first.\n")
        return 2
    extra_payload = _parse_extra_payload(args.extra_payload)
    run_preprocess_step(
        manifest=manifest,
        endpoint=args.endpoint,
        timeout_s=args.timeout,
        extra_payload=extra_payload,
        use_stream=not args.no_stream,
        verbose=args.debug,
        tmp_dir=args.tmp_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        retry=args.retry,
        retry_sleep_s=args.retry_sleep,
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: translate
# ---------------------------------------------------------------------------

def cmd_translate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.tmp_dir)
    if not manifest.get("series"):
        sys.stderr.write("[ERROR] No manifest found. Run 'input' step first.\n")
        return 2
    if not manifest.get("preprocess_done"):
        sys.stderr.write("[WARN] Preprocess step not marked done. Proceeding anyway.\n")
    extra_payload = _parse_extra_payload(args.extra_payload)
    run_translate_step(
        manifest=manifest,
        endpoint=args.endpoint,
        timeout_s=args.timeout,
        extra_payload=extra_payload,
        use_stream=not args.no_stream,
        verbose=args.debug,
        tmp_dir=args.tmp_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        chunk_size=args.chunk_size,
        repetition_penalty=args.repetition_penalty,
        vocab_path=args.vocab,
        retry=args.retry,
        retry_sleep_s=args.retry_sleep,
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: postprocess
# ---------------------------------------------------------------------------

def cmd_postprocess(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.tmp_dir)
    if not manifest.get("series"):
        sys.stderr.write("[ERROR] No manifest found. Run 'input' step first.\n")
        return 2
    if not manifest.get("translate_done"):
        sys.stderr.write("[WARN] Translate step not marked done. Proceeding anyway.\n")
    extra_payload = _parse_extra_payload(args.extra_payload)
    run_postprocess_step(
        manifest=manifest,
        endpoint=args.endpoint,
        timeout_s=args.timeout,
        extra_payload=extra_payload,
        use_stream=not args.no_stream,
        verbose=args.debug,
        tmp_dir=args.tmp_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        chunk_size=args.chunk_size,
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: proofread
# ---------------------------------------------------------------------------

def cmd_proofread(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.tmp_dir)
    if not manifest.get("series"):
        sys.stderr.write("[ERROR] No manifest found. Run 'input' step first.\n")
        return 2
    if not manifest.get("postprocess_done"):
        sys.stderr.write("[WARN] Postprocess step not marked done. Proceeding anyway.\n")
    extra_payload = _parse_extra_payload(args.extra_payload)
    run_proofread_step(
        manifest=manifest,
        endpoint=args.endpoint,
        timeout_s=args.timeout,
        extra_payload=extra_payload,
        use_stream=not args.no_stream,
        verbose=args.debug,
        tmp_dir=args.tmp_dir,
        out_dir=args.out_dir,
        suffix=args.suffix,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        context_radius=args.context_radius,
        retry=args.retry,
        retry_sleep_s=args.retry_sleep,
    )
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="srt-translate",
        description="SRT subtitle translation pipeline (5 steps).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to run")

    # --- run (full pipeline) ---
    p_run = subparsers.add_parser("run", help="Run full pipeline")
    _add_common_args(p_run)
    p_run.add_argument("inputs", nargs="+", help="Input .srt files, directories, or globs")
    p_run.add_argument("--out-dir", default="out", help="Output directory (default: out)")
    p_run.add_argument("--suffix", default=".zh.srt", help="Output filename suffix (default: .zh.srt)")
    p_run.add_argument("--no-group", action="store_true", help="Disable series grouping")
    p_run.add_argument("--chunk-size", type=int, default=10, help="Lines per translation chunk (default: 10)")
    p_run.add_argument("--repetition-penalty", type=float, default=1.3, help="Repetition penalty (default: 1.3)")
    p_run.add_argument("--vocab", default="vocab.txt", help="External vocabulary file (default: vocab.txt)")
    p_run.set_defaults(func=cmd_run)

    # --- input ---
    p_input = subparsers.add_parser("input", help="Step 1: Input expansion and series grouping")
    _add_common_args(p_input)
    p_input.add_argument("inputs", nargs="+", help="Input .srt files, directories, or globs")
    p_input.add_argument("--no-group", action="store_true", help="Disable series grouping")
    p_input.set_defaults(func=cmd_input)

    # --- preprocess ---
    p_pre = subparsers.add_parser("preprocess", help="Step 2: ASR fix, context, term extraction")
    _add_common_args(p_pre)
    p_pre.set_defaults(func=cmd_preprocess)

    # --- translate ---
    p_trans = subparsers.add_parser("translate", help="Step 3: Chunked translation")
    _add_common_args(p_trans)
    p_trans.add_argument("--chunk-size", type=int, default=10, help="Lines per translation chunk (default: 10)")
    p_trans.add_argument("--repetition-penalty", type=float, default=1.3, help="Repetition penalty (default: 1.3)")
    p_trans.add_argument("--vocab", default="vocab.txt", help="External vocabulary file (default: vocab.txt)")
    p_trans.set_defaults(func=cmd_translate)

    # --- postprocess ---
    p_post = subparsers.add_parser("postprocess", help="Step 4: Flag unfit translations")
    _add_common_args(p_post)
    p_post.add_argument("--chunk-size", type=int, default=20, help="Lines per flagging chunk (default: 20)")
    p_post.set_defaults(func=cmd_postprocess)

    # --- proofread ---
    p_proof = subparsers.add_parser("proofread", help="Step 5: Correct flagged lines, final review")
    _add_common_args(p_proof)
    p_proof.add_argument("--out-dir", default="out", help="Output directory (default: out)")
    p_proof.add_argument("--suffix", default=".zh.srt", help="Output filename suffix (default: .zh.srt)")
    p_proof.add_argument("--context-radius", type=int, default=50, help="Context lines around flagged line (default: 50)")
    p_proof.set_defaults(func=cmd_proofread)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 2

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
