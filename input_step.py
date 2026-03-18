#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
input_step.py — Step 1: Input expansion, series grouping, tmp folder setup.

Expands file/directory/glob inputs, warns about non-SRT files,
groups files by series using the LLM, creates tmp folder structure,
and writes the initial manifest.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from typing import List, Tuple, Optional, Dict, Any

from core import (
    call_llm,
    expand_inputs,
    save_manifest,
    TmpPaths,
)


# ---------------------------------------------------------------------------
# Non-SRT warning
# ---------------------------------------------------------------------------

def warn_non_srt(paths: List[str]) -> List[str]:
    """Filter non-.srt files, print warnings, return only .srt paths."""
    srt_paths: List[str] = []
    for p in paths:
        if p.lower().endswith(".srt"):
            srt_paths.append(p)
        else:
            sys.stderr.write(f"[WARN] Skipping non-SRT file: {p}\n")
    return srt_paths


# ---------------------------------------------------------------------------
# Series grouping (extracted from main.py, adapted for call_llm)
# ---------------------------------------------------------------------------

def group_files_by_series(
    file_paths: List[str],
    endpoint: str,
    timeout_s: int,
    extra_payload: Optional[Dict[str, Any]],
    use_stream: bool = True,
    verbose: bool = False,
) -> List[Tuple[str, List[str]]]:
    """Use the LLM to group subtitle files by series and sort by episode order."""
    if len(file_paths) <= 2:
        return [("All", file_paths)]

    # Build basename-to-path lookup (handle duplicate basenames)
    basenames = [os.path.basename(p) for p in file_paths]
    has_dupes = len(set(basenames)) < len(basenames)

    if has_dupes:
        common = os.path.commonpath(file_paths)
        display_names = [os.path.relpath(p, common) for p in file_paths]
    else:
        display_names = basenames

    name_to_path: Dict[str, str] = {}
    for display, full in zip(display_names, file_paths):
        name_to_path[display] = full

    file_list_text = "\n".join(display_names)

    system = (
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
    )
    user = f"Filenames:\n{file_list_text}"

    try:
        sys.stderr.write(f"[INFO] Grouping {len(file_paths)} files by series...\n")
        if verbose:
            sys.stderr.write(f"  [DETAIL] Files to group:\n")
            for dn in display_names:
                sys.stderr.write(f"    {dn}\n")

        raw = call_llm(
            system=system,
            user=user,
            endpoint=endpoint,
            timeout_s=min(timeout_s, 60),
            extra_payload=extra_payload,
            use_stream=use_stream,
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

        for series, paths in groups:
            sys.stderr.write(f'[GROUP] "{series}" ({len(paths)} files)\n')

        return groups

    except Exception as e:
        sys.stderr.write(f"[WARN] Series grouping failed: {e}. Treating all files as one group.\n")
        return [("All", file_paths)]


# ---------------------------------------------------------------------------
# Tmp folder setup
# ---------------------------------------------------------------------------

def setup_tmp_folder(
    series_groups: List[Tuple[str, List[str]]],
    tmp_dir: str,
) -> dict:
    """Create tmp dirs, copy source SRTs, return manifest dict."""
    os.makedirs(tmp_dir, exist_ok=True)

    manifest: dict = {
        "series": [],
    }

    for series_name, file_paths in series_groups:
        paths = TmpPaths(tmp_dir, series_name)
        os.makedirs(paths.series_dir, exist_ok=True)

        files_info: List[dict] = []
        for fp in file_paths:
            stem = os.path.splitext(os.path.basename(fp))[0]
            dest = paths.source_srt(stem)
            shutil.copy2(fp, dest)
            files_info.append({
                "original": os.path.abspath(fp),
                "stem": stem,
            })

        manifest["series"].append({
            "name": series_name,
            "dir": paths.series_dir,
            "files": files_info,
        })

    return manifest


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_input_step(
    inputs: List[str],
    endpoint: str,
    timeout_s: int = 300,
    extra_payload: Optional[Dict[str, Any]] = None,
    use_stream: bool = True,
    verbose: bool = False,
    tmp_dir: str = "./tmp",
    no_group: bool = False,
) -> dict:
    """Step 1: expand inputs → warn non-SRT → group → setup tmp → write manifest."""
    # Expand
    all_paths = expand_inputs(inputs)
    if not all_paths:
        sys.stderr.write("[ERROR] No valid files found.\n")
        return {}

    # Filter non-SRT
    srt_paths = warn_non_srt(all_paths)
    if not srt_paths:
        sys.stderr.write("[ERROR] No .srt files found.\n")
        return {}

    sys.stderr.write(f"[INPUT] {len(srt_paths)} SRT files found.\n")

    # Group by series
    if no_group:
        groups: List[Tuple[str, List[str]]] = [("All", srt_paths)]
    else:
        groups = group_files_by_series(
            srt_paths, endpoint, timeout_s, extra_payload,
            use_stream=use_stream, verbose=verbose,
        )

    # Setup tmp folder
    manifest = setup_tmp_folder(groups, tmp_dir)

    # Save manifest
    save_manifest(tmp_dir, manifest)
    sys.stderr.write(f"[INPUT] Manifest written to {tmp_dir}/manifest.json\n")

    return manifest
