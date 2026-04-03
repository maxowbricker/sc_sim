#!/usr/bin/env python3
"""
Find duplicate PDFs under a directory.

1) Default: group by SHA-256 of raw file bytes (identical downloads only).

2) With --by-doi: also group by DOI extracted from the PDF (same article, possibly
   different bytes — e.g. different "Open Access Support" footers on the first page).

This script only reports duplicates; it does not delete or move files.

Byte-only usage:
  python scripts/find_duplicate_pdfs.py "/path/to/papers of relevance"

Include same-DOI / different-hash pairs (recommended for ACM-style variants):
  python scripts/find_duplicate_pdfs.py "/path/to/papers" --by-doi

Optional (clearer DOI extraction on compressed streams):
  pip install -r scripts/requirements-pdf-dedup.txt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict

# DOI: https://www.doi.org/doi_handbook/2_Numbering.html#2.2
# Practical pattern (ASCII DOIs in PDFs)
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\)\]\}\"\'>]+", re.IGNORECASE)

# Try to strip trailing punctuation DOI regex might have captured
_DOI_TRAIL = re.compile(r"[\s\.\,\;\:\)\]\}\"\'>]+$")


def _normalize_doi(raw: str) -> str:
    s = raw.strip()
    s = _DOI_TRAIL.sub("", s)
    return s.lower()


def _try_pypdf_extract(path: str, max_pages: int = 2) -> str | None:
    """Prefer the article DOI: first match on page 0, then page 1, then metadata strings."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(path)
    except Exception:
        return None

    def first_doi(text: str) -> str | None:
        if not text:
            return None
        m = DOI_PATTERN.search(text)
        return _normalize_doi(m.group(0)) if m else None

    n = min(max_pages, len(reader.pages))
    for i in range(n):
        try:
            t = reader.pages[i].extract_text() or ""
        except Exception:
            t = ""
        d = first_doi(t)
        if d:
            return d

    meta = reader.metadata
    if meta:
        for _k, v in (meta or {}).items():
            if isinstance(v, str):
                d = first_doi(v)
                if d:
                    return d
    return None


def _fallback_doi_from_bytes(path: str, max_read: int = 3_000_000) -> str | None:
    """
    Best-effort: search the first max_read bytes for an ASCII DOI.
    Works when DOI appears uncompressed; may miss compressed first pages or pick a reference DOI.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(max_read)
    except OSError:
        return None
    text = data.decode("latin-1", errors="ignore")
    m = DOI_PATTERN.search(text)
    if m:
        return _normalize_doi(m.group(0))
    return None


def extract_primary_doi(path: str) -> str | None:
    """Single canonical DOI per file, or None."""
    d = _try_pypdf_extract(path)
    if d:
        return d
    return _fallback_doi_from_bytes(path)


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iter_pdf_paths(
    root: str,
    ext: str,
    min_size: int,
) -> tuple[list[str], int]:
    """Returns (paths, skipped_small_count)."""
    paths: list[str] = []
    skipped_small = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        if "/." in dirpath or dirpath.endswith("/.") or os.path.basename(dirpath).startswith("."):
            continue
        for name in filenames:
            if name.startswith("."):
                continue
            if not name.lower().endswith(ext):
                continue
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError as e:
                print(f"warning: skip (stat): {path}: {e}", file=sys.stderr)
                continue
            if size < min_size:
                skipped_small += 1
                continue
            paths.append(path)
    return paths, skipped_small


def report_byte_duplicates(hash_to_paths: dict[str, list[str]]) -> int:
    duplicate_groups = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    redundant_count = sum(len(paths) - 1 for paths in duplicate_groups.values())

    print(f"Unique SHA-256 values: {len(hash_to_paths)}")
    print(f"Duplicate groups (same bytes, 2+ paths): {len(duplicate_groups)}")
    print(f"Redundant copies (one keeper per group): {redundant_count}")
    print()

    if not duplicate_groups:
        print("No duplicate PDFs found (by byte identity).")
        return 0

    for digest in sorted(duplicate_groups, key=lambda d: min(duplicate_groups[d])):
        paths = sorted(duplicate_groups[digest])
        print(f"SHA-256: {digest}")
        print(f"  Count: {len(paths)}")
        for p in paths:
            try:
                sz = os.path.getsize(p)
            except OSError:
                sz = -1
            print(f"    {sz:>12}  {p}")
        print()
    return 0


def report_doi_groups(
    paths: list[str],
    path_to_hash: dict[str, str],
) -> None:
    try:
        import pypdf  # noqa: F401
        _has_pypdf = True
    except ImportError:
        _has_pypdf = False

    print("=" * 72)
    print("DOI-based grouping (same article, possibly different file bytes)")
    print(
        "Extraction: "
        + ("pypdf (pages 0–1, then metadata)" if _has_pypdf else "byte-scan fallback — pip install pypdf for reliable text")
    )
    print()

    doi_to_paths: dict[str, list[str]] = defaultdict(list)
    no_doi: list[str] = []
    for p in paths:
        doi = extract_primary_doi(p)
        if doi:
            doi_to_paths[doi].append(p)
        else:
            no_doi.append(p)

    multi = {d: sorted(ps) for d, ps in doi_to_paths.items() if len(ps) > 1}
    different_bytes = 0
    for _doi, ps in multi.items():
        if len({path_to_hash[p] for p in ps}) > 1:
            different_bytes += 1

    with_doi = len(paths) - len(no_doi)
    print(f"PDFs where a DOI was detected: {with_doi} / {len(paths)}")
    print(f"DOIs that appear on 2+ files: {len(multi)}")
    print(f"  Of those, groups with different SHA-256 (e.g. OA footer / re-download): {different_bytes}")
    print(f"PDFs with no DOI detected: {len(no_doi)}")
    print()

    if not multi:
        print("No DOI collisions (same DOI on multiple files).")
        if no_doi:
            print("(Many PDFs may lack an extractable DOI without pypdf; try: pip install pypdf)")
        print()
        return

    for doi in sorted(multi, key=lambda d: min(multi[d])):
        ps = multi[doi]
        hashes = [path_to_hash[p] for p in ps]
        unique_h = len(set(hashes))
        print(f"DOI: {doi}")
        print(f"  Files: {len(ps)}  |  Distinct SHA-256: {unique_h}")
        if unique_h > 1:
            print("  Note: same DOI but different bytes — e.g. OA footer, download timestamp, or re-export.")
        for p in ps:
            try:
                sz = os.path.getsize(p)
            except OSError:
                sz = -1
            print(f"    {path_to_hash[p][:16]}…  {sz:>12}  {p}")
        print()

    if no_doi and len(no_doi) <= 30:
        print("Files with no DOI detected (first 30):")
        for p in sorted(no_doi)[:30]:
            print(f"  {p}")
        if len(no_doi) > 30:
            print(f"  ... and {len(no_doi) - 30} more")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report duplicate PDFs: by SHA-256 bytes and optionally by DOI."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan recursively (default: current directory).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        metavar="BYTES",
        help="Skip files smaller than this many bytes (default: 0).",
    )
    parser.add_argument(
        "--ext",
        default=".pdf",
        help="File extension to match, case-insensitive (default: .pdf).",
    )
    parser.add_argument(
        "--by-doi",
        action="store_true",
        help="Also group by DOI (catches same paper with different bytes, e.g. ACM OA footers).",
    )
    args = parser.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    ext = args.ext.lower()
    if not ext.startswith("."):
        ext = "." + ext

    paths, skipped_small = iter_pdf_paths(root, ext, args.min_size)

    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    path_to_hash: dict[str, str] = {}

    for path in paths:
        try:
            digest = sha256_file(path)
        except OSError as e:
            print(f"warning: skip (read): {path}: {e}", file=sys.stderr)
            continue
        path_to_hash[path] = digest
        hash_to_paths[digest].append(path)

    scanned = len(path_to_hash)

    print(f"Root: {root}")
    print(f"PDFs scanned: {scanned}")
    if skipped_small:
        print(f"Skipped (smaller than --min-size): {skipped_small}")
    print()

    report_byte_duplicates(hash_to_paths)

    if args.by_doi:
        if not path_to_hash:
            print("No PDFs to analyze for DOI.")
            return 0
        report_doi_groups(list(path_to_hash.keys()), path_to_hash)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
