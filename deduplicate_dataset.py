#!/usr/bin/env python3
"""Audit and materialize a globally deduplicated image dataset.

The default operation is deliberately non-destructive: it scans every supported
image below ``--root`` and writes an auditable manifest.  Exact byte duplicates
and exact decoded-pixel duplicates are safe to resolve automatically.  pHash is
used only to *propose* near-duplicate pairs unless ``--accept-verified-near`` is
explicitly supplied.

Examples
--------
Run a complete audit (recommended first command)::

    python deduplicate_dataset.py --root data_raw --out dedup_audit

Create a clean hard-linked dataset using exact duplicates only::

    python deduplicate_dataset.py --root data_raw --out dedup_audit \
        --materialize data_deduplicated --link-mode hardlink

After visually checking ``near_duplicate_pairs.csv`` and contact sheets, include
the conservatively verified near-duplicate pairs::

    python deduplicate_dataset.py --root data_raw --out dedup_audit_verified \
        --accept-verified-near --materialize data_deduplicated_verified

Important policy decisions
--------------------------
* Matching is global, not per class.  This is required to expose label conflicts.
* Every member of a cross-class duplicate group is excluded.  Selecting one of
  conflicting labels silently would create label noise.
* Within a single class, one deterministic, highest-resolution representative is
  retained.
* The source tree is never modified.  Materialization writes a new tree.
* Weak pHash candidates never drive deletion on their own.  Independently
  verified near-duplicate edges are resolved as connected components so no
  accepted duplicate edge can remain between two retained images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import imagehash
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage import exposure
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}
CACHE_VERSION = 1
@dataclass(frozen=True)
class ImageRecord:
    index: int
    rel_path: str
    class_name: str
    extension: str
    file_size: int
    mtime_ns: int
    sha256: str
    pixel_sha256: str
    width: int
    height: int
    phash: str
    dhash: str
    error: str


@dataclass(frozen=True)
class PairMetrics:
    left: int
    right: int
    phash_distance: int
    dhash_distance: int
    mae: float
    rmse: float
    correlation: float
    verified: bool
    error: str = ""
    equalized_correlation: float = math.nan
    orb_matches: int = 0
    orb_inliers: int = 0
    orb_inlier_ratio: float = 0.0
    verification_rule: str = ""


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def discover_images(root: Path) -> list[Path]:
    paths = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def class_from_path(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else "__root__"


def _hash_image_worker(task: tuple[int, str, str, str, int, int]) -> ImageRecord:
    index, path_str, rel_path, class_name, file_size, mtime_ns = task
    path = Path(path_str)
    sha256 = ""
    pixel_sha256 = ""
    width = height = 0
    phash = dhash = ""
    error = ""
    try:
        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        with Image.open(BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
        width, height = image.size
        pixel_hasher = hashlib.sha256()
        pixel_hasher.update(f"RGB:{width}x{height}:".encode("ascii"))
        pixel_hasher.update(image.tobytes())
        pixel_sha256 = pixel_hasher.hexdigest()
        phash = str(imagehash.phash(image, hash_size=8))
        dhash = str(imagehash.dhash(image, hash_size=8))
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # preserve the audit instead of aborting on one file
        error = f"unexpected {type(exc).__name__}: {exc}"
    return ImageRecord(
        index=index,
        rel_path=rel_path,
        class_name=class_name,
        extension=path.suffix.lower(),
        file_size=file_size,
        mtime_ns=mtime_ns,
        sha256=sha256,
        pixel_sha256=pixel_sha256,
        width=width,
        height=height,
        phash=phash,
        dhash=dhash,
        error=error,
    )


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS image_hashes (
            rel_path TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            extension TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            pixel_sha256 TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            phash TEXT NOT NULL,
            dhash TEXT NOT NULL,
            error TEXT NOT NULL
        )
        """
    )
    return conn


def _record_from_cache(index: int, row: Sequence[object]) -> ImageRecord:
    return ImageRecord(
        index=index,
        rel_path=str(row[0]),
        class_name=str(row[1]),
        extension=str(row[2]),
        file_size=int(row[3]),
        mtime_ns=int(row[4]),
        sha256=str(row[5]),
        pixel_sha256=str(row[6]),
        width=int(row[7]),
        height=int(row[8]),
        phash=str(row[9]),
        dhash=str(row[10]),
        error=str(row[11]),
    )


def analyze_images(
    paths: Sequence[Path], root: Path, cache_path: Path, workers: int
) -> tuple[list[ImageRecord], int]:
    conn = open_cache(cache_path)
    cached: dict[str, Sequence[object]] = {}
    query = (
        "SELECT rel_path,class_name,extension,file_size,mtime_ns,sha256,"
        "pixel_sha256,width,height,phash,dhash,error "
        "FROM image_hashes WHERE version=?"
    )
    for row in conn.execute(query, (CACHE_VERSION,)):
        cached[str(row[0])] = row

    records: list[ImageRecord | None] = [None] * len(paths)
    tasks: list[tuple[int, str, str, str, int, int]] = []
    cache_hits = 0
    for index, path in enumerate(paths):
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        row = cached.get(rel)
        if row is not None and int(row[3]) == stat.st_size and int(row[4]) == stat.st_mtime_ns:
            records[index] = _record_from_cache(index, row)
            cache_hits += 1
        else:
            tasks.append((
                index, str(path), rel, class_from_path(path, root),
                stat.st_size, stat.st_mtime_ns,
            ))

    if tasks:
        print(f"Hashing/decoding {len(tasks):,} images with {workers} worker(s) ...", flush=True)
        if workers == 1:
            iterator: Iterable[ImageRecord] = map(_hash_image_worker, tasks)
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            iterator = pool.map(_hash_image_worker, tasks, chunksize=16)
        try:
            for done, rec in enumerate(iterator, 1):
                records[rec.index] = rec
                conn.execute(
                    """
                    INSERT OR REPLACE INTO image_hashes VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        rec.rel_path, CACHE_VERSION, rec.class_name, rec.extension,
                        rec.file_size, rec.mtime_ns, rec.sha256, rec.pixel_sha256,
                        rec.width, rec.height, rec.phash, rec.dhash, rec.error,
                    ),
                )
                if done % 1000 == 0:
                    conn.commit()
                if done % 5000 == 0 or done == len(tasks):
                    print(f"  analyzed {done:,}/{len(tasks):,}", flush=True)
        finally:
            if workers != 1:
                pool.shutdown()
        conn.commit()
    conn.close()
    assert all(r is not None for r in records)
    return [r for r in records if r is not None], cache_hits


def group_by_value(records: Sequence[ImageRecord], field: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for rec in records:
        value = getattr(rec, field)
        if value:
            groups[value].append(rec.index)
    return groups


def build_exact_components(
    records: Sequence[ImageRecord],
) -> tuple[UnionFind, dict[tuple[int, int], set[str]], dict[str, int]]:
    uf = UnionFind(len(records))
    edge_reasons: dict[tuple[int, int], set[str]] = defaultdict(set)
    stats: dict[str, int] = {}
    for field, reason in (("sha256", "exact_file"), ("pixel_sha256", "exact_pixels")):
        groups = group_by_value(records, field)
        duplicate_groups = [idxs for idxs in groups.values() if len(idxs) > 1]
        stats[f"{reason}_groups"] = len(duplicate_groups)
        stats[f"{reason}_members"] = sum(len(x) for x in duplicate_groups)
        for idxs in duplicate_groups:
            anchor = idxs[0]
            for other in idxs[1:]:
                uf.union(anchor, other)
                edge_reasons[tuple(sorted((anchor, other)))].add(reason)
    return uf, edge_reasons, stats


def components_from_uf(uf: UnionFind, n: int) -> list[list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return sorted((sorted(v) for v in groups.values()), key=lambda x: x[0])


def segment_ranges(total_bits: int, segments: int) -> list[tuple[int, int]]:
    base, extra = divmod(total_bits, segments)
    ranges: list[tuple[int, int]] = []
    offset = 0
    for i in range(segments):
        size = base + (1 if i < extra else 0)
        ranges.append((offset, size))
        offset += size
    return ranges


def find_phash_candidates(hashes: Sequence[int], threshold: int) -> list[tuple[int, int]]:
    """Find every pair within a 64-bit Hamming threshold using segment LSH."""
    if len(hashes) < 2:
        return []
    segments = threshold + 1
    buckets_by_segment: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(segments)]
    ranges = segment_ranges(64, segments)
    for index, value in enumerate(hashes):
        for seg, (offset, size) in enumerate(ranges):
            buckets_by_segment[seg][(value >> offset) & ((1 << size) - 1)].append(index)

    candidates: set[tuple[int, int]] = set()
    for seg, buckets in enumerate(buckets_by_segment, 1):
        for members in buckets.values():
            if len(members) < 2:
                continue
            for pos, left in enumerate(members[:-1]):
                left_hash = hashes[left]
                for right in members[pos + 1:]:
                    if (left_hash ^ hashes[right]).bit_count() <= threshold:
                        candidates.add((left, right))
        print(f"  pHash LSH segment {seg}/{segments}: {len(candidates):,} candidates", flush=True)
    return sorted(candidates)


def _load_comparison_array(path: str, size: int = 128) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def _array_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = left.ravel().astype(np.float32, copy=False)
    b = right.ravel().astype(np.float32, copy=False)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def _pair_metrics_worker(
    task: tuple[
        int, int, str, str, int, int, float, float, float, float, int, float
    ]
) -> PairMetrics:
    (
        left, right, left_path, right_path, phash_distance, dhash_distance,
        max_mae, max_rmse, min_correlation, min_equalized_correlation,
        min_orb_inliers, min_orb_inlier_ratio,
    ) = task
    try:
        a = _load_comparison_array(left_path)
        b = _load_comparison_array(right_path)
        delta = a - b
        mae = float(np.mean(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        gray_a = np.mean(a, axis=2).ravel()
        gray_b = np.mean(b, axis=2).ravel()
        centered_a = gray_a - float(gray_a.mean())
        centered_b = gray_b - float(gray_b.mean())
        denom = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
        correlation = float(np.dot(centered_a, centered_b) / denom) if denom > 1e-12 else 0.0
        strict_pixel_verified = (
            mae <= max_mae
            and rmse <= max_rmse
            and correlation >= min_correlation
        )
        low_error_verified = mae <= max_mae and rmse <= max_rmse
        equalized_correlation = math.nan
        orb_matches = 0
        orb_inliers = 0
        orb_inlier_ratio = 0.0
        verification_rule = (
            "direct_pixels" if strict_pixel_verified
            else "low_error" if low_error_verified
            else ""
        )
        verified = low_error_verified

        if not verified:
            # Histogram equalization makes this path robust to exposure changes.
            equalized_a = exposure.equalize_hist(gray_a.reshape(a.shape[:2]))
            equalized_b = exposure.equalize_hist(gray_b.reshape(b.shape[:2]))
            equalized_correlation = _array_correlation(equalized_a, equalized_b)
            try:
                detector_a = ORB(n_keypoints=300, fast_threshold=0.03)
                detector_b = ORB(n_keypoints=300, fast_threshold=0.03)
                detector_a.detect_and_extract(gray_a.reshape(a.shape[:2]))
                detector_b.detect_and_extract(gray_b.reshape(b.shape[:2]))
                matches = match_descriptors(
                    detector_a.descriptors,
                    detector_b.descriptors,
                    cross_check=True,
                    max_ratio=0.8,
                )
                orb_matches = int(len(matches))
                if orb_matches >= 4:
                    source = detector_a.keypoints[matches[:, 0]][:, ::-1]
                    destination = detector_b.keypoints[matches[:, 1]][:, ::-1]
                    _, inliers = ransac(
                        (source, destination),
                        ProjectiveTransform,
                        min_samples=4,
                        residual_threshold=1.5,
                        max_trials=100,
                        rng=np.random.default_rng(0),
                    )
                    orb_inliers = int(inliers.sum()) if inliers is not None else 0
                    orb_inlier_ratio = orb_inliers / orb_matches
            except (RuntimeError, ValueError):
                # Low-texture images may not contain enough ORB features.
                pass
            if (
                equalized_correlation >= min_equalized_correlation
                and orb_inliers >= min_orb_inliers
                and orb_inlier_ratio >= min_orb_inlier_ratio
            ):
                verified = True
                verification_rule = "orb_ransac"
        return PairMetrics(
            left, right, phash_distance, dhash_distance,
            mae, rmse, correlation, verified, "", equalized_correlation,
            orb_matches, orb_inliers, orb_inlier_ratio, verification_rule,
        )
    except Exception as exc:
        return PairMetrics(
            left, right, phash_distance, dhash_distance,
            math.nan, math.nan, math.nan, False,
            f"{type(exc).__name__}: {exc}",
        )


def verify_candidates(
    candidate_pairs: Sequence[tuple[int, int]],
    representatives: Sequence[int],
    records: Sequence[ImageRecord],
    root: Path,
    workers: int,
    max_dhash: int,
    max_mae: float,
    max_rmse: float,
    min_correlation: float,
    min_equalized_correlation: float,
    min_orb_inliers: int,
    min_orb_inlier_ratio: float,
) -> list[PairMetrics]:
    tasks: list[tuple[
        int, int, str, str, int, int, float, float, float, float, int, float
    ]] = []
    for left_local, right_local in candidate_pairs:
        left = representatives[left_local]
        right = representatives[right_local]
        phash_distance = (
            int(records[left].phash, 16) ^ int(records[right].phash, 16)
        ).bit_count()
        dhash_distance = (
            int(records[left].dhash, 16) ^ int(records[right].dhash, 16)
        ).bit_count()
        # dHash is a cheap second, independent guard before image decoding.
        if dhash_distance > max_dhash:
            tasks.append((
                left, right, "", "", phash_distance, dhash_distance,
                max_mae, max_rmse, min_correlation, min_equalized_correlation,
                min_orb_inliers, min_orb_inlier_ratio,
            ))
            continue
        tasks.append((
            left, right,
            str(root / records[left].rel_path),
            str(root / records[right].rel_path),
            phash_distance, dhash_distance,
            max_mae, max_rmse, min_correlation,
            min_equalized_correlation, min_orb_inliers, min_orb_inlier_ratio,
        ))

    def guarded(task: tuple[
        int, int, str, str, int, int, float, float, float, float, int, float
    ]) -> PairMetrics:
        if not task[2]:
            return PairMetrics(
                task[0], task[1], task[4], task[5],
                math.nan, math.nan, math.nan, False, "dhash_guard",
            )
        return _pair_metrics_worker(task)

    # ``guarded`` is local and therefore not picklable; resolve guarded pairs here.
    ready: list[PairMetrics] = []
    actual: list[tuple[
        int, int, str, str, int, int, float, float, float, float, int, float
    ]] = []
    for task in tasks:
        if not task[2]:
            ready.append(guarded(task))
        else:
            actual.append(task)
    if actual:
        print(f"Verifying {len(actual):,} near-duplicate candidates ...", flush=True)
        if workers == 1:
            iterator: Iterable[PairMetrics] = map(_pair_metrics_worker, actual)
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            iterator = pool.map(_pair_metrics_worker, actual, chunksize=16)
        try:
            for done, result in enumerate(iterator, 1):
                ready.append(result)
                if done % 5000 == 0 or done == len(actual):
                    print(f"  verified {done:,}/{len(actual):,}", flush=True)
        finally:
            if workers != 1:
                pool.shutdown()
    return sorted(ready, key=lambda p: (p.left, p.right))


def clique_cover(nodes: Sequence[int], verified_edges: set[tuple[int, int]]) -> list[list[int]]:
    """Deterministic disjoint clique cover; singleton cliques are omitted."""
    remaining = set(nodes)
    clusters: list[list[int]] = []
    for seed in sorted(nodes):
        if seed not in remaining:
            continue
        cluster = [seed]
        remaining.remove(seed)
        for candidate in sorted(remaining):
            if all(tuple(sorted((member, candidate))) in verified_edges for member in cluster):
                cluster.append(candidate)
        for member in cluster[1:]:
            remaining.remove(member)
        if len(cluster) > 1:
            clusters.append(cluster)
    return clusters


def choose_representative(indices: Sequence[int], records: Sequence[ImageRecord]) -> int:
    return min(
        indices,
        key=lambda i: (
            bool(records[i].error),
            -(records[i].width * records[i].height),
            -records[i].file_size,
            records[i].rel_path,
        ),
    )


def resolve_groups(
    records: Sequence[ImageRecord],
    exact_components: Sequence[Sequence[int]],
    verified_near_edges: Sequence[tuple[int, int]],
    accept_verified_near: bool,
) -> tuple[list[str], list[str], list[str], dict[int, list[int]]]:
    status = ["keep" if not rec.error else "exclude_decode_error" for rec in records]
    reason = ["unique" if not rec.error else rec.error for rec in records]
    group_id = ["" for _ in records]

    final_uf = UnionFind(len(records))
    for component in exact_components:
        if len(component) > 1:
            for member in component[1:]:
                final_uf.union(component[0], member)
    if accept_verified_near:
        for left, right in verified_near_edges:
            final_uf.union(left, right)

    components = [c for c in components_from_uf(final_uf, len(records)) if len(c) > 1]
    groups: dict[int, list[int]] = {}
    for ordinal, component in enumerate(components, 1):
        classes = {records[i].class_name for i in component}
        gid = f"D{ordinal:06d}"
        groups[ordinal] = list(component)
        for i in component:
            group_id[i] = gid
        if len(classes) > 1:
            for i in component:
                if not records[i].error:
                    status[i] = "exclude_label_conflict"
                    reason[i] = "duplicate group spans classes: " + "|".join(sorted(classes))
        else:
            representative = choose_representative(component, records)
            for i in component:
                if records[i].error:
                    continue
                if i == representative:
                    status[i] = "keep"
                    reason[i] = "duplicate representative"
                else:
                    status[i] = "exclude_duplicate"
                    reason[i] = f"represented by {records[representative].rel_path}"
    return status, reason, group_id, groups


def build_visual_groups(
    records: Sequence[ImageRecord],
    exact_components: Sequence[Sequence[int]],
    pair_metrics: Sequence[PairMetrics],
) -> tuple[list[str], list[list[int]]]:
    """Group every exact or pHash-candidate pair for leakage-safe splitting.

    A visual group is intentionally broader than an accepted duplicate group.
    A false-positive edge may make a training group larger, but it cannot inflate
    validation/test performance by placing a suspiciously similar pair across
    splits.  These IDs are *not* patient IDs.
    """
    uf = UnionFind(len(records))
    for component in exact_components:
        for member in component[1:]:
            uf.union(component[0], member)
    for pair in pair_metrics:
        uf.union(pair.left, pair.right)
    components = [c for c in components_from_uf(uf, len(records)) if len(c) > 1]
    group_ids = ["" for _ in records]
    for ordinal, component in enumerate(components, 1):
        gid = f"V{ordinal:06d}"
        for index in component:
            group_ids[index] = gid
    return group_ids, components


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def materialize_dataset(
    root: Path,
    destination: Path,
    records: Sequence[ImageRecord],
    status: Sequence[str],
    mode: str,
) -> dict[str, int]:
    if destination.resolve() == root.resolve():
        raise ValueError("--materialize must not be the source directory")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            "--materialize destination must be empty; this prevents stale files "
            f"from an older deduplication run: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    for rec, state in zip(records, status):
        if state != "keep":
            continue
        source = root / rec.rel_path
        target = destination / rec.rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            counts["already_exists"] += 1
        elif mode == "hardlink":
            os.link(source, target)
            counts["image_hardlinked"] += 1
        elif mode == "copy":
            shutil.copy2(source, target)
            counts["image_copied"] += 1
        elif mode == "symlink":
            target.symlink_to(source.resolve())
            counts["image_symlinked"] += 1
        else:
            raise ValueError(f"unknown link mode: {mode}")

        json_source = source.with_suffix(".json")
        json_target = target.with_suffix(".json")
        if not json_source.exists():
            counts["json_missing"] += 1
        elif json_target.exists() or json_target.is_symlink():
            counts["json_already_exists"] += 1
        elif mode == "hardlink":
            os.link(json_source, json_target)
            counts["json_hardlinked"] += 1
        elif mode == "copy":
            shutil.copy2(json_source, json_target)
            counts["json_copied"] += 1
        else:
            json_target.symlink_to(json_source.resolve())
            counts["json_symlinked"] += 1
    return dict(counts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data_raw"))
    parser.add_argument("--out", type=Path, default=Path("dedup_audit"))
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--phash-threshold", type=int, default=5)
    parser.add_argument("--max-dhash", type=int, default=10)
    parser.add_argument("--max-mae", type=float, default=0.03)
    parser.add_argument("--max-rmse", type=float, default=0.06)
    parser.add_argument("--min-correlation", type=float, default=0.985)
    parser.add_argument("--min-equalized-correlation", type=float, default=0.90)
    parser.add_argument("--min-orb-inliers", type=int, default=80)
    parser.add_argument("--min-orb-inlier-ratio", type=float, default=0.80)
    parser.add_argument(
        "--exact-only", action="store_true",
        help="skip pHash candidate generation and verification",
    )
    parser.add_argument(
        "--accept-verified-near", action="store_true",
        help="include conservative verified near-duplicate components in exclusions",
    )
    parser.add_argument("--materialize", type=Path)
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    args = parser.parse_args(argv)
    if args.phash_threshold < 0 or args.phash_threshold > 63:
        parser.error("--phash-threshold must be between 0 and 63")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.accept_verified_near and args.exact_only:
        parser.error("--accept-verified-near cannot be combined with --exact-only")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    out = args.out.resolve()
    if not root.is_dir():
        raise SystemExit(f"dataset root does not exist: {root}")
    out.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.monotonic()

    paths = discover_images(root)
    if not paths:
        raise SystemExit(f"no supported images below: {root}")
    extension_counts = Counter(p.suffix.lower() for p in paths)
    print(f"Discovered {len(paths):,} images: {dict(sorted(extension_counts.items()))}")

    records, cache_hits = analyze_images(paths, root, out / "hash_cache.sqlite3", args.workers)
    errors = [r for r in records if r.error]
    print(f"Cache hits: {cache_hits:,}; decode/read errors: {len(errors):,}")

    exact_uf, _edge_reasons, exact_stats = build_exact_components(records)
    exact_components = components_from_uf(exact_uf, len(records))
    exact_duplicate_components = [c for c in exact_components if len(c) > 1]
    print(f"Exact duplicate components: {len(exact_duplicate_components):,}")

    pair_metrics: list[PairMetrics] = []
    near_cliques: list[list[int]] = []
    verified_edges: set[tuple[int, int]] = set()
    if not args.exact_only:
        representatives = [choose_representative(c, records) for c in exact_components]
        valid_representatives = [i for i in representatives if records[i].phash and records[i].dhash]
        phashes = [int(records[i].phash, 16) for i in valid_representatives]
        print(f"Searching {len(phashes):,} exact-component representatives with pHash ...")
        local_pairs = find_phash_candidates(phashes, args.phash_threshold)
        pair_metrics = verify_candidates(
            local_pairs, valid_representatives, records, root, args.workers,
            args.max_dhash, args.max_mae, args.max_rmse, args.min_correlation,
            args.min_equalized_correlation, args.min_orb_inliers,
            args.min_orb_inlier_ratio,
        )
        verified_edges = {
            tuple(sorted((p.left, p.right))) for p in pair_metrics if p.verified
        }
        near_nodes = sorted({i for edge in verified_edges for i in edge})
        near_cliques = clique_cover(near_nodes, verified_edges)
        print(
            f"pHash candidates: {len(pair_metrics):,}; "
            f"conservatively verified pairs: {len(verified_edges):,}; "
            f"near cliques: {len(near_cliques):,}"
        )

    status, reason, duplicate_group, groups = resolve_groups(
        records, exact_components, sorted(verified_edges), args.accept_verified_near
    )
    visual_group, visual_components = build_visual_groups(
        records, exact_components, pair_metrics
    )

    manifest_path = out / "dedup_manifest.csv"
    write_csv(
        manifest_path,
        (
            "path", "class", "extension", "file_size", "width", "height",
            "sha256", "pixel_sha256", "phash", "dhash", "status", "reason",
            "duplicate_group", "visual_group", "error",
        ),
        (
            (
                rec.rel_path, rec.class_name, rec.extension, rec.file_size,
                rec.width, rec.height, rec.sha256, rec.pixel_sha256,
                rec.phash, rec.dhash, status[rec.index], reason[rec.index],
                duplicate_group[rec.index], visual_group[rec.index], rec.error,
            )
            for rec in records
        ),
    )

    pair_path = out / "near_duplicate_pairs.csv"
    write_csv(
        pair_path,
        (
            "left_path", "right_path", "left_class", "right_class",
            "phash_distance", "dhash_distance", "mae", "rmse", "correlation",
            "equalized_correlation", "orb_matches", "orb_inliers",
            "orb_inlier_ratio", "verified", "verification_rule",
            "verification_error",
        ),
        (
            (
                records[p.left].rel_path, records[p.right].rel_path,
                records[p.left].class_name, records[p.right].class_name,
                p.phash_distance, p.dhash_distance,
                "" if math.isnan(p.mae) else f"{p.mae:.8f}",
                "" if math.isnan(p.rmse) else f"{p.rmse:.8f}",
                "" if math.isnan(p.correlation) else f"{p.correlation:.8f}",
                "" if math.isnan(p.equalized_correlation) else f"{p.equalized_correlation:.8f}",
                p.orb_matches, p.orb_inliers, f"{p.orb_inlier_ratio:.8f}",
                int(p.verified), p.verification_rule, p.error,
            )
            for p in pair_metrics
        ),
    )

    group_path = out / "duplicate_groups.csv"
    write_csv(
        group_path,
        ("group", "path", "class", "status", "reason", "group_size", "group_classes"),
        (
            (
                duplicate_group[i], records[i].rel_path, records[i].class_name,
                status[i], reason[i], len(component),
                "|".join(sorted({records[j].class_name for j in component})),
            )
            for component in groups.values()
            for i in component
        ),
    )

    visual_group_path = out / "visual_groups.csv"
    write_csv(
        visual_group_path,
        ("visual_group", "path", "class", "status", "group_size", "group_classes"),
        (
            (
                visual_group[i], records[i].rel_path, records[i].class_name,
                status[i], len(component),
                "|".join(sorted({records[j].class_name for j in component})),
            )
            for component in visual_components
            for i in component
        ),
    )

    materialized: dict[str, int] = {}
    if args.materialize is not None:
        materialized = materialize_dataset(
            root, args.materialize.resolve(), records, status, args.link_mode
        )

    status_counts = Counter(status)
    class_before = Counter(r.class_name for r in records)
    class_after = Counter(r.class_name for r, s in zip(records, status) if s == "keep")
    summary = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "root": str(root),
        "out": str(out),
        "parameters": {
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "workers": args.workers,
            "phash_threshold": args.phash_threshold,
            "max_dhash": args.max_dhash,
            "max_mae": args.max_mae,
            "max_rmse": args.max_rmse,
            "min_correlation": args.min_correlation,
            "min_equalized_correlation": args.min_equalized_correlation,
            "min_orb_inliers": args.min_orb_inliers,
            "min_orb_inlier_ratio": args.min_orb_inlier_ratio,
            "exact_only": args.exact_only,
            "accept_verified_near": args.accept_verified_near,
            "cross_class_policy": "exclude_all",
            "accepted_near_resolution": "verified_edge_connected_components",
            "all_candidate_visual_grouping": "connected_components",
        },
        "inventory": {
            "images": len(records),
            "extensions": dict(sorted(extension_counts.items())),
            "classes_before": dict(sorted(class_before.items())),
            "classes_after": dict(sorted(class_after.items())),
            "cache_hits": cache_hits,
            "decode_errors": len(errors),
        },
        "exact": exact_stats,
        "near": {
            "candidate_pairs": len(pair_metrics),
            "verified_pairs": sum(p.verified for p in pair_metrics),
            "verified_cliques": len(near_cliques),
            "verification_rules": dict(sorted(Counter(
                p.verification_rule or "rejected" for p in pair_metrics
            ).items())),
            "accepted_for_exclusion": bool(args.accept_verified_near),
        },
        "visual_groups_for_splitting": {
            "groups": len(visual_components),
            "members": sum(len(c) for c in visual_components),
            "largest_group": max((len(c) for c in visual_components), default=0),
            "includes_all_phash_candidates": True,
            "patient_identity_guaranteed": False,
        },
        "result": {
            "status_counts": dict(sorted(status_counts.items())),
            "duplicate_groups": len(groups),
            "kept_images": status_counts.get("keep", 0),
            "excluded_images": len(records) - status_counts.get("keep", 0),
        },
        "materialization": {
            "destination": str(args.materialize.resolve()) if args.materialize else None,
            "link_mode": args.link_mode if args.materialize else None,
            "counts": materialized,
        },
        "outputs": {
            "manifest": str(manifest_path),
            "duplicate_groups": str(group_path),
            "visual_groups": str(visual_group_path),
            "near_duplicate_pairs": str(pair_path),
            "cache": str(out / "hash_cache.sqlite3"),
        },
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary["result"], ensure_ascii=False, indent=2))
    print(f"Audit written to: {out}")
    if pair_metrics and not args.accept_verified_near:
        print(
            "Verified near-duplicate pairs were reported but not excluded. "
            "Review them before rerunning with --accept-verified-near."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
