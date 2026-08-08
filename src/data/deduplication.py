"""Exact and perceptual duplicate detection without platform-specific code."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageOps


@dataclass(frozen=True)
class DuplicateGroup:
    sample_ids: tuple[str, ...]
    max_pair_distance: int


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: str | Path, hash_size: int = 8) -> str:
    """Return a 64-bit dHash by default; EXIF orientation is normalized."""

    if hash_size <= 0:
        raise ValueError("hash_size must be positive")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        pixels = image.tobytes()
    bits = []
    row_width = hash_size + 1
    for row in range(hash_size):
        offset = row * row_width
        bits.extend(pixels[offset + col] > pixels[offset + col + 1] for col in range(hash_size))
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:0{(hash_size * hash_size + 3) // 4}x}"


def hamming_distance(left_hash: str, right_hash: str) -> int:
    if len(left_hash) != len(right_hash):
        raise ValueError("perceptual hashes must have equal length")
    try:
        return (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("perceptual hashes must be hexadecimal") from exc


def find_near_duplicate_groups(
    hashes: Mapping[str, str], max_distance: int = 5
) -> tuple[DuplicateGroup, ...]:
    """Find connected components of images whose dHash distance is within a threshold."""

    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")
    sample_ids = sorted(hashes)
    parent = {sample_id: sample_id for sample_id in sample_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    distances: dict[tuple[str, str], int] = {}
    for index, left in enumerate(sample_ids):
        for right in sample_ids[index + 1 :]:
            distance = hamming_distance(hashes[left], hashes[right])
            distances[(left, right)] = distance
            if distance <= max_distance:
                union(left, right)

    components: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        components.setdefault(find(sample_id), []).append(sample_id)

    groups = []
    for members in components.values():
        if len(members) < 2:
            continue
        pair_distances = [
            distances[(left, right)]
            for index, left in enumerate(members)
            for right in members[index + 1 :]
        ]
        groups.append(DuplicateGroup(tuple(sorted(members)), max(pair_distances)))
    return tuple(sorted(groups, key=lambda group: group.sample_ids))


def cross_split_leakage_rate(
    groups: Sequence[DuplicateGroup], split_by_sample: Mapping[str, str], total_samples: int
) -> float:
    """Fraction of samples involved in a near-duplicate group spanning multiple splits."""

    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    if total_samples == 0:
        return 0.0
    leaked: set[str] = set()
    for group in groups:
        known_members = [sample_id for sample_id in group.sample_ids if sample_id in split_by_sample]
        splits = {split_by_sample[sample_id] for sample_id in known_members}
        if len(splits) > 1:
            leaked.update(known_members)
    return len(leaked) / total_samples
