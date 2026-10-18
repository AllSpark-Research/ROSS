#!/usr/bin/env python3
"""Compile grounded character intervals into masks over the original token IDs.

The compiler never edits or retokenizes the saved sequence. It reconstructs
UTF-8 byte offsets from a ByteLevel tokenizer, verifies the tokenizer decode,
and expands a character interval to every overlapping non-special token.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Callable, Iterable


def load_bytelevel_codec(tokenizer_json: str | Path):
    """Return the tokenizer, an exact token-to-bytes function, and added IDs."""
    from tokenizers import Tokenizer

    path = Path(tokenizer_json)
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("decoder", {}).get("type") != "ByteLevel":
        raise ValueError("exact byte alignment requires a ByteLevel tokenizer")
    id_to_piece = {token_id: piece for piece, token_id in spec["model"]["vocab"].items()}
    added = {entry["id"]: entry["content"] for entry in spec.get("added_tokens", [])}
    id_to_piece.update(added)

    native = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    inverse = {chr(value): value for value in native}
    offset = 0
    for value in range(256):
        if value not in native:
            inverse[chr(256 + offset)] = value
            offset += 1

    def token_bytes(token_id: int) -> bytes:
        if token_id in added:
            return added[token_id].encode("utf-8")
        return bytes(inverse[char] for char in id_to_piece[token_id])

    return Tokenizer.from_file(str(path)), token_bytes, set(added)


def decode_regions(
    tokens: list[int],
    response_length: int,
    response_loss_mask: list[int],
    tokenizer,
    token_bytes: Callable[[int], bytes],
):
    """Recover alternating context/generated regions from the saved base mask."""
    if response_length != len(response_loss_mask) or response_length > len(tokens):
        raise ValueError("response length and response loss mask are inconsistent")
    base = [0] * (len(tokens) - response_length) + list(response_loss_mask)
    if set(base) - {0, 1}:
        raise ValueError("base loss mask must be binary")
    pieces = [token_bytes(token_id) for token_id in tokens]
    byte_offsets = [0]
    for piece in pieces:
        byte_offsets.append(byte_offsets[-1] + len(piece))
    raw = b"".join(pieces)
    decoded = raw.decode("utf-8", errors="strict")
    if decoded != tokenizer.decode(tokens, skip_special_tokens=False):
        raise ValueError("byte reconstruction differs from tokenizer decode")

    regions = {}
    counters = {0: 0, 1: 0}
    start = 0
    while start < len(tokens):
        end = start + 1
        while end < len(tokens) and base[end] == base[start]:
            end += 1
        kind = base[start]
        counters[kind] += 1
        region_id = ("G" if kind else "C") + f"{counters[kind]:03d}"
        text = raw[byte_offsets[start]:byte_offsets[end]].decode("utf-8", errors="strict")
        regions[region_id] = {
            "text": text,
            "token_start": start,
            "token_end": end,
            "byte_start": byte_offsets[start],
            "byte_end": byte_offsets[end],
            "trainable": bool(kind),
        }
        start = end
    if "".join(region["text"] for region in regions.values()) != decoded:
        raise ValueError("region reconstruction changed the saved sequence")
    return base, byte_offsets, regions


def _token_interval(region, text: str, start: int, end: int, byte_offsets):
    if not region["trainable"] or not 0 <= start < end <= len(text):
        raise ValueError("invalid interval or non-trainable region")
    byte_start = region["byte_start"] + len(text[:start].encode("utf-8"))
    byte_end = region["byte_start"] + len(text[:end].encode("utf-8"))
    token_start = bisect.bisect_right(byte_offsets, byte_start) - 1
    token_end = bisect.bisect_left(byte_offsets, byte_end)
    if not region["token_start"] <= token_start < token_end <= region["token_end"]:
        raise ValueError("character interval escapes its source region")
    return token_start, token_end, byte_start, byte_end


def compile_exclusions(
    tokens: list[int],
    base_mask: list[int],
    byte_offsets: list[int],
    regions: dict,
    intervals: Iterable[dict],
    added_token_ids: set[int],
):
    """Set loss to zero for agentic exclusion intervals."""
    final = list(base_mask)
    compiled = []
    for interval in intervals:
        region = regions[interval["region_id"]]
        text = region["text"]
        start, end = interval["char_start"], interval["char_end"]
        token_start, token_end, byte_start, byte_end = _token_interval(
            region, text, start, end, byte_offsets
        )
        protected = 0
        for index in range(token_start, token_end):
            if not base_mask[index]:
                raise ValueError("exclusion overlaps a context-only token")
            if tokens[index] in added_token_ids:
                protected += 1
            else:
                final[index] = 0
        compiled.append({
            **interval,
            "token_start": token_start,
            "token_end": token_end,
            "protected_special_tokens": protected,
            "boundary_expansion_bytes":
                (byte_start - byte_offsets[token_start]) + (byte_offsets[token_end] - byte_end),
        })
    if any(value > base for value, base in zip(final, base_mask)):
        raise ValueError("compiled mask enables a context-only token")
    return final, compiled


def compile_keeps(
    tokens: list[int],
    base_mask: list[int],
    byte_offsets: list[int],
    regions: dict,
    region_id: str,
    keep_intervals: Iterable[tuple[int, int]],
    added_token_ids: set[int],
):
    """Keep loss only on selected single-turn spans in one generated region."""
    region = regions[region_id]
    if not region["trainable"]:
        raise ValueError("keep spans must belong to a generated region")
    final = list(base_mask)
    for index in range(region["token_start"], region["token_end"]):
        if tokens[index] not in added_token_ids:
            final[index] = 0
    compiled = []
    for start, end in keep_intervals:
        token_start, token_end, byte_start, byte_end = _token_interval(
            region, region["text"], start, end, byte_offsets
        )
        for index in range(token_start, token_end):
            if base_mask[index] and tokens[index] not in added_token_ids:
                final[index] = 1
        compiled.append({
            "region_id": region_id,
            "char_start": start,
            "char_end": end,
            "token_start": token_start,
            "token_end": token_end,
            "boundary_expansion_bytes":
                (byte_start - byte_offsets[token_start]) + (byte_offsets[token_end] - byte_end),
        })
    if any(value > base for value, base in zip(final, base_mask)):
        raise ValueError("compiled mask enables a context-only token")
    return final, compiled
