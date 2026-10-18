"""Offline teacher-probe manifests used by TG-OPD reliability gating."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Unsupported value in offline probe fingerprint: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def _field(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(name, default)
    return getattr(sample, name, default)


def sample_fingerprint(sample: Any) -> str:
    """Return a stable identity independent of rollout order and group index."""
    payload = {
        "prompt": _field(sample, "prompt", ""),
        "label": _field(sample, "label"),
        "metadata": _field(sample, "metadata", {}) or {},
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reward_value(sample: Any, reward_key: str | None) -> float:
    reward = _field(sample, "reward")
    if isinstance(reward, dict):
        if reward_key is None:
            if len(reward) != 1:
                raise ValueError(f"reward_key is required for reward keys {sorted(reward)}")
            reward = next(iter(reward.values()))
        else:
            try:
                reward = reward[reward_key]
            except KeyError as exc:
                raise ValueError(f"Reward does not contain key {reward_key!r}: {sorted(reward)}") from exc
    if reward is None:
        raise ValueError("Offline probe sample has no reward")
    return float(reward)


def _rollout_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def _load_rollout_samples(path: Path) -> list[Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"Unexpected rollout dump format: {path}")
    return payload["samples"]


def build_manifest(
    *,
    rollout_dir: str,
    output_path: str,
    probe_n: int,
    min_pass: int,
    reward_threshold: float,
    reward_key: str | None,
    dataset_path: str,
    teacher_checkpoint: str,
    probe_max_response_len: int,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if probe_n <= 0 or not 0 < min_pass <= probe_n:
        raise ValueError("Require probe_n > 0 and min_pass in [1, probe_n]")

    rollout_paths = sorted(Path(rollout_dir).glob("rollout_*.pt"), key=_rollout_sort_key)
    if not rollout_paths:
        raise FileNotFoundError(f"No rollout_*.pt files found in {rollout_dir}")

    grouped: dict[int, list[Any]] = defaultdict(list)
    for path in rollout_paths:
        for sample in _load_rollout_samples(path):
            group_index = _field(sample, "group_index")
            if group_index is None:
                raise ValueError(f"Sample in {path} has no group_index")
            grouped[int(group_index)].append(sample)

    records: dict[str, dict[str, Any]] = {}
    duplicate_fingerprints = 0
    for group_index, samples in sorted(grouped.items()):
        if len(samples) != probe_n:
            raise ValueError(
                f"group_index={group_index} has {len(samples)} samples, expected exactly {probe_n}"
            )
        fingerprints = {sample_fingerprint(sample) for sample in samples}
        if len(fingerprints) != 1:
            raise ValueError(f"group_index={group_index} contains multiple prompt fingerprints")
        fingerprint = fingerprints.pop()
        scores = [_reward_value(sample, reward_key) for sample in samples]
        pass_count = sum(score >= reward_threshold for score in scores)
        record = {
            "scores": scores,
            "pass_count": pass_count,
            "reliable": pass_count >= min_pass,
            "source_group_indices": [group_index],
        }
        previous = records.get(fingerprint)
        if previous is not None:
            if previous["scores"] != record["scores"] or previous["reliable"] != record["reliable"]:
                raise ValueError(
                    f"Fingerprint {fingerprint} occurs in multiple groups with conflicting probe results"
                )
            previous["source_group_indices"].append(group_index)
            duplicate_fingerprints += 1
        else:
            records[fingerprint] = record

    pass_count_distribution = Counter(record["pass_count"] for record in records.values())
    reliable_count = sum(record["reliable"] for record in records.values())
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rollout_dir": str(Path(rollout_dir).resolve()),
        "rollout_file_count": len(rollout_paths),
        "dataset_path": dataset_path,
        "dataset_sha256": file_sha256(dataset_path),
        "teacher_checkpoint": teacher_checkpoint,
        "probe_n": probe_n,
        "min_pass": min_pass,
        "reward_threshold": reward_threshold,
        "reward_key": reward_key,
        "probe_max_response_len": probe_max_response_len,
        "source_group_count": len(grouped),
        "record_count": len(records),
        "duplicate_fingerprints": duplicate_fingerprints,
        "reliable_count": reliable_count,
        "fallback_count": len(records) - reliable_count,
        "pass_count_distribution": {
            str(key): pass_count_distribution.get(key, 0) for key in range(probe_n + 1)
        },
    }
    metadata.update(extra_metadata or {})
    manifest = {"schema_version": SCHEMA_VERSION, "metadata": metadata, "records": records}

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_suffix(output.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    os.replace(temp_path, output)
    return manifest


class OfflineProbeManifest:
    def __init__(
        self,
        path: str,
        *,
        probe_n: int,
        min_pass: int,
        reward_threshold: float,
        sample_k: int | None = None,
        sample_min_pass: int | None = None,
        sample_seed: int = 42,
    ) -> None:
        """Load an offline probe manifest.

        If ``sample_k`` is set (and < probe_n), for each record we
        deterministically sub-sample ``sample_k`` scores out of the recorded
        ``probe_n`` and recompute ``pass_count`` / ``reliable`` using
        ``sample_min_pass``. This lets us cheaply simulate an online
        ``TG_OPD_PROBE_N=sample_k`` gate off a manifest that was pre-computed at
        a larger K. Because the teacher is frozen throughout training, the
        sub-sample is statistically identical in distribution to fresh online
        rollouts. Sampling uses a per-fingerprint deterministic seed for
        reproducibility.
        """
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported offline probe manifest schema {manifest.get('schema_version')}; "
                f"expected {SCHEMA_VERSION}"
            )
        metadata = manifest.get("metadata") or {}
        expected = {
            "probe_n": probe_n,
            "min_pass": min_pass,
            "reward_threshold": reward_threshold,
        }
        actual = {key: metadata.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"Offline probe manifest settings mismatch: expected {expected}, got {actual}")
        records = manifest.get("records")
        if not isinstance(records, dict) or not records:
            raise ValueError(f"Offline probe manifest has no records: {path}")

        # Optional deterministic K_T sub-sampling to simulate a smaller online probe.
        if sample_k is not None and sample_k < probe_n:
            if sample_k <= 0:
                raise ValueError(f"sample_k must be positive, got {sample_k}")
            if sample_min_pass is None:
                raise ValueError("sample_min_pass is required when sample_k is set")
            if not 0 < sample_min_pass <= sample_k:
                raise ValueError(f"sample_min_pass must be in [1, sample_k]={sample_k}, got {sample_min_pass}")
            import random as _random
            resampled: dict[str, dict[str, Any]] = {}
            reliable_after = 0
            for fp, rec in records.items():
                scores = rec.get("scores")
                if not isinstance(scores, list) or len(scores) != probe_n:
                    raise ValueError(
                        f"Record {fp} missing/malformed 'scores' (expected len={probe_n}); "
                        "cannot sub-sample"
                    )
                # deterministic per-fingerprint seed for reproducibility
                rng = _random.Random(sample_seed ^ (int(fp[:16], 16) & 0xFFFFFFFF))
                sub_scores = rng.sample(scores, sample_k)
                sub_pass_count = sum(1 for s in sub_scores if s >= reward_threshold)
                sub_reliable = sub_pass_count >= sample_min_pass
                new_rec = dict(rec)
                new_rec["sample_scores"] = sub_scores
                new_rec["sample_pass_count"] = sub_pass_count
                new_rec["pass_count"] = sub_pass_count       # overwrite for downstream lookup
                new_rec["reliable"] = sub_reliable            # overwrite for downstream lookup
                resampled[fp] = new_rec
                if sub_reliable:
                    reliable_after += 1
            records = resampled
            # Amend metadata for logging/debug
            metadata = dict(metadata)
            metadata["_sample_k"] = sample_k
            metadata["_sample_min_pass"] = sample_min_pass
            metadata["_sample_seed"] = sample_seed
            metadata["_reliable_count_after_subsample"] = reliable_after
            metadata["_fallback_count_after_subsample"] = len(records) - reliable_after

        self.path = path
        self.metadata = metadata
        self.records = records

    def lookup(self, sample: Any) -> dict[str, Any]:
        fingerprint = sample_fingerprint(sample)
        try:
            return self.records[fingerprint]
        except KeyError as exc:
            raise KeyError(
                f"Offline probe manifest miss for fingerprint={fingerprint}, "
                f"sample_index={_field(sample, 'index')}, group_index={_field(sample, 'group_index')}"
            ) from exc

    def validate_samples(self, samples: Iterable[Any]) -> dict[str, int]:
        total = 0
        missing = 0
        fingerprints: set[str] = set()
        for sample in samples:
            total += 1
            fingerprint = sample_fingerprint(sample)
            fingerprints.add(fingerprint)
            if fingerprint not in self.records:
                missing += 1
        return {"total": total, "unique": len(fingerprints), "missing": missing}
