# SPDX-License-Identifier: Apache-2.0

"""Replay LMCache traces across multiple raw-block byte windows.

The default mode is plan-only.  Destructive writes to a raw NVMe namespace only
run when ``--allow-destructive-device-write`` is supplied and the confirmation
gate is satisfied.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import argparse
import datetime as dt
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time

try:
    # Third Party
    import yaml
except Exception:  # pragma: no cover - PyYAML is present in normal test envs.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

# First Party
from benchmarks.agentic_mp_trace.replay.command_builder import (  # noqa: E402
    storage_class_for_entry,
    trace_entries,
    trace_id_for_entry,
)
from benchmarks.fdp_waf_stress.run_fdp_waf_stress import (  # noqa: E402
    analyze_trace_footprint,
    extract_media_write_bytes,
)


SIZE_RE = re.compile(r"^\s*(\d+)\s*([KMGT]i?B?|B)?\s*$", re.IGNORECASE)
VALID_WINDOW_POLICIES = ("equal", "explicit")
VALID_STOP_POLICIES = ("timeout", "total_written_size", "iterations")
VALID_RUH_ASSIGNMENTS = ("mixed", "per_app")
VALID_PLACEMENTS = ("fixed", "random")
VALID_WORKLOAD_KEYS = ("storage_class", "dataset_adapter", "trace_name")
VALID_LAUNCH_POLICIES = ("simultaneous", "random_jitter")
VALID_GPU_IO_MODES = ("none", "cpu_stage", "gds_if_supported")
END_CONDITION_ACTUAL_WRITTEN_SOURCE = "lmcache_successful_write_physical_bytes"
END_CONDITION_HOST_WRITE_SOURCE = "host_write_bytes_delta"
RAW_BLOCK_ACCOUNTING_FIELDS = (
    "store_attempted_count",
    "store_attempted_logical_bytes",
    "store_existing_hit_count",
    "store_existing_hit_logical_bytes",
    "store_committed_count",
    "store_committed_logical_bytes",
    "eviction_count",
    "eviction_logical_bytes",
    "data_write_logical_bytes",
    "data_write_payload_physical_bytes",
    "data_write_header_physical_bytes",
    "data_write_physical_bytes",
    "metadata_write_logical_bytes",
    "metadata_write_physical_bytes",
    "total_write_logical_bytes",
    "total_write_physical_bytes",
    "media_write_logical_bytes",
    "media_write_physical_bytes",
)


@dataclass(frozen=True)
class TraceWorkload:
    trace_id: str
    trace_path: str
    workload_key: str
    storage_class: str
    dataset_adapter: str
    entry: dict[str, Any]
    estimated_store_bytes: int | None = None
    duration_seconds: float | None = None
    record_count: int | None = None
    store_count: int | None = None


@dataclass(frozen=True)
class RuhPolicy:
    use_fdp: bool
    available_ruhs: list[int]
    data_ruhs: list[int]
    metadata_ruhs: list[int]
    warning: str | None = None


@dataclass(frozen=True)
class WindowLayout:
    index: int
    base_offset_bytes: int
    capacity_bytes: int
    metadata_reserved_bytes: int
    usable_capacity_bytes: int
    estimated_slot_count: int
    meta_magic: str


@dataclass
class WindowPlan:
    index: int
    trace_id: str
    trace_path: str
    workload_key: str
    storage_class: str
    base_offset_bytes: int
    capacity_bytes: int
    metadata_reserved_bytes: int
    usable_capacity_bytes: int
    slot_bytes: int
    estimated_slot_count: int
    meta_magic: str
    fdp_data_ruh_ids: list[int]
    fdp_metadata_ruh_ids: list[int]
    command: list[str]
    command_text: str
    commands: list[list[str]]
    command_texts: list[str]
    workloads: list[dict[str, str]]
    output_dir: str
    jsonl_out: str
    num_store_workers: int
    num_lookup_workers: int
    num_load_workers: int
    raw_block_core_status: dict[str, Any] | None = None
    io_accounting: dict[str, Any] | None = None
    launch_delay_sec: float = 0.0


ACTIVE_PROCS: list[subprocess.Popen] = []
LAST_HOST_WRITE_COUNTER_SOURCE: str | None = None


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_size(value: str | int, *, allow_auto: bool = False) -> int | str:
    if isinstance(value, int):
        if value < 0:
            raise ValueError("size must be non-negative")
        return value
    text = str(value).strip()
    if allow_auto and text.lower() == "auto":
        return "auto"
    match = SIZE_RE.match(text)
    if not match:
        raise ValueError(f"invalid size value: {value!r}")
    number = int(match.group(1))
    suffix = (match.group(2) or "B").lower()
    multipliers = {
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "kib": 1024,
        "m": 1000**2,
        "mb": 1000**2,
        "mib": 1024**2,
        "g": 1000**3,
        "gb": 1000**3,
        "gib": 1024**3,
        "t": 1000**4,
        "tb": 1000**4,
        "tib": 1024**4,
    }
    return number * multipliers[suffix]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def align_down(value: int, align: int) -> int:
    return value - (value % align)


def align_up(value: int, align: int) -> int:
    remainder = value % align
    return value if remainder == 0 else value + align - remainder


def parse_csv_sizes(value: str) -> list[int]:
    if not value:
        return []
    return [int(parse_size(part)) for part in value.split(",") if part.strip()]


def parse_ruh_ids(value: str | None) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    ids = [int(part.strip(), 0) for part in str(value).split(",") if part.strip()]
    validate_ruh_id_list(ids)
    return ids


def validate_ruh_id_list(ids: list[int]) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError("RUH IDs must not contain duplicates")
    for ruh_id in ids:
        if ruh_id < 0 or ruh_id > 0xFFFF:
            raise ValueError("RUH IDs must fit uint16")


def load_manifest(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text()
    if Path(path).suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML trace manifests")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("trace manifest must decode to a mapping")
    return data


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_yaml(path: str | Path, payload: Any) -> None:
    if yaml is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))


def detect_device_capacity(block_device_path: str) -> int:
    if shutil.which("blockdev"):
        proc = subprocess.run(
            ["blockdev", "--getsize64", block_device_path],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return int(proc.stdout.strip())
    if shutil.which("nvme"):
        proc = subprocess.run(
            ["nvme", "id-ns", block_device_path],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode == 0:
            match = re.search(r"nsze\s*:\s*([0-9a-fA-Fx]+)", proc.stdout)
            lba_match = re.search(r"lbaf\s+\d+\s*:\s*ms:\d+\s+lbads:(\d+)", proc.stdout)
            if match and lba_match:
                return int(match.group(1), 0) * (1 << int(lba_match.group(1)))
    raise RuntimeError(
        f"could not auto-detect capacity for {block_device_path}; "
        "pass --device-capacity-bytes explicitly"
    )


def validate_ng_device_path(path: str, *, use_uring_cmd: bool) -> None:
    if not use_uring_cmd:
        return
    try:
        st = os.stat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{path} does not exist") from exc
    if not stat.S_ISCHR(st.st_mode):
        raise ValueError(
            "--use-uring-cmd requires a character NVMe namespace device "
            f"such as /dev/ng1n1, got {path}"
        )
    if not re.match(r"^/dev/ng\d+n\d+$", path):
        raise ValueError(f"--device-path should look like /dev/ngXnY, got {path}")


def resolve_ruh_policy(
    *,
    use_fdp: bool,
    ruh_ids: str | None,
    ruh_count: int | None,
    ruh_start_id: int,
    metadata_ruh_ids: str,
) -> RuhPolicy:
    if not use_fdp:
        return RuhPolicy(False, [], [], [])
    if ruh_ids:
        available = parse_ruh_ids(ruh_ids)
    else:
        count = 1 if ruh_count is None else int(ruh_count)
        if count < 1:
            raise ValueError("--ruh-count must be positive")
        available = list(range(int(ruh_start_id), int(ruh_start_id) + count))
        validate_ruh_id_list(available)
    if not available:
        raise ValueError("FDP enabled but no RUHs are available")

    warning = None
    if metadata_ruh_ids == "auto":
        if len(available) == 1:
            metadata = [available[0]]
            data = [available[0]]
            warning = (
                "Only one RUH configured; data and metadata cannot be separated."
            )
        else:
            metadata = [available[-1]]
            data = available[:-1]
    else:
        metadata = parse_ruh_ids(metadata_ruh_ids)
        missing = sorted(set(metadata) - set(available))
        if missing:
            raise ValueError(f"metadata RUHs are not in available RUHs: {missing}")
        data = [ruh for ruh in available if ruh not in set(metadata)]
        if not data:
            data = list(metadata)
            warning = "All RUHs are metadata RUHs; data will share metadata RUHs."
    return RuhPolicy(True, available, data, metadata, warning)


def assigned_data_ruhs(
    *,
    policy: RuhPolicy,
    assignment: str,
    window_index: int,
    workload_key: str,
    app_to_ruhs: dict[str, list[int]],
) -> list[int]:
    if not policy.use_fdp:
        return []
    if assignment == "mixed":
        return [policy.data_ruhs[window_index % len(policy.data_ruhs)]]
    if assignment != "per_app":
        raise ValueError(f"unknown RUH assignment: {assignment}")
    if workload_key not in app_to_ruhs:
        keys = sorted(app_to_ruhs)
        assigned = policy.data_ruhs[len(keys) % len(policy.data_ruhs)]
        app_to_ruhs[workload_key] = [assigned]
    return list(app_to_ruhs[workload_key])


def load_per_app_ruh_map(value: str | None) -> dict[str, list[int]]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
        else:
            payload = load_manifest(path)
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--per-app-ruh-map must be a JSON object or mapping file")
    result = {str(key): [int(item) for item in value] for key, value in payload.items()}
    for ids in result.values():
        validate_ruh_id_list(ids)
    return result


def metadata_reserved_bytes(
    *,
    use_fdp: bool,
    metadata_ruh_ids: list[int],
    meta_total_bytes: int,
) -> int:
    return meta_total_bytes * (len(metadata_ruh_ids) if use_fdp else 1)


def compute_usable_capacity(
    *,
    capacity_bytes: int,
    slot_bytes: int,
    metadata_reserved_bytes: int,
) -> tuple[int, int]:
    if capacity_bytes <= metadata_reserved_bytes:
        raise ValueError("capacity_bytes must be larger than reserved metadata bytes")
    usable = align_down(capacity_bytes - metadata_reserved_bytes, slot_bytes)
    slots = usable // slot_bytes
    if slots < 1:
        raise ValueError("raw-block window has no usable data slots")
    return usable, slots


def make_meta_magic(index: int) -> str:
    if index < 0 or index > 99_999_999:
        raise ValueError("window index cannot fit meta_magic")
    return f"MW{index + 1:06d}"


def compute_windows(
    *,
    total_capacity: int,
    start_offset: int,
    num_windows: int,
    policy: str,
    explicit_capacities: list[int],
    stride_value: int | str,
    guard_bytes: int,
    block_align: int,
    slot_bytes: int,
    meta_total_bytes: int,
    use_fdp: bool,
    metadata_ruh_ids: list[int],
    safety_check: bool = True,
) -> list[WindowLayout]:
    if num_windows <= 0:
        raise ValueError("--num-windows must be positive")
    if start_offset % block_align:
        raise ValueError("start offset must be block aligned")
    if guard_bytes % block_align:
        raise ValueError("guard bytes must be block aligned")
    reserved = metadata_reserved_bytes(
        use_fdp=use_fdp,
        metadata_ruh_ids=metadata_ruh_ids,
        meta_total_bytes=meta_total_bytes,
    )
    windows: list[WindowLayout] = []
    if policy == "equal":
        available = total_capacity - start_offset
        if available <= 0:
            raise ValueError("start offset is beyond device capacity")
        raw_stride = available // num_windows
        stride = align_down(raw_stride, block_align)
        if stride <= guard_bytes:
            raise ValueError("available capacity is too small for guard bytes")
        capacity = align_down(stride - guard_bytes, block_align)
        bases = [start_offset + index * stride for index in range(num_windows)]
        capacities = [capacity] * num_windows
    elif policy == "explicit":
        if len(explicit_capacities) != num_windows:
            raise ValueError("--window-capacity-bytes-list length must equal windows")
        bases = []
        capacities = explicit_capacities
        cursor = start_offset
        for capacity in capacities:
            bases.append(cursor)
            cursor += align_up(capacity + guard_bytes, block_align)
    else:
        raise ValueError(f"unknown window capacity policy: {policy}")

    if stride_value != "auto":
        stride = int(stride_value)
        if stride % block_align:
            raise ValueError("window stride must be block aligned")
        bases = [start_offset + index * stride for index in range(num_windows)]

    for index, (base, capacity) in enumerate(zip(bases, capacities, strict=True)):
        if capacity % block_align:
            raise ValueError("window capacity must be block aligned")
        usable, slots = compute_usable_capacity(
            capacity_bytes=capacity,
            slot_bytes=slot_bytes,
            metadata_reserved_bytes=reserved,
        )
        windows.append(
            WindowLayout(
                index=index,
                base_offset_bytes=base,
                capacity_bytes=capacity,
                metadata_reserved_bytes=reserved,
                usable_capacity_bytes=usable,
                estimated_slot_count=slots,
                meta_magic=make_meta_magic(index),
            )
        )
    if safety_check:
        validate_windows(
            windows,
            total_capacity=total_capacity,
            block_align=block_align,
        )
    return windows


def validate_windows(
    windows: list[WindowLayout],
    *,
    total_capacity: int,
    block_align: int,
) -> None:
    ranges = []
    seen_magic: set[str] = set()
    for window in windows:
        if window.base_offset_bytes % block_align:
            raise ValueError(f"window {window.index} base offset is not aligned")
        if window.capacity_bytes % block_align:
            raise ValueError(f"window {window.index} capacity is not aligned")
        if window.base_offset_bytes + window.capacity_bytes > total_capacity:
            raise ValueError(f"window {window.index} exceeds device capacity")
        if window.meta_magic in seen_magic:
            raise ValueError("meta_magic values must be unique")
        seen_magic.add(window.meta_magic)
        ranges.append(
            (
                window.base_offset_bytes,
                window.base_offset_bytes + window.capacity_bytes,
                window.index,
            )
        )
    for prev, current in zip(sorted(ranges), sorted(ranges)[1:], strict=False):
        if current[0] < prev[1]:
            raise ValueError(f"windows overlap: {prev[2]} and {current[2]}")


def workload_key_for_entry(entry: dict[str, Any], index: int, key_type: str) -> str:
    if key_type == "storage_class":
        return storage_class_for_entry(entry)
    if key_type == "dataset_adapter":
        dataset = entry.get("dataset") or {}
        return str(dataset.get("adapter") or dataset.get("family") or "unknown")
    if key_type == "trace_name":
        return trace_id_for_entry(entry, index)
    raise ValueError(f"unsupported workload key: {key_type}")


def trace_stats(entry: dict[str, Any], trace_path: str) -> dict[str, Any]:
    stats = entry.get("trace_stats")
    if isinstance(stats, dict) and stats:
        return stats
    try:
        footprint = analyze_trace_footprint(trace_path)
    except Exception:
        return {}
    return {
        "estimated_store_bytes": footprint.estimated_total_store_bytes,
        "duration_seconds": footprint.duration_seconds,
        "record_count": footprint.record_count,
        "store_count": footprint.store_count,
    }


def select_workloads(
    manifest: dict[str, Any],
    *,
    num_workloads: int,
    workload_key: str,
    workload_filter: str | None,
    placement: str,
    seed: int,
) -> list[TraceWorkload]:
    if num_workloads <= 0:
        raise ValueError("--num-workloads must be positive")
    pattern = re.compile(workload_filter) if workload_filter else None
    entries = trace_entries(manifest)
    candidates: list[TraceWorkload] = []
    seen_keys: set[str] = set()
    for index, entry in enumerate(entries):
        trace_path = str(entry.get("trace_path") or "")
        trace_id = trace_id_for_entry(entry, index)
        storage_class = storage_class_for_entry(entry)
        dataset = entry.get("dataset") or {}
        dataset_adapter = str(dataset.get("adapter") or dataset.get("family") or "")
        key = workload_key_for_entry(entry, index, workload_key)
        haystack = " ".join([trace_id, trace_path, key, storage_class, dataset_adapter])
        if pattern and not pattern.search(haystack):
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        stats = trace_stats(entry, trace_path)
        candidates.append(
            TraceWorkload(
                trace_id=trace_id,
                trace_path=trace_path,
                workload_key=key,
                storage_class=storage_class,
                dataset_adapter=dataset_adapter,
                entry=entry,
                estimated_store_bytes=_optional_int(stats.get("estimated_store_bytes")),
                duration_seconds=_optional_float(stats.get("duration_seconds")),
                record_count=_optional_int(stats.get("record_count")),
                store_count=_optional_int(stats.get("store_count")),
            )
        )
    if len(candidates) < num_workloads:
        raise ValueError(
            f"requested {num_workloads} workloads but only {len(candidates)} are "
            "available after filtering"
        )
    if placement == "random":
        rng = random.Random(seed)
        return rng.sample(candidates, num_workloads)
    if placement != "fixed":
        raise ValueError(f"unknown application placement: {placement}")
    return candidates[:num_workloads]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def assign_workloads_to_windows(
    workloads: list[TraceWorkload],
    *,
    num_windows: int,
    placement: str,
    seed: int,
    allow_multiplexing: bool,
) -> list[list[TraceWorkload]]:
    if len(workloads) > num_windows and not allow_multiplexing:
        raise ValueError(
            "num_workloads > num_windows requires --allow-workload-multiplexing"
        )
    ordered = list(workloads)
    if placement == "random":
        rng = random.Random(seed)
        rng.shuffle(ordered)
    if len(ordered) <= num_windows:
        return [[ordered[index % len(ordered)]] for index in range(num_windows)]
    groups: list[list[TraceWorkload]] = [[] for _ in range(num_windows)]
    for index, workload in enumerate(ordered):
        groups[index % num_windows].append(workload)
    return groups


def build_adapter_json(
    *,
    device_path: str,
    window: WindowLayout,
    slot_bytes: int,
    block_align: int,
    header_bytes: int,
    meta_total_bytes: int,
    use_uring: bool,
    use_uring_cmd: bool,
    use_odirect: bool,
    use_fdp: bool,
    fdp_data_ruh_ids: list[int],
    fdp_metadata_ruh_ids: list[int],
    fdp_metadata_mode: str,
    num_store_workers: int,
    num_lookup_workers: int,
    num_load_workers: int,
) -> dict[str, Any]:
    adapter = {
        "type": "raw_block",
        "device_path": device_path,
        "slot_bytes": slot_bytes,
        "base_offset_bytes": window.base_offset_bytes,
        "capacity_bytes": window.capacity_bytes,
        "meta_total_bytes": meta_total_bytes,
        "meta_magic": window.meta_magic,
        "block_align": block_align,
        "header_bytes": header_bytes,
        "use_odirect": use_odirect,
        "use_uring": use_uring,
        "use_uring_cmd": use_uring_cmd,
        "use_fdp": use_fdp,
        "num_store_workers": num_store_workers,
        "num_lookup_workers": num_lookup_workers,
        "num_load_workers": num_load_workers,
    }
    if use_fdp:
        adapter["fdp_data_ruh_ids"] = fdp_data_ruh_ids
        adapter["fdp_metadata_ruh_ids"] = fdp_metadata_ruh_ids
        adapter["fdp_metadata_mode"] = fdp_metadata_mode
    return adapter


def command_to_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_replay_command(
    *,
    replay_binary: str,
    trace_path: str,
    salt_suffix: str,
    l1_size_gb: float,
    block_align: int,
    output_dir: str,
    jsonl_out: str,
    adapter_json: dict[str, Any],
) -> list[str]:
    command = shlex.split(replay_binary)
    command.extend(
        [
            "trace",
            "replay",
            trace_path,
            "--replay-cache-salt-suffix",
            salt_suffix,
            "--l1-size-gb",
            str(l1_size_gb),
            "--no-l1-use-lazy",
            "--l1-align-bytes",
            str(block_align),
            "--eviction-policy",
            "noop",
            "--l2-store-policy",
            "skip_l1",
            "--output-dir",
            output_dir,
            "--json",
            "--jsonl-out",
            jsonl_out,
            "--disable-metrics",
            "--quiet",
            "--l2-adapter",
            json.dumps(adapter_json, separators=(",", ":")),
        ]
    )
    return command


def validate_stop_policy(args: argparse.Namespace) -> None:
    if args.stop_policy == "timeout" and args.duration_seconds is None:
        raise ValueError("--stop-policy timeout requires --duration-seconds")
    if args.stop_policy == "iterations" and args.iterations is None:
        raise ValueError("--stop-policy iterations requires --iterations")
    if args.stop_policy == "total_written_size":
        if (
            args.target_host_write_bytes is None
            and args.target_host_write_multiplier is None
        ):
            raise ValueError(
                "--stop-policy total_written_size requires --target-host-write-bytes "
                "or --target-host-write-multiplier"
            )
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be non-negative")


def estimate_iteration_bytes(workloads: list[TraceWorkload]) -> int:
    total = 0
    for workload in workloads:
        total += int(workload.estimated_store_bytes or 0)
    return max(1, total)


def flatten_workload_groups(
    workload_groups: list[list[TraceWorkload]],
) -> list[TraceWorkload]:
    return [workload for group in workload_groups for workload in group]


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "trace"


def estimate_runtime_seconds(
    workloads: list[TraceWorkload],
    *,
    iterations: int | None,
    warmup_iterations: int,
    stop_policy: str,
    duration_seconds: int | None,
) -> int | None:
    if stop_policy == "timeout":
        return duration_seconds
    max_duration = max((w.duration_seconds or 0.0) for w in workloads)
    if max_duration <= 0 or iterations is None:
        return None
    return math.ceil(max_duration * (iterations + warmup_iterations))


def resolve_plan(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    validate_stop_policy(args)
    device_capacity = (
        detect_device_capacity(args.block_device_path)
        if args.device_capacity_bytes == "auto"
        else int(args.device_capacity_bytes)
    )
    use_fdp = bool(args.use_fdp)
    use_uring_cmd = bool(args.use_uring_cmd or use_fdp)
    use_uring = bool(args.use_uring or use_uring_cmd)
    validate_ng_device_path(args.device_path, use_uring_cmd=use_uring_cmd)

    ruh_policy = resolve_ruh_policy(
        use_fdp=use_fdp,
        ruh_ids=args.ruh_ids,
        ruh_count=args.ruh_count,
        ruh_start_id=args.ruh_start_id,
        metadata_ruh_ids=args.metadata_ruh_ids,
    )
    if args.per_app_ruh_map:
        app_to_ruhs = load_per_app_ruh_map(args.per_app_ruh_map)
        for key, ids in app_to_ruhs.items():
            missing = sorted(set(ids) - set(ruh_policy.available_ruhs))
            if missing:
                raise ValueError(f"app {key!r} maps to unavailable RUHs {missing}")
    else:
        app_to_ruhs = {}

    workloads = select_workloads(
        manifest,
        num_workloads=args.num_workloads,
        workload_key=args.workload_key,
        workload_filter=args.workload_filter,
        placement=args.application_placement,
        seed=args.seed,
    )
    workload_groups = assign_workloads_to_windows(
        workloads,
        num_windows=args.num_windows,
        placement=args.application_placement,
        seed=args.seed,
        allow_multiplexing=args.allow_workload_multiplexing,
    )
    window_workloads = flatten_workload_groups(workload_groups)
    windows = compute_windows(
        total_capacity=device_capacity,
        start_offset=args.start_offset_bytes,
        num_windows=args.num_windows,
        policy=args.window_capacity_policy,
        explicit_capacities=args.window_capacity_bytes_list,
        stride_value=args.window_stride_bytes,
        guard_bytes=args.guard_bytes,
        block_align=args.block_align,
        slot_bytes=args.slot_bytes,
        meta_total_bytes=args.meta_total_bytes,
        use_fdp=use_fdp,
        metadata_ruh_ids=ruh_policy.metadata_ruhs,
        safety_check=args.capacity_safety_check,
    )

    total_window_capacity = sum(window.capacity_bytes for window in windows)
    if args.target_host_write_bytes is not None:
        target_host_write_bytes = int(args.target_host_write_bytes)
    elif args.target_host_write_multiplier is not None:
        target_host_write_bytes = (
            total_window_capacity * args.target_host_write_multiplier
        )
    else:
        target_host_write_bytes = None

    estimated_iterations_for_target = None
    if args.stop_policy == "total_written_size":
        target = int(target_host_write_bytes or 1)
        estimated_iterations_for_target = max(
            1,
            math.ceil(target / estimate_iteration_bytes(window_workloads)),
        )
    iterations = args.iterations
    runtime_iterations = iterations
    if args.stop_policy == "total_written_size" and runtime_iterations is None:
        runtime_iterations = estimated_iterations_for_target

    rng = random.Random(args.seed)
    warnings: list[str] = []
    if ruh_policy.warning:
        warnings.append(ruh_policy.warning)
    if args.gpu_io_mode == "gds_if_supported":
        raise ValueError(
            "gds_if_supported requested, but this branch does not implement a "
            "GPU-direct raw-block path. Use cpu_stage."
        )
    if args.gpu_io_mode == "cpu_stage":
        warnings.append(
            "cpu_stage selected: replay uses LMCache CPU/L1 staging plus async "
            "StoreController and RawBlockL2Adapter store workers."
        )

    window_plans: list[WindowPlan] = []
    for window, workload_group in zip(windows, workload_groups, strict=True):
        workload = workload_group[0]
        data_ruhs = assigned_data_ruhs(
            policy=ruh_policy,
            assignment=args.ruh_assignment,
            window_index=window.index,
            workload_key=workload.workload_key,
            app_to_ruhs=app_to_ruhs,
        )
        launch_delay = (
            rng.uniform(0, args.launch_jitter_sec)
            if args.launch_policy == "random_jitter"
            else 0.0
        )
        window_dir = os.fspath(
            Path(args.output_dir) / f"per_window/window_{window.index}"
        )
        adapter_json = build_adapter_json(
            device_path=args.device_path,
            window=window,
            slot_bytes=args.slot_bytes,
            block_align=args.block_align,
            header_bytes=args.header_bytes,
            meta_total_bytes=args.meta_total_bytes,
            use_uring=use_uring,
            use_uring_cmd=use_uring_cmd,
            use_odirect=args.use_odirect,
            use_fdp=use_fdp,
            fdp_data_ruh_ids=data_ruhs,
            fdp_metadata_ruh_ids=ruh_policy.metadata_ruhs if use_fdp else [],
            fdp_metadata_mode=args.fdp_metadata_mode,
            num_store_workers=args.num_store_workers,
            num_lookup_workers=args.num_lookup_workers,
            num_load_workers=args.num_load_workers,
        )
        commands: list[list[str]] = []
        command_texts: list[str] = []
        workload_summaries: list[dict[str, str]] = []
        jsonl_out = os.fspath(Path(window_dir) / "records.jsonl")
        for workload_index, group_workload in enumerate(workload_group):
            workload_output_dir = window_dir
            workload_jsonl_out = jsonl_out
            if len(workload_group) > 1:
                workload_dir_name = (
                    f"workload_{workload_index:02d}_"
                    f"{safe_path_part(group_workload.trace_id)}"
                )
                workload_output_dir = os.fspath(
                    Path(window_dir) / workload_dir_name
                )
                workload_jsonl_out = os.fspath(
                    Path(workload_output_dir) / "records.jsonl"
                )
            salt = (
                f"{args.run_id}.{group_workload.trace_id}.w{window.index}."
                f"m{workload_index}.iter_{{iteration}}"
            )
            command = build_replay_command(
                replay_binary=args.replay_binary,
                trace_path=group_workload.trace_path,
                salt_suffix=salt,
                l1_size_gb=args.l1_size_gb,
                block_align=args.block_align,
                output_dir=workload_output_dir,
                jsonl_out=workload_jsonl_out,
                adapter_json=adapter_json,
            )
            commands.append(command)
            command_texts.append(command_to_text(command))
            workload_summaries.append(
                {
                    "trace_id": group_workload.trace_id,
                    "trace_path": group_workload.trace_path,
                    "workload_key": group_workload.workload_key,
                    "storage_class": group_workload.storage_class,
                    "dataset_adapter": group_workload.dataset_adapter,
                    "output_dir": workload_output_dir,
                    "jsonl_out": workload_jsonl_out,
                }
            )
        window_plans.append(
            WindowPlan(
                index=window.index,
                trace_id=workload.trace_id,
                trace_path=workload.trace_path,
                workload_key=workload.workload_key,
                storage_class=workload.storage_class,
                base_offset_bytes=window.base_offset_bytes,
                capacity_bytes=window.capacity_bytes,
                metadata_reserved_bytes=window.metadata_reserved_bytes,
                usable_capacity_bytes=window.usable_capacity_bytes,
                slot_bytes=args.slot_bytes,
                estimated_slot_count=window.estimated_slot_count,
                meta_magic=window.meta_magic,
                fdp_data_ruh_ids=data_ruhs,
                fdp_metadata_ruh_ids=ruh_policy.metadata_ruhs if use_fdp else [],
                command=commands[0],
                command_text=command_texts[0],
                commands=commands,
                command_texts=command_texts,
                workloads=workload_summaries,
                output_dir=window_dir,
                jsonl_out=jsonl_out,
                num_store_workers=args.num_store_workers,
                num_lookup_workers=args.num_lookup_workers,
                num_load_workers=args.num_load_workers,
                launch_delay_sec=launch_delay,
            )
        )

    estimated_runtime = estimate_runtime_seconds(
        window_workloads,
        iterations=runtime_iterations,
        warmup_iterations=args.warmup_iterations,
        stop_policy=args.stop_policy,
        duration_seconds=args.duration_seconds,
    )
    plan = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "device_path": args.device_path,
        "block_device_path": args.block_device_path,
        "device_capacity_bytes": device_capacity,
        "start_offset_bytes": args.start_offset_bytes,
        "num_windows": args.num_windows,
        "num_workloads": args.num_workloads,
        "window_capacity_policy": args.window_capacity_policy,
        "guard_bytes": args.guard_bytes,
        "block_align": args.block_align,
        "slot_bytes": args.slot_bytes,
        "header_bytes": args.header_bytes,
        "meta_total_bytes": args.meta_total_bytes,
        "use_fdp": use_fdp,
        "use_uring": use_uring,
        "use_uring_cmd": use_uring_cmd,
        "use_odirect": args.use_odirect,
        "ruh_assignment": args.ruh_assignment,
        "available_ruhs": ruh_policy.available_ruhs,
        "metadata_ruh_ids": ruh_policy.metadata_ruhs if use_fdp else [],
        "application_placement": args.application_placement,
        "launch_policy": args.launch_policy,
        "max_active_windows": args.max_active_windows,
        "stop_policy": args.stop_policy,
        "duration_seconds": args.duration_seconds,
        "iterations": iterations,
        "estimated_iterations_for_target": estimated_iterations_for_target,
        "warmup_iterations": args.warmup_iterations,
        "target_host_write_bytes": target_host_write_bytes,
        "target_host_write_multiplier": args.target_host_write_multiplier,
        "total_written_size_end_condition_preference": [
            END_CONDITION_ACTUAL_WRITTEN_SOURCE,
            END_CONDITION_HOST_WRITE_SOURCE,
        ],
        "total_configured_window_capacity_bytes": total_window_capacity,
        "estimated_usable_raw_block_data_capacity_bytes": sum(
            window.usable_capacity_bytes for window in windows
        ),
        "estimated_bytes_per_iteration": estimate_iteration_bytes(window_workloads),
        "estimated_runtime_seconds": estimated_runtime,
        "estimated_runtime": (
            "unknown" if estimated_runtime is None else f"{estimated_runtime}s"
        ),
        "gpu_io_mode": args.gpu_io_mode,
        "num_store_workers": args.num_store_workers,
        "num_lookup_workers": args.num_lookup_workers,
        "num_load_workers": args.num_load_workers,
        "media_write_counter_command": args.media_write_counter_command,
        "io_path": (
            "lmcache trace replay -> StoreController / PrefetchController -> "
            "RawBlockL2Adapter -> RawBlockCore -> Rust raw-block I/O -> "
            "NVMe namespace"
        ),
        "warnings": warnings,
        "windows": [asdict(window) for window in window_plans],
    }
    return plan


def write_plan_artifacts(plan: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_plan.json", plan)
    write_yaml(output / "resolved_plan.yaml", plan)
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for window in plan["windows"]:
        commands.append(f"# window {window['index']}: {window['trace_id']}")
        for command_text in window.get("command_texts", [window["command_text"]]):
            commands.append(command_text.replace("{iteration}", "0000"))
        commands.append("")
    commands_path = output / "commands.sh"
    commands_path.write_text("\n".join(commands) + "\n")
    commands_path.chmod(0o755)


def print_run_plan(plan: dict[str, Any]) -> None:
    print("Resolved raw-block replay plan")
    print(f"  device_path: {plan['device_path']}")
    print(f"  block_device_path: {plan['block_device_path']}")
    print(f"  device_capacity_bytes: {plan['device_capacity_bytes']}")
    print(f"  start_offset_bytes: {plan['start_offset_bytes']}")
    print(f"  num_windows: {plan['num_windows']}")
    print(
        "  total_configured_window_capacity_bytes: "
        f"{plan['total_configured_window_capacity_bytes']}"
    )
    print(
        "  estimated_usable_raw_block_data_capacity_bytes: "
        f"{plan['estimated_usable_raw_block_data_capacity_bytes']}"
    )
    print(f"  target_host_write_bytes: {plan['target_host_write_bytes']}")
    print(
        "  total_written_size_end_condition_preference: "
        f"{plan['total_written_size_end_condition_preference']}"
    )
    print(f"  use_odirect: {plan['use_odirect']}")
    print(f"  stop_policy: {plan['stop_policy']}")
    print(f"  warmup_iterations: {plan['warmup_iterations']}")
    print(f"  estimated_runtime: {plan['estimated_runtime']}")
    for warning in plan["warnings"]:
        print(f"  warning: {warning}")
    for window in plan["windows"]:
        print(
            "  window {index}: trace={trace_id} app={workload_key} "
            "base={base_offset_bytes} capacity={capacity_bytes} "
            "usable={usable_capacity_bytes} slots={estimated_slot_count} "
            "data_ruhs={fdp_data_ruh_ids} metadata_ruhs={fdp_metadata_ruh_ids}".format(
                **window
            )
        )


def confirm_destructive_run(plan: dict[str, Any], *, yes: bool, allow: bool) -> bool:
    if not allow:
        return False
    if yes:
        return True
    end_offset = max(
        window["base_offset_bytes"] + window["capacity_bytes"]
        for window in plan["windows"]
    )
    print(
        f"This run may write approximately {plan['target_host_write_bytes']} bytes "
        f"to {plan['device_path']}."
    )
    print(f"Estimated runtime: {plan['estimated_runtime']}.")
    print(
        f"It will use {plan['num_windows']} raw-block windows from "
        f"{plan['start_offset_bytes']} to {end_offset}."
    )
    response = input("Type RUN to continue: ")
    return response == "RUN"


def capture_host_write_bytes(block_device_path: str) -> int | None:
    global LAST_HOST_WRITE_COUNTER_SOURCE
    LAST_HOST_WRITE_COUNTER_SOURCE = None
    if not shutil.which("nvme"):
        return capture_sysfs_host_write_bytes(block_device_path)
    proc = subprocess.run(
        ["nvme", "smart-log", block_device_path],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode == 0:
        match = re.search(r"Data Units Written\s*:\s*([0-9,]+)", proc.stdout)
        if match:
            LAST_HOST_WRITE_COUNTER_SOURCE = "nvme_smart_log_data_units_written"
            return int(match.group(1).replace(",", "")) * 512_000
    return capture_sysfs_host_write_bytes(block_device_path)


def capture_sysfs_host_write_bytes(block_device_path: str) -> int | None:
    global LAST_HOST_WRITE_COUNTER_SOURCE
    stat_path = Path("/sys/class/block") / Path(block_device_path).name / "stat"
    try:
        fields = stat_path.read_text().split()
    except OSError:
        return None
    if len(fields) < 7:
        return None
    try:
        sectors_written = int(fields[6])
    except ValueError:
        return None
    LAST_HOST_WRITE_COUNTER_SOURCE = "sysfs_block_stat_write_sectors"
    return sectors_written * 512


def capture_media_write_bytes(command: str | None) -> int | None:
    if not command:
        return None
    proc = subprocess.run(
        command,
        shell=True,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return None
    try:
        payload: Any = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = proc.stdout
    return extract_media_write_bytes(payload)


def parse_failed_records(jsonl_path: str) -> int | None:
    path = Path(jsonl_path)
    if not path.exists():
        return None
    failed = 0
    for line in path.read_text(errors="replace").splitlines():
        try:
            if json.loads(line).get("failed"):
                failed += 1
        except json.JSONDecodeError:
            continue
    return failed


def window_failed_records(window: dict[str, Any]) -> int | None:
    paths = [workload["jsonl_out"] for workload in window.get("workloads", [])]
    if not paths:
        paths = [window["jsonl_out"]]
    total = 0
    found = False
    for path in paths:
        failed = parse_failed_records(path)
        if failed is None:
            continue
        found = True
        total += failed
    return total if found else None


def _empty_raw_block_accounting() -> dict[str, int]:
    return {field: 0 for field in RAW_BLOCK_ACCOUNTING_FIELDS}


def _add_raw_block_accounting(
    total: dict[str, int],
    accounting: dict[str, Any] | None,
) -> dict[str, int]:
    if not accounting:
        return total
    values: dict[str, int] = {}
    for field in RAW_BLOCK_ACCOUNTING_FIELDS:
        try:
            values[field] = int(accounting.get(field, 0) or 0)
        except (TypeError, ValueError):
            values[field] = 0
    if values["total_write_physical_bytes"] == 0:
        values["total_write_physical_bytes"] = values["media_write_physical_bytes"]
    if values["total_write_logical_bytes"] == 0:
        values["total_write_logical_bytes"] = values["media_write_logical_bytes"]
    if values["data_write_physical_bytes"] == 0:
        payload = values["data_write_payload_physical_bytes"]
        header = values["data_write_header_physical_bytes"]
        values["data_write_physical_bytes"] = payload + header
    for field, value in values.items():
        total[field] += value
    return total


def extract_raw_block_accounting(status: dict[str, Any]) -> dict[str, int]:
    total = _empty_raw_block_accounting()
    for adapter in status.get("l2_adapters", []):
        core = adapter.get("core", {}) if isinstance(adapter, dict) else {}
        accounting = core.get("io_accounting", {}) if isinstance(core, dict) else {}
        _add_raw_block_accounting(total, accounting)
    return total


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def window_io_accounting(window: dict[str, Any]) -> dict[str, int] | None:
    output_dirs = [
        Path(workload["output_dir"]) for workload in window.get("workloads", [])
    ]
    if not output_dirs:
        output_dirs = [Path(window["output_dir"])]
    total = _empty_raw_block_accounting()
    found = False
    for output_dir in sorted(set(output_dirs)):
        status = _read_json_if_exists(output_dir / "storage_manager_status.json")
        if status is None:
            continue
        _add_raw_block_accounting(total, extract_raw_block_accounting(status))
        found = True
    return total if found else None


def aggregate_result_io_accounting(results: list[dict[str, Any]]) -> dict[str, int]:
    total = _empty_raw_block_accounting()
    for result in results:
        _add_raw_block_accounting(total, result.get("io_accounting"))
    return total


def measurement_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if int(result.get("iteration", -1)) >= 0]


def total_written_size_progress(
    results: list[dict[str, Any]],
    *,
    host_write_delta: int | None,
    defer_host_fallback_until_result: bool = False,
) -> tuple[int | None, str]:
    measured_results = measurement_results(results)
    lmcache_accounting_available = any(
        result.get("io_accounting") is not None for result in measured_results
    )
    if lmcache_accounting_available:
        accounting = aggregate_result_io_accounting(measured_results)
        return (
            int(accounting["total_write_physical_bytes"]),
            END_CONDITION_ACTUAL_WRITTEN_SOURCE,
        )
    if defer_host_fallback_until_result and not measured_results:
        return None, END_CONDITION_ACTUAL_WRITTEN_SOURCE
    return host_write_delta, END_CONDITION_HOST_WRITE_SOURCE


def run_iteration(
    plan: dict[str, Any],
    *,
    iteration: int,
    timeout_seconds: int | None,
    stop_check: Callable[[list[dict[str, Any]]], bool] | None = None,
) -> list[dict[str, Any]]:
    procs: list[tuple[subprocess.Popen, Path, dict[str, Any], Any]] = []
    results: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
    active_windows = plan["windows"]
    max_active = min(plan["max_active_windows"], len(active_windows))
    pending = list(active_windows)

    def launch(window: dict[str, Any]) -> None:
        delay = float(window.get("launch_delay_sec") or 0)
        if delay > 0:
            time.sleep(delay)
        window_dir = Path(window["output_dir"])
        window_dir.mkdir(parents=True, exist_ok=True)
        for workload in window.get("workloads", []):
            Path(workload["output_dir"]).mkdir(parents=True, exist_ok=True)
        rendered_commands = [
            [
                part.replace("{iteration}", f"{iteration:04d}")
                for part in command
            ]
            for command in window.get("commands", [window["command"]])
        ]
        if len(rendered_commands) == 1:
            popen_command = rendered_commands[0]
        else:
            popen_command = [
                "bash",
                "-lc",
                " && ".join(command_to_text(command) for command in rendered_commands),
            ]
        log_path = window_dir / f"iteration_{iteration:04d}.log"
        log = log_path.open("w")
        proc = subprocess.Popen(popen_command, stdout=log, stderr=subprocess.STDOUT)
        ACTIVE_PROCS.append(proc)
        procs.append((proc, log_path, window, log))

    while pending or procs:
        while pending and len(procs) < max_active:
            launch(pending.pop(0))
        for proc, log_path, window, log in list(procs):
            if proc.poll() is None:
                continue
            log.close()
            if proc in ACTIVE_PROCS:
                ACTIVE_PROCS.remove(proc)
            procs.remove((proc, log_path, window, log))
            results.append(
                {
                    "window_index": window["index"],
                    "iteration": iteration,
                    "exit_code": proc.returncode,
                    "log_path": os.fspath(log_path),
                    "jsonl_out": window["jsonl_out"],
                    "failed_records": window_failed_records(window),
                    "io_accounting": window_io_accounting(window),
                    "ended_at": utc_now(),
                }
            )
        if deadline and time.monotonic() >= deadline:
            for proc, _, _, _ in procs:
                if proc.poll() is None:
                    proc.terminate()
            break
        if stop_check is not None and stop_check(results):
            for proc, _, _, _ in procs:
                if proc.poll() is None:
                    proc.terminate()
            break
        time.sleep(0.05)
    for proc, log_path, window, log in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
        if proc in ACTIVE_PROCS:
            ACTIVE_PROCS.remove(proc)
        results.append(
            {
                "window_index": window["index"],
                "iteration": iteration,
                "exit_code": proc.returncode,
                "log_path": os.fspath(log_path),
                "jsonl_out": window["jsonl_out"],
                "failed_records": window_failed_records(window),
                "io_accounting": window_io_accounting(window),
                "ended_at": utc_now(),
            }
        )
    return results


def run_plan(plan: dict[str, Any], output_dir: str) -> dict[str, Any]:
    stop_requested = False

    def _signal_handler(signum, frame):  # noqa: ANN001
        del signum, frame
        nonlocal stop_requested
        stop_requested = True
        for proc in list(ACTIVE_PROCS):
            if proc.poll() is None:
                proc.terminate()

    old_sigint = signal.signal(signal.SIGINT, _signal_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
    before = capture_host_write_bytes(plan["block_device_path"])
    media_before = capture_media_write_bytes(plan.get("media_write_counter_command"))
    results: list[dict[str, Any]] = []
    completed_iterations = 0
    try:
        for iteration in range(plan["warmup_iterations"]):
            if stop_requested:
                break
            results.extend(
                run_iteration(
                    plan,
                    iteration=-(iteration + 1),
                    timeout_seconds=None,
                )
            )
        measurement_start = capture_host_write_bytes(plan["block_device_path"])
        media_measurement_start = capture_media_write_bytes(
            plan.get("media_write_counter_command")
        )
        last_counter_check = 0.0
        latest_host_write_bytes: int | None = measurement_start
        latest_end_condition_bytes: int | None = None

        def target_reached_during_iteration(
            iteration_results: list[dict[str, Any]],
        ) -> bool:
            nonlocal last_counter_check
            nonlocal latest_host_write_bytes
            nonlocal latest_end_condition_bytes
            if plan["stop_policy"] != "total_written_size":
                return False
            if plan["target_host_write_bytes"] is None:
                return False
            now = time.monotonic()
            if now - last_counter_check < 5.0:
                return False
            last_counter_check = now
            latest_host_write_bytes = capture_host_write_bytes(
                plan["block_device_path"]
            )
            host_write_delta = (
                latest_host_write_bytes - measurement_start
                if latest_host_write_bytes is not None
                and measurement_start is not None
                else None
            )
            latest_end_condition_bytes = total_written_size_progress(
                [*results, *iteration_results],
                host_write_delta=host_write_delta,
                defer_host_fallback_until_result=True,
            )[0]
            return (
                latest_end_condition_bytes is not None
                and latest_end_condition_bytes >= plan["target_host_write_bytes"]
            )

        iteration = 0
        while True:
            if stop_requested:
                break
            if plan["stop_policy"] == "iterations" and iteration >= int(
                plan["iterations"]
            ):
                break
            if (
                plan["stop_policy"] == "total_written_size"
                and plan["iterations"] is not None
                and iteration >= int(plan["iterations"])
            ):
                break
            if plan["stop_policy"] == "timeout" and iteration > 0:
                break
            timeout = (
                plan["duration_seconds"] if plan["stop_policy"] == "timeout" else None
            )
            results.extend(
                run_iteration(
                    plan,
                    iteration=iteration,
                    timeout_seconds=timeout,
                    stop_check=target_reached_during_iteration,
                )
            )
            completed_iterations += 1
            current = latest_host_write_bytes or capture_host_write_bytes(
                plan["block_device_path"]
            )
            host_write_delta = (
                current - measurement_start
                if current is not None and measurement_start is not None
                else None
            )
            latest_end_condition_bytes = total_written_size_progress(
                results,
                host_write_delta=host_write_delta,
            )[0]
            if plan["stop_policy"] == "timeout":
                break
            if (
                plan["stop_policy"] == "total_written_size"
                and plan["target_host_write_bytes"] is not None
                and latest_end_condition_bytes is not None
                and latest_end_condition_bytes >= plan["target_host_write_bytes"]
            ):
                break
            iteration += 1
        after = capture_host_write_bytes(plan["block_device_path"])
        media_after = capture_media_write_bytes(plan.get("media_write_counter_command"))
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
    host_delta = (
        after - measurement_start
        if after is not None and measurement_start is not None
        else None
    )
    media_delta = (
        media_after - media_measurement_start
        if media_after is not None and media_measurement_start is not None
        else None
    )
    measured_results = measurement_results(results)
    lmcache_io_accounting = aggregate_result_io_accounting(measured_results)
    total_written_size_end_condition_bytes, total_written_size_end_condition_source = (
        total_written_size_progress(
            results,
            host_write_delta=host_delta,
        )
    )
    total_written_size_target_reached = (
        total_written_size_end_condition_bytes is not None
        and plan["target_host_write_bytes"] is not None
        and total_written_size_end_condition_bytes >= plan["target_host_write_bytes"]
    )
    lmcache_successful_write_physical_bytes = int(
        lmcache_io_accounting["total_write_physical_bytes"]
    )
    host_vs_lmcache_successful_physical_delta = (
        host_delta - lmcache_successful_write_physical_bytes
        if host_delta is not None
        else None
    )
    host_vs_lmcache_successful_physical_ratio = (
        host_delta / lmcache_successful_write_physical_bytes
        if host_delta is not None and lmcache_successful_write_physical_bytes > 0
        else None
    )
    waf = None
    waf_status = "unavailable: media/NAND write counter missing"
    if host_delta is None:
        waf_status = "unavailable: host write counter missing"
    elif host_delta <= 0:
        waf_status = "unavailable: host write delta is zero"
    elif media_delta is not None:
        waf = media_delta / host_delta
        waf_status = "available"
    summary = {
        "run_id": plan["run_id"],
        "device_path": plan["device_path"],
        "block_device_path": plan["block_device_path"],
        "completed_iterations": completed_iterations,
        "host_write_bytes_before": before,
        "host_write_bytes_delta": host_delta,
        "host_write_counter_source": LAST_HOST_WRITE_COUNTER_SOURCE,
        "media_write_bytes_before": media_before,
        "media_write_bytes_delta": media_delta,
        "lmcache_io_accounting": lmcache_io_accounting,
        "lmcache_store_attempted_logical_bytes": lmcache_io_accounting[
            "store_attempted_logical_bytes"
        ],
        "lmcache_store_committed_logical_bytes": lmcache_io_accounting[
            "store_committed_logical_bytes"
        ],
        "lmcache_eviction_count": lmcache_io_accounting["eviction_count"],
        "lmcache_eviction_logical_bytes": lmcache_io_accounting[
            "eviction_logical_bytes"
        ],
        "lmcache_successful_data_write_physical_bytes": lmcache_io_accounting[
            "data_write_physical_bytes"
        ],
        "lmcache_successful_metadata_write_physical_bytes": lmcache_io_accounting[
            "metadata_write_physical_bytes"
        ],
        "lmcache_successful_write_physical_bytes": (
            lmcache_successful_write_physical_bytes
        ),
        "host_vs_lmcache_successful_physical_delta_bytes": (
            host_vs_lmcache_successful_physical_delta
        ),
        "host_vs_lmcache_successful_physical_ratio": (
            host_vs_lmcache_successful_physical_ratio
        ),
        "target_host_write_bytes": plan["target_host_write_bytes"],
        "target_host_write_bytes_reached": (
            host_delta is not None
            and plan["target_host_write_bytes"] is not None
            and host_delta >= plan["target_host_write_bytes"]
        ),
        "total_written_size_end_condition_source": (
            total_written_size_end_condition_source
        ),
        "total_written_size_end_condition_bytes": (
            total_written_size_end_condition_bytes
        ),
        "total_written_size_target_bytes": plan["target_host_write_bytes"],
        "total_written_size_target_reached": total_written_size_target_reached,
        "waf": waf,
        "waf_status": waf_status,
        "windows": plan["windows"],
        "results": results,
        "exit_codes": [result["exit_code"] for result in results],
    }
    write_json(Path(output_dir) / "summary.json", summary)
    write_summary_md(Path(output_dir) / "summary.md", summary)
    return summary


def write_summary_md(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Multi-Window Raw-Block Replay Summary",
        "",
        f"- run_id: {summary['run_id']}",
        f"- device_path: {summary['device_path']}",
        f"- block_device_path: {summary['block_device_path']}",
        f"- completed_iterations: {summary['completed_iterations']}",
        f"- host_write_bytes_delta: {summary['host_write_bytes_delta']}",
        f"- host_write_counter_source: {summary['host_write_counter_source']}",
        f"- media_write_bytes_delta: {summary['media_write_bytes_delta']}",
        "- lmcache_store_attempted_logical_bytes: "
        f"{summary['lmcache_store_attempted_logical_bytes']}",
        "- lmcache_store_committed_logical_bytes: "
        f"{summary['lmcache_store_committed_logical_bytes']}",
        f"- lmcache_eviction_count: {summary['lmcache_eviction_count']}",
        "- lmcache_eviction_logical_bytes: "
        f"{summary['lmcache_eviction_logical_bytes']}",
        "- lmcache_successful_data_write_physical_bytes: "
        f"{summary['lmcache_successful_data_write_physical_bytes']}",
        "- lmcache_successful_metadata_write_physical_bytes: "
        f"{summary['lmcache_successful_metadata_write_physical_bytes']}",
        "- lmcache_successful_write_physical_bytes: "
        f"{summary['lmcache_successful_write_physical_bytes']}",
        "- host_vs_lmcache_successful_physical_delta_bytes: "
        f"{summary['host_vs_lmcache_successful_physical_delta_bytes']}",
        "- host_vs_lmcache_successful_physical_ratio: "
        f"{summary['host_vs_lmcache_successful_physical_ratio']}",
        "- target_host_write_bytes_reached: "
        f"{summary['target_host_write_bytes_reached']}",
        "- total_written_size_end_condition_source: "
        f"{summary['total_written_size_end_condition_source']}",
        "- total_written_size_end_condition_bytes: "
        f"{summary['total_written_size_end_condition_bytes']}",
        "- total_written_size_target_bytes: "
        f"{summary['total_written_size_target_bytes']}",
        "- total_written_size_target_reached: "
        f"{summary['total_written_size_target_reached']}",
        f"- waf: {summary['waf']}",
        f"- waf_status: {summary['waf_status']}",
        "",
        "| window | trace | base_offset_bytes | capacity_bytes | "
        "usable_capacity_bytes | data RUHs | metadata RUHs |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for window in summary["windows"]:
        lines.append(
            f"| {window['index']} | {window['trace_id']} | "
            f"{window['base_offset_bytes']} | {window['capacity_bytes']} | "
            f"{window['usable_capacity_bytes']} | {window['fdp_data_ruh_ids']} | "
            f"{window['fdp_metadata_ruh_ids']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-manifest", required=True)
    parser.add_argument("--device-path", required=True)
    parser.add_argument("--block-device-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-capacity-bytes", default="auto")
    parser.add_argument("--start-offset-bytes", default="0")
    parser.add_argument("--num-windows", type=int, required=True)
    parser.add_argument(
        "--window-capacity-policy",
        choices=VALID_WINDOW_POLICIES,
        default="equal",
    )
    parser.add_argument("--window-capacity-bytes-list", default="")
    parser.add_argument("--window-stride-bytes", default="auto")
    parser.add_argument("--guard-bytes", default="0")
    parser.add_argument("--block-align", type=int, default=4096)
    parser.add_argument("--slot-bytes", default="4MiB")
    parser.add_argument("--meta-total-bytes", default="64MiB")
    parser.add_argument("--header-bytes", default="4096")
    parser.add_argument(
        "--capacity-safety-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num-workloads", type=int, required=True)
    parser.add_argument(
        "--workload-key",
        choices=VALID_WORKLOAD_KEYS,
        default="storage_class",
    )
    parser.add_argument("--workload-filter", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--launch-policy",
        choices=VALID_LAUNCH_POLICIES,
        default="simultaneous",
    )
    parser.add_argument("--launch-jitter-sec", type=float, default=0.0)
    parser.add_argument(
        "--application-placement",
        choices=VALID_PLACEMENTS,
        default="fixed",
    )
    parser.add_argument("--allow-workload-multiplexing", action="store_true")
    parser.add_argument("--use-fdp", type=parse_bool, default=False)
    parser.add_argument("--use-uring", type=parse_bool, default=True)
    parser.add_argument("--use-uring-cmd", type=parse_bool, default=False)
    parser.add_argument("--use-odirect", type=parse_bool, default=False)
    parser.add_argument("--ruh-count", type=int, default=None)
    parser.add_argument("--ruh-start-id", type=int, default=0)
    parser.add_argument("--ruh-ids", default=None)
    parser.add_argument("--metadata-ruh-ids", default="auto")
    parser.add_argument(
        "--ruh-assignment",
        choices=VALID_RUH_ASSIGNMENTS,
        default="mixed",
    )
    parser.add_argument("--per-app-ruh-map", default=None)
    parser.add_argument("--fdp-metadata-mode", choices=("per_ruh",), default="per_ruh")
    parser.add_argument(
        "--stop-policy",
        choices=VALID_STOP_POLICIES,
        default="iterations",
    )
    parser.add_argument("--duration-seconds", type=int, default=None)
    parser.add_argument("--target-host-write-bytes", default=None)
    parser.add_argument("--target-host-write-multiplier", type=int, default=None)
    parser.add_argument("--media-write-counter-command", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--num-store-workers", type=int, default=2)
    parser.add_argument("--num-load-workers", type=int, default=4)
    parser.add_argument("--num-lookup-workers", type=int, default=1)
    parser.add_argument("--max-active-windows", type=int, default=None)
    parser.add_argument(
        "--gpu-io-mode",
        choices=VALID_GPU_IO_MODES,
        default="none",
    )
    parser.add_argument(
        "--replay-binary",
        default="uv run --no-sync lmcache",
    )
    parser.add_argument("--l1-size-gb", type=float, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--allow-destructive-device-write", action="store_true")
    parser.add_argument("--run-id", default="multiwin001")
    args = parser.parse_args(argv)

    args.device_capacity_bytes = parse_size(
        args.device_capacity_bytes,
        allow_auto=True,
    )
    args.start_offset_bytes = int(parse_size(args.start_offset_bytes))
    args.window_capacity_bytes_list = parse_csv_sizes(args.window_capacity_bytes_list)
    args.window_stride_bytes = parse_size(args.window_stride_bytes, allow_auto=True)
    args.guard_bytes = int(parse_size(args.guard_bytes))
    args.slot_bytes = int(parse_size(args.slot_bytes))
    args.meta_total_bytes = int(parse_size(args.meta_total_bytes))
    args.header_bytes = int(parse_size(args.header_bytes))
    if args.target_host_write_bytes is not None:
        args.target_host_write_bytes = int(parse_size(args.target_host_write_bytes))
    if args.max_active_windows is None:
        args.max_active_windows = args.num_windows
    if args.max_active_windows <= 0:
        parser.error("--max-active-windows must be positive")
    if args.launch_jitter_sec < 0:
        parser.error("--launch-jitter-sec must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.trace_manifest)
    plan = resolve_plan(args, manifest)
    plan["max_active_windows"] = args.max_active_windows
    write_plan_artifacts(plan, args.output_dir)
    print_run_plan(plan)

    default_plan_only = not args.allow_destructive_device_write
    if args.dry_run or args.plan_only or default_plan_only:
        return 0

    if not confirm_destructive_run(
        plan,
        yes=args.yes,
        allow=args.allow_destructive_device_write,
    ):
        print("Aborted; destructive run was not confirmed.", file=sys.stderr)
        return 2
    run_plan(plan, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
