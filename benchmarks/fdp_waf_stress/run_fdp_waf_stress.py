# SPDX-License-Identifier: Apache-2.0

"""Concurrent FDP WAF stress harness for LMCache trace replay."""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time

# Third Party
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

DEFAULT_META_TOTAL_BYTES = 64 * 1024 * 1024
DEFAULT_SLOT_HEADER_BYTES = 4096
DEFAULT_RUN_ID = "waf001"
DEFAULT_OUTPUT_ROOT = "/mnt/hc-ssd/lmcache-fdp-waf-stress"
DEFAULT_SAMPLE_INTERVAL_SECONDS = 300
DEFAULT_TARGET_HOST_WRITE_POLL_SECONDS = 60
VALID_MODES = ("mixed", "separated", "no_fdp")
WAF_SAMPLE_TSV_COLUMNS = (
    "timestamp",
    "fdp_host_write_bytes",
    "fdp_media_write_bytes",
    "fdp_waf",
    "device_write_multiplier",
    "sample_status",
)
WAF_SAMPLE_COLUMN_WIDTHS = {
    "timestamp": 20,
    "fdp_host_write_bytes": 22,
    "fdp_media_write_bytes": 22,
    "fdp_waf": 10,
    "device_write_multiplier": 24,
    "sample_status": 14,
}


@dataclass
class TraceFootprint:
    trace_path: str
    record_count: int = 0
    store_count: int = 0
    retrieve_prefetch_count: int = 0
    unique_object_key_count: int = 0
    estimated_total_store_bytes: int = 0
    estimated_max_object_bytes: int = 0
    duration_seconds: float = 0.0
    exact_byte_estimate: bool = True
    warnings: list[str] = field(default_factory=list)


@dataclass
class WorkerSpec:
    name: str
    class_name: str
    trace_path: str
    worker_index: int
    worker_global_index: int
    base_offset_bytes: int
    capacity_bytes: int
    slot_bytes: int
    l1_size_gb: float
    meta_magic: str
    use_fdp: bool
    fdp_data_ruh_ids: list[int] = field(default_factory=list)
    fdp_metadata_ruh_ids: list[int] = field(default_factory=list)
    trace_footprint: TraceFootprint | None = None


@dataclass
class ReplayRunResult:
    worker_global_index: int
    worker_name: str
    worker_index: int
    iteration: int
    phase: str
    command: list[str]
    log_path: str
    output_dir: str
    jsonl_path: str
    exit_code: int
    records_failed: int | None
    started_at: str
    ended_at: str


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("_", "")
        if not cleaned:
            return None
        try:
            return int(cleaned, 0)
        except ValueError:
            match = re.search(r"[-+]?\d[\d,_]*", cleaned)
            if match:
                return int(match.group(0).replace(",", "").replace("_", ""))
    return None


def _prod(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _required_slot_bytes_for_payload(
    payload_bytes: int,
    *,
    block_align: int,
    header_bytes: int = DEFAULT_SLOT_HEADER_BYTES,
) -> int:
    return _align_up(payload_bytes, block_align) + header_bytes


def _slot_header_bytes(config: dict[str, Any]) -> int:
    return int(config.get("global", {}).get("header_bytes", DEFAULT_SLOT_HEADER_BYTES))


def _dtype_size(dtype: Any) -> int | None:
    size = getattr(dtype, "itemsize", None)
    if isinstance(size, int) and size > 0:
        return size
    name = str(dtype).replace("torch.", "")
    table = {
        "bool": 1,
        "uint8": 1,
        "int8": 1,
        "float8_e4m3fn": 1,
        "float8_e5m2": 1,
        "int16": 2,
        "uint16": 2,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "int64": 8,
        "uint64": 8,
        "float64": 8,
        "complex64": 8,
        "complex128": 16,
    }
    return table.get(name)


def _layout_size_bytes(layout_desc: Any) -> int | None:
    shapes = getattr(layout_desc, "shapes", None)
    dtypes = getattr(layout_desc, "dtypes", None)
    if not shapes or not dtypes or len(shapes) != len(dtypes):
        return None
    total = 0
    for shape, dtype in zip(shapes, dtypes, strict=True):
        item_size = _dtype_size(dtype)
        if item_size is None:
            return None
        total += _prod([int(dim) for dim in shape]) * item_size
    return total


def _object_key_id(key: Any) -> str:
    chunk_hash = getattr(key, "chunk_hash", b"")
    if isinstance(chunk_hash, bytes):
        chunk_hash_text = chunk_hash.hex()
    else:
        chunk_hash_text = str(chunk_hash)
    return "|".join(
        [
            chunk_hash_text,
            str(getattr(key, "model_name", "")),
            str(getattr(key, "kv_rank", "")),
            str(getattr(key, "cache_salt", "")),
        ]
    )


def _import_trace_runtime():
    try:
        # Work around the local OTel logger-provider issue seen in this checkout.
        # Third Party
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider

        set_logger_provider(LoggerProvider())
    except Exception:
        pass

    # First Party
    from lmcache.v1.mp_observability.trace import codecs
    from lmcache.v1.mp_observability.trace.reader import TraceReader

    return TraceReader, codecs


def analyze_trace_footprint(
    trace_path: str,
    *,
    fallback_object_bytes: int | None = None,
) -> TraceFootprint:
    """Estimate storage pressure from a storage-level ``.lct`` trace."""
    footprint = TraceFootprint(trace_path=trace_path)
    if not os.path.exists(trace_path):
        footprint.exact_byte_estimate = False
        footprint.warnings.append(f"trace file does not exist: {trace_path}")
        return footprint

    TraceReader, codecs = _import_trace_runtime()
    unique_store_keys: set[str] = set()

    with TraceReader(trace_path) as reader:
        for record in reader.records():
            footprint.record_count += 1
            footprint.duration_seconds = max(
                footprint.duration_seconds,
                float(getattr(record, "t_mono", 0.0)),
            )
            qualname = record.qualname
            try:
                args = codecs.decode_args(record.args)
            except Exception as exc:
                footprint.exact_byte_estimate = False
                footprint.warnings.append(
                    f"failed to decode record {footprint.record_count}: {exc}"
                )
                continue

            keys = args.get("keys") or []
            if not isinstance(keys, list):
                keys = list(keys)

            if qualname.endswith(".reserve_write"):
                object_bytes = _layout_size_bytes(args.get("layout_desc"))
                if object_bytes is None:
                    object_bytes = fallback_object_bytes
                    footprint.exact_byte_estimate = False
                if object_bytes is None:
                    continue
                footprint.store_count += len(keys)
                footprint.estimated_total_store_bytes += object_bytes * len(keys)
                footprint.estimated_max_object_bytes = max(
                    footprint.estimated_max_object_bytes,
                    object_bytes,
                )
                for key in keys:
                    unique_store_keys.add(_object_key_id(key))
            elif qualname.endswith(".submit_prefetch_task") or qualname.endswith(
                ".read_prefetched_results.__enter__"
            ):
                footprint.retrieve_prefetch_count += len(keys) or 1

    footprint.unique_object_key_count = len(unique_store_keys)
    if not footprint.exact_byte_estimate:
        footprint.warnings.append(
            "byte estimates are conservative because at least one layout "
            "descriptor could not be decoded"
        )
    return footprint


def load_yaml_config(path: str) -> dict[str, Any]:
    with open(path) as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    if "workloads" not in data or not isinstance(data["workloads"], list):
        raise ValueError("config must contain a workloads list")
    return data


def _as_command_prefix(value: Any) -> list[str]:
    if value is None:
        return ["lmcache"]
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError("global.replay_binary must be a string or list of strings")


def resolve_output_dir(
    config: dict[str, Any],
    *,
    mode: str,
    run_id: str,
    output_dir: str | None,
) -> str:
    if output_dir:
        return output_dir
    root = config.get("global", {}).get("output_root") or DEFAULT_OUTPUT_ROOT
    return os.path.join(str(root), f"{run_id}-{mode}")


def _mode_config(config: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {', '.join(VALID_MODES)}")
    modes = config.get("modes", {})
    mode_cfg = modes.get(mode)
    if not isinstance(mode_cfg, dict):
        raise ValueError(f"config.modes.{mode} must be defined")
    return mode_cfg


def expand_ruh_ids(spec: Any) -> list[int]:
    """Expand FDP RUH specs into explicit integer IDs.

    Accepted forms:
    - ``[0, 1, 2]``
    - ``{"start": 0, "count": 128}``
    - ``{"range": [96, 127]}`` where the end is inclusive
    - ``"0-3,8,10"``
    """
    if spec is None:
        return []
    if isinstance(spec, bool):
        raise ValueError("RUH IDs must be integers, not booleans")
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        ids: list[int] = []
        for raw_part in spec.split(","):
            part = raw_part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text.strip(), 0)
                end = int(end_text.strip(), 0)
                if end < start:
                    raise ValueError(f"invalid descending RUH range {part!r}")
                ids.extend(range(start, end + 1))
            else:
                ids.append(int(part, 0))
        return ids
    if isinstance(spec, list):
        ids = []
        for item in spec:
            ids.extend(expand_ruh_ids(item))
        return ids
    if isinstance(spec, dict):
        if "start" in spec and "count" in spec:
            start = int(spec["start"])
            count = int(spec["count"])
            if count < 0:
                raise ValueError("RUH count must be >= 0")
            return list(range(start, start + count))
        if "range" in spec:
            bounds = spec["range"]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError("RUH range must be a two-item list")
            start = int(bounds[0])
            end = int(bounds[1])
            if end < start:
                raise ValueError("RUH range end must be >= start")
            return list(range(start, end + 1))
    raise ValueError(f"unsupported RUH ID spec: {spec!r}")


def _validate_ruh_ids(ids: list[int]) -> list[int]:
    if len(set(ids)) != len(ids):
        raise ValueError("FDP RUH ID lists must not contain duplicates")
    for ruh_id in ids:
        if ruh_id < 0 or ruh_id > 0xFFFF:
            raise ValueError("FDP RUH IDs must fit in uint16")
    return ids


def _resolve_worker_ruhs(
    config: dict[str, Any],
    workload: dict[str, Any],
    mode: str,
) -> tuple[bool, list[int], list[int]]:
    mode_cfg = _mode_config(config, mode)
    use_fdp = bool(mode_cfg.get("use_fdp", False))
    if not use_fdp:
        return False, [], []

    workload_class = str(workload.get("class", ""))
    data_ruhs = workload.get("fdp_data_ruh_ids")
    metadata_ruhs = workload.get("fdp_metadata_ruh_ids")

    if data_ruhs is None:
        data_ruhs = mode_cfg.get("default_data_ruhs")
    if metadata_ruhs is None:
        metadata_ruhs = mode_cfg.get("default_metadata_ruhs")

    classes = mode_cfg.get("classes", {})
    if (data_ruhs is None or metadata_ruhs is None) and workload_class in classes:
        class_cfg = classes[workload_class]
        data_ruhs = data_ruhs if data_ruhs is not None else class_cfg.get("data_ruhs")
        metadata_ruhs = (
            metadata_ruhs
            if metadata_ruhs is not None
            else class_cfg.get("metadata_ruhs")
        )

    if data_ruhs is None or metadata_ruhs is None:
        raise ValueError(
            f"mode {mode!r} has no FDP RUH mapping for workload class "
            f"{workload_class!r}"
        )
    return (
        True,
        _validate_ruh_ids(expand_ruh_ids(data_ruhs)),
        _validate_ruh_ids(expand_ruh_ids(metadata_ruhs)),
    )


def make_meta_magic(run_id: str, worker_global_index: int) -> str:
    if worker_global_index < 0 or worker_global_index > 0xFFFF:
        raise ValueError("worker index cannot fit into an 8-byte meta_magic")
    run_hash = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:4].upper()
    return f"{run_hash}{worker_global_index:04X}"


def make_salt_suffix(
    run_id: str,
    mode: str,
    worker: WorkerSpec,
    iteration: int,
) -> str:
    return f"{run_id}.{mode}.{worker.name}.w{worker.worker_index}.iter_{iteration:04d}"


def _validate_alignment(name: str, value: int, block_align: int) -> None:
    if value % block_align != 0:
        raise ValueError(f"{name}={value} is not aligned to {block_align}")


def expand_workers(
    config: dict[str, Any],
    mode: str,
    *,
    run_id: str,
) -> list[WorkerSpec]:
    block_align = int(config.get("block_align", 4096))
    windows = config.get("windows", {})
    start_offset = int(windows.get("start_offset_bytes", 0))
    stride = int(windows.get("window_stride_bytes", 0))
    default_capacity = int(windows.get("default_capacity_bytes", 0))
    auto_assign = bool(windows.get("auto_assign", True))
    if auto_assign and stride <= 0:
        raise ValueError("windows.window_stride_bytes must be > 0")
    if default_capacity <= 0:
        raise ValueError("windows.default_capacity_bytes must be > 0")

    workers: list[WorkerSpec] = []
    for workload in config.get("workloads", []):
        if not isinstance(workload, dict):
            raise ValueError("each workload must be a mapping")
        name = str(workload["name"])
        concurrency = int(workload.get("concurrency", 1))
        if concurrency <= 0:
            raise ValueError(f"workload {name}: concurrency must be > 0")
        capacity = int(workload.get("capacity_bytes", default_capacity))
        slot_bytes = int(workload["slot_bytes"])
        _validate_alignment(f"{name}.capacity_bytes", capacity, block_align)
        _validate_alignment(f"{name}.slot_bytes", slot_bytes, block_align)
        use_fdp, data_ruhs, metadata_ruhs = _resolve_worker_ruhs(
            config,
            workload,
            mode,
        )

        for local_index in range(concurrency):
            global_index = len(workers)
            if auto_assign:
                base_offset = start_offset + global_index * stride
            else:
                base_offset = int(workload["base_offset_bytes"]) + local_index * stride
            _validate_alignment("base_offset_bytes", base_offset, block_align)
            workers.append(
                WorkerSpec(
                    name=name,
                    class_name=str(workload.get("class", "")),
                    trace_path=str(workload["trace_path"]),
                    worker_index=local_index,
                    worker_global_index=global_index,
                    base_offset_bytes=base_offset,
                    capacity_bytes=capacity,
                    slot_bytes=slot_bytes,
                    l1_size_gb=float(workload["l1_size_gb"]),
                    meta_magic=make_meta_magic(run_id, global_index + 1),
                    use_fdp=use_fdp,
                    fdp_data_ruh_ids=list(data_ruhs),
                    fdp_metadata_ruh_ids=list(metadata_ruhs),
                )
            )

    validate_windows(workers)
    if len({worker.meta_magic for worker in workers}) != len(workers):
        raise ValueError("meta_magic values are not unique")
    return workers


def validate_windows(workers: list[WorkerSpec]) -> None:
    ranges = sorted(
        (
            worker.base_offset_bytes,
            worker.base_offset_bytes + worker.capacity_bytes,
            worker,
        )
        for worker in workers
    )
    for (_, prev_end, prev_worker), (start, _, worker) in zip(
        ranges,
        ranges[1:],
        strict=False,
    ):
        if start < prev_end:
            raise ValueError(
                "raw-block windows overlap: "
                f"{prev_worker.name}[{prev_worker.worker_index}] and "
                f"{worker.name}[{worker.worker_index}]"
            )


def _global_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    return bool(config.get("global", {}).get(key, default))


def _global_int(config: dict[str, Any], key: str, default: int) -> int:
    return int(config.get("global", {}).get(key, default))


def build_l2_adapter(worker: WorkerSpec, config: dict[str, Any]) -> dict[str, Any]:
    adapter: dict[str, Any] = {
        "type": "raw_block",
        "device_path": config["device_path"],
        "slot_bytes": worker.slot_bytes,
        "base_offset_bytes": worker.base_offset_bytes,
        "capacity_bytes": worker.capacity_bytes,
        "meta_total_bytes": _global_int(
            config,
            "meta_total_bytes",
            DEFAULT_META_TOTAL_BYTES,
        ),
        "meta_magic": worker.meta_magic,
        "use_odirect": _global_bool(config, "use_odirect", False),
        "use_uring": _global_bool(config, "use_uring", True),
        "use_uring_cmd": _global_bool(config, "use_uring_cmd", True),
        "use_fdp": worker.use_fdp,
    }
    if "block_align" in config:
        adapter["block_align"] = int(config["block_align"])
    if worker.use_fdp:
        adapter["fdp_data_ruh_ids"] = list(worker.fdp_data_ruh_ids)
        adapter["fdp_metadata_ruh_ids"] = list(worker.fdp_metadata_ruh_ids)
    return adapter


def build_replay_command(
    worker: WorkerSpec,
    config: dict[str, Any],
    *,
    mode: str,
    run_id: str,
    iteration: int,
    worker_output_dir: str,
    jsonl_path: str,
) -> list[str]:
    global_cfg = config.get("global", {})
    cmd = _as_command_prefix(global_cfg.get("replay_binary"))
    cmd.extend(["trace", "replay", worker.trace_path])
    cmd.extend(
        [
            "--replay-cache-salt-suffix",
            make_salt_suffix(run_id, mode, worker, iteration),
            "--l1-size-gb",
            str(worker.l1_size_gb),
            "--eviction-policy",
            str(global_cfg.get("eviction_policy", "noop")),
            "--l2-store-policy",
            str(global_cfg.get("l2_store_policy", "skip_l1")),
            "--l1-align-bytes",
            str(global_cfg.get("l1_align_bytes", config.get("block_align", 4096))),
            "--output-dir",
            worker_output_dir,
            "--json",
            "--jsonl-out",
            jsonl_path,
        ]
    )
    if bool(global_cfg.get("disable_metrics", True)):
        cmd.append("--disable-metrics")
    if bool(global_cfg.get("quiet", True)):
        cmd.append("--quiet")
    adapter_json = json.dumps(build_l2_adapter(worker, config), separators=(",", ":"))
    cmd.extend(["--l2-adapter", adapter_json])
    return cmd


def _reserved_metadata_bytes(worker: WorkerSpec, config: dict[str, Any]) -> int:
    meta_total = _global_int(config, "meta_total_bytes", DEFAULT_META_TOTAL_BYTES)
    if worker.use_fdp:
        return meta_total * max(1, len(worker.fdp_metadata_ruh_ids))
    return meta_total


def attach_trace_footprints(
    workers: list[WorkerSpec],
    config: dict[str, Any],
    *,
    dry_run: bool,
) -> list[str]:
    warnings: list[str] = []
    by_trace_and_slot: dict[tuple[str, int], TraceFootprint] = {}
    for worker in workers:
        cache_key = (worker.trace_path, worker.slot_bytes)
        footprint = by_trace_and_slot.get(cache_key)
        if footprint is None:
            fallback = max(1, worker.slot_bytes - int(config.get("block_align", 4096)))
            footprint = analyze_trace_footprint(
                worker.trace_path,
                fallback_object_bytes=fallback,
            )
            by_trace_and_slot[cache_key] = footprint
        worker.trace_footprint = footprint
        if footprint.estimated_max_object_bytes > 0:
            required_slot_bytes = _required_slot_bytes_for_payload(
                footprint.estimated_max_object_bytes,
                block_align=int(config.get("block_align", 4096)),
                header_bytes=_slot_header_bytes(config),
            )
            if worker.slot_bytes < required_slot_bytes:
                warnings.append(
                    f"{worker.name}[{worker.worker_index}]: slot_bytes "
                    f"{worker.slot_bytes} is smaller than max payload "
                    f"{footprint.estimated_max_object_bytes} + raw-block header; "
                    f"using {required_slot_bytes}"
                )
                worker.slot_bytes = required_slot_bytes
        for warning in footprint.warnings:
            message = f"{worker.name}: {warning}"
            if "does not exist" in warning and not dry_run:
                raise FileNotFoundError(warning)
            warnings.append(message)

        unique_bytes = footprint.estimated_total_store_bytes
        if footprint.unique_object_key_count and footprint.store_count:
            avg_bytes = max(1, unique_bytes // max(1, footprint.store_count))
            unique_bytes = avg_bytes * footprint.unique_object_key_count
        if unique_bytes > 0 and worker.capacity_bytes > unique_bytes * 4:
            warnings.append(
                f"{worker.name}[{worker.worker_index}]: capacity_bytes "
                f"{worker.capacity_bytes} is more than 4x estimated unique "
                f"store bytes {unique_bytes}; churn may be weak"
            )
        min_capacity = worker.slot_bytes + _reserved_metadata_bytes(worker, config)
        if (
            footprint.estimated_max_object_bytes
            and worker.capacity_bytes < min_capacity
        ):
            warnings.append(
                f"{worker.name}[{worker.worker_index}]: capacity_bytes "
                f"{worker.capacity_bytes} is smaller than estimated max object "
                f"+ raw-block header + metadata reservation {min_capacity}; "
                "replay may fail"
            )
    return warnings


def write_json(path: str | Path, payload: Any) -> None:
    with open(path, "w") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def append_text(path: str | Path, text: str) -> None:
    with open(path, "a") as file_obj:
        file_obj.write(text)


def write_text(path: str | Path, text: str) -> None:
    with open(path, "w") as file_obj:
        file_obj.write(text)


def write_yaml(path: str | Path, payload: Any) -> None:
    with open(path, "w") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False)


def command_to_text(cmd: list[str]) -> str:
    return shlex.join(cmd)


def print_dry_run(
    workers: list[WorkerSpec],
    config: dict[str, Any],
    *,
    mode: str,
    run_id: str,
    output_dir: str,
    iterations: int,
    warmup_iterations: int,
    duration_seconds: int | None,
    target_write_multiplier: float | None,
    target_write_bytes: int | None,
    target_device_capacity_bytes: int | None,
) -> list[str]:
    lines = [
        f"mode={mode}",
        f"run_id={run_id}",
        f"output_dir={output_dir}",
        f"worker_count={len(workers)}",
        "",
    ]
    total_preview_iterations = warmup_iterations + (iterations or 1)
    if target_write_bytes is not None:
        lines.append(f"target_write_multiplier={target_write_multiplier}")
        lines.append(f"target_device_capacity_bytes={target_device_capacity_bytes}")
        lines.append(f"target_write_bytes={target_write_bytes}")
        total_preview_iterations = warmup_iterations + 1
    elif duration_seconds is not None:
        lines.append(f"duration_seconds={duration_seconds}")
        total_preview_iterations = warmup_iterations + 1
    commands: list[str] = []
    for worker in workers:
        lines.append(
            "window "
            f"worker={worker.worker_global_index} "
            f"name={worker.name} "
            f"w={worker.worker_index} "
            f"base={worker.base_offset_bytes} "
            f"capacity={worker.capacity_bytes} "
            f"meta_magic={worker.meta_magic} "
            f"use_fdp={str(worker.use_fdp).lower()} "
            f"data_ruhs={worker.fdp_data_ruh_ids} "
            f"metadata_ruhs={worker.fdp_metadata_ruh_ids}"
        )
    lines.append("")
    for iteration in range(total_preview_iterations):
        phase = "warmup" if iteration < warmup_iterations else "measurement"
        for worker in workers:
            worker_dir = os.path.join(
                output_dir,
                "worker_logs",
                f"{worker.worker_global_index:03d}",
                f"{phase}_{iteration:04d}",
            )
            jsonl = os.path.join(worker_dir, "records.jsonl")
            cmd = build_replay_command(
                worker,
                config,
                mode=mode,
                run_id=run_id,
                iteration=iteration,
                worker_output_dir=worker_dir,
                jsonl_path=jsonl,
            )
            command_text = command_to_text(cmd)
            commands.append(command_text)
            lines.append(command_text)
    print("\n".join(lines))
    return commands


def _parse_json_maybe(stdout: str) -> Any:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def run_capture(
    cmd: list[str] | str,
    *,
    shell: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    started = _utc_now()
    try:
        proc = subprocess.run(
            cmd,
            shell=shell,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        return {
            "command": cmd if isinstance(cmd, str) else command_to_text(cmd),
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "json": _parse_json_maybe(stdout),
            "started_at": started,
            "ended_at": _utc_now(),
        }
    except Exception as exc:
        return {
            "command": cmd if isinstance(cmd, str) else command_to_text(cmd),
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "json": None,
            "started_at": started,
            "ended_at": _utc_now(),
        }


def detect_block_device_capacity_bytes(block_device_path: str | None) -> int | None:
    if not block_device_path:
        return None
    device_name = os.path.basename(os.path.realpath(str(block_device_path)))
    sysfs_size = os.path.join("/sys/class/block", device_name, "size")
    try:
        with open(sysfs_size) as file_obj:
            sectors = _safe_int(file_obj.read())
        if sectors is not None and sectors > 0:
            return sectors * 512
    except OSError:
        pass

    if shutil.which("blockdev"):
        result = run_capture(["blockdev", "--getsize64", str(block_device_path)])
        parsed = _safe_int(result.get("stdout"))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def extract_host_write_bytes(smart_payload: Any) -> int | None:
    if not isinstance(smart_payload, dict):
        return None
    value = smart_payload.get("data_units_written")
    parsed = _safe_int(value)
    if parsed is None:
        return None
    return parsed * 512_000


def capture_host_write_bytes(config: dict[str, Any]) -> int | None:
    block_device = config.get("block_device_path")
    if not shutil.which("nvme") or not block_device:
        return None
    smart = run_capture(["nvme", "smart-log", str(block_device), "--json"])
    return extract_host_write_bytes(smart.get("json"))


def extract_fdp_stats_bytes(payload: Any) -> dict[str, int]:
    stats: dict[str, int] = {}
    if payload is None:
        return stats
    if isinstance(payload, dict):
        field_names = {
            "host_write_bytes": (
                "host_write_bytes",
                "host_bytes_with_metadata_written",
                "hbmw",
            ),
            "media_write_bytes": (
                "media_write_bytes",
                "media_bytes_with_metadata_written",
                "mbmw",
            ),
            "media_bytes_erased": (
                "media_bytes_erased",
                "mbe",
            ),
        }
        for output_key, input_keys in field_names.items():
            for input_key in input_keys:
                parsed = _safe_int(payload.get(input_key))
                if parsed is not None:
                    stats[output_key] = parsed
                    break
        for value in payload.values():
            nested = extract_fdp_stats_bytes(value)
            stats = {**nested, **stats}
        return stats
    if isinstance(payload, list):
        for item in payload:
            stats.update(extract_fdp_stats_bytes(item))
        return stats
    if not isinstance(payload, str):
        return stats

    patterns = {
        "host_write_bytes": (
            r"Host Bytes with Metadata Written \(HBMW\):\s*(\d[\d,_]*)",
            r"^\s*hbmw:\s*\[\s*(\d[\d,_]*)",
        ),
        "media_write_bytes": (
            r"Media Bytes with Metadata Written \(MBMW\):\s*(\d[\d,_]*)",
            r"^\s*mbmw:\s*\[\s*(\d[\d,_]*)",
        ),
        "media_bytes_erased": (
            r"Media Bytes Erased \(MBE\):\s*(\d[\d,_]*)",
            r"^\s*mbe:\s*\[\s*(\d[\d,_]*)",
        ),
    }
    for output_key, regexes in patterns.items():
        for regex in regexes:
            match = re.search(regex, payload, re.MULTILINE)
            if match:
                parsed = _safe_int(match.group(1))
                if parsed is not None:
                    stats[output_key] = parsed
                    break
    return stats


def _capture_xnvme_fdp_stats(block_device: str) -> dict[str, Any] | None:
    if not shutil.which("xnvme"):
        return None
    result = run_capture(["xnvme", "log-fdp-stats", str(block_device), "--lsi", "0x1"])
    if result.get("returncode") == 0 and extract_fdp_stats_bytes(result.get("stdout")):
        result["source"] = "xnvme log-fdp-stats"
        return result
    return None


def _capture_nvme_cli_fdp_stats(block_device: str) -> dict[str, Any] | None:
    if not shutil.which("nvme"):
        return None
    result = run_capture(
        ["sudo", "-n", "nvme", "fdp", "stats", str(block_device), "-e", "1"]
    )
    result["source"] = "nvme fdp stats"
    return result


def capture_fdp_stats(config: dict[str, Any]) -> dict[str, Any] | None:
    block_device = config.get("block_device_path")
    if not block_device:
        return None
    measurement_cfg = config.get("measurement", {})
    command = measurement_cfg.get("fdp_stats_command")
    if command:
        return run_capture(str(command), shell=True)

    backend = str(measurement_cfg.get("fdp_stats_backend", "auto")).lower()
    if backend not in {"auto", "xnvme", "nvme"}:
        raise ValueError(
            "measurement.fdp_stats_backend must be one of: auto, xnvme, nvme"
        )

    if backend == "xnvme":
        return _capture_xnvme_fdp_stats(str(block_device))
    if backend == "nvme":
        return _capture_nvme_cli_fdp_stats(str(block_device))

    return _capture_xnvme_fdp_stats(str(block_device)) or _capture_nvme_cli_fdp_stats(
        str(block_device)
    )


def extract_media_write_bytes(payload: Any) -> int | None:
    if payload is None:
        return None
    if isinstance(payload, (str, int, float)):
        return _safe_int(payload)
    if isinstance(payload, dict):
        keys = (
            "media_write_bytes",
            "media_bytes_written",
            "nand_write_bytes",
            "nand_bytes_written",
            "physical_media_write_bytes",
            "physical_media_bytes_written",
        )
        for key in keys:
            parsed = _safe_int(payload.get(key))
            if parsed is not None:
                return parsed
        for value in payload.values():
            parsed = extract_media_write_bytes(value)
            if parsed is not None:
                return parsed
    if isinstance(payload, list):
        for item in payload:
            parsed = extract_media_write_bytes(item)
            if parsed is not None:
                return parsed
    return None


def capture_measurement(config: dict[str, Any], label: str) -> dict[str, Any]:
    measurement_cfg = config.get("measurement", {})
    if not bool(measurement_cfg.get("enabled", True)):
        return {"label": label, "enabled": False, "captured_at": _utc_now()}

    block_device = config.get("block_device_path")
    result: dict[str, Any] = {
        "label": label,
        "enabled": True,
        "captured_at": _utc_now(),
        "block_device_path": block_device,
        "warnings": [],
    }

    if bool(measurement_cfg.get("collect_nvme_smart", True)):
        if shutil.which("nvme") and block_device:
            smart = run_capture(["nvme", "smart-log", str(block_device), "--json"])
            result["nvme_smart_log"] = smart
            result["host_write_bytes"] = extract_host_write_bytes(smart.get("json"))
        else:
            result["warnings"].append("nvme CLI or block_device_path unavailable")

    fdp_stats = capture_fdp_stats(config)
    if fdp_stats is not None:
        result.setdefault("fdp_logs", {})["stats"] = fdp_stats
        parsed_fdp_stats = extract_fdp_stats_bytes(fdp_stats.get("json"))
        if not parsed_fdp_stats:
            parsed_fdp_stats = extract_fdp_stats_bytes(fdp_stats.get("stdout"))
        if "host_write_bytes" in parsed_fdp_stats:
            result["host_write_bytes"] = parsed_fdp_stats["host_write_bytes"]
        if "media_write_bytes" in parsed_fdp_stats:
            result["media_write_bytes"] = parsed_fdp_stats["media_write_bytes"]
        if "media_bytes_erased" in parsed_fdp_stats:
            result["media_bytes_erased"] = parsed_fdp_stats["media_bytes_erased"]
    else:
        result["warnings"].append("FDP stats unavailable; skipped FDP stats")

    vendor_command = measurement_cfg.get("vendor_media_write_command")
    if vendor_command:
        vendor = run_capture(str(vendor_command), shell=True)
        result["vendor_media_write"] = vendor
        parsed_json = vendor.get("json")
        parsed_fdp_stats = extract_fdp_stats_bytes(parsed_json)
        if "host_write_bytes" in parsed_fdp_stats:
            result["host_write_bytes"] = parsed_fdp_stats["host_write_bytes"]
        media_bytes = parsed_fdp_stats.get("media_write_bytes")
        if media_bytes is None:
            media_bytes = extract_media_write_bytes(parsed_json)
        if media_bytes is None:
            media_bytes = extract_media_write_bytes(vendor.get("stdout"))
        if media_bytes is not None:
            result["media_write_bytes"] = media_bytes
    elif "media_write_bytes" not in result:
        result["media_write_bytes"] = None
        result["warnings"].append("vendor media-write command not configured")

    if bool(measurement_cfg.get("collect_fdp_logs", True)):
        if shutil.which("xnvme") and block_device:
            fdp_logs = result.setdefault("fdp_logs", {})
            fdp_logs.setdefault("stats", fdp_stats)
            fdp_logs["ruhu"] = run_capture(
                [
                    "xnvme",
                    "log-ruhu",
                    str(block_device),
                    "--lsi",
                    "0x1",
                    "--limit",
                    "16",
                ]
            )
            fdp_logs["ruhs"] = run_capture(
                ["xnvme", "fdp-ruhs", str(block_device), "--limit", "16"]
            )
        elif not shutil.which("nvme") or not block_device:
            result["warnings"].append("FDP log tools unavailable; skipped FDP logs")

    return result


def _waf_from_deltas(
    host_delta: int | None,
    media_delta: int | None,
) -> float | None:
    if host_delta is None or host_delta <= 0 or media_delta is None:
        return None
    return media_delta / host_delta


def _format_sample_timestamp(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) >= 19 and "T" in text:
        return text[:19].replace("T", " ")
    return text


def _format_sample_value(column: str, value: Any) -> str:
    if value is None:
        return ""
    if column == "timestamp":
        return _format_sample_timestamp(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_sample_row(values: dict[str, Any]) -> str:
    cells = []
    for column in WAF_SAMPLE_TSV_COLUMNS:
        value = _format_sample_value(column, values.get(column))
        cells.append(value.ljust(WAF_SAMPLE_COLUMN_WIDTHS[column]))
    return "  ".join(cells).rstrip()


def build_waf_sample(
    *,
    sample: dict[str, Any],
    baseline: dict[str, Any],
    previous: dict[str, Any],
    target_device_capacity_bytes: int | None = None,
) -> dict[str, Any]:
    interval_host_delta = _delta(sample, previous, "host_write_bytes")
    interval_media_delta = _delta(sample, previous, "media_write_bytes")
    cumulative_host_delta = _delta(sample, baseline, "host_write_bytes")
    device_write_multiplier = None
    if (
        cumulative_host_delta is not None
        and target_device_capacity_bytes is not None
        and target_device_capacity_bytes > 0
    ):
        device_write_multiplier = cumulative_host_delta / target_device_capacity_bytes

    sample_status = "updated"
    output_host_delta = interval_host_delta
    output_media_delta = interval_media_delta
    if interval_host_delta is None or interval_media_delta is None:
        sample_status = "unavailable"
    elif interval_host_delta == 0 and interval_media_delta == 0:
        sample_status = "stale"
        output_host_delta = None
        output_media_delta = None

    return {
        "timestamp": sample.get("captured_at"),
        "fdp_host_write_bytes": output_host_delta,
        "fdp_media_write_bytes": output_media_delta,
        "fdp_waf": _waf_from_deltas(output_host_delta, output_media_delta),
        "device_write_multiplier": device_write_multiplier,
        "sample_status": sample_status,
    }


def waf_sample_to_tsv(sample: dict[str, Any]) -> str:
    return _format_sample_row(sample)


def initialize_waf_samples(path: str | Path) -> None:
    header = _format_sample_row({column: column for column in WAF_SAMPLE_TSV_COLUMNS})
    separator = _format_sample_row(
        {
            column: "-" * WAF_SAMPLE_COLUMN_WIDTHS[column]
            for column in WAF_SAMPLE_TSV_COLUMNS
        }
    )
    write_text(path, header + "\n" + separator + "\n")


def start_waf_sampler(
    *,
    config: dict[str, Any],
    output_dir: str,
    baseline: dict[str, Any],
    interval_seconds: int,
    target_device_capacity_bytes: int | None = None,
) -> tuple[threading.Event | None, threading.Thread | None]:
    if interval_seconds <= 0:
        return None, None

    samples_path = os.path.join(output_dir, "waf_samples.tsv")
    initialize_waf_samples(samples_path)
    stop_event = threading.Event()

    def _sample_loop() -> None:
        previous = baseline
        sample_index = 0
        while not stop_event.wait(interval_seconds):
            sample_index += 1
            sample = capture_measurement(config, f"sample_{sample_index:04d}")
            append_text(
                samples_path,
                waf_sample_to_tsv(
                    build_waf_sample(
                        sample=sample,
                        baseline=baseline,
                        previous=previous,
                        target_device_capacity_bytes=target_device_capacity_bytes,
                    )
                )
                + "\n",
            )
            previous = sample

    thread = threading.Thread(
        target=_sample_loop,
        name="fdp-waf-sampler",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def _count_failed_jsonl(path: str) -> int | None:
    if not os.path.exists(path):
        return None
    failed = 0
    with open(path) as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if bool(payload.get("failed", False)):
                failed += 1
    return failed


def run_replay_round(
    workers: list[WorkerSpec],
    config: dict[str, Any],
    *,
    mode: str,
    run_id: str,
    output_dir: str,
    iteration: int,
    phase: str,
) -> list[ReplayRunResult]:
    process_items: list[
        tuple[subprocess.Popen, WorkerSpec, str, str, str, list[str], str]
    ] = []
    for worker in workers:
        worker_dir = os.path.join(
            output_dir,
            "worker_logs",
            f"{worker.worker_global_index:03d}",
            f"{phase}_{iteration:04d}",
        )
        os.makedirs(worker_dir, exist_ok=True)
        jsonl_path = os.path.join(worker_dir, "records.jsonl")
        log_path = os.path.join(worker_dir, "replay.log")
        cmd = build_replay_command(
            worker,
            config,
            mode=mode,
            run_id=run_id,
            iteration=iteration,
            worker_output_dir=worker_dir,
            jsonl_path=jsonl_path,
        )
        log_fh = open(log_path, "w")
        started_at = _utc_now()
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_fh.close()
        process_items.append(
            (proc, worker, worker_dir, jsonl_path, log_path, cmd, started_at)
        )

    results: list[ReplayRunResult] = []
    for item in process_items:
        proc, worker, worker_dir, jsonl_path, log_path, cmd, started_at = item
        exit_code = proc.wait()
        results.append(
            ReplayRunResult(
                worker_global_index=worker.worker_global_index,
                worker_name=worker.name,
                worker_index=worker.worker_index,
                iteration=iteration,
                phase=phase,
                command=cmd,
                log_path=log_path,
                output_dir=worker_dir,
                jsonl_path=jsonl_path,
                exit_code=exit_code,
                records_failed=_count_failed_jsonl(jsonl_path),
                started_at=started_at,
                ended_at=_utc_now(),
            )
        )
    return results


def run_single_worker_replay(
    worker: WorkerSpec,
    config: dict[str, Any],
    *,
    mode: str,
    run_id: str,
    output_dir: str,
    iteration: int,
    phase: str,
) -> ReplayRunResult:
    worker_dir = os.path.join(
        output_dir,
        "worker_logs",
        f"{worker.worker_global_index:03d}",
        f"{phase}_{iteration:04d}",
    )
    os.makedirs(worker_dir, exist_ok=True)
    jsonl_path = os.path.join(worker_dir, "records.jsonl")
    log_path = os.path.join(worker_dir, "replay.log")
    cmd = build_replay_command(
        worker,
        config,
        mode=mode,
        run_id=run_id,
        iteration=iteration,
        worker_output_dir=worker_dir,
        jsonl_path=jsonl_path,
    )
    started_at = _utc_now()
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        exit_code = proc.wait()
    return ReplayRunResult(
        worker_global_index=worker.worker_global_index,
        worker_name=worker.name,
        worker_index=worker.worker_index,
        iteration=iteration,
        phase=phase,
        command=cmd,
        log_path=log_path,
        output_dir=worker_dir,
        jsonl_path=jsonl_path,
        exit_code=exit_code,
        records_failed=_count_failed_jsonl(jsonl_path),
        started_at=started_at,
        ended_at=_utc_now(),
    )


def run_workers_until_deadline(
    workers: list[WorkerSpec],
    config: dict[str, Any],
    *,
    mode: str,
    run_id: str,
    output_dir: str,
    start_iteration: int,
    duration_seconds: int,
) -> tuple[list[ReplayRunResult], int]:
    deadline = time.monotonic() + duration_seconds

    def _worker_loop(worker: WorkerSpec) -> list[ReplayRunResult]:
        worker_results: list[ReplayRunResult] = []
        iteration = start_iteration
        while time.monotonic() < deadline:
            worker_results.append(
                run_single_worker_replay(
                    worker,
                    config,
                    mode=mode,
                    run_id=run_id,
                    output_dir=output_dir,
                    iteration=iteration,
                    phase="measurement",
                )
            )
            iteration += 1
        return worker_results

    all_results: list[ReplayRunResult] = []
    max_worker_iterations = 0
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(_worker_loop, worker) for worker in workers]
        for future in as_completed(futures):
            worker_results = future.result()
            all_results.extend(worker_results)
            max_worker_iterations = max(max_worker_iterations, len(worker_results))
    return all_results, max_worker_iterations


def run_workers_until_host_write_target(
    workers: list[WorkerSpec],
    config: dict[str, Any],
    *,
    mode: str,
    run_id: str,
    output_dir: str,
    start_iteration: int,
    baseline_host_write_bytes: int,
    target_write_bytes: int,
    poll_interval_seconds: int,
) -> tuple[list[ReplayRunResult], int]:
    stop_event = threading.Event()
    poll_interval = max(1, int(poll_interval_seconds))

    def _monitor_loop() -> None:
        while not stop_event.wait(poll_interval):
            current = capture_host_write_bytes(config)
            if current is None:
                continue
            if current - baseline_host_write_bytes >= target_write_bytes:
                stop_event.set()
                return

    def _worker_loop(worker: WorkerSpec) -> list[ReplayRunResult]:
        worker_results: list[ReplayRunResult] = []
        iteration = start_iteration
        while not stop_event.is_set():
            worker_results.append(
                run_single_worker_replay(
                    worker,
                    config,
                    mode=mode,
                    run_id=run_id,
                    output_dir=output_dir,
                    iteration=iteration,
                    phase="measurement",
                )
            )
            iteration += 1
        return worker_results

    monitor = threading.Thread(
        target=_monitor_loop,
        name="fdp-waf-target-monitor",
        daemon=True,
    )
    monitor.start()

    all_results: list[ReplayRunResult] = []
    max_worker_iterations = 0
    try:
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = [executor.submit(_worker_loop, worker) for worker in workers]
            for future in as_completed(futures):
                worker_results = future.result()
                all_results.extend(worker_results)
                max_worker_iterations = max(
                    max_worker_iterations,
                    len(worker_results),
                )
    finally:
        stop_event.set()
        monitor.join(timeout=1)
    return all_results, max_worker_iterations


def run_iterations(
    workers: list[WorkerSpec],
    config: dict[str, Any],
    *,
    mode: str,
    run_id: str,
    output_dir: str,
    warmup_iterations: int,
    measurement_iterations: int | None,
    duration_seconds: int | None,
    target_write_bytes: int | None = None,
    target_write_baseline_bytes: int | None = None,
    target_write_poll_seconds: int = DEFAULT_TARGET_HOST_WRITE_POLL_SECONDS,
    start_iteration: int = 0,
) -> tuple[list[ReplayRunResult], int]:
    all_results: list[ReplayRunResult] = []
    for iteration in range(start_iteration, start_iteration + warmup_iterations):
        all_results.extend(
            run_replay_round(
                workers,
                config,
                mode=mode,
                run_id=run_id,
                output_dir=output_dir,
                iteration=iteration,
                phase="warmup",
            )
        )

    measurement_start_iter = start_iteration + warmup_iterations
    completed_measurement_iterations = 0
    if target_write_bytes is not None:
        if target_write_baseline_bytes is None:
            raise ValueError("target host-write mode requires a baseline counter")
        target_results, completed_measurement_iterations = (
            run_workers_until_host_write_target(
                workers,
                config,
                mode=mode,
                run_id=run_id,
                output_dir=output_dir,
                start_iteration=measurement_start_iter,
                baseline_host_write_bytes=target_write_baseline_bytes,
                target_write_bytes=target_write_bytes,
                poll_interval_seconds=target_write_poll_seconds,
            )
        )
        all_results.extend(target_results)
    elif duration_seconds is not None:
        duration_results, completed_measurement_iterations = run_workers_until_deadline(
            workers,
            config,
            mode=mode,
            run_id=run_id,
            output_dir=output_dir,
            start_iteration=measurement_start_iter,
            duration_seconds=duration_seconds,
        )
        all_results.extend(duration_results)
    else:
        assert measurement_iterations is not None
        for offset in range(measurement_iterations):
            iteration = measurement_start_iter + offset
            all_results.extend(
                run_replay_round(
                    workers,
                    config,
                    mode=mode,
                    run_id=run_id,
                    output_dir=output_dir,
                    iteration=iteration,
                    phase="measurement",
                )
            )
            completed_measurement_iterations += 1
    return all_results, completed_measurement_iterations


def _delta(after: dict[str, Any], before: dict[str, Any], key: str) -> int | None:
    after_value = _safe_int(after.get(key))
    before_value = _safe_int(before.get(key))
    if after_value is None or before_value is None:
        return None
    return after_value - before_value


def build_summary(
    *,
    config: dict[str, Any],
    workers: list[WorkerSpec],
    mode: str,
    run_id: str,
    warmup_iterations: int,
    measurement_iterations: int,
    measurement_after_warmup: dict[str, Any],
    measurement_after: dict[str, Any],
    run_results: list[ReplayRunResult],
    warnings: list[str],
    target_write_multiplier: float | None = None,
    target_device_capacity_bytes: int | None = None,
    target_write_bytes: int | None = None,
) -> dict[str, Any]:
    host_delta = _delta(
        measurement_after,
        measurement_after_warmup,
        "host_write_bytes",
    )
    media_delta = _delta(
        measurement_after,
        measurement_after_warmup,
        "media_write_bytes",
    )
    waf = None
    waf_status = "unavailable"
    if host_delta is not None and host_delta > 0 and media_delta is not None:
        waf = media_delta / host_delta
        waf_status = "available"
    elif host_delta is None:
        waf_status = "unavailable: host write counter missing"
    elif host_delta <= 0:
        waf_status = "unavailable: host write delta is zero"
    elif media_delta is None:
        waf_status = "unavailable: media/NAND write counter missing"

    fdp_stats_host_write_bytes = _safe_int(measurement_after.get("host_write_bytes"))
    fdp_stats_media_write_bytes = _safe_int(measurement_after.get("media_write_bytes"))
    fdp_stats_cumulative_waf = None
    if (
        fdp_stats_host_write_bytes is not None
        and fdp_stats_host_write_bytes > 0
        and fdp_stats_media_write_bytes is not None
    ):
        fdp_stats_cumulative_waf = (
            fdp_stats_media_write_bytes / fdp_stats_host_write_bytes
        )
    if waf is None and fdp_stats_cumulative_waf is not None:
        waf = fdp_stats_cumulative_waf
        waf_status = "available: cumulative FDP stats"

    measurement_results = [
        result for result in run_results if result.phase == "measurement"
    ]
    result_by_worker: dict[int, list[ReplayRunResult]] = {}
    for result in measurement_results:
        result_by_worker.setdefault(result.worker_global_index, []).append(result)

    worker_summaries = []
    for worker in workers:
        worker_results = result_by_worker.get(worker.worker_global_index, [])
        exit_codes = [result.exit_code for result in worker_results]
        failed_counts = [
            result.records_failed
            for result in worker_results
            if result.records_failed is not None
        ]
        worker_summaries.append(
            {
                "name": worker.name,
                "worker_index": worker.worker_index,
                "worker_global_index": worker.worker_global_index,
                "trace_path": worker.trace_path,
                "base_offset_bytes": worker.base_offset_bytes,
                "capacity_bytes": worker.capacity_bytes,
                "slot_bytes": worker.slot_bytes,
                "meta_magic": worker.meta_magic,
                "fdp_data_ruh_ids": worker.fdp_data_ruh_ids,
                "fdp_metadata_ruh_ids": worker.fdp_metadata_ruh_ids,
                "records_failed": sum(failed_counts) if failed_counts else None,
                "exit_code": max(exit_codes) if exit_codes else None,
            }
        )

    return {
        "mode": mode,
        "run_id": run_id,
        "device_path": config.get("device_path"),
        "block_device_path": config.get("block_device_path"),
        "worker_count": len(workers),
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "host_write_bytes_delta": host_delta,
        "media_write_bytes_delta": media_delta,
        "waf": waf,
        "waf_status": waf_status,
        "fdp_stats_host_write_bytes": fdp_stats_host_write_bytes,
        "fdp_stats_media_write_bytes": fdp_stats_media_write_bytes,
        "fdp_stats_cumulative_waf": fdp_stats_cumulative_waf,
        "target_write_multiplier": target_write_multiplier,
        "target_device_capacity_bytes": target_device_capacity_bytes,
        "target_write_bytes": target_write_bytes,
        "workers": worker_summaries,
        "warnings": warnings,
    }


def build_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        f"# LMCache FDP WAF Stress Summary ({summary['mode']})",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- device_path: `{summary['device_path']}`",
        f"- block_device_path: `{summary['block_device_path']}`",
        f"- worker_count: {summary['worker_count']}",
        f"- warmup_iterations: {summary['warmup_iterations']}",
        f"- measurement_iterations: {summary['measurement_iterations']}",
        f"- host_write_bytes_delta: {summary['host_write_bytes_delta']}",
        f"- media_write_bytes_delta: {summary['media_write_bytes_delta']}",
        f"- target_write_bytes: {summary.get('target_write_bytes')}",
        f"- target_device_capacity_bytes: "
        f"{summary.get('target_device_capacity_bytes')}",
        f"- target_write_multiplier: {summary.get('target_write_multiplier')}",
        f"- waf: {summary['waf']}",
        f"- waf_status: {summary['waf_status']}",
        f"- fdp_stats_host_write_bytes: {summary.get('fdp_stats_host_write_bytes')}",
        f"- fdp_stats_media_write_bytes: {summary.get('fdp_stats_media_write_bytes')}",
        f"- fdp_stats_cumulative_waf: {summary.get('fdp_stats_cumulative_waf')}",
        "",
        "## Workers",
        "",
        "| worker | trace | base_offset_bytes | capacity_bytes | "
        "FDP data | FDP metadata | exit | failed |",
        "|---|---|---:|---:|---|---|---:|---:|",
    ]
    for worker in summary["workers"]:
        lines.append(
            "| "
            f"{worker['name']}[{worker['worker_index']}] | "
            f"{worker['trace_path']} | "
            f"{worker['base_offset_bytes']} | "
            f"{worker['capacity_bytes']} | "
            f"{worker['fdp_data_ruh_ids']} | "
            f"{worker['fdp_metadata_ruh_ids']} | "
            f"{worker['exit_code']} | "
            f"{worker['records_failed']} |"
        )
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    lines.append("")
    return "\n".join(lines)


def _workers_payload(workers: list[WorkerSpec]) -> list[dict[str, Any]]:
    payload = []
    for worker in workers:
        item = asdict(worker)
        if worker.trace_footprint is not None:
            item["trace_footprint"] = asdict(worker.trace_footprint)
        payload.append(item)
    return payload


def _write_initial_outputs(
    *,
    output_dir: str,
    config: dict[str, Any],
    workers: list[WorkerSpec],
    commands: list[str],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_yaml(os.path.join(output_dir, "run_config.resolved.yaml"), config)
    write_json(os.path.join(output_dir, "workers.json"), _workers_payload(workers))
    write_text(os.path.join(output_dir, "commands.txt"), "\n".join(commands) + "\n")


def _all_planned_commands(
    workers: list[WorkerSpec],
    config: dict[str, Any],
    *,
    mode: str,
    run_id: str,
    output_dir: str,
    warmup_iterations: int,
    measurement_iterations: int | None,
    duration_seconds: int | None,
    target_write_bytes: int | None,
) -> list[str]:
    count = warmup_iterations + (measurement_iterations or 1)
    if target_write_bytes is not None or duration_seconds is not None:
        count = warmup_iterations + 1
    commands = []
    for iteration in range(count):
        phase = "warmup" if iteration < warmup_iterations else "measurement"
        for worker in workers:
            worker_dir = os.path.join(
                output_dir,
                "worker_logs",
                f"{worker.worker_global_index:03d}",
                f"{phase}_{iteration:04d}",
            )
            cmd = build_replay_command(
                worker,
                config,
                mode=mode,
                run_id=run_id,
                iteration=iteration,
                worker_output_dir=worker_dir,
                jsonl_path=os.path.join(worker_dir, "records.jsonl"),
            )
            commands.append(command_to_text(cmd))
    return commands


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--mode", required=True, choices=VALID_MODES)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--duration-seconds", type=int, default=None)
    parser.add_argument(
        "--sample-interval-seconds",
        type=int,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
        help=(
            "Record timestamp, host-write delta, media-write delta, and WAF "
            "to waf_samples.tsv every N seconds during measurement; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--target-write-multiplier",
        default=None,
        type=float,
        help=(
            "Run measurement until host-write delta reaches device capacity "
            "multiplied by this value. When enabled, duration/iteration "
            "measurement limits are ignored."
        ),
    )
    parser.add_argument(
        "--target-write-poll-seconds",
        type=int,
        default=DEFAULT_TARGET_HOST_WRITE_POLL_SECONDS,
        help="Host-write target polling interval in seconds.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations < 0:
        raise ValueError("--iterations must be >= 0")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be >= 0")
    if args.duration_seconds is not None:
        if args.duration_seconds <= 0:
            raise ValueError("--duration-seconds must be > 0")
        if (
            args.target_write_multiplier is None
            and args.iterations != parser.get_default("iterations")
        ):
            raise ValueError(
                "--iterations and --duration-seconds are mutually exclusive"
            )
    if args.sample_interval_seconds < 0:
        raise ValueError("--sample-interval-seconds must be >= 0")
    if args.target_write_multiplier is not None and args.target_write_multiplier <= 0:
        raise ValueError("--target-write-multiplier must be > 0")
    if args.target_write_poll_seconds <= 0:
        raise ValueError("--target-write-poll-seconds must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    config = copy.deepcopy(config)
    config.setdefault("global", {})
    config["global"].setdefault("output_root", DEFAULT_OUTPUT_ROOT)
    output_dir = resolve_output_dir(
        config,
        mode=args.mode,
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    workers = expand_workers(config, args.mode, run_id=args.run_id)
    warnings = attach_trace_footprints(workers, config, dry_run=args.dry_run)
    target_multiplier = args.target_write_multiplier
    target_device_capacity_bytes = None
    target_write_bytes = None
    if target_multiplier is not None:
        target_device_capacity_bytes = detect_block_device_capacity_bytes(
            config.get("block_device_path")
        )
        if target_device_capacity_bytes is None:
            raise ValueError(
                "could not detect block device capacity for "
                f"{config.get('block_device_path')!r}"
            )
        target_write_bytes = int(target_device_capacity_bytes * target_multiplier)
        if target_write_bytes <= 0:
            raise ValueError("target host-write byte count must be > 0")

    if args.dry_run:
        commands = print_dry_run(
            workers,
            config,
            mode=args.mode,
            run_id=args.run_id,
            output_dir=output_dir,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            duration_seconds=args.duration_seconds,
            target_write_multiplier=target_multiplier,
            target_write_bytes=target_write_bytes,
            target_device_capacity_bytes=target_device_capacity_bytes,
        )
        _write_initial_outputs(
            output_dir=output_dir,
            config=config,
            workers=workers,
            commands=commands,
        )
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        return 0

    os.makedirs(os.path.join(output_dir, "worker_logs"), exist_ok=True)
    planned_commands = _all_planned_commands(
        workers,
        config,
        mode=args.mode,
        run_id=args.run_id,
        output_dir=output_dir,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.iterations,
        duration_seconds=args.duration_seconds,
        target_write_bytes=target_write_bytes,
    )
    _write_initial_outputs(
        output_dir=output_dir,
        config=config,
        workers=workers,
        commands=planned_commands,
    )
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    measurement_before = capture_measurement(config, "before_warmup")
    write_json(os.path.join(output_dir, "measurement_before.json"), measurement_before)

    warmup_results, _ = run_iterations(
        workers,
        config,
        mode=args.mode,
        run_id=args.run_id,
        output_dir=output_dir,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=0,
        duration_seconds=None,
    )

    measurement_after_warmup = capture_measurement(config, "after_warmup")
    write_json(
        os.path.join(output_dir, "measurement_after_warmup.json"),
        measurement_after_warmup,
    )
    target_write_baseline = None
    if target_write_bytes is not None:
        target_write_baseline = _safe_int(
            measurement_after_warmup.get("host_write_bytes")
        )
        if target_write_baseline is None:
            raise ValueError(
                "target host-write mode requires nvme smart-log host counter"
            )

    sampler_stop, sampler_thread = start_waf_sampler(
        config=config,
        output_dir=output_dir,
        baseline=measurement_after_warmup,
        interval_seconds=args.sample_interval_seconds,
        target_device_capacity_bytes=target_device_capacity_bytes,
    )
    try:
        measurement_results, completed_measurement_iterations = run_iterations(
            workers,
            config,
            mode=args.mode,
            run_id=args.run_id,
            output_dir=output_dir,
            warmup_iterations=0,
            measurement_iterations=args.iterations,
            duration_seconds=args.duration_seconds,
            target_write_bytes=target_write_bytes,
            target_write_baseline_bytes=target_write_baseline,
            target_write_poll_seconds=args.target_write_poll_seconds,
            start_iteration=args.warmup_iterations,
        )
    finally:
        if sampler_stop is not None:
            sampler_stop.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=10)

    measurement_after = capture_measurement(config, "after_measurement")
    write_json(os.path.join(output_dir, "measurement_after.json"), measurement_after)

    all_results = warmup_results + measurement_results
    write_json(
        os.path.join(output_dir, "run_results.json"),
        [asdict(result) for result in all_results],
    )
    summary = build_summary(
        config=config,
        workers=workers,
        mode=args.mode,
        run_id=args.run_id,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=completed_measurement_iterations,
        measurement_after_warmup=measurement_after_warmup,
        measurement_after=measurement_after,
        run_results=all_results,
        warnings=warnings,
        target_write_multiplier=target_multiplier,
        target_device_capacity_bytes=target_device_capacity_bytes,
        target_write_bytes=target_write_bytes,
    )
    write_json(os.path.join(output_dir, "summary.json"), summary)
    write_text(os.path.join(output_dir, "summary.md"), build_summary_md(summary))

    failed = any(result.exit_code != 0 for result in all_results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
