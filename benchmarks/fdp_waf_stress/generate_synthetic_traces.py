# SPDX-License-Identifier: Apache-2.0

"""Generate CPU-only synthetic LMCache storage traces for FDP WAF stress."""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import os
import random
import struct
import sys
import time

# Third Party
import torch
import yaml

try:
    # Work around the local OTel logger-provider issue seen in this checkout.
    # Third Party
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider

    set_logger_provider(LoggerProvider())
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey  # noqa: E402
from lmcache.v1.mp_observability.trace import codecs  # noqa: E402
from lmcache.v1.mp_observability.trace.format import (  # noqa: E402
    FORMAT_VERSION,
    MAGIC,
    TRACE_SCHEMA_VERSION,
    Header,
    Record,
    encode_header,
    encode_record,
)

DEFAULT_ROOT = "/mnt/hc-ssd/lmcache-fdp-waf-stress"
DEFAULT_DEVICE_PATH = "/dev/ng1n1"
DEFAULT_BLOCK_DEVICE_PATH = "/dev/nvme1n1"
DEFAULT_START_OFFSET_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_WINDOW_STRIDE_BYTES = 2_147_483_648
DEFAULT_META_TOTAL_BYTES = 67_108_864
_LEN_STRUCT = struct.Struct(">I")
_SM_PREFIX = "lmcache.v1.distributed.storage_manager.StorageManager"


@dataclass(frozen=True)
class TraceRecipe:
    name: str
    class_name: str
    file_name: str
    model_name: str
    object_bytes: int
    batch_keys: int
    store_batches: int
    prefetch_batches: int
    prefetch_width: int
    concurrency: int
    slot_bytes: int
    capacity_bytes: int
    l1_size_gb: int
    placement_ranks: int
    notes: str


@dataclass
class GeneratedTraceSummary:
    name: str
    class_name: str
    path: str
    record_count: int
    store_count: int
    prefetch_count: int
    unique_object_key_count: int
    estimated_store_bytes: int
    estimated_max_object_bytes: int
    duration_seconds: float


class TraceWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self.record_count = 0
        self._t_mono = 0.0
        self._t_wall_start = time.time()
        self._t_mono_start = time.monotonic()
        self._fh = open(path, "wb")
        header = Header(
            magic=MAGIC,
            format_version=FORMAT_VERSION,
            level="storage",
            trace_schema_version=TRACE_SCHEMA_VERSION,
            t_mono_start=self._t_mono_start,
            t_wall_start=self._t_wall_start,
            sm_config_json="",
            sm_config_digest="",
        )
        self._write_frame(encode_header(header))

    def close(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()

    def record(self, qualname: str, args: dict[str, Any], *, gap_s: float) -> None:
        self._t_mono += gap_s
        record = Record(
            t_mono=self._t_mono,
            t_wall=self._t_wall_start + self._t_mono,
            qualname=qualname,
            args=codecs.encode_args(args),
        )
        self._write_frame(encode_record(record))
        self.record_count += 1

    @property
    def duration_seconds(self) -> float:
        return self._t_mono

    def _write_frame(self, frame: bytes) -> None:
        self._fh.write(_LEN_STRUCT.pack(len(frame)) + frame)


def _layout(object_bytes: int) -> MemoryLayoutDesc:
    if object_bytes % 2 != 0:
        raise ValueError("object_bytes must be even for float16 layout")
    return MemoryLayoutDesc(
        shapes=[torch.Size([object_bytes // 2])],
        dtypes=[torch.float16],
    )


def _chunk_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def _kv_rank(placement_rank: int, placement_ranks: int) -> int:
    rank_count = max(1, min(placement_ranks, 256))
    rank = int(placement_rank % rank_count)
    return ObjectKey.ComputeKVRank(
        world_size=rank_count,
        global_rank=rank,
        local_world_size=rank_count,
        local_rank=rank,
    )


def _make_key(recipe: TraceRecipe, key_id: int) -> ObjectKey:
    placement_rank = key_id % max(1, recipe.placement_ranks)
    return ObjectKey(
        chunk_hash=_chunk_hash(f"{recipe.name}:{key_id}"),
        model_name=recipe.model_name,
        kv_rank=_kv_rank(placement_rank, recipe.placement_ranks),
    )


def _record_store(
    writer: TraceWriter,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
    *,
    gap_s: float,
) -> None:
    writer.record(
        f"{_SM_PREFIX}.reserve_write",
        {"keys": keys, "layout_desc": layout, "mode": "new"},
        gap_s=gap_s,
    )
    writer.record(f"{_SM_PREFIX}.finish_write", {"keys": keys}, gap_s=gap_s)


def _record_prefetch(
    writer: TraceWriter,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
    *,
    request_id: str,
    gap_s: float,
) -> None:
    writer.record(
        f"{_SM_PREFIX}.submit_prefetch_task",
        {
            "keys": keys,
            "layout_desc": layout,
            "extra_count": 0,
            "external_request_id": request_id,
        },
        gap_s=gap_s,
    )


def _recipes(scale: str, ruh_count: int) -> list[TraceRecipe]:
    if ruh_count != 8 and ruh_count < 128:
        raise ValueError("--ruh-count must be 8 or at least 128")

    stress = scale != "smoke"
    return [
        TraceRecipe(
            name="llama8b_chat_chunk256",
            class_name="hot_churn",
            file_name="llama8b_chat_chunk256.lct",
            model_name="llama8b-chat",
            object_bytes=4 * 1024 * 1024,
            batch_keys=4,
            store_batches=640 if stress else 8,
            prefetch_batches=80 if stress else 2,
            prefetch_width=4,
            concurrency=2,
            slot_bytes=4 * 1024 * 1024,
            capacity_bytes=1 * 1024 * 1024 * 1024,
            l1_size_gb=4,
            placement_ranks=ruh_count,
            notes="small model chat-like hot churn",
        ),
        TraceRecipe(
            name="llama70b_longctx_chunk1024",
            class_name="large_model",
            file_name="llama70b_longctx_chunk1024.lct",
            model_name="llama70b-longctx",
            object_bytes=32 * 1024 * 1024,
            batch_keys=2,
            store_batches=160 if stress else 4,
            prefetch_batches=40 if stress else 1,
            prefetch_width=8,
            concurrency=1,
            slot_bytes=32 * 1024 * 1024,
            capacity_bytes=2 * 1024 * 1024 * 1024,
            l1_size_gb=8,
            placement_ranks=ruh_count,
            notes="large model long-context objects",
        ),
        TraceRecipe(
            name="rag_shared_prefix_chunk512",
            class_name="cold_rag",
            file_name="rag_shared_prefix_chunk512.lct",
            model_name="rag-shared-prefix",
            object_bytes=16 * 1024 * 1024,
            batch_keys=3,
            store_batches=256 if stress else 8,
            prefetch_batches=256 if stress else 8,
            prefetch_width=12,
            concurrency=2,
            slot_bytes=16 * 1024 * 1024,
            capacity_bytes=1 * 1024 * 1024 * 1024,
            l1_size_gb=4,
            placement_ranks=ruh_count,
            notes="RAG/shared-prefix trace with repeated prefix prefetches",
        ),
        TraceRecipe(
            name="random_prompts_chunk128",
            class_name="hot_churn",
            file_name="random_prompts_chunk128.lct",
            model_name="random-prompts",
            object_bytes=2 * 1024 * 1024,
            batch_keys=4,
            store_batches=1024 if stress else 8,
            prefetch_batches=32 if stress else 2,
            prefetch_width=2,
            concurrency=3,
            slot_bytes=2 * 1024 * 1024,
            capacity_bytes=1 * 1024 * 1024 * 1024,
            l1_size_gb=2,
            placement_ranks=ruh_count,
            notes="random prompts with low reuse and small chunks",
        ),
        TraceRecipe(
            name="metadata_heavy_small_objects",
            class_name="metadata_heavy",
            file_name="metadata_heavy_small_objects.lct",
            model_name="metadata-heavy",
            object_bytes=1 * 1024 * 1024,
            batch_keys=1,
            store_batches=8192 if stress else 16,
            prefetch_batches=0,
            prefetch_width=1,
            concurrency=1,
            slot_bytes=1 * 1024 * 1024,
            capacity_bytes=512 * 1024 * 1024,
            l1_size_gb=2,
            placement_ranks=ruh_count,
            notes="many small objects to pressure checkpoint metadata",
        ),
    ]


def _scale_recipes(recipes: list[TraceRecipe], scale: int) -> list[TraceRecipe]:
    """Scale stress recipes without changing individual operation sizes."""
    return [
        replace(
            recipe,
            store_batches=recipe.store_batches * scale,
            prefetch_batches=recipe.prefetch_batches * scale,
            capacity_bytes=recipe.capacity_bytes * scale,
        )
        for recipe in recipes
    ]


def _parse_scale(value: str) -> tuple[str, int, bool]:
    """Parse a legacy preset or a positive stress scaling factor."""
    normalized = value.lower()
    if normalized == "smoke":
        return normalized, 1, True
    if normalized == "stress":
        return normalized, 1, False
    try:
        scale = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--scale must be smoke, stress, or a positive integer"
        ) from exc
    if scale < 1:
        raise argparse.ArgumentTypeError("--scale must be >= 1")
    return f"{scale}x", scale, False


def _packed_layout(recipes: list[TraceRecipe]) -> dict[str, Any]:
    """Return the contiguous packed LBA windows for generated workloads."""
    base_offset = DEFAULT_START_OFFSET_BYTES
    workers: list[dict[str, Any]] = []
    for recipe in recipes:
        for worker_index in range(recipe.concurrency):
            end_offset = base_offset + recipe.capacity_bytes
            workers.append(
                {
                    "name": recipe.name,
                    "worker_index": worker_index,
                    "base_offset_bytes": base_offset,
                    "end_offset_bytes": end_offset,
                    "capacity_bytes": recipe.capacity_bytes,
                }
            )
            base_offset = end_offset
    return {
        "allocation": "packed",
        "worker_count": len(workers),
        "capacity_sum_bytes": sum(worker["capacity_bytes"] for worker in workers),
        "start_offset_bytes": DEFAULT_START_OFFSET_BYTES,
        "end_offset_bytes": base_offset,
        "span_bytes": base_offset - DEFAULT_START_OFFSET_BYTES,
        "alignment_padding_bytes": 0,
        "workers": workers,
    }


def generate_trace(
    recipe: TraceRecipe,
    output_dir: str,
    *,
    seed: int,
) -> dict[str, Any]:
    path = os.path.join(output_dir, recipe.file_name)
    rng = random.Random(seed + sum(recipe.name.encode("utf-8")))
    layout = _layout(recipe.object_bytes)
    all_keys: list[ObjectKey] = []
    unique_ids: set[str] = set()
    store_count = 0
    prefetch_count = 0
    key_id = 0

    writer = TraceWriter(path)
    try:
        for batch_idx in range(recipe.store_batches):
            keys = [
                _make_key(recipe, key_id + offset)
                for offset in range(recipe.batch_keys)
            ]
            key_id += recipe.batch_keys
            all_keys.extend(keys)
            unique_ids.update(_key_id(key) for key in keys)
            store_count += len(keys)
            _record_store(writer, keys, layout, gap_s=0.001)

            if recipe.prefetch_batches:
                prefetch_period = max(
                    1,
                    recipe.store_batches // recipe.prefetch_batches,
                )
                if batch_idx % prefetch_period == 0:
                    width = min(recipe.prefetch_width, len(all_keys))
                    prefetch_keys = rng.sample(all_keys, width)
                    prefetch_count += len(prefetch_keys)
                    _record_prefetch(
                        writer,
                        prefetch_keys,
                        layout,
                        request_id=f"{recipe.name}-pf-{batch_idx}",
                        gap_s=0.001,
                    )
    finally:
        writer.close()

    summary = GeneratedTraceSummary(
        name=recipe.name,
        class_name=recipe.class_name,
        path=path,
        record_count=writer.record_count,
        store_count=store_count,
        prefetch_count=prefetch_count,
        unique_object_key_count=len(unique_ids),
        estimated_store_bytes=store_count * recipe.object_bytes,
        estimated_max_object_bytes=recipe.object_bytes,
        duration_seconds=writer.duration_seconds,
    )
    return asdict(summary)


def _key_id(key: ObjectKey) -> str:
    return "|".join(
        [
            key.chunk_hash.hex(),
            key.model_name,
            str(key.kv_rank),
            key.cache_salt,
        ]
    )


def _uv_replay_binary() -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-c",
        "\n".join(
            [
                "from opentelemetry._logs import set_logger_provider",
                "from opentelemetry.sdk._logs import LoggerProvider",
                "set_logger_provider(LoggerProvider())",
                "import sys",
                "from lmcache.cli.main import main",
                'sys.argv = ["lmcache"] + sys.argv[1:]',
                "main()",
            ]
        ),
    ]


def _build_modes(ruh_count: int) -> dict[str, Any]:
    """Build RUH mappings for either the 128-RUH or compact profile."""
    if ruh_count < 8:
        raise ValueError("generated config requires ruh_count >= 8")
    if ruh_count >= 128:
        return {
            "mixed": {
                "use_fdp": True,
                "default_data_ruhs": {"start": 0, "count": ruh_count},
                "default_metadata_ruhs": {"start": 120, "count": 4},
            },
            "separated": {
                "use_fdp": True,
                "classes": {
                    "hot_churn": {
                        "data_ruhs": {"start": 0, "count": 32},
                        "metadata_ruhs": {"start": 100, "count": 4},
                    },
                    "cold_rag": {
                        "data_ruhs": {"start": 32, "count": 32},
                        "metadata_ruhs": {"start": 104, "count": 4},
                    },
                    "metadata_heavy": {
                        "data_ruhs": {"start": 64, "count": 32},
                        "metadata_ruhs": {"start": 108, "count": 4},
                    },
                    "large_model": {
                        "data_ruhs": {"start": 32, "count": 32},
                        "metadata_ruhs": {"start": 112, "count": 4},
                    },
                    "small_model": {
                        "data_ruhs": {"start": 0, "count": 32},
                        "metadata_ruhs": {"start": 116, "count": 4},
                    },
                },
            },
            "no_fdp": {"use_fdp": False},
        }

    if ruh_count != 8:
        raise ValueError("generated config requires ruh_count to be 8 or >= 128")
    return {
        "mixed": {
            "use_fdp": True,
            "default_data_ruhs": [0, 1, 2, 3, 4, 5, 6],
            "default_metadata_ruhs": [7],
        },
        "separated": {
            "use_fdp": True,
            "classes": {
                "hot_churn": {"data_ruhs": [0, 1], "metadata_ruhs": [7]},
                "cold_rag": {"data_ruhs": [2, 3], "metadata_ruhs": [7]},
                "metadata_heavy": {"data_ruhs": [4], "metadata_ruhs": [7]},
                "large_model": {"data_ruhs": [5, 6], "metadata_ruhs": [7]},
                "small_model": {"data_ruhs": [0, 1], "metadata_ruhs": [7]},
            },
        },
        "no_fdp": {"use_fdp": False},
    }


def build_config(
    recipes: list[TraceRecipe],
    *,
    trace_dir: str,
    output_root: str,
    ruh_count: int,
    device_path: str,
    block_device_path: str,
    meta_total_bytes: int,
) -> dict[str, Any]:
    modes = _build_modes(ruh_count)
    return {
        "device_path": device_path,
        "block_device_path": block_device_path,
        "block_align": 4096,
        "global": {
            "l2_store_policy": "skip_l1",
            "eviction_policy": "noop",
            "disable_metrics": True,
            "quiet": True,
            "l1_align_bytes": 4096,
            "meta_total_bytes": meta_total_bytes,
            "use_odirect": False,
            "use_uring": True,
            "use_uring_cmd": True,
            "output_root": output_root,
            "replay_binary": _uv_replay_binary(),
        },
        "measurement": {
            "enabled": True,
            "collect_nvme_smart": True,
            "collect_fdp_logs": True,
            "vendor_media_write_command": None,
        },
        "windows": {
            "start_offset_bytes": DEFAULT_START_OFFSET_BYTES,
            "allocation": "packed",
            "alignment_bytes": 4096,
            "default_capacity_bytes": 1 * 1024 * 1024 * 1024,
            "auto_assign": True,
        },
        "modes": modes,
        "workloads": [
            {
                "name": recipe.name,
                "class": recipe.class_name,
                "trace_path": os.path.join(trace_dir, recipe.file_name),
                "concurrency": recipe.concurrency,
                "slot_bytes": recipe.slot_bytes,
                "capacity_bytes": recipe.capacity_bytes,
                "l1_size_gb": recipe.l1_size_gb,
            }
            for recipe in recipes
        ],
    }


def build_manifest(recipes: list[TraceRecipe], trace_dir: str) -> dict[str, Any]:
    return {
        "traces": [
            {
                "name": recipe.name,
                "class": recipe.class_name,
                "path": os.path.join(trace_dir, recipe.file_name),
                "notes": recipe.notes,
            }
            for recipe in recipes
        ],
        "collection_notes": [
            "These are synthetic storage-level traces generated without a GPU.",
            "Replay writes synthetic payloads based on recorded layout descriptors.",
            "Model names only identify storage behavior families.",
        ],
    }


def write_yaml(path: str, payload: Any) -> None:
    with open(path, "w") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", default=None, help="Directory for .lct files.")
    parser.add_argument("--config-out", default=None)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--device-path", default=DEFAULT_DEVICE_PATH)
    parser.add_argument("--block-device-path", default=DEFAULT_BLOCK_DEVICE_PATH)
    parser.add_argument("--ruh-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--scale", type=_parse_scale, default=_parse_scale("stress"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scale_name, scale_factor, use_smoke_recipe = args.scale
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "traces"
    scale_label = "smoke" if use_smoke_recipe else str(scale_factor)
    config_out = (
        Path(args.config_out)
        if args.config_out
        else root / f"config.{args.ruh_count}ruh.{scale_label}scale.yaml"
    )
    manifest_out = (
        Path(args.manifest_out)
        if args.manifest_out
        else root / "trace_manifest.generated.yaml"
    )
    summary_out = (
        Path(args.summary_out)
        if args.summary_out
        else root / "trace_generation_summary.json"
    )
    trace_dir = os.path.abspath(output_dir)
    output_root = os.path.abspath(config_out.parent)
    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    recipes = _recipes("smoke" if use_smoke_recipe else "stress", args.ruh_count)
    if not use_smoke_recipe:
        recipes = _scale_recipes(recipes, scale_factor)
    meta_total_bytes = DEFAULT_META_TOTAL_BYTES * scale_factor
    traces = [generate_trace(recipe, trace_dir, seed=args.seed) for recipe in recipes]
    layout = _packed_layout(recipes)
    config = build_config(
        recipes,
        trace_dir=trace_dir,
        output_root=output_root,
        ruh_count=args.ruh_count,
        device_path=args.device_path,
        block_device_path=args.block_device_path,
        meta_total_bytes=meta_total_bytes,
    )
    manifest = build_manifest(recipes, trace_dir)
    estimated_store_bytes = sum(item["estimated_store_bytes"] for item in traces)
    estimated_concurrent_store_bytes = sum(
        item["estimated_store_bytes"] * recipe.concurrency
        for item, recipe in zip(traces, recipes, strict=True)
    )
    summary = {
        "scale": {"input": scale_name, "factor": scale_factor},
        "ruh_count": args.ruh_count,
        "paths": {
            "root": os.path.abspath(root),
            "device_path": args.device_path,
            "block_device_path": args.block_device_path,
        },
        "trace_dir": trace_dir,
        "config_path": os.path.abspath(config_out),
        "manifest_path": os.path.abspath(manifest_out),
        "estimated_store_bytes": estimated_store_bytes,
        "estimated_store_bytes_with_concurrency": estimated_concurrent_store_bytes,
        "worker_lba_layout": layout,
        "traces": traces,
    }

    write_yaml(os.fspath(config_out), config)
    write_yaml(os.fspath(manifest_out), manifest)
    write_json(os.fspath(summary_out), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
