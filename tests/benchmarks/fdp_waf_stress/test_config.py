# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Third Party
import pytest
import yaml

# First Party
from benchmarks.fdp_waf_stress.run_fdp_waf_stress import (
    expand_ruh_ids,
    expand_workers,
    load_yaml_config,
    make_meta_magic,
    make_salt_suffix,
    validate_windows,
)


def _config(
    tmp_path,
    *,
    stride=1024 * 1024,
    capacity=1024 * 1024,
    allocation="fixed_stride",
):
    trace = tmp_path / "missing.lct"
    return {
        "device_path": "/dev/ng1n1",
        "block_device_path": "/dev/nvme1n1",
        "block_align": 4096,
        "global": {
            "replay_binary": "lmcache",
            "l2_store_policy": "skip_l1",
            "eviction_policy": "noop",
        },
        "measurement": {"enabled": False},
        "windows": {
            "start_offset_bytes": 4096 * 1024,
            "window_stride_bytes": stride,
            "default_capacity_bytes": capacity,
            "allocation": allocation,
            "auto_assign": True,
        },
        "modes": {
            "mixed": {
                "use_fdp": True,
                "default_data_ruhs": [0, 1],
                "default_metadata_ruhs": [2],
            },
            "separated": {
                "use_fdp": True,
                "classes": {
                    "hot_churn": {
                        "data_ruhs": [0, 1],
                        "metadata_ruhs": [2],
                    },
                    "cold_rag": {
                        "data_ruhs": [3, 4],
                        "metadata_ruhs": [5],
                    },
                },
            },
            "no_fdp": {"use_fdp": False},
        },
        "workloads": [
            {
                "name": "hot",
                "class": "hot_churn",
                "trace_path": os.fspath(trace),
                "concurrency": 2,
                "slot_bytes": 4096 * 8,
                "capacity_bytes": capacity,
                "l1_size_gb": 1,
            },
            {
                "name": "cold",
                "class": "cold_rag",
                "trace_path": os.fspath(trace),
                "concurrency": 1,
                "slot_bytes": 4096 * 16,
                "capacity_bytes": capacity,
                "l1_size_gb": 2,
            },
        ],
    }


def test_yaml_parsing(tmp_path):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as file_obj:
        yaml.safe_dump(_config(tmp_path), file_obj)

    loaded = load_yaml_config(os.fspath(config_path))

    assert loaded["device_path"] == "/dev/ng1n1"
    assert len(loaded["workloads"]) == 2


def test_mode_resolution_mixed(tmp_path):
    workers = expand_workers(_config(tmp_path), "mixed")

    assert len(workers) == 3
    assert all(worker.use_fdp for worker in workers)
    assert all(worker.fdp_data_ruh_ids == [0, 1] for worker in workers)
    assert all(worker.fdp_metadata_ruh_ids == [2] for worker in workers)


def test_mode_resolution_separated(tmp_path):
    workers = expand_workers(_config(tmp_path), "separated")

    hot_workers = [worker for worker in workers if worker.name == "hot"]
    cold_worker = [worker for worker in workers if worker.name == "cold"][0]
    assert all(worker.fdp_data_ruh_ids == [0, 1] for worker in hot_workers)
    assert cold_worker.fdp_data_ruh_ids == [3, 4]
    assert cold_worker.fdp_metadata_ruh_ids == [5]


def test_mode_resolution_no_fdp(tmp_path):
    workers = expand_workers(_config(tmp_path), "no_fdp")

    assert all(not worker.use_fdp for worker in workers)
    assert all(worker.fdp_data_ruh_ids == [] for worker in workers)
    assert all(worker.fdp_metadata_ruh_ids == [] for worker in workers)


def test_auto_window_allocation(tmp_path):
    config = _config(tmp_path)
    workers = expand_workers(config, "mixed")

    assert [worker.worker_global_index for worker in workers] == [0, 1, 2]
    assert [worker.base_offset_bytes for worker in workers] == [
        4096 * 1024,
        4096 * 1024 + 1024 * 1024,
        4096 * 1024 + 2 * 1024 * 1024,
    ]
    validate_windows(workers)


def test_packed_window_allocation(tmp_path):
    config = _config(tmp_path, allocation="packed")
    config["workloads"][0]["capacity_bytes"] = 2 * 1024 * 1024
    config["workloads"][1]["capacity_bytes"] = 3 * 1024 * 1024

    workers = expand_workers(config, "mixed")

    assert [worker.base_offset_bytes for worker in workers] == [
        4096 * 1024,
        4096 * 1024 + 2 * 1024 * 1024,
        4096 * 1024 + 4 * 1024 * 1024,
    ]
    assert workers[-1].base_offset_bytes + workers[-1].capacity_bytes == (
        4096 * 1024 + 7 * 1024 * 1024
    )
    validate_windows(workers)


def test_invalid_window_allocation(tmp_path):
    config = _config(tmp_path, allocation="unknown")

    with pytest.raises(ValueError, match="windows.allocation"):
        expand_workers(config, "mixed")


def test_overlap_detection(tmp_path):
    config = _config(tmp_path, stride=512 * 1024, capacity=1024 * 1024)

    with pytest.raises(ValueError, match="overlap"):
        expand_workers(config, "mixed")


def test_unique_meta_magic_generation(tmp_path):
    workers = expand_workers(_config(tmp_path), "mixed")

    assert make_meta_magic("waf001", 1) == "83E40001"
    assert len({worker.meta_magic for worker in workers}) == len(workers)
    assert all(len(worker.meta_magic) == 8 for worker in workers)


def test_unique_replay_cache_salt_suffix(tmp_path):
    workers = expand_workers(_config(tmp_path), "mixed")

    salts = {
        make_salt_suffix("waf001", "mixed", worker, iteration)
        for worker in workers
        for iteration in range(3)
    }

    assert len(salts) == len(workers) * 3
    assert "waf001.mixed.hot.w0.iter_0001" in salts


def test_expand_ruh_id_ranges():
    assert expand_ruh_ids({"start": 100, "count": 4}) == [100, 101, 102, 103]
    assert expand_ruh_ids({"range": [120, 123]}) == [120, 121, 122, 123]
    assert expand_ruh_ids("0-2,100") == [0, 1, 2, 100]
