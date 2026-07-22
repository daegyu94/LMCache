# SPDX-License-Identifier: Apache-2.0

# Standard
import json
import os
import sys
import threading
import time

# Third Party
import yaml

# First Party
from benchmarks.fdp_waf_stress.run_fdp_waf_stress import (
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    ActiveReplayProcesses,
    build_l2_adapter,
    build_replay_command,
    build_waf_sample,
    capture_application_write_bytes,
    expand_workers,
    extract_host_write_bytes,
    extract_media_write_bytes,
    format_target_write_progress,
    main,
    parse_args,
    run_single_worker_replay,
    summarize_l2_latencies,
    waf_sample_to_tsv,
)
import benchmarks.fdp_waf_stress.run_fdp_waf_stress as runner


def _config(tmp_path):
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
            "window_stride_bytes": 1024 * 1024,
            "default_capacity_bytes": 1024 * 1024,
            "auto_assign": True,
        },
        "modes": {
            "mixed": {
                "use_fdp": True,
                "default_data_ruhs": [0, 1],
                "default_metadata_ruhs": [2],
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
                "capacity_bytes": 1024 * 1024,
                "l1_size_gb": 1,
            },
            {
                "name": "cold",
                "class": "cold_rag",
                "trace_path": os.fspath(trace),
                "concurrency": 1,
                "slot_bytes": 4096 * 16,
                "capacity_bytes": 1024 * 1024,
                "l1_size_gb": 2,
            },
        ],
    }


def test_l2_adapter_json_construction_mixed(tmp_path):
    config = _config(tmp_path)
    worker = expand_workers(config, "mixed")[0]

    adapter = build_l2_adapter(worker, config)

    assert adapter["type"] == "raw_block"
    assert adapter["device_path"] == "/dev/ng1n1"
    assert adapter["base_offset_bytes"] == worker.base_offset_bytes
    assert adapter["capacity_bytes"] == worker.capacity_bytes
    assert adapter["meta_magic"] == "WF000001"
    assert adapter["use_fdp"] is True
    assert adapter["fdp_data_ruh_ids"] == [0, 1]
    assert adapter["fdp_metadata_ruh_ids"] == [2]


def test_l2_adapter_json_construction_no_fdp_omits_ruhs(tmp_path):
    config = _config(tmp_path)
    worker = expand_workers(config, "no_fdp")[0]

    adapter = build_l2_adapter(worker, config)

    assert adapter["use_fdp"] is False
    assert "fdp_data_ruh_ids" not in adapter
    assert "fdp_metadata_ruh_ids" not in adapter


def test_replay_command_contains_required_flags(tmp_path):
    config = _config(tmp_path)
    worker = expand_workers(config, "mixed")[0]
    cmd = build_replay_command(
        worker,
        config,
        mode="mixed",
        run_id="waf001",
        iteration=4,
        worker_output_dir=os.fspath(tmp_path / "worker"),
        jsonl_path=os.fspath(tmp_path / "records.jsonl"),
    )

    assert cmd[:3] == ["lmcache", "trace", "replay"]
    assert "--replay-cache-salt-suffix" in cmd
    salt = cmd[cmd.index("--replay-cache-salt-suffix") + 1]
    assert salt == "waf001.mixed.hot.w0.iter_0004"
    adapter = json.loads(cmd[cmd.index("--l2-adapter") + 1])
    assert adapter["type"] == "raw_block"
    assert adapter["fdp_data_ruh_ids"] == [0, 1]
    assert "--jsonl-out" in cmd
    assert adapter["latency_log_path"] == os.fspath(
        tmp_path / "worker/l2_latency.jsonl"
    )


def test_dry_run_command_output(tmp_path, capsys):
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    with open(config_path, "w") as file_obj:
        yaml.safe_dump(config, file_obj)

    exit_code = main(
        [
            "--config",
            os.fspath(config_path),
            "--mode",
            "mixed",
            "--iterations",
            "1",
            "--warmup-iterations",
            "0",
            "--output-dir",
            os.fspath(output_dir),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "worker_count=3" in captured.out
    assert "--l2-adapter" in captured.out
    assert "waf001.mixed.hot.w0.iter_0000" in captured.out
    assert (output_dir / "commands.txt").exists()
    assert (output_dir / "workers.json").exists()


def test_measurement_parser_fallback_without_vendor_media_counter():
    smart = {"data_units_written": "10"}

    assert extract_host_write_bytes(smart) == 5_120_000
    assert extract_media_write_bytes(None) is None
    assert extract_media_write_bytes({"nested": {"nand_write_bytes": "1234"}}) == 1234


def test_waf_sample_tsv_uses_interval_deltas_without_multiplier_column():
    baseline = {
        "captured_at": "2026-07-07T00:00:00+00:00",
        "host_write_bytes": 1_000_000_000,
        "media_write_bytes": 2_000_000_000,
    }
    previous = {
        "captured_at": "2026-07-07T00:05:00+00:00",
        "host_write_bytes": 1_500_000_000,
        "media_write_bytes": 2_750_000_000,
    }
    sample = {
        "captured_at": "2026-07-07T00:10:00+00:00",
        "host_write_bytes": 2_000_000_000,
        "media_write_bytes": 3_500_000_000,
    }

    result = build_waf_sample(
        sample=sample,
        baseline=baseline,
        previous=previous,
        target_device_capacity_bytes=2_000_000_000,
    )

    assert result["timestamp"] == "2026-07-07T00:10:00+00:00"
    assert result["fdp_host_write_mb"] == 500.0
    assert result["fdp_media_write_mb"] == 750.0
    assert result["fdp_waf"] == 1.5
    assert result["device_write_multiplier"] == 0.5
    assert result["sample_status"] == "updated"
    assert waf_sample_to_tsv(result).split() == [
        "2026-07-07",
        "00:10:00",
        "500.00",
        "750.00",
        "1.500",
        "updated",
    ]


def test_target_write_progress_shows_percent_amount_and_multiplier():
    gib = 1024**3

    progress = format_target_write_progress(
        written_bytes=3 * gib,
        target_write_bytes=6 * gib,
        device_capacity_bytes=2 * gib,
        width=10,
        elapsed_seconds=3661,
    )

    assert "[#####-----]" in progress
    assert "50.00%" in progress
    assert "3.00 GiB / 6.00 GiB" in progress
    assert "(1.500x / 3.000x)" in progress
    assert "elapsed=01:01:01" in progress


def test_target_write_progress_caps_bar_at_100_percent_after_overshoot():
    gib = 1024**3

    progress = format_target_write_progress(
        written_bytes=7 * gib,
        target_write_bytes=6 * gib,
        device_capacity_bytes=2 * gib,
        width=10,
    )

    assert "[##########]" in progress
    assert "100.00%" in progress
    assert "7.00 GiB / 6.00 GiB" in progress


def test_application_write_bytes_uses_final_status_over_live_progress(tmp_path):
    worker_root = tmp_path / "worker_logs"
    finished = worker_root / "000" / "measurement_0000"
    active = worker_root / "001" / "measurement_0000"
    finished.mkdir(parents=True)
    active.mkdir(parents=True)

    def status(total_write_physical_bytes):
        return {
            "l2_adapters": [
                {
                    "core": {
                        "io_accounting": {
                            "total_write_physical_bytes": total_write_physical_bytes,
                        }
                    }
                }
            ]
        }

    (finished / "storage_manager_progress.json").write_text(json.dumps(status(100)))
    (finished / "storage_manager_status.json").write_text(json.dumps(status(200)))
    (active / "storage_manager_progress.json").write_text(json.dumps(status(300)))

    assert capture_application_write_bytes(os.fspath(tmp_path)) == 500


def test_waf_sample_marks_stale_interval_when_deltas_are_zero():
    baseline = {
        "captured_at": "2026-07-07T00:00:00+00:00",
        "host_write_bytes": 1_000,
        "media_write_bytes": 2_000,
    }
    previous = {
        "captured_at": "2026-07-07T00:05:00+00:00",
        "host_write_bytes": 1_500,
        "media_write_bytes": 2_750,
    }
    sample = {
        "captured_at": "2026-07-07T00:10:00+00:00",
        "host_write_bytes": 1_500,
        "media_write_bytes": 2_750,
    }

    result = build_waf_sample(sample=sample, baseline=baseline, previous=previous)

    assert result["fdp_host_write_mb"] is None
    assert result["fdp_media_write_mb"] is None
    assert result["fdp_waf"] is None
    assert result["sample_status"] == "stale"
    assert waf_sample_to_tsv(result).split() == [
        "2026-07-07",
        "00:10:00",
        "stale",
    ]


def test_sample_interval_defaults_to_five_minutes_and_can_be_disabled(tmp_path):
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as file_obj:
        yaml.safe_dump(config, file_obj)

    default_args = parse_args(["--config", os.fspath(config_path), "--mode", "mixed"])
    disabled_args = parse_args(
        [
            "--config",
            os.fspath(config_path),
            "--mode",
            "mixed",
            "--sample-interval-seconds",
            "0",
        ]
    )

    assert default_args.sample_interval_seconds == DEFAULT_SAMPLE_INTERVAL_SECONDS
    assert disabled_args.sample_interval_seconds == 0


def test_target_write_multiplier_is_opt_in(tmp_path):
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as file_obj:
        yaml.safe_dump(config, file_obj)

    default_args = parse_args(["--config", os.fspath(config_path), "--mode", "mixed"])
    target_args = parse_args(
        [
            "--config",
            os.fspath(config_path),
            "--mode",
            "mixed",
            "--duration-seconds",
            "30",
            "--iterations",
            "2",
            "--target-write-multiplier",
            "5",
        ]
    )

    assert default_args.target_write_multiplier is None
    assert target_args.target_write_multiplier == 5


def test_target_write_multiplier_dry_run_uses_device_capacity(
    tmp_path,
    capsys,
    monkeypatch,
):
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    with open(config_path, "w") as file_obj:
        yaml.safe_dump(config, file_obj)
    monkeypatch.setattr(runner, "detect_block_device_capacity_bytes", lambda _: 1024)

    exit_code = main(
        [
            "--config",
            os.fspath(config_path),
            "--mode",
            "mixed",
            "--duration-seconds",
            "30",
            "--target-write-multiplier",
            "5",
            "--output-dir",
            os.fspath(output_dir),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "target_write_multiplier=5.0" in captured.out
    assert "target_device_capacity_bytes=1024" in captured.out
    assert "target_write_bytes=5120" in captured.out
    assert "duration_seconds=30" not in captured.out


def test_target_sigterm_is_success_and_keeps_completed_jsonl(tmp_path, monkeypatch):
    config = _config(tmp_path)
    worker = expand_workers(config, "mixed")[0]
    started_path = tmp_path / "started"

    def fake_replay_command(*_args, jsonl_path, **_kwargs):
        script = (
            "import pathlib, time; "
            f"pathlib.Path({json.dumps(os.fspath(started_path))}).touch(); "
            f'f = open({json.dumps(jsonl_path)}, "w"); '
            'f.write("{\\"failed\\": false}\\n"); f.flush(); '
            "time.sleep(60)"
        )
        return [sys.executable, "-c", script]

    monkeypatch.setattr(runner, "build_replay_command", fake_replay_command)
    active_processes = ActiveReplayProcesses()
    results = []

    def run_worker():
        results.append(
            run_single_worker_replay(
                worker,
                config,
                mode="mixed",
                run_id="test",
                output_dir=os.fspath(tmp_path / "out"),
                iteration=0,
                phase="measurement",
                active_processes=active_processes,
            )
        )

    thread = threading.Thread(target=run_worker)
    thread.start()
    deadline = time.monotonic() + 5
    while not started_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started_path.exists()

    active_processes.terminate_all(grace_seconds=2)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(results) == 1
    assert results[0].terminated_by_target is True
    assert results[0].process_exit_code != 0
    assert results[0].exit_code == 0
    assert results[0].records_failed == 0


def test_summarize_l2_latencies_uses_measurement_data_io_only(tmp_path):
    output_dir = tmp_path / "measurement"
    output_dir.mkdir()
    samples = [
        {"metric": "raw_block_write", "latency_ms": 1.0, "io_class": "data"},
        {"metric": "raw_block_write", "latency_ms": 3.0, "io_class": "data"},
        {"metric": "raw_block_write", "latency_ms": 99.0, "io_class": "metadata"},
        {
            "metric": "raw_block_write",
            "latency_ms": 10.0,
            "io_class": "data",
            "failed": True,
        },
        {"metric": "l2_e2e_read", "latency_ms": 4.0},
    ]
    latency_path = output_dir / "l2_latency.jsonl"
    latency_path.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    result = runner.ReplayRunResult(
        worker_global_index=0,
        worker_name="hot",
        worker_index=0,
        iteration=0,
        phase="measurement",
        command=[],
        log_path="",
        output_dir=os.fspath(output_dir),
        jsonl_path="",
        exit_code=0,
        records_failed=0,
        started_at="",
        ended_at="",
    )

    summary = summarize_l2_latencies([result])

    assert summary["raw_block_write"]["count"] == 2
    assert summary["raw_block_write"]["error_count"] == 1
    assert summary["raw_block_write"]["avg_ms"] == 2.0
    assert summary["raw_block_write"]["p90_ms"] == 3.0
    assert summary["raw_block_write"]["p99_ms"] == 3.0
    assert summary["l2_e2e_read"]["count"] == 1
