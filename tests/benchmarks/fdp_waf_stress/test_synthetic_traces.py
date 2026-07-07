# SPDX-License-Identifier: Apache-2.0

# Standard
import json
import os

# First Party
from benchmarks.fdp_waf_stress.generate_synthetic_traces import main as generate_main
from benchmarks.fdp_waf_stress.run_fdp_waf_stress import (
    DEFAULT_SLOT_HEADER_BYTES,
    analyze_trace_footprint,
    expand_workers,
    load_yaml_config,
    main as harness_main,
)


def test_generate_synthetic_traces_roundtrip_and_dry_run(tmp_path):
    trace_dir = tmp_path / "traces"
    config_path = tmp_path / "config.128ruh.yaml"
    manifest_path = tmp_path / "trace_manifest.generated.yaml"
    summary_path = tmp_path / "trace_generation_summary.json"

    exit_code = generate_main(
        [
            "--output-dir",
            os.fspath(trace_dir),
            "--config-out",
            os.fspath(config_path),
            "--manifest-out",
            os.fspath(manifest_path),
            "--summary-out",
            os.fspath(summary_path),
            "--ruh-count",
            "128",
            "--scale",
            "smoke",
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text())
    assert len(summary["traces"]) == 5
    assert all(os.path.exists(item["path"]) for item in summary["traces"])

    first_trace = summary["traces"][0]["path"]
    footprint = analyze_trace_footprint(first_trace)
    assert footprint.record_count > 0
    assert footprint.store_count > 0
    assert footprint.estimated_total_store_bytes > 0

    config = load_yaml_config(os.fspath(config_path))
    workers = expand_workers(config, "separated")
    assert len(workers) >= 8
    assert max(max(worker.fdp_metadata_ruh_ids) for worker in workers) > 100
    assert max(max(worker.fdp_data_ruh_ids) for worker in workers) > 60

    dry_run_dir = tmp_path / "dry-run"
    dry_run_exit = harness_main(
        [
            "--config",
            os.fspath(config_path),
            "--mode",
            "separated",
            "--iterations",
            "1",
            "--warmup-iterations",
            "0",
            "--output-dir",
            os.fspath(dry_run_dir),
            "--dry-run",
        ]
    )

    assert dry_run_exit == 0
    assert (dry_run_dir / "commands.txt").exists()
    assert (dry_run_dir / "workers.json").exists()
    workers_payload = json.loads((dry_run_dir / "workers.json").read_text())
    assert workers_payload
    assert all(
        worker["slot_bytes"]
        >= worker["trace_footprint"]["estimated_max_object_bytes"]
        + DEFAULT_SLOT_HEADER_BYTES
        for worker in workers_payload
    )
