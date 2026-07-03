#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DURATION_SEC="${DURATION_SEC:-3600}"
CONFIG_PATH="${CONFIG_PATH:-$HOME/tmp/lmcache-fdp-waf-stress/config.4ruh.waf.slots.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/tmp/lmcache-fdp-waf-stress/log}"
SAMPLE_INTERVAL_SECONDS="${SAMPLE_INTERVAL_SECONDS:-}"
SWEEP="${SWEEP:-fdp_separated}"

usage() {
  cat <<'EOF'
Usage:
  run_fdp_waf_stress.example.sh [--duration-seconds SEC] [--sample-interval-seconds SEC] [--config PATH] [--sweep LIST]

Sweep names:
  no_fdp
  fdp_mixed
  fdp_separated
  all

Examples:
  bash benchmarks/fdp_waf_stress/run_fdp_waf_stress.example.sh --sweep fdp_separated --duration-seconds 1800 --sample-interval-seconds 300
  bash benchmarks/fdp_waf_stress/run_fdp_waf_stress.example.sh --sweep no_fdp,fdp_mixed,fdp_separated
  bash benchmarks/fdp_waf_stress/run_fdp_waf_stress.example.sh --sweep all
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration-seconds)
      DURATION_SEC="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --sample-interval-seconds)
      SAMPLE_INTERVAL_SECONDS="$2"
      shift 2
      ;;
    --sweep)
      SWEEP="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if (( DURATION_SEC % 3600 == 0 )); then
  DURATION_LABEL="$((DURATION_SEC / 3600))h"
elif (( DURATION_SEC % 60 == 0 )); then
  DURATION_LABEL="$((DURATION_SEC / 60))m"
else
  DURATION_LABEL="${DURATION_SEC}s"
fi

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_INTERVAL_SECONDS}" ]]; then
  SAMPLE_ARGS=(--sample-interval-seconds "${SAMPLE_INTERVAL_SECONDS}")
fi

if [[ "${SWEEP}" == "all" ]]; then
  SWEEP="no_fdp,fdp_mixed,fdp_separated"
fi

IFS=',' read -r -a SWEEP_ITEMS <<< "${SWEEP}"

cd "${REPO_ROOT}"

for item in "${SWEEP_ITEMS[@]}"; do
  case "${item}" in
    no_fdp)
      MODE="no_fdp"
      LABEL="no_fdp"
      ;;
    fdp_mixed)
      MODE="mixed"
      LABEL="fdp_mixed"
      ;;
    fdp_separated)
      MODE="separated"
      LABEL="fdp_separated"
      ;;
    *)
      echo "unknown sweep item: ${item}" >&2
      usage >&2
      exit 2
      ;;
  esac

  RUN_ID="waf-${LABEL}-${DURATION_LABEL}-$(date +%Y%m%d-%H%M%S)"
  OUTPUT_DIR="$OUTPUT_ROOT/waf-${LABEL}-${DURATION_LABEL}-${RUN_ID}"

  echo "==> ${LABEL}: mode=${MODE}, duration=${DURATION_SEC}s"
  sudo -S -p '' env HOME="$HOME" \
    PATH=/tmp/waf-bin:"$HOME"/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH="$HOME"/opt/xnvme/lib/x86_64-linux-gnu:"$HOME"/opt/xnvme/lib \
    uv run --no-sync python benchmarks/fdp_waf_stress/run_fdp_waf_stress.py \
      --config "${CONFIG_PATH}" \
      --mode "${MODE}" \
      --duration-seconds "${DURATION_SEC}" \
      "${SAMPLE_ARGS[@]}" \
      --warmup-iterations 0 \
      --run-id "${RUN_ID}" \
      --output-dir "${OUTPUT_DIR}"

  echo "summary: ${OUTPUT_DIR}/summary.md"
done
