# SPDX-License-Identifier: Apache-2.0
"""CPU-only synthetic L2 adapter benchmark."""

# Local
from .workload import WorkloadSpec, build_workload

__all__ = ["WorkloadSpec", "build_workload"]
