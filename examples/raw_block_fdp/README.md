# Raw Block FDP Example

This directory contains a standalone diagnostic example for checking raw-block
FDP support on a real NVMe FDP namespace.

Use the NVMe character namespace device, such as `/dev/ng0n1`, not the block
namespace path `/dev/nvme0n1`. `io_uring_cmd` requires the character device.

## Device Policy Smoke

`device_policy_smoke.py` has two modes:

- Read-only discovery: query FDP reclaim unit handle status and validate an
  optional handle subset.
- Policy write smoke: create `RawBlockL2Adapter`, store a few small objects, and
  print the FDP placement map populated by the selected policy.

### Read-only Discovery

The default mode opens the NVMe namespace character device read-only, queries
FDP reclaim unit handle status, validates optional placement handles, and prints
JSON. It does not write raw-block data.

```bash
~/.venv/py312/bin/python examples/raw_block_fdp/device_policy_smoke.py \
  --device-path /dev/ng0n1 \
  --placement-handles 0,1
```

Omit `--placement-handles` to select all placement handles reported by the
device.

### Policy Write Smoke

To verify that an adapter policy map is populated in `RawBlockL2Adapter`, add
`--exercise-writes`. This mode writes raw-block metadata and data slots to the
target device, so only run it on a disposable/test namespace. It requires
`--placement-handles` so the smoke test does not claim every FDP handle reported
by the device.

Rank isolation:

```bash
~/.venv/py312/bin/python examples/raw_block_fdp/device_policy_smoke.py \
  --device-path /dev/ng0n1 \
  --policy rank_isolation \
  --placement-handles 0,1 \
  --exercise-writes \
  --i-understand-this-writes-to-device
```

Domain isolation:

```bash
~/.venv/py312/bin/python examples/raw_block_fdp/device_policy_smoke.py \
  --device-path /dev/ng0n1 \
  --policy domain_isolation \
  --placement-handles 0,1 \
  --exercise-writes \
  --i-understand-this-writes-to-device
```

Model isolation:

```bash
~/.venv/py312/bin/python examples/raw_block_fdp/device_policy_smoke.py \
  --device-path /dev/ng0n1 \
  --policy model_isolation \
  --placement-handles 0,1 \
  --exercise-writes \
  --i-understand-this-writes-to-device
```

The JSON output includes `placement_handle_to_ruh_id`, selected handles, initial
adapter FDP status, store result, and final adapter FDP status. Check these
fields in `final_policy_status`:

- `fdp_local_rank_to_placement` for `rank_isolation`
- `fdp_cache_salt_to_placement` for `domain_isolation`
- `fdp_model_to_placement` for `model_isolation`
