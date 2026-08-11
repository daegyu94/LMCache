FS (native)
===========

A file-system L2 adapter backed by the native C++ ``LMCacheFSClient``
wrapped with ``NativeConnectorL2Adapter``.  I/O is dispatched through a
C++ worker-thread pool with eventfd-driven completions, giving a true
I/O queue depth on a single Python thread.

**Required fields:**

- ``base_path``: Directory for storing KV cache files.

**Optional fields:**

- ``num_workers`` (int, default ``4``, > 0): Number of C++ worker threads
  inside the connector.  This is the real I/O queue depth -- raise to
  push throughput on filesystems whose aggregate BW exceeds per-stream
  BW.
- ``relative_tmp_dir`` (str, default ``""``): Relative sub-directory for
  temporary files during writes (atomic rename on completion).
- ``use_odirect`` (bool, default ``false``): Bypass the page cache via
  ``O_DIRECT``.  Required to measure real disk bandwidth.  The connector
  uses a reusable aligned buffer when the caller's address or object length
  does not meet the target filesystem's direct-I/O requirements.
- ``read_ahead_size`` (int, optional): Trigger filesystem readahead by
  issuing a warm-up read of this many bytes at open time.
- ``max_capacity_gb`` (float, default ``0``): Maximum L2 capacity in GB
  for client-side usage tracking.  Default ``0`` disables tracking.
- ``io_log_interval_sec`` (float, default ``0``): Emit cumulative I/O
  progress logs at this interval in seconds.  ``0`` disables progress logs.

.. important::

   With ``use_odirect: true``, every data-file open uses ``O_DIRECT``; the
   connector does not fall back to buffered I/O.  It queries Linux
   ``STATX_DIOALIGN`` when available, otherwise uses the filesystem block
   size.  Unaligned data is copied through one reusable aligned buffer per
   worker, and non-aligned object lengths are zero-padded on disk.  This
   requires extra memory copies only when the caller's buffer is unsuitable.
   ``read_ahead_size`` is ignored in direct-I/O mode.

   The target filesystem and kernel must support ``O_DIRECT``.  Unsupported
   filesystems or direct-I/O syscall failures are reported instead of being
   retried with buffered I/O.

   The adapter always emits one aggregate I/O summary when it closes.  The
   summary and optional progress logs are written to the LMCache process log
   and include logical bytes, direct/buffered operation counts, errors, and
   average/maximum latency.  The ``*_bytes`` fields are logical payload bytes;
   ``*_direct_bytes`` additionally includes alignment padding issued to the
   direct-I/O syscall.  For example, set ``io_log_interval_sec`` to ``5`` to
   emit a cumulative progress line every five seconds.

**Configuration examples:**

.. code-block:: bash

    # Basic native FS adapter
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2"}'

    # Many worker threads for a parallel filesystem (e.g. GPFS, Lustre)
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "num_workers": 32}'

    # O_DIRECT for real-disk benchmarking
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "num_workers": 32, "use_odirect": true}'

    # Aggregate summary plus cumulative progress every five seconds
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "use_odirect": true, "io_log_interval_sec": 5}'

**Buffer-only mode example.**  L1 acts as a pure write buffer that
absorbs the peak burst of in-flight chunks while the C++ worker pool
drains them to disk; nothing is retained in L1 once a store completes:

.. code-block:: bash

    lmcache server \
        --host 0.0.0.0 --port 5555 \
        --max-workers 32 \
        --l1-size-gb 32 --l1-use-lazy \
        --eviction-policy noop \
        --l2-store-policy skip_l1 \
        --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "num_workers": 32, "use_odirect": true}'
