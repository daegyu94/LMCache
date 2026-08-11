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
  keeps this mode enabled for every operation and uses an aligned temporary
  buffer when the caller's buffer is not suitable for direct I/O.
- ``read_ahead_size`` (int, optional): Trigger filesystem readahead by
  issuing a warm-up read of this many bytes at open time.
- ``max_capacity_gb`` (float, default ``0``): Maximum L2 capacity in GB
  for client-side usage tracking.  Default ``0`` disables tracking.

.. important::

   ``O_DIRECT`` requires the buffer address, transfer length, and file
   offset to be aligned to the filesystem block size.  The connector
   queries that block size at construction time, keeps ``O_DIRECT`` on for
   all reads and writes, and uses a block-aligned bounce buffer for an
   unaligned caller buffer.  A non-block-sized object is zero-padded in the
   file and trimmed back to its logical size on read.  Therefore a fresh
   directory can use arbitrary object sizes without falling back to buffered
   I/O.  Existing files written by an older direct-I/O implementation must
   be rewritten if their physical size is not block-aligned.

**Configuration examples:**

.. code-block:: bash

    # Basic native FS adapter
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2"}'

    # Many worker threads for a parallel filesystem (e.g. GPFS, Lustre)
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "num_workers": 32}'

    # O_DIRECT for real-disk benchmarking
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "num_workers": 32, "use_odirect": true}'

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
