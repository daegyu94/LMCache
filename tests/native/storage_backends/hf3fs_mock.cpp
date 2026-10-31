// SPDX-License-Identifier: Apache-2.0
// SDK test double: POSIX payload I/O, tracked ownership, no real 3FS service.
#include "hf3fs_mock.h"
#include <hf3fs_usrbio.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace hf3fs_mock {
std::string mount_point;
std::atomic<int> created{0};
std::atomic<int> destroyed{0};
std::atomic<int> duplicate_destroys{0};
std::atomic<int> deregistered{0};
std::atomic<bool> hold_io{false};
std::atomic<bool> io_entered{false};
std::atomic<bool> io_released{false};
std::atomic<bool> exit_on_cleanup{false};
}

namespace {
std::mutex resource_mutex;
std::unordered_set<void*> resources;
thread_local hf3fs_cqe pending;

void* create_resource() {
  auto* ptr = new char[4096];
  std::lock_guard<std::mutex> lock(resource_mutex);
  resources.insert(ptr);
  ++hf3fs_mock::created;
  return ptr;
}

void destroy_resource(void* ptr) {
  // The shutdown test stops at worker-local resource destruction, after
  // do_single_set has accessed the connector's members.
  if (hf3fs_mock::exit_on_cleanup) ::_exit(0);
  std::lock_guard<std::mutex> lock(resource_mutex);
  if (resources.erase(ptr) == 0) {
    ++hf3fs_mock::duplicate_destroys;
    return;
  }
  ++hf3fs_mock::destroyed;
  delete[] static_cast<char*>(ptr);
}
}

extern "C" {
int hf3fs_extract_mount_point(char* out, int size, const char*) {
  const auto& path = hf3fs_mock::mount_point;
  if (path.size() + 1 <= static_cast<size_t>(size)) {
    std::memcpy(out, path.c_str(), path.size() + 1);
  }
  return static_cast<int>(path.size() + 1);
}

int hf3fs_iorcreate4(hf3fs_ior* ring, const char*, int, bool, int, int, int,
                      uint64_t) {
  ring->iorh = create_resource();
  return 0;
}
void hf3fs_iordestroy(hf3fs_ior* ring) { destroy_resource(ring->iorh); }

int hf3fs_iovcreate(hf3fs_iov* iov, const char*, size_t size, size_t, int) {
  // Tests transfer at most a few bytes; do not allocate the production-sized
  // 100 MiB–2 GiB buffer for SDK contract tests.
  iov->base = static_cast<uint8_t*>(create_resource());
  iov->size = size;
  return 0;
}
void hf3fs_iovdestroy(hf3fs_iov* iov) { destroy_resource(iov->base); }
int hf3fs_reg_fd(int, uint64_t) { return 0; }
void hf3fs_dereg_fd(int) { ++hf3fs_mock::deregistered; }

int hf3fs_prep_io(const hf3fs_ior* ring, const hf3fs_iov* iov, bool read,
                    void* ptr, int fd, size_t offset, uint64_t length,
                    const void*) {
  {
    std::lock_guard<std::mutex> lock(resource_mutex);
    if (!resources.count(ring->iorh) || !resources.count(iov->base)) return -EFAULT;
  }
  pending.result = read ? ::pread(fd, ptr, length, offset)
                        : ::pwrite(fd, ptr, length, offset);
  if (pending.result < 0) pending.result = -errno;
  return 0;
}
int hf3fs_submit_ios(const hf3fs_ior*) { return 0; }
int hf3fs_wait_for_ios(const hf3fs_ior*, hf3fs_cqe* cqe, int, int,
                         const timespec*) {
  if (hf3fs_mock::hold_io) {
    hf3fs_mock::io_entered = true;
    while (!hf3fs_mock::io_released) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  *cqe = pending;
  return 1;
}
}
