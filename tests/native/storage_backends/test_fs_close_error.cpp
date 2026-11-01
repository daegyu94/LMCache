// SPDX-License-Identifier: Apache-2.0
#include "csrc/storage_backends/fs/connector.h"
#include <poll.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

static std::atomic<bool> fail_close{false};
extern "C" int __real_close(int fd);
extern "C" int __wrap_close(int fd) {
  const int flags = fcntl(fd, F_GETFL);
  const bool inject = flags >= 0 && (flags & O_ACCMODE) == O_WRONLY &&
                      fail_close.exchange(false);
  const int result = __real_close(fd);
  if (inject) {
    errno = ENOSPC;
    return -1;
  }
  return result;
}

static void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

static lmcache::connector::Completion wait(
    lmcache::connector::FSConnector& client, uint64_t id) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    for (auto& result : client.drain_completions()) {
      require(result.future_id == id, "unexpected completion");
      return result;
    }
    pollfd pfd{client.event_fd(), POLLIN, 0};
    ::poll(&pfd, 1, 10);
  }
  throw std::runtime_error("completion timed out");
}

int main(int argc, char** argv) {
  require(argc == 2, "pass a scratch directory");
  for (const std::string tmp : {std::string(), std::string("staging")}) {
    const auto root = std::filesystem::path(argv[1]) /
                      (tmp.empty() ? "default" : "separate");
    lmcache::connector::FSConnector client(root.string(), 1, tmp);
    const std::string key = "model@00000000@0@deadbeef";
    std::string payload = "test";
    fail_close = true;
    auto result = wait(client, client.submit_batch_set(
        {key}, {payload.data()}, {payload.size()}, payload.size()));
    require(!result.ok, "close error incorrectly reported store success");
    require(result.error.find("close after write failed") != std::string::npos,
            "close failure reason was lost");
    require(!std::filesystem::exists(root / "model@0x00000000@0@deadbeef.data"),
            "failed store published a final file");
    const auto staging = tmp.empty() ? root : root / tmp;
    require(std::filesystem::is_empty(staging), "failed store leaked a temp file");
    result = wait(client, client.submit_batch_set(
        {key}, {payload.data()}, {payload.size()}, payload.size()));
    require(result.ok, "store did not recover after a failed close");
    client.close();
  }
}
