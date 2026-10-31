// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <string>

namespace hf3fs_mock {
extern std::string mount_point;
extern std::atomic<int> created;
extern std::atomic<int> destroyed;
extern std::atomic<int> duplicate_destroys;
extern std::atomic<int> deregistered;
extern std::atomic<bool> hold_io;
extern std::atomic<bool> io_entered;
extern std::atomic<bool> io_released;
extern std::atomic<bool> exit_on_cleanup;
}
