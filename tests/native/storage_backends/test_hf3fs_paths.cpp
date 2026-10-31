// SPDX-License-Identifier: Apache-2.0
#include "csrc/storage_backends/hf3fs/connector.h"
#include "completion.h"
#include "hf3fs_mock.h"
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
  require(argc == 2, "pass a scratch directory");
  const std::filesystem::path root(argv[1]);
  const auto mount = root / "mount";
  const auto outside = root / "mount-other";
  std::filesystem::create_directories(mount / "inside");
  std::filesystem::create_directories(outside);
  std::filesystem::create_directory_symlink(outside, mount / "escape");
  hf3fs_mock::mount_point = mount.string();
  for (const auto& path : {outside, mount / ".." / "mount-other",
                           mount / "escape"}) {
    bool rejected = false;
    try {
      lmcache::connector::Hf3fsConnector client(mount.string(), path.string(),
                                                1, 256, 0, -1, 104857600, 200,
                                                false);
      client.close();
    } catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find("not a subdirectory") !=
                 std::string::npos;
    }
    require(rejected, "base path outside the mount was accepted");
  }
  for (const auto& path : {mount, mount / "inside" / ".." / "inside"}) {
    lmcache::connector::Hf3fsConnector client(mount.string(), path.string(),
                                              1, 256, 0, -1, 104857600, 200,
                                              false);
    client.close();
  }
}
