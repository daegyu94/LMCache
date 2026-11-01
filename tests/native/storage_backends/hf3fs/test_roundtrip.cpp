// SPDX-License-Identifier: Apache-2.0
// Store, load, look up, and delete one object through Hf3fsConnector, with
// and without the key buffer, and check every SDK handle is released once.
#include "csrc/storage_backends/hf3fs/connector.h"
#include "hf3fs_mock.h"
#include "test_util.h"
#include <string>
#include <vector>

namespace {
std::vector<uint8_t> exists(lmcache::connector::Hf3fsConnector& client,
                            const std::string& key) {
  auto completion =
      wait_for_completion(client, client.submit_batch_exists({key}));
  require(completion.ok, "exists failed");
  return completion.result_bytes;
}
}  // namespace

int main(int argc, char** argv) {
  require(argc == 2, "pass an empty scratch directory");
  hf3fs_mock::mount_point = argv[1];
  const std::string key = "model@00000000@0@1234";
  for (bool key_buffer : {false, true}) {
    lmcache::connector::Hf3fsConnector client(argv[1], argv[1], 2, 256, 0, -1,
                                              104857600, 200, key_buffer);
    require(exists(client, key) == std::vector<uint8_t>{0},
            "absent key reported present");

    char payload[] = "test";
    auto set = wait_for_completion(
        client, client.submit_batch_set({key}, {payload}, {4}, 4));
    require(set.ok, "store failed");
    require(exists(client, key) == std::vector<uint8_t>{1},
            "stored key reported absent");

    char loaded[4]{};
    auto get = wait_for_completion(
        client, client.submit_batch_get({key}, {loaded}, {4}, 4));
    require(get.ok && get.result_bytes == std::vector<uint8_t>{1} &&
                std::string(loaded, 4) == "test",
            "load did not return the stored payload");

    auto del = wait_for_completion(client, client.submit_batch_delete({key}));
    require(del.ok && del.result_bytes == std::vector<uint8_t>{1},
            "delete failed");
    require(exists(client, key) == std::vector<uint8_t>{0},
            "deleted key reported present");
    client.close();
  }
  require(hf3fs_mock::created > 0, "workers created no SDK resources");
  require(hf3fs_mock::created == hf3fs_mock::destroyed,
          "connector leaked SDK resources");
  require(hf3fs_mock::duplicate_destroys == 0,
          "connector released SDK resources twice");
}
