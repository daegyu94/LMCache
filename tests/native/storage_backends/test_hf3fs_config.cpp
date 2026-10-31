// SPDX-License-Identifier: Apache-2.0
#include "csrc/storage_backends/hf3fs/connector.h"
#include "completion.h"
#include "hf3fs_mock.h"
#include <string>

int main(int argc, char** argv) {
  require(argc == 2, "pass an existing mount directory");
  hf3fs_mock::mount_point = argv[1];
  for (int entries : {127, 1025}) {
    bool rejected = false;
    try {
      lmcache::connector::Hf3fsConnector client(argv[1], argv[1], 1, entries);
      client.close();
    } catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find("ior_entries") != std::string::npos;
    }
    require(rejected, "out-of-range ring entry count accepted");
  }
  for (int depth : {-129, 129}) {
    bool rejected = false;
    try {
      lmcache::connector::Hf3fsConnector client(argv[1], argv[1], 1, 256, depth);
      client.close();
    } catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find("io_depth") != std::string::npos;
    }
    require(rejected, "out-of-range I/O depth accepted");
  }
  for (size_t size : {size_t{0}, size_t{104857599}, size_t{2147483649}}) {
    bool rejected = false;
    try {
      lmcache::connector::Hf3fsConnector client(argv[1], argv[1], 1, 256, 0,
                                                -1, size);
      client.close();
    } catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find("iov_size") != std::string::npos;
    }
    require(rejected, "out-of-range I/O buffer size accepted");
  }
  require(hf3fs_mock::created == 0, "invalid config allocated SDK resources");
  for (int entries : {128, 1024}) {
    for (int depth : {-128, 128}) {
      for (size_t size : {size_t{104857600}, size_t{2147483648}}) {
        lmcache::connector::Hf3fsConnector client(argv[1], argv[1], 1, entries,
                                                  depth, -1, size, 200, false);
        client.close();
      }
    }
  }
  require(hf3fs_mock::created == hf3fs_mock::destroyed,
          "accepted config leaked SDK resources");
  require(hf3fs_mock::duplicate_destroys == 0,
          "accepted config destroyed resources twice");
}
