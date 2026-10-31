// SPDX-License-Identifier: Apache-2.0
#include "csrc/storage_backends/fs/connector.h"
#include "completion.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
  require(argc == 2, "pass an empty scratch directory");
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  const std::string key = "model@00000000@0@0102";
  const auto path = root / "model@0x00000000@0@0102.data";
  for (int readers : {0, 2}) {
    for (size_t readahead : {size_t{0}, size_t{2}}) {
      lmcache::connector::FSConnector client(root.string(), 1, "", false,
                                             readahead, readers);
      for (size_t size : {size_t{5}, size_t{3}, size_t{4}}) {
        {
          std::ofstream file(path, std::ios::binary | std::ios::trunc);
          file << std::string(size, 'x');
        }
        std::array<char, 4> destination{'?', '?', '?', '?'};
        const auto id = client.submit_batch_get(
            {key}, {destination.data()}, {destination.size()},
            destination.size());
        const auto result = wait_for_completion(client, id);
        require(result.result_bytes.size() == 1, "missing per-key result");
        require(bool(result.result_bytes[0]) == (size == destination.size()),
                "wrong-sized file was accepted or exact-sized file rejected");
        if (size == destination.size()) {
          require(std::string(destination.data(), destination.size()) == "xxxx",
                  "loaded payload differs");
        } else {
          require(std::string(destination.data(), destination.size()) == "????",
                  "wrong-sized file modified the destination");
        }
      }
      client.close();
    }
  }
}
