// SPDX-License-Identifier: Apache-2.0
#include "csrc/storage_backends/keys.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require_equal(const std::string& actual, const std::string& expected) {
  if (actual != expected) {
    throw std::runtime_error(
        "filename encoding differs from established bytes");
  }
}

void require_error(const std::string& key, bool sharded,
                   const std::string& expected) {
  try {
    lmcache::connector::key_to_filename(key, sharded);
  } catch (const std::runtime_error& error) {
    require_equal(error.what(), expected);
    return;
  }
  throw std::runtime_error("invalid key did not raise");
}

}  // namespace

int main() {
  using lmcache::connector::key_to_filename;
  using namespace std::string_literals;
  const std::array<std::pair<std::string, std::string>, 8> cases{{
      {"model@0000002a@3@deadbeef", "model@0x0000002a@3@deadbeef.data"},
      {"org/model@0000002a@3@deadbeef@tenant",
       "org-SEP-model@0x0000002a@3@deadbeef@tenant.data"},
      {"org/model@r@g@abcd@", "org-SEP-model@0xr@g@abcd.data"},
      {"组织/模型@r@g@abcd@租户", "组织-SEP-模型@0xr@g@abcd@租户.data"},
      {"//model/@r@g@abcd", "-SEP--SEP-model-SEP-@0xr@g@abcd.data"},
      {"@r@@abcd", "@0xr@@abcd.data"},
      {"m\0x@r@g@abcd"s, "m\0x@0xr@g@abcd.data"s},
      {"m-SEP-x@r@g@abcd", "m-SEP-x@0xr@g@abcd.data"},
  }};
  for (const auto& [key, expected] : cases) {
    require_equal(key_to_filename(key), expected);
    const std::string prefix =
        key.find("deadbeef") != std::string::npos ? "de/ad/" : "ab/cd/";
    require_equal(key_to_filename(key, true), prefix + expected);
  }

  const std::array<std::string, 9> malformed{{
      "", "model", "m@r", "m@r@g", "m@r@g@abcd@salt@extra",
      "m@r@g@abcd@@", "m@r@g@abcd@salt@", "@@@@@", "@@@@@@",
  }};
  for (const auto& key : malformed) {
    const std::string message =
        "FSConnector: malformed key (expected 4 or 5 '@'-separated fields): " +
        key;
    require_error(key, false, message);
    require_error(key, true, message);
  }

  for (const std::string hash : {"", "0", "01", "012"}) {
    const std::string key = "m@r@g@" + hash;
    require_equal(key_to_filename(key), "m@0xr@g@" + hash + ".data");
    require_error(key, true,
                  "shard_dirs requires a chunk hash of at least two bytes");
  }
  std::string source = "m@r@g@abcd";
  const std::string owned = key_to_filename(source);
  source.assign(10000, 'x');
  require_equal(owned, "m@0xr@g@abcd.data");
  require_equal(key_to_filename(std::string("m@r@g@abcd")), owned);
  std::cout << "PASS filename encoding: bytes, errors, sharding, ownership\n";
}
