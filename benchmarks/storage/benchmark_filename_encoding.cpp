// SPDX-License-Identifier: Apache-2.0
// Build the same benchmark against baseline and candidate checkouts:
// c++ -std=c++17 -O2 -I. benchmarks/storage/benchmark_filename_encoding.cpp
// Results cover CPU filename encoding only, without file I/O or serving.
#include "csrc/storage_backends/keys.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>

namespace {
bool count_allocations = false;
size_t allocation_count = 0;
}  // namespace

void* operator new(std::size_t size) {
  if (count_allocations) ++allocation_count;
  if (void* pointer = std::malloc(size == 0 ? 1 : size)) return pointer;
  throw std::bad_alloc();
}

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

int main() {
  std::vector<std::string> keys;
  for (const std::string model :
       {"m", "meta-llama/Llama-3.1-8B-Instruct", "组织/模型"}) {
    for (const std::string salt : {"", "@tenant-01", "@"}) {
      for (size_t length : {8, 64, 128}) {
        keys.push_back(model + "@00000000@0@" + std::string(length, 'a') +
                       salt);
      }
    }
  }
  constexpr size_t iterations = 200000;
  size_t checksum = 0;
  std::cout << "[";
  for (int round = 0; round < 9; ++round) {
    allocation_count = 0;
    count_allocations = true;
    for (const auto& key : keys) {
      checksum += lmcache::connector::key_to_filename(key).size();
    }
    count_allocations = false;
    const double allocations_per_key =
        static_cast<double>(allocation_count) / keys.size();
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
      checksum +=
          lmcache::connector::key_to_filename(keys[i % keys.size()]).size();
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             end - start)
                             .count();
    if (round != 0) std::cout << ',';
    std::cout << "{\"round\":" << round
              << ",\"ns_per_key\":" << static_cast<double>(elapsed) / iterations
              << ",\"allocations_per_key\":" << allocations_per_key << '}';
  }
  std::cout << "]\n";
  return checksum == 0;
}
