// SPDX-License-Identifier: Apache-2.0

#include "connector.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace lmcache {
namespace connector {

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

std::string FSConnector::replace_all(const std::string& str,
                                     const std::string& from,
                                     const std::string& to) {
  std::string result = str;
  size_t pos = 0;
  while ((pos = result.find(from, pos)) != std::string::npos) {
    result.replace(pos, from.size(), to);
    pos += to.size();
  }
  return result;
}

std::string FSConnector::key_to_filename(const std::string& key) {
  // Input key format (from _object_key_to_string):
  //   Unsalted: <model_name>@<kv_rank_hex>@<chunk_hash_hex>
  //   Salted  : <model_name>@<kv_rank_hex>@<chunk_hash_hex>@<cache_salt>
  //
  // Output filename (matching fs_l2_adapter.py._object_key_to_filename):
  //   Unsalted: <model_name_safe>@0x<kv_rank_hex>@<chunk_hash_hex>.data
  //   Salted  :
  //   <model_name_safe>@0x<kv_rank_hex>@<chunk_hash_hex>@<cache_salt>.data
  //
  // The unsalted 3-field shape is bit-identical to the pre-cache_salt
  // format, so existing cache directories remain valid.
  //
  // NOTE: both model_name and cache_salt are forbidden from containing
  // '@' (invariant enforced on the Python side), so splitting on '@'
  // is unambiguous — no marker, no rsplit.

  // Split on '@' — must yield 3 (unsalted) or 4 (salted) fields.
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == KEY_SEP) {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() != 3 && parts.size() != 4) {
    throw std::runtime_error(
        "FSConnector: malformed key (expected 3 or 4 '@'-separated fields): " +
        key);
  }

  const std::string& model_name = parts[0];
  const std::string& kv_rank_hex = parts[1];
  const std::string& chunk_hash = parts[2];
  const std::string cache_salt = parts.size() == 4 ? parts[3] : std::string();

  // Replace '/' with '-SEP-' for filesystem safety
  std::string safe_model = replace_all(model_name, "/", PATH_SLASH_REPLACEMENT);

  // Emit filename. Salt is appended at the tail so the unsalted shape
  // matches what older builds wrote to disk.
  std::string result;
  result.reserve(safe_model.size() + kv_rank_hex.size() + chunk_hash.size() +
                 cache_salt.size() + 32);
  result += safe_model;
  result += KEY_SEP;
  result += "0x";
  result += kv_rank_hex;
  result += KEY_SEP;
  result += chunk_hash;
  if (!cache_salt.empty()) {
    result += KEY_SEP;
    result += cache_salt;
  }
  result += FILE_EXT;
  return result;
}

// ---------------------------------------------------------------
// read/write helpers
// ---------------------------------------------------------------

static void write_all(int fd, const void* data, size_t len) {
  size_t written = 0;
  const char* ptr = static_cast<const char*>(data);
  while (written < len) {
    ssize_t n = ::write(fd, ptr + written, len - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("write failed: " + std::string(strerror(errno)));
    }
    if (n == 0) {
      throw std::runtime_error("write returned 0");
    }
    written += static_cast<size_t>(n);
  }
}

static size_t read_all(int fd, void* buf, size_t len) {
  size_t total = 0;
  char* ptr = static_cast<char*>(buf);
  while (total < len) {
    ssize_t n = ::read(fd, ptr + total, len - total);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    }
    if (n == 0) break;  // EOF
    total += static_cast<size_t>(n);
  }
  return total;
}

static bool is_aligned_for_direct_io(const void* buffer, size_t alignment) {
  return alignment > 0 &&
         reinterpret_cast<std::uintptr_t>(buffer) % alignment == 0;
}

static size_t round_up_to_alignment(size_t value, size_t alignment) {
  if (alignment == 0) {
    throw std::runtime_error("O_DIRECT alignment is zero");
  }
  const size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const size_t padding = alignment - remainder;
  if (value > std::numeric_limits<size_t>::max() - padding) {
    throw std::runtime_error("O_DIRECT length alignment overflow");
  }
  return value + padding;
}

class AlignedBuffer {
 public:
  AlignedBuffer(size_t size, size_t alignment) {
    void* data = nullptr;
    const int rc = ::posix_memalign(&data, alignment, size);
    if (rc != 0) {
      throw std::runtime_error("posix_memalign failed: " +
                               std::string(std::strerror(rc)));
    }
    data_ = data;
  }

  ~AlignedBuffer() { std::free(data_); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  void* data() { return data_; }
  const void* data() const { return data_; }

 private:
  void* data_ = nullptr;
};

// ---------------------------------------------------------------
// FSConnector
// ---------------------------------------------------------------

FSConnector::FSConnector(std::string base_path, int num_workers,
                         std::string relative_tmp_dir, bool use_odirect,
                         size_t read_ahead_size)
    : ConnectorBase(num_workers),
      base_path_(std::move(base_path)),
      relative_tmp_dir_(std::move(relative_tmp_dir)),
      use_odirect_(use_odirect),
      disk_block_size_(0),
      read_ahead_size_(read_ahead_size) {
  // Create base directory
  std::filesystem::create_directories(base_path_);

  // Create tmp directory if configured
  if (!relative_tmp_dir_.empty()) {
    auto tmp_path = std::filesystem::path(base_path_) / relative_tmp_dir_;
    std::filesystem::create_directories(tmp_path);
  }

  // Query disk block size for O_DIRECT
  if (use_odirect_) {
#ifndef O_DIRECT
    throw std::runtime_error(
        "use_odirect=true but this platform does not define O_DIRECT");
#else
    struct statvfs st;
    if (statvfs(base_path_.c_str(), &st) != 0) {
      throw std::runtime_error("statvfs failed for O_DIRECT path " +
                               base_path_ + ": " + std::strerror(errno));
    }
    disk_block_size_ = st.f_bsize;
    if (disk_block_size_ == 0) {
      throw std::runtime_error(
          "filesystem reported a zero block size for O_DIRECT path " +
          base_path_);
    }
#endif
  }

  start_workers();  // IMPORTANT: call at END of constructor
}

FSConnector::~FSConnector() { close(); }

WorkerFSConn FSConnector::create_connection() {
  WorkerFSConn conn;
  conn.base_path = base_path_;
  if (!relative_tmp_dir_.empty()) {
    conn.tmp_dir = std::filesystem::path(base_path_) / relative_tmp_dir_;
  }
  conn.use_odirect = use_odirect_;
  conn.disk_block_size = disk_block_size_;
  conn.read_ahead_size = read_ahead_size_;
  return conn;
}

void FSConnector::do_single_get(WorkerFSConn& conn, const std::string& key,
                                void* buf, size_t len, size_t chunk_size) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;

  const bool direct_io = conn.use_odirect;
  const size_t io_len =
      direct_io ? round_up_to_alignment(len, conn.disk_block_size) : len;
  std::unique_ptr<AlignedBuffer> bounce;
  void* io_buf = buf;
  if (direct_io &&
      (io_len != len || !is_aligned_for_direct_io(buf, conn.disk_block_size))) {
    bounce = std::make_unique<AlignedBuffer>(io_len, conn.disk_block_size);
    io_buf = bounce->data();
  }

  int flags = O_RDONLY;
#ifdef O_DIRECT
  if (direct_io) {
    flags |= O_DIRECT;
  }
#endif

  int fd = ::open(file_path.c_str(), flags);
  if (fd < 0) {
    throw std::runtime_error("open for read failed: " + file_path.string() +
                             ": " + strerror(errno));
  }

  try {
    size_t n;
    if (!direct_io && conn.read_ahead_size > 0 && len > conn.read_ahead_size) {
      // Trigger filesystem readahead with a small initial
      // read, then read the remainder.
      size_t ra = conn.read_ahead_size;
      size_t n_head = read_all(fd, io_buf, ra);
      if (n_head < ra) {
        // Short read on the head portion — treat as
        // incomplete
        n = n_head;
      } else {
        size_t n_tail = read_all(fd, static_cast<char*>(io_buf) + ra, len - ra);
        n = n_head + n_tail;
      }
    } else {
      n = read_all(fd, io_buf, io_len);
    }
    if (n != io_len) {
      throw std::runtime_error("incomplete read for " + file_path.string() +
                               ": expected " + std::to_string(io_len) +
                               ", got " + std::to_string(n));
    }
    if (direct_io && io_buf != buf) {
      std::memcpy(buf, io_buf, len);
    }
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);
}

void FSConnector::do_single_set(WorkerFSConn& conn, const std::string& key,
                                const void* buf, size_t len,
                                size_t chunk_size) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;

  // Skip if already stored on disk
  if (std::filesystem::exists(file_path)) {
    return;
  }

  // Determine temp file path
  std::filesystem::path tmp_path;
  if (!conn.tmp_dir.empty()) {
    tmp_path = conn.tmp_dir / filename;
  } else {
    tmp_path = file_path;
    tmp_path.replace_extension(TMP_EXT);
  }

  const bool direct_io = conn.use_odirect;
  const size_t io_len =
      direct_io ? round_up_to_alignment(len, conn.disk_block_size) : len;
  std::unique_ptr<AlignedBuffer> bounce;
  const void* io_buf = buf;
  if (direct_io &&
      (io_len != len || !is_aligned_for_direct_io(buf, conn.disk_block_size))) {
    bounce = std::make_unique<AlignedBuffer>(io_len, conn.disk_block_size);
    std::memcpy(bounce->data(), buf, len);
    if (io_len > len) {
      std::memset(static_cast<char*>(bounce->data()) + len, 0, io_len - len);
    }
    io_buf = bounce->data();
  }

  int flags = O_CREAT | O_WRONLY | O_TRUNC;
#ifdef O_DIRECT
  if (direct_io) {
    flags |= O_DIRECT;
  }
#endif

  int fd = ::open(tmp_path.c_str(), flags, 0644);
  if (fd < 0) {
    throw std::runtime_error("open for write failed: " + tmp_path.string() +
                             ": " + strerror(errno));
  }

  try {
    write_all(fd, io_buf, io_len);
  } catch (...) {
    ::close(fd);
    // Clean up temp file on failure
    std::filesystem::remove(tmp_path);
    throw;
  }
  ::close(fd);

  // Atomic rename: tmp -> final
  std::error_code ec;
  std::filesystem::rename(tmp_path, file_path, ec);
  if (ec) {
    // Try to clean up, but prioritize reporting the original error.
    std::error_code remove_ec;
    std::filesystem::remove(tmp_path, remove_ec);
    throw std::runtime_error("rename failed: " + tmp_path.string() + " -> " +
                             file_path.string() + ": " + ec.message());
  }
}

bool FSConnector::do_single_exists(WorkerFSConn& conn, const std::string& key) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;
  return std::filesystem::exists(file_path);
}

bool FSConnector::do_single_delete(WorkerFSConn& conn, const std::string& key) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;
  std::error_code ec;
  return std::filesystem::remove(file_path, ec);
}

}  // namespace connector
}  // namespace lmcache
