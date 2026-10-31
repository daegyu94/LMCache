// SPDX-License-Identifier: Apache-2.0
#include "csrc/storage_backends/hf3fs/connector.h"
#include "completion.h"
#include "hf3fs_mock.h"
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <string>
#include <utility>

int main(int argc, char** argv) {
  require(argc == 2, "pass an existing mount directory");
  const int before = hf3fs_mock::destroyed;
  const int fd = ::open("/dev/null", O_RDONLY);
  require(fd >= 0, "open failed");
  {
    lmcache::connector::WorkerHf3fsConn source;
    require(hf3fs_iorcreate4(&source.read_ior, "/mock", 256, true, 0, 200,
                              -1, 0) == 0, "ring initialization failed");
    source.read_ior_initialized = true;
    require(hf3fs_iovcreate(&source.read_iov, "/mock", 4096, 0, -1) == 0,
            "buffer initialization failed");
    source.read_iov_initialized = true;
    require(hf3fs_iorcreate4(&source.write_ior, "/mock", 256, false, 0, 200,
                              -1, 0) == 0, "write ring initialization failed");
    source.write_ior_initialized = true;
    require(hf3fs_iovcreate(&source.write_iov, "/mock", 4096, 0, -1) == 0,
            "write buffer initialization failed");
    source.write_iov_initialized = true;
    source.fd = fd;
    require(hf3fs_reg_fd(fd, 0) <= 0, "fd registration failed");
    source.registered = true;
    source.file_path = "/mock/object";
    {
      auto destination = std::move(source);
      require(hf3fs_mock::destroyed == before, "move released owned resources");
      require(destination.fd == fd && destination.registered &&
                  destination.file_path == "/mock/object",
              "move lost file state");
    }
    require(hf3fs_mock::destroyed == before + 4,
            "destination did not release transferred resources");
    require(::fcntl(fd, F_GETFD) == -1 && errno == EBADF,
            "destination did not close transferred fd");
  }
  require(hf3fs_mock::duplicate_destroys == 0,
          "source destructor released transferred resources again");
  require(hf3fs_mock::deregistered == 1, "fd deregistered more than once");
  // A partially initialized connection also has to be movable.
  {
    lmcache::connector::WorkerHf3fsConn empty;
    auto destination = std::move(empty);
  }
  // With -fno-elide-constructors this exercises create_connection's actual
  // named return; the worker must retain live handles after that return.
  hf3fs_mock::mount_point = argv[1];
  {
    lmcache::connector::Hf3fsConnector client(argv[1], argv[1], 1, 256, 0,
                                              -1, 104857600, 200, false);
    const std::string key = "model@00000000@0@1234";
    char payload[] = "test";
    auto id = client.submit_batch_set({key}, {payload}, {4}, 4);
    require(wait_for_completion(client, id).ok, "store used invalid SDK handles");
    char destination[4]{};
    id = client.submit_batch_get({key}, {destination}, {4}, 4);
    auto completion = wait_for_completion(client, id);
    require(completion.result_bytes.size() == 1 && completion.result_bytes[0] &&
                std::string(destination, 4) == "test",
            "load used invalid SDK handles");
    client.close();
  }
  require(hf3fs_mock::created == hf3fs_mock::destroyed,
          "worker leaked SDK resources");
  require(hf3fs_mock::duplicate_destroys == 0,
          "worker released SDK resources twice");
}
