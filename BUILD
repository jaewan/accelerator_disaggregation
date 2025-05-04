load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_proto_grpc//cpp:defs.bzl", "cpp_proto_library", "cpp_grpc_library")


package(
    default_visibility = ["//visibility:public"],
    features = ["cpp17"],
)

config_setting(
    name = "use_cpp17",
    values = {"cpp_version": "c++17"},
)

# Proto definitions
proto_library(
    name = "remote_proto",
    srcs = ["proto/remote.proto"],
)

proto_library(
    name = "remote_execution_proto",
    srcs = ["proto/remote_execution.proto"],
)

cpp_proto_library(
    name = "remote_execution_cc_proto",
    protos = [":remote_execution_proto"],
)

cpp_grpc_library(
    name = "remote_execution_grpc_cc",
    protos = [":remote_execution_proto"],
)

# genrule(
#     name = "remote_execution_grpc_gen",
#     srcs = ["proto/remote_execution.proto"],
#     outs = [
#         "remote_execution.pb.h",
#         "remote_execution.pb.cc",
#         "remote_execution.grpc.pb.h", 
#         "remote_execution.grpc.pb.cc"
#     ],
#     cmd = "protoc --plugin=protoc-gen-grpc=/usr/bin/grpc_cpp_plugin --grpc_out=$(GENDIR) --cpp_out=$(GENDIR) --proto_path=proto $(SRCS)",
# )

# cc_library(
#     name = "remote_execution_grpc_cc",
#     srcs = [
#         "remote_execution.pb.cc",
#         "remote_execution.grpc.pb.cc"
#     ],
#     hdrs = [
#         "remote_execution.pb.h",
#         "remote_execution.grpc.pb.h"
#     ],
#     deps = ["@com_github_grpc_grpc//:grpc++"], 
# )

# C++ core libraries
cc_library(
    name = "remote_device_lib",
    srcs = ["csrc/remote_device.cc"],
    hdrs = ["csrc/remote_device.h"],
    copts = [
        "-std=c++17",
        "-fPIC",
        "-D_GLIBCXX_USE_CXX11_ABI=0",  # Match PyTorch ABI
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=remote_cuda_ext",
    ],
    deps = [
        "@libtorch",
		"@spdlog//:spdlog",
    ],
    includes = [
        "@libtorch//:include",
        "@libtorch//:include/torch/csrc/api/include",
    ],
	features = ["cpp17"],
)

cc_library(
    name = "remote_dispatch_lib",
    srcs = ["csrc/remote_dispatch.cc"],
    hdrs = ["csrc/remote_dispatch.h"],
    deps = [
        ":remote_execution_grpc_cc",
        ":remote_execution_cc_proto",
        ":remote_device_lib",
        "@libtorch",
		"@spdlog//:spdlog",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

# Python extension module
pybind_extension(
    name = "remote_cuda_ext",
    srcs = ["csrc/python_bindings.cc"],
    deps = [
        ":remote_device_lib",
        ":remote_dispatch_lib",
        "@libtorch",
        "@spdlog//:spdlog",
    ],
    copts = [
        "-std=c++17",
        "-fPIC",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-DSPDLOG_COMPILED_LIB",
    ],
    features = ["cpp17"],
)

# Python package
py_library(
    name = "remote_cuda",
    srcs = glob(["remote_cuda/*.py"]),
    data = [":remote_cuda_ext.so"],
    imports = ["."],
    deps = [
        requirement("torch"),
    ],
)

# Example application
py_binary(
    name = "example",
    srcs = ["example.py"],
    deps = [
        ":remote_cuda",
        requirement("torch"),
    ],
)

load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "benchmark_library",
    srcs = [
        "benchmark/server-runner.cu",
        "benchmark/common.h"]
)

cc_binary(
    name = "benchmark_server",
    srcs = [
        "benchmark/server.cc",
        "benchmark/common.cc",
        "benchmark/common.h",
        "benchmark/server-runner.h"],
    #copts = ["-x cuda"],
    linkopts = ["-lcudart"],
    deps = ["benchmark_library"]
)

cc_binary(
    name = "benchmark_client",
    srcs = [
        "benchmark/client.cc",
        "benchmark/common.cc",
        "benchmark/common.h"]
)