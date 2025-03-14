load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

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
