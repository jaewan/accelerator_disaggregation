package(default_visibility = ["//visibility:public"])

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library")


cc_library(
    name = "remote_gpu_extension",
    srcs = ["src/remote_gpu_extension.cpp"],
    hdrs = ["include/remote_gpu_extension.h"],
    deps = [
        "@libtorch//:libtorch",
    ],
    copts = ["-std=c++17"],
)

pybind_extension(
    name = "remote_gpu_extension_binding",
    srcs = ["src/python_bindings.cpp"],
    deps = [
        ":remote_gpu_extension",
        "@libtorch//:libtorch",
    ],
)

py_library(
    name = "remote_gpu_extension_py",
    srcs = ["python/remote_gpu_extension.py"],
    data = [":remote_gpu_extension_binding.so"],
)


#     For example pytorch
py_binary(
    name = "basic_example",
    srcs = ["examples/basic_example.py"],
    deps = [
        ":remote_gpu_extension_py",
    ],
)

py_binary(
    name = "remote_server",
    srcs = ["python/remote_server.py"],
)
