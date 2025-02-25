package(default_visibility = ["//visibility:public"])

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library")

cc_library(
    name = "remote_gpu_extension",
    srcs = ["src/remote_gpu_extension.cc"],
    hdrs = ["include/remote_gpu_extension.h"],
    strip_include_prefix = "include",
    deps = [
        "@libtorch//:libtorch",
        "@local_config_python//:python_headers",  # Add Python headers for Libtorch
    ],
    copts = [
        "-std=c++17",
        "-D_GLIBCXX_USE_CXX11_ABI=0",	# Forces old C++11 ABI for compatibility with pre-built PyTorch binaries
										# PyTorch is typically built with this flag, so we need to match it
		 "-fPIC",
        "-DTORCH_API_INCLUDE_EXTENSION_H", #Tells compiler this is a Pytorch extension
        "-DTORCH_EXTENSION_NAME=remote_gpu_extension",# Stes python module name Used when importing the module in Python
    ],
    linkopts = ["-pthread"],
)

pybind_extension(
    name = "remote_gpu_extension_binding",
    srcs = ["src/python_bindings.cc"],
    deps = [
        ":remote_gpu_extension",  # Inherit Python headers from remote_gpu_extension
        "@libtorch//:libtorch",
    ],
    copts = [
        "-std=c++17",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
)

py_library(
    name = "remote_gpu_extension_py",
    srcs = ["python/remote_gpu_extension.py"],
    data = [":remote_gpu_extension_binding.so"],
)

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
