package(default_visibility = ["//visibility:public"])

cc_library(
    name = "remote_gpu_extension",
    srcs = ["src/remote_gpu_extension.cpp"],
    hdrs = ["include/remote_gpu_extension.h"],
    deps = [
        "@libtorch//:libtorch",
    ],
	copts = ["-std=c++17"],
)

py_library(
    name = "remote_gpu_extension_py",
    srcs = ["python/remote_gpu_extension.py"],
    deps = [
        ":remote_gpu_extension",
    ],
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
