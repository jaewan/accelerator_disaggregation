load("@rules_cc//cc:defs.bzl", "cc_library")

cc_binary(
    name = "dpdk_test",
    srcs = ["dpdk_test.cc"],
    deps = ["@dpdk//:dpdk_lib"],
    copts = ["-std=c++17"],
)
