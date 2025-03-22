cc_binary(
    name = "dpdk-server",
    srcs = ["src/server.c"],
    deps = [
        "@dpdk//:dpdk",
        "@dpdk_config//:dpdk_config",
        "@dpdk_arch//:dpdk_arch",
        "@dpdk_build//:dpdk_build",
    ],
    copts = [
        "-O3",
        "-g",
        "-Wall",
        "-Wextra",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "dpdk-client",
    srcs = ["src/client.c"],
    deps = [
        "@dpdk//:dpdk",
        "@dpdk_config//:dpdk_config",
        "@dpdk_arch//:dpdk_arch",
        "@dpdk_build//:dpdk_build",
    ],
) 
