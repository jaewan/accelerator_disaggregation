# third_party/libtorch.BUILD
cc_library(
    name = "libtorch",
    srcs = glob([
        "lib/*.so",
        "lib/*.a",
    ]),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
