# spdlog.BUILD
cc_library(
    name = "spdlog",
    srcs = glob([
        "src/*.cpp",
    ]),
    hdrs = glob([
        "include/**/*.h",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    defines = [
        "SPDLOG_COMPILED_LIB",  # This is important!
        "SPDLOG_SHARED_LIB",    # This too for shared library
    ],
    copts = [
        "-fPIC",
    ],
)
