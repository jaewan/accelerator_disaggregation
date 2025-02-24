# WORKSPACE
workspace(name = "accelerator_disaggregation") #TODO change this once this project is named

# Load Bazel rules for Python

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


http_archive(
    name = "rules_python",
    urls = ["https://github.com/bazelbuild/rules_python/releases/download/0.24.0/rules_python-0.24.0.tar.gz"],
    sha256 = "0019dfc4b32d63c1392aa264aed2253c1e0c2fb09216f8e2cc269bbfb8bb49b5",
)

# Load Python dependencies correctly
load("@rules_python//python:repositories.bzl", "rules_python_dependencies", "python_register_toolchains")
rules_python_dependencies()
python_register_toolchains()



http_archive(
    name = "rules_cc",
    sha256 = "a2c1b3a0d11f8929c5b28a25276a812b0566126b5e2e4a7a13c4d160a07d56a9",
    strip_prefix = "rules_cc-0.0.7",
    urls = [
        "https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.0.7.tar.gz",
    ],
)



# Load gRPC (if needed)
#http_archive(
#    name = "com_github_grpc_grpc",
#    sha256 = "e1353f1f9cf9b9e93011a3c813a20dbd6a9995e768b3b163fa1d3e0a41bb3b95",
#    strip_prefix = "grpc-1.41.0",
#    urls = [
#        "https://github.com/grpc/grpc/archive/refs/tags/v1.41.0.tar.gz",
#    ],
#)

# Load additional Bazel dependencies if required
http_archive(
    name = "bazel_skylib",
    sha256 = "6e82e61ccbd1cdb9b5e980b9a2858c2a53f13a8d20f2f0e289d563d08f669779",
    strip_prefix = "bazel-skylib-1.0.3",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/archive/refs/tags/1.0.3.tar.gz",
    ],
)
