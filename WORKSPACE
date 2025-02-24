# WORKSPACE
workspace(name = "accelerator_disaggregation")

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Bazel Skylib
http_archive(
    name = "bazel_skylib",
    sha256 = "6e82e61ccbd1cdb9b5e980b9a2858c2a53f13a8d20f2f0e289d563d08f669779",
    strip_prefix = "bazel-skylib-1.0.3",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/refs/tags/1.0.3.tar.gz"],
)

# Python rules
http_archive(
    name = "rules_python",
    sha256 = "94750828b18044533e98a129003b6a68001204038dc4749f40b195b24c38f49f",
    strip_prefix = "rules_python-0.21.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.21.0/rules_python-0.21.0.tar.gz",
)

# Python toolchain setup
load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python3_8",
    python_version = "3.8",
)

# Pybind11 setup
http_archive(
    name = "pybind11_bazel",
    sha256 = "fec6281e4109115c5157ca720b8fe20c8f655f773172290b03f57353c11869c2",
    strip_prefix = "pybind11_bazel-72cbbf1fbc830e487e3012862b7b720001b70672",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/72cbbf1fbc830e487e3012862b7b720001b70672.zip"],
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    sha256 = "5d8c4c5dda428d3a944ba3d2a5212cb988c2fae4670d58075a5a49075a6ca315",
    strip_prefix = "pybind11-2.10.3",
    urls = ["https://github.com/pybind/pybind11/archive/v2.10.3.tar.gz"],
)

# Configure pybind11
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

# C++ rules
http_archive(
    name = "rules_cc",
    sha256 = "eb389b5b74862a3d310ee9d6c63348388223b384ae4423ff0fd286fcd123942d",
    strip_prefix = "rules_cc-0.0.7",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.0.7.tar.gz"],
)

# Libtorch - using local_repository instead of new_local_repository
local_repository(
    name = "libtorch",
    path = "libtorch",
)
