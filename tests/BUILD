load("@rules_python//python:defs.bzl", "py_test")
load("@pip//:requirements.bzl", "requirement")

# Python Unit tests
# run "bazel test //tests:test_device"
py_test(
    name = "test_device",
    srcs = ["test_device.py"],
    deps = [
        "//:remote_cuda",
        requirement("numpy"),
    ],
    imports = [".."],  # Add parent directory to Python path
)
