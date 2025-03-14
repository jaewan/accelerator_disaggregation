load("@rules_python//python:defs.bzl", "py_test")

# Python Unit tests
# run "bazel test //:test_device"
py_test(
    name = "test_device",
    srcs = ["test_device.py"],
    deps = [
        "//:remote_cuda",
    ],
    imports = [".."],  # Add parent directory to Python path
)
