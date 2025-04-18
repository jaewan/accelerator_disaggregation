load("@rules_python//python:defs.bzl", "py_test")
load("@pip//:requirements.bzl", "requirement")

# Python Unit tests
# run "bazel test //tests:*"

py_test(
    name = "test_device",
    srcs = ["test_device.py"],
    deps = [
        "//:remote_cuda",
        requirement("numpy"),
    ],
    imports = [".."],  # Add parent directory to Python path
)

py_test(
    name = "test_remote_cuda",
    srcs = ["test_remote_cuda.py"],
    deps = [
        "//:remote_cuda",
        requirement("pytest"),
        requirement("numpy"),
    ],
    args = ["-v"],
    imports = [".."],
)

py_test(
    name = "test_real_workload",
    srcs = ["test_real_workload.py"],
    deps = [
        "//:remote_cuda",
        requirement("pytest"),
        requirement("numpy"),
        requirement("torchvision"),
    ],
    imports = [".."],
)

py_test(
    name = "test_simple_model",
    srcs = ["test_simple_model.py"],
    deps = [
        "//:remote_cuda",
        requirement("pytest"),
        requirement("numpy"),
    ],
    imports = [".."],
)