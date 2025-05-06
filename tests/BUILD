load("@rules_python//python:defs.bzl", "py_test")
load("@pip//:requirements.bzl", "requirement")

# Python Unit tests
# run "bazel test //tests:*"

py_test(
    name = "test_device",
    srcs = ["test_device.py"],
    data = ["//:remote_execution_server"],
    env = {"SERVER_BIN": "$(location //:remote_execution_server)"},
    deps = [
        "//:remote_cuda",
        requirement("numpy"),
    ],
    imports = [".."],  # Add parent directory to Python path
)

py_test(
    name = "test_remote_cuda",
    srcs = ["test_remote_cuda.py"],
    data = ["//:remote_execution_server"],
    env = {"SERVER_BIN": "$(location //:remote_execution_server)"},
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
    data = ["//:remote_execution_server"],
    env = {"SERVER_BIN": "$(location //:remote_execution_server)"},
    deps = [
        "//:remote_cuda",
        requirement("pytest"),
        requirement("numpy"),
        requirement("torchvision"),
    ],
    imports = [".."],
)

py_test(
    name = "test_simple_model_inference",
    srcs = ["test_simple_model_inference.py"],
    data = ["//:remote_execution_server"],
    env = {"SERVER_BIN": "$(location //:remote_execution_server)"},
    deps = [
        "//:remote_cuda",
        requirement("pytest"),
        requirement("numpy"),
    ],
    imports = [".."],
)

py_test(
    name = "test_simple_model_train",
    srcs = ["test_simple_model_train.py"],
    data = ["//:remote_execution_server"],
    env = {"SERVER_BIN": "$(location //:remote_execution_server)"},
    deps = [
        "//:remote_cuda",
        requirement("pytest"),
        requirement("numpy"),
    ],
    imports = [".."],
)

py_test(
    name = "test_bert_inference",
    srcs = ["test_bert_inference.py"],
    data = ["//:remote_execution_server"],
    env = {"SERVER_BIN": "$(location //:remote_execution_server)"},
    deps = [
        "//:remote_cuda",
        requirement("pytest"),
        requirement("numpy"),
        requirement("transformers"),
    ],
    imports = [".."],
)