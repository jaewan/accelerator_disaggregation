load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _detect_cuda_version(repository_ctx):
    result = repository_ctx.execute(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if result.return_code != 0:
        return "cpu"
    
    cuda_version = result.stdout.strip().split(".")[0]
    cuda_version = int(cuda_version)
    
    if cuda_version >= 525:
        return "cu121"
    elif cuda_version >= 510:
        return "cu118"
    elif cuda_version >= 450:
        return "cu116"
    return "cpu"

def _libtorch_repository_impl(repository_ctx):
    cuda_setting = repository_ctx.attr.cuda
    torch_version = repository_ctx.attr.torch_version

    if cuda_setting == "auto":
        cuda_tag = _detect_cuda_version(repository_ctx)
    else:
        cuda_tag = cuda_setting

    # Construct download URL
    if cuda_tag == "cpu":
        url = "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-{version}%2Bcpu.zip".format(
            version = torch_version
        )
    else:
        url = "https://download.pytorch.org/libtorch/{cuda}/libtorch-shared-with-deps-{version}%2B{cuda}.zip".format(
            cuda = cuda_tag,
            version = torch_version
        )

    # Download and extract libtorch
    repository_ctx.download_and_extract(
        url = url,
        sha256 = "", # You should add proper sha256 for reproducibility
        stripPrefix = "libtorch",
    )

    # Create BUILD file
    repository_ctx.file("BUILD", """
cc_library(
    name = "libtorch",
    srcs = glob([
        "lib/*.so*",
        "lib/*.dylib*",
    ]),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.cuh",
        "include/**/*.inc",
        "include/**/*.inl",
    ]),
    includes = [
        "include",
        "include/torch",
        "include/torch/csrc",
        "include/torch/csrc/api/include",
        "include/torch/csrc/api",
        "include/torch/csrc/utils",
    ],
    deps = ["@local_config_python//:python_headers"],
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-fPIC",
    ],
    visibility = ["//visibility:public"],
    linkstatic = 1,
)
""")

libtorch_repository = repository_rule(
    implementation = _libtorch_repository_impl,
    attrs = {
        "cuda": attr.string(default = "auto"),
        "torch_version": attr.string(default = "2.5.1"),
    },
)
