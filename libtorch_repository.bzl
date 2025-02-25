def _libtorch_repository_rule_impl(repository_ctx):
    # Get LIBTORCH_URL from the environment, or use a default value
    libtorch_url = repository_ctx.os.environ.get(
        "LIBTORCH_URL",
        "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip"
    )

    if libtorch_url.endswith("/libtorch-shared-with-deps-%2B.zip") or "//" in libtorch_url:
        fail("LIBTORCH_URL is invalid: {}".format(libtorch_url))

    # Use the URL to fetch and set up the libtorch dependency
    repository_ctx.download_and_extract(
        url = libtorch_url,
        stripPrefix = "libtorch",
    )

    # Create a BUILD file for libtorch
    repository_ctx.file(
        path = repository_ctx.path("BUILD"),
        content = """
cc_library(
    name = "libtorch",
    srcs = glob(["lib/*.so*"]),
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
        """
    )

# Declare the custom repository rule
libtorch_repository = repository_rule(
    implementation = _libtorch_repository_rule_impl,
    environ = ["LIBTORCH_URL"],  # Declare the environment variable dependency
)

