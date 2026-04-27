include(ExternalProject)

ExternalProject_Add(
    ext_eigen
    PREFIX eigen
    # Commit point: https://gitlab.com/libeigen/eigen/-/merge_requests/716
    # GitHub mirror (same commit) avoids GitLab HTTP 403 / Cloudflare blocks for curl.
    URL https://github.com/eigenteam/eigen-git-mirror/archive/da7909592376c893dabbc4b6453a8ffe46b1eb8e.tar.gz
    URL_HASH SHA256=6fd3e995fb6b0bfc6300fb680aad7610eb5246fb8f9098e647a6a7dca7297352
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/eigen"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_eigen SOURCE_DIR)
set(EIGEN_INCLUDE_DIRS ${SOURCE_DIR}/Eigen)
