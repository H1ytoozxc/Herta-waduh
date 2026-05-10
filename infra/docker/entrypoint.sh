#!/usr/bin/env sh
# Container entrypoint. Passes all CLI arguments straight to `herta`.
# Exists as a seam so operators can insert init logic (e.g. secret fetching)
# without rebuilding the image.
set -eu

exec /usr/local/bin/herta "$@"
