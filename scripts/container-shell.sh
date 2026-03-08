#!/usr/bin/env bash
# Run this to get an interactive shell inside the dev container with GPU access.
# Requires Docker (or Podman) with the compose plugin installed.
set -euo pipefail

# Navigate to project root (parent of scripts/)
cd "$(dirname "$0")/.."

# Prefer docker compose; fall back to podman compose if docker is not found
if command -v docker &>/dev/null; then
    docker compose run --rm dev bash
elif command -v podman &>/dev/null; then
    podman compose run --rm dev bash
else
    echo "Error: Neither docker nor podman found. Install one to use the dev container." >&2
    exit 1
fi
