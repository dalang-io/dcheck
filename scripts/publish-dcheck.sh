#!/bin/bash
# Publish the dcheck release channel on https://wayang.dalang.io/dcheck/:
# /dcheck/install.sh, /dcheck/LATEST and /dcheck/v<ver>/{tarballs,SHA256SUMS}.
#
# Usage: ./scripts/publish-dcheck.sh
# Env:
#   HOST             ssh target (default: root@10.0.0.251)
#   REMOTE_DIR       served directory (default: /root/wayang.dalang.io/public)
#   RELEASE_DIR      prebuilt artifacts (default: build with release-dcheck.sh)
#   FORCE_REPUBLISH  1 to overwrite an already published version
#
# The landing page is deployed separately from the wayangos repo
# (scripts/deploy-site.sh there), which leaves /dcheck/ alone.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-root@10.0.0.251}"
REMOTE_DIR="${REMOTE_DIR:-/root/wayang.dalang.io/public}"
VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$REPO_DIR/Cargo.toml" | head -1)"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

RELEASE_DIR="${RELEASE_DIR:-}"
if [ -z "$RELEASE_DIR" ]; then
    RELEASE_DIR="$STAGE/release"
    OUT_DIR="$RELEASE_DIR" "$REPO_DIR/scripts/release-dcheck.sh"
fi
# Never publish a LATEST that some platform cannot download.
required="x86_64-unknown-linux-musl aarch64-unknown-linux-musl"
[ "$(uname -s)" = "Darwin" ] && required="$required aarch64-apple-darwin x86_64-apple-darwin"
for target in $required; do
    pkg="$RELEASE_DIR/dcheck-$VERSION-$target.tar.gz"
    [ -f "$pkg" ] || { echo "ERROR: missing $pkg — not publishing" >&2; exit 1; }
done

echo "=== staging dcheck $VERSION ==="
CHANNEL="$STAGE/dcheck"
mkdir -p "$CHANNEL/v$VERSION"
cp "$RELEASE_DIR"/dcheck-"$VERSION"-*.tar.gz "$RELEASE_DIR/SHA256SUMS" "$CHANNEL/v$VERSION/"
cp "$REPO_DIR/install.sh" "$CHANNEL/install.sh"
printf '%s\n' "$VERSION" > "$CHANNEL/LATEST"

echo "=== uploading to $HOST:$REMOTE_DIR/dcheck ==="
# Releases are immutable: Cloudflare caches the tarballs (max-age 4 h, per
# PoP), so re-uploading a version makes clients see stale bytes and fail the
# checksum. Bump the version instead (FORCE_REPUBLISH=1 to override).
# shellcheck disable=SC2029 # the paths are meant to expand locally
if [ -z "${FORCE_REPUBLISH:-}" ] && ssh "$HOST" "test -e '$REMOTE_DIR/dcheck/v$VERSION'"; then
    echo "ERROR: dcheck $VERSION is already published — bump the version in Cargo.toml" >&2
    exit 1
fi
# shellcheck disable=SC2029
ssh "$HOST" "mkdir -p '$REMOTE_DIR/dcheck'"
# No --delete: older releases stay (clients may pin DCHECK_VERSION).
rsync -rlptz --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r "$CHANNEL/" "$HOST:$REMOTE_DIR/dcheck/"
echo "done."
