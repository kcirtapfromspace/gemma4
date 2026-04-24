#!/usr/bin/env bash
# Build the LiteRtLmCore.xcframework from an upstream LiteRT-LM checkout.
#
# Usage:
#   LITERTLM_UPSTREAM=/tmp/c11-litertlm/LiteRT-LM ./scripts/build_xcframework.sh
#
# Requirements:
#   - bazelisk on PATH (Bazel 7.6.1 resolved via .bazelversion)
#   - Xcode 15+ (CMake is NOT required; pure Bazel build)
#   - ~30 GB free disk for the Bazel cache + LiteRT / TensorFlow downloads
#
# Output:
#   apps/mobile/litertlm-swift/build/LiteRtLmCore.xcframework

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$THIS_DIR/.." && pwd)"
OVERLAY_SRC="$THIS_DIR/bazel_overlay"
SHIM_SRC="$PKG_DIR/Sources/LiteRtLmCShim"
OUT_DIR="$PKG_DIR/build"

: "${LITERTLM_UPSTREAM:=/tmp/c11-litertlm/LiteRT-LM}"

if [ ! -d "$LITERTLM_UPSTREAM" ]; then
  echo "error: LITERTLM_UPSTREAM=$LITERTLM_UPSTREAM not found" >&2
  echo "hint:  git clone https://github.com/google-ai-edge/LiteRT-LM.git /tmp/c11-litertlm/LiteRT-LM" >&2
  exit 1
fi

OVERLAY_DST="$LITERTLM_UPSTREAM/apps/litertlm_swift_shim"

echo "==> staging overlay at $OVERLAY_DST"
rm -rf "$OVERLAY_DST"
mkdir -p "$OVERLAY_DST/include"
cp "$OVERLAY_SRC/BUILD.bazel"                 "$OVERLAY_DST/BUILD"
cp "$SHIM_SRC/litertlm_c_shim.cc"             "$OVERLAY_DST/"
cp "$SHIM_SRC/include/litertlm_c_shim.h"      "$OVERLAY_DST/include/"

cd "$LITERTLM_UPSTREAM"

echo "==> bazel build //apps/litertlm_swift_shim:LiteRtLmCore (ios_arm64 + ios_sim_arm64)"
# The xcframework rule internally fans out to both device and simulator
# slices; --config=ios is a no-op but keeps logs consistent.
bazelisk --output_user_root=/tmp/c11-bazel-cache \
  build --config=ios \
  //apps/litertlm_swift_shim:LiteRtLmCore

echo "==> copying xcframework to $OUT_DIR"
mkdir -p "$OUT_DIR"
XCFW="$LITERTLM_UPSTREAM/bazel-bin/apps/litertlm_swift_shim/LiteRtLmCore.xcframework"
rm -rf "$OUT_DIR/LiteRtLmCore.xcframework"
cp -R "$XCFW" "$OUT_DIR/"

echo "==> done:  $OUT_DIR/LiteRtLmCore.xcframework"
du -sh "$OUT_DIR/LiteRtLmCore.xcframework"
