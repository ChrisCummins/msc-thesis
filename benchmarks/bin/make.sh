#!/usr/bin/env bash

. benchlib.sh

set -ue
benchmark="$1"

# Go to benchmark build directory.
cd "$BIN_DIR/$benchmark"

# Compile new binary.
echo "$(date)" > "$BUILDLOG"
echo "In directory $(pwd)" >> "$BUILDLOG"
make clean >>"$BUILDLOG" 2>&1
make >>"$BUILDLOG" 2>&1

# Get checksum.
checksum="$(sha1sum $benchmark | awk '{print $1}')"

echo $checksum
