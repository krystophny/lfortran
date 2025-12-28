#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/lfortran"
    exit 2
fi

LFORTRAN_BIN="$(realpath "$1")"
if [[ ! -x "$LFORTRAN_BIN" ]]; then
    echo "ERROR: not executable: $LFORTRAN_BIN"
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAPACK_SRC="${ROOT_DIR}/../lapack"

if [[ ! -d "$LAPACK_SRC" ]]; then
    echo "ERROR: lapack source directory not found at: $LAPACK_SRC"
    exit 2
fi

OUT_DIR="/tmp/lfortran-dev/lapack-mre-sdrvls-$$"
BUILD_DIR="${OUT_DIR}/build"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

CONFIG_LOG="${LOG_DIR}/configure.log"
BUILD_LOG="${LOG_DIR}/build_xlintsts.log"
REPRO_LOG="${LOG_DIR}/repro_sdrvls.log"

echo "Using lfortran: $LFORTRAN_BIN"
echo "Using lapack:   $LAPACK_SRC"
echo "Out dir:        $OUT_DIR"
echo

cmake -S "$LAPACK_SRC" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_Fortran_COMPILER="$LFORTRAN_BIN" \
  -DCMAKE_Fortran_FLAGS="--fixed-form-infer --implicit-interface --implicit-typing --legacy-array-sections --separate-compilation" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_INDEX64=OFF \
  -DBUILD_INDEX64_EXT_API=OFF \
  -DBUILD_COMPLEX=OFF \
  -DBUILD_COMPLEX16=OFF \
  -DBUILD_TESTING=ON \
  2>&1 | tee "$CONFIG_LOG"

set +e
cmake --build "$BUILD_DIR" --target xlintsts -j32 2>&1 | tee "$BUILD_LOG"
BUILD_RC="${PIPESTATUS[0]}"
set -e

echo
echo "Build exit code: $BUILD_RC"

if [[ ! -f "$BUILD_LOG" ]]; then
    echo "ERROR: build log not found at: $BUILD_LOG"
    exit 1
fi

cmd="$(grep -n "sdrvls.f-pp.f" "$BUILD_LOG" | grep -m 1 "lfortran" | sed -E 's/^[0-9]+://')"
if [[ -z "${cmd}" ]]; then
    echo "ERROR: could not find sdrvls compile command in build log: $BUILD_LOG"
    exit 1
fi

echo
echo "sdrvls compile command (from build log):"
echo "$cmd"
echo

echo "Re-running the command to reproduce the ICE (output captured):"
(cd "$BUILD_DIR" && bash -lc "$cmd") 2>&1 | tee "$REPRO_LOG"
