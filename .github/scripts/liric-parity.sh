#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/liric-parity-common.sh
source "${SCRIPT_DIR}/liric-parity-common.sh"

PROFILE="${1:-quick}"
case "${PROFILE}" in
    quick|exhaustive)
        ;;
    *)
        echo "Usage: .github/scripts/liric-parity.sh [quick|exhaustive]"
        exit 1
        ;;
esac

ensure_layout
ensure_linux_deps
build_liric_release
build_lfortran_with_liric
prepare_liric_runtime_archive
ensure_llvm_dwarfdump
run_lfortran_unit_subset
prepare_integration_build_tree
run_integration_common

if [[ "${PROFILE}" == "quick" ]]; then
    run_quick_repo_checks
else
    run_integration_exhaustive_tail
fi
