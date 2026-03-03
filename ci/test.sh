#!/usr/bin/env shell
# This is a cross-platform `shell` script.

set -ex

echo "Running SHELL"

LFORTRAN_BIN="${LFORTRAN_BIN:-src/bin/lfortran}"
LFORTRAN_WITH_LIRIC="${LFORTRAN_WITH_LIRIC:-0}"
LFORTRAN_ITEST_FAMILY_SET="${LFORTRAN_ITEST_FAMILY_SET:-default}"
LFORTRAN_RUN_REF_TESTS="${LFORTRAN_RUN_REF_TESTS:-auto}"
LFORTRAN_RUN_ITEST_CMAKE="${LFORTRAN_RUN_ITEST_CMAKE:-auto}"
LFORTRAN_INSTALL_SERVER_TEST_DEPS="${LFORTRAN_INSTALL_SERVER_TEST_DEPS:-auto}"

[[ -x "${LFORTRAN_BIN}" ]] || {
    echo "lfortran executable not found: ${LFORTRAN_BIN}" >&2
    exit 1
}

# Ensure run_tests.py resolves the intended binary.
export PATH="$(cd "$(dirname "${LFORTRAN_BIN}")" && pwd):${PATH}"

if [[ "${LFORTRAN_WITH_LIRIC}" == "1" ]]; then
    export LFORTRAN_REF_POLICY="${LFORTRAN_REF_POLICY:-with_liric}"
    export LFORTRAN_REF_SKIP_IR="${LFORTRAN_REF_SKIP_IR:-llvm}"
    export LFORTRAN_REF_SKIP_DBG="${LFORTRAN_REF_SKIP_DBG:-run_dbg}"
    export LFORTRAN_NO_LINK_MODE="${LFORTRAN_NO_LINK_MODE:-1}"
    export LFORTRAN_NO_LINK_MODULE_EMPTY_OBJECTS="${LFORTRAN_NO_LINK_MODULE_EMPTY_OBJECTS:-1}"

    [[ "${LFORTRAN_NO_LINK_MODE}" == "1" ]] || {
        echo "LFORTRAN_NO_LINK_MODE must remain 1 for WITH_LIRIC lanes" >&2
        exit 1
    }
    if [[ "${LIRIC_COMPILE_MODE:-}" == "llvm" ]]; then
        echo "LIRIC_COMPILE_MODE=llvm is disallowed in WITH_LIRIC lanes" >&2
        exit 1
    fi
    [[ -n "${LIRIC_RUNTIME_BC:-}" ]] || {
        echo "LIRIC_RUNTIME_BC must be set for WITH_LIRIC no-link mode" >&2
        exit 1
    }
    [[ -f "${LIRIC_RUNTIME_BC}" ]] || {
        echo "LIRIC_RUNTIME_BC file is missing: ${LIRIC_RUNTIME_BC}" >&2
        exit 1
    }
fi

if [[ "${LFORTRAN_RUN_REF_TESTS}" == "auto" ]]; then
    if [[ "${LFORTRAN_WITH_LIRIC}" == "1" || "${LFORTRAN_LLVM_VERSION:-}" == "11" ]]; then
        LFORTRAN_RUN_REF_TESTS="yes"
    else
        LFORTRAN_RUN_REF_TESTS="no"
    fi
fi

if [[ "${LFORTRAN_RUN_ITEST_CMAKE}" == "auto" ]]; then
    if [[ "${LFORTRAN_WITH_LIRIC}" == "1" ]]; then
        LFORTRAN_RUN_ITEST_CMAKE="no"
    else
        LFORTRAN_RUN_ITEST_CMAKE="yes"
    fi
fi

if [[ "${LFORTRAN_INSTALL_SERVER_TEST_DEPS}" == "auto" ]]; then
    if [[ "${LFORTRAN_WITH_LIRIC}" == "1" ]]; then
        LFORTRAN_INSTALL_SERVER_TEST_DEPS="no"
    else
        LFORTRAN_INSTALL_SERVER_TEST_DEPS="yes"
    fi
fi

run_itest_family() {
    local family="$1"
    case "$family" in
        llvm_base)
            ./run_tests.py -b llvm llvm2 llvm_rtlib llvm_nopragma llvm_integer_8 llvmImplicit
            ;;
        llvm_sc)
            ./run_tests.py -b llvm -sc
            ;;
        llvm_fast)
            ./run_tests.py -b llvm2 llvm_rtlib llvm_nopragma llvm_integer_8 -f
            if [[ "${LFORTRAN_LLVM_VERSION:-}" == "11" ]]; then
                ./run_tests.py -b llvm llvmImplicit -f -nf16
            else
                ./run_tests.py -b llvm llvmImplicit -f
            fi
            ;;
        llvm_submodule)
            ./run_tests.py -b llvm_submodule
            ;;
        llvm_submodule_sc)
            ./run_tests.py -b llvm_submodule -sc
            ;;
        llvm_single_invocation)
            ./run_tests.py -b llvm_single_invocation
            ;;
        llvm_std_f23)
            ./run_tests.py -b llvm --std=f23
            if [[ "${LFORTRAN_LLVM_VERSION:-}" == "11" ]]; then
                ./run_tests.py -b llvm -f --std=f23 -nf16
            else
                ./run_tests.py -b llvm -f --std=f23
            fi
            ;;
        *)
            echo "Unsupported integration test family: ${family}" >&2
            exit 1
            ;;
    esac
}

select_itest_families() {
    case "${LFORTRAN_ITEST_FAMILY_SET}" in
        default|quick)
            ITEST_FAMILIES=(llvm_base llvm_sc llvm_fast llvm_submodule llvm_submodule_sc)
            ;;
        extended|llvm21)
            ITEST_FAMILIES=(
                llvm_base
                llvm_sc
                llvm_fast
                llvm_submodule
                llvm_submodule_sc
                llvm_single_invocation
                llvm_std_f23
            )
            ;;
        *)
            echo "Unsupported LFORTRAN_ITEST_FAMILY_SET: ${LFORTRAN_ITEST_FAMILY_SET}" >&2
            exit 1
            ;;
    esac
}

# Run some simple compilation tests, works everywhere:
"${LFORTRAN_BIN}" --version
# Compile and link separately
"${LFORTRAN_BIN}" -c examples/expr2.f90 -o expr2.o
"${LFORTRAN_BIN}" -o expr2 expr2.o
./expr2

# Compile C and Fortran
"${LFORTRAN_BIN}" -c integration_tests/modules_15b.f90 -o modules_15b.o
"${LFORTRAN_BIN}" -c integration_tests/modules_15.f90 -o modules_15.o

if [[ $WIN == "1" ]]; then # Windows
    cl /MD /c integration_tests/modules_15c.c /Fomodules_15c.o
elif [[ $MACOS == "1" ]]; then # macOS
    clang -c integration_tests/modules_15c.c -o modules_15c.o
else # Linux
    gcc -c integration_tests/modules_15c.c -o modules_15c.o
fi

"${LFORTRAN_BIN}" modules_15.o modules_15b.o modules_15c.o -o modules_15
./modules_15


# Compile and link in one step
"${LFORTRAN_BIN}" integration_tests/intrinsics_04s.f90 -o intrinsics_04s
./intrinsics_04s

"${LFORTRAN_BIN}" integration_tests/intrinsics_04.f90 -o intrinsics_04
./intrinsics_04


# Run all tests (does not work on Windows yet):
cmake --version
if [[ $WIN != "1" ]]; then
    # using debugging option i.e. `-x` causes incorrect assignment
    set +x
    if [[ $MACOS == "1" ]]; then
        # macOS ARM64 runners have 3 cores; higher parallelism overwhelms them
        NPROC=3
    else
        # this works fine on Linux
        NPROC=$(nproc)
    fi
    # we turn on the debugging again
    set -x
    echo "NPROC: ${NPROC}"

    if [[ "${LFORTRAN_RUN_REF_TESTS}" == "yes" ]]; then
        ./run_tests.py
    fi

    if [[ -n "${LFORTRAN_CTEST_DIR:-}" ]]; then
        if [[ "${LFORTRAN_WITH_LIRIC}" == "1" ]]; then
            LFORTRAN_UNIT_BIN="${LFORTRAN_CTEST_DIR}/src/lfortran/tests/test_lfortran"
            if [[ -x "${LFORTRAN_UNIT_BIN}" ]]; then
                "${LFORTRAN_UNIT_BIN}" --source-file-exclude='*test_llvm.cpp'
                ctest --test-dir "${LFORTRAN_CTEST_DIR}" --output-on-failure -E '^test_lfortran$'
            else
                ctest --test-dir "${LFORTRAN_CTEST_DIR}" --output-on-failure
            fi
        else
            ctest --test-dir "${LFORTRAN_CTEST_DIR}" --output-on-failure
        fi
    fi

    cd integration_tests
    if [[ "${LFORTRAN_RUN_ITEST_CMAKE}" == "yes" ]]; then
        LFORTRAN_BIN_ABS="$(cd "$(dirname "${LFORTRAN_BIN}")" && pwd)/$(basename "${LFORTRAN_BIN}")"
        rm -rf build-lfortran-llvm
        mkdir build-lfortran-llvm
        cd build-lfortran-llvm
        FC="${LFORTRAN_BIN_ABS}" cmake -DLFORTRAN_BACKEND=llvm -DCURRENT_BINARY_DIR=. ..
        make -j${NPROC}
        ctest -L llvm -j${NPROC}
        cd ..
    else
        echo "Skipping integration_tests CMake/ctest stage (LFORTRAN_RUN_ITEST_CMAKE=${LFORTRAN_RUN_ITEST_CMAKE})"
    fi

    select_itest_families
    for family in "${ITEST_FAMILIES[@]}"; do
        run_itest_family "${family}"
    done
    cd ..

    if [[ "${LFORTRAN_INSTALL_SERVER_TEST_DEPS}" == "yes" ]]; then
        python3 -m pip install src/server/tests tests/server
    else
        echo "Skipping server test dependency install (LFORTRAN_INSTALL_SERVER_TEST_DEPS=${LFORTRAN_INSTALL_SERVER_TEST_DEPS})"
    fi
    # NOTE: `--full-trace` tends to print excessively long stack traces. Please
    # re-enable it if needed:
    # pytest -vv --showlocals --full-trace --capture=no --timeout=5 tests/server
#    pytest -vv --showlocals --capture=no --timeout=5 tests/server
fi
