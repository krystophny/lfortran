# Repository Guidelines

This file is for LLM agents and new contributors to have a single point of detailed 
reference how to contribute to the project.

## Project Structure & Module Organization
- `src/`: core sources
  - `libasr/`: ASR + utilities, passes, verification
  - `lfortran/`: parser, semantics, drivers, backends
  - `runtime/`: Fortran runtime (built via CMake)
  - `server/`: language server
- `tests/`, `integration_tests/`: unit/E2E suites
- `doc/`: docs & manpages (site generated from here)
- `examples/`, `grammar/`, `cmake/`, `ci/`, `share/`: supporting assets

## Prerequisites
- Tools: CMake (>=3.10), Ninja, Git, Python (>=3.8), GCC/Clang.
- Generators: re2c, bison (needed for build0/codegen).
- Libraries: zlib; optional: LLVM dev, libunwind, RapidJSON, fmt, xeus/xeus-zmq, Pandoc.

## Build, Test, and Development Commands
- Typical dev config (Ninja + LLVM):
  - `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DWITH_LLVM=ON`
  - `cmake --build build -j`
- Release build: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DWITH_LLVM=ON`
- Tests: `./run_tests.py` (compiler); `cd integration_tests && ./run_tests.py -j16`
- Local integration tip (LLVM): some evaluator/diagnostic paths require precise
  location info. When running integration tests locally with the LLVM backend,
  prefer injecting the flag via environment rather than changing CMake:
  - `cd integration_tests && FFLAGS="--debug-with-line-column" ./run_tests.py -j8`
  This mirrors CI closely while enabling line/column emission.

## Quick Smoke Test
- We usually build with LLVM enabled (`-DWITH_LLVM=ON`).
- AST/ASR (no LLVM): `build/src/bin/lfortran --show-ast examples/expr2.f90`
- Run program (LLVM): `build/src/bin/lfortran examples/expr2.f90 && ./a.out`

## Architecture & Scope
- AST (syntax) ↔ ASR (semantic, valid-only). See `doc/src/design.md`.
- Pipeline: parse → semantics → ASR passes → codegen (LLVM/C/C++/x86/WASM).
- Prefer `src/libasr/pass` and existing utils; avoid duplicate helpers/APIs.

## Git Remotes & Issues
- Upstream: `lfortran/lfortran` on GitHub (canonical repo and issues).
- Fork workflow: `origin` is your fork; submit draft PRs to upstream and mark ready for review.
- Setup:
  - `git remote add upstream git@github.com:lfortran/lfortran.git`
  - `git fetch upstream --tags`
  - `git checkout -b feature/xyz && git push -u origin feature/xyz`
 - PRs: always target `upstream/main`. Keep feature branches rebased on `upstream/main`:
   - `git fetch upstream && git rebase upstream/main && git push --force-with-lease`
   - Ensure the `upstream` remote exists (see setup) when working from a fork.

## Coding Style & Naming Conventions
- C/C++: C++17; follow the existing formatting in the file to be consistent; use 4 spaces for indentation
- Names: lower_snake_case files; concise CMake target names.
- No commented-out code.

## Testing Guidelines
- Full coverage required: every behavior change must come with tests that fail before your change and pass after. Do not merge without a full local pass of unit and integration suites.

### Unit/Reference Tests (`tests/`)
- Purpose: fast, deterministic checks of AST/ASR/codegen outputs and CLI behaviors.
- Add a source file under `tests/` (use `.f90` or `.f` for fixed-form). Do not normalize line endings that the test depends on; `.gitattributes` preserves files in `tests/**`.
- Register the test in `tests/tests.toml` with a `[[test]]` table:
  - `filename = "foo.f90"` (relative to `tests/`)
  - Enable outputs to check, e.g. `ast = true`, `asr = true`, `llvm = true`, `obj = true`, `run = true`, etc.
  - For fixed-form (`.f`), `run_tests.py` automatically adds `--fixed-form` for AST/ASR.
  - For multi-file module use, set `extrafiles = "mod1.f90,mod2.f90"` (these are precompiled before running the main test).
- Generate or refresh references only as needed; see "Reference Generation: Best Practices" below for the recommended single‑test workflow.
- Run unit tests locally before committing: `./run_tests.py -j16` (use `-s` for sequential if debugging).

#### Reference Generation: Best Practices
- Prefer single‑test updates to avoid accidental mass changes:
  - `./run_tests.py -t ../integration_tests/your_test.f90 -u -s`
- Only stage the new files for your test under `tests/reference/`:
  - `git add tests/reference/*your_test*`
- Avoid a blanket `-u` unless you purposely intend to refresh all references.
- For tests referring to files outside `tests/` (e.g., `../integration_tests/...`), the reference files are still written to `tests/reference/`.
- If you mistakenly removed many refs, restore and re‑generate just your test:
  - `git restore --worktree tests/reference`
  - Re‑run the single‑test `-u` command above, then stage only your new refs.

### Integration Tests (`integration_tests/`)
- Purpose: build-and-run end-to-end programs across backends/configurations via CMake/CTest.
- Add a `.f90` program under `integration_tests/` and wire it through the existing CMake/test macros.
  - Prefer using the existing `RUN_UTIL` macro in `integration_tests/CMakeLists.txt` rather than ad-hoc commands.
  - Avoid custom test generation in CMake; place real sources in the tree and check them in.
- CI‑parity (recommended): run with the same env and scripts CI uses
  - Use micromamba with `ci/environment.yml` to match toolchain (LLVM, etc.).
  - Set env like CI and call the same helper scripts:
    - `export LFORTRAN_CMAKE_GENERATOR=Ninja`
    - `export ENABLE_RUNTIME_STACKTRACE=yes` (Linux/macOS)
    - Build: `shell ci/build.sh`
    - Quick integration run (LLVM):
      - `shell ci/test.sh` (runs a CMake+CTest LLVM pass and runner passes)
      - or: `cd integration_tests && ./run_tests.py -b llvm && ./run_tests.py -b llvm -f -nf16`
  - GFortran pass: `cd integration_tests && ./run_tests.py -b gfortran`
  - Other backends as in CI:
    - `./run_tests.py -b cpp c c_nopragma` and `-f`
    - `./run_tests.py -b wasm` and `-f`
    - `./run_tests.py -b llvm_omp` / `target_offload` / `fortran -j1`

- Minimal local (without micromamba):
  - Build: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DWITH_LLVM=ON -DWITH_RUNTIME_STACKTRACE=yes`
  - Run: `cd integration_tests && ./run_tests.py -b llvm && ./run_tests.py -b llvm -f -nf16`
  - If diagnostics need line/column mapping during local debugging, inject:
    `FFLAGS="--debug-with-line-column" ./run_tests.py -b llvm`
- If builds fail with messages about missing debug info or line/column emission:
  - Install LLVM tools so `llvm-dwarfdump` is available (e.g., `sudo pacman -S llvm`,
    `apt install llvm`, or `conda install -c conda-forge llvm-tools`).
  - Rebuild with runtime stacktraces if needed:
    `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DWITH_LLVM=ON -DWITH_RUNTIME_STACKTRACE=yes -DWITH_UNWIND=ON`
  - Run integration with LFortran flags injected via env:
    `FFLAGS="--debug-with-line-column" ./run_tests.py -j8`

### Local Troubleshooting
- Modfile version mismatch: if you see "Incompatible format: LFortran Modfile...",
  remove stale module files generated by earlier versions:
  - `find tests -name '*.mod' -delete && find integration_tests -name '*.mod' -delete`
  - Also remove stale integration build dirs to ensure a clean configure:
    `rm -rf integration_tests/build-* integration_tests/test-*`
  Ensure the current `build/src/bin` is first on `PATH` when running tests.

### Common Commands
- Run all tests: `ctest` and `./run_tests.py -j16`
- Run a specific test: `./run_tests.py -t pattern -s`

References
- Developer docs: `doc/src/installation.md` (Tests section) and `doc/src/progress.md`.
- Online docs: https://docs.lfortran.org/en/installation/ (see Tests: run, update, integration).

## Commit & Pull Request Guidelines
- Commits: small, single-topic, imperative (e.g., "fix: handle BOZ constants").
- PRs target `upstream/main`; reference issues (`fixes #123`), explain rationale.
- Include test evidence (commands + summary); ensure CI passes.
- Do not commit generated artifacts, large binaries, or local configs.
 - Before draft PR: all local tests pass.
 - Open a draft PR; mark ready only when finished.
