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
- Tests: `./run_tests.py -j16` (compiler); `cd integration_tests && ./run_tests.py -j16`

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
- Generate reference outputs the first time or when outputs intentionally change:
  - `./run_tests.py -u` (updates all) or `./run_tests.py -t foo.f90 -u -s` (single test).
- Run unit tests locally before committing: `./run_tests.py -j16` (use `-s` for sequential if debugging).

### Integration Tests (`integration_tests/`)
- Purpose: build-and-run end-to-end programs across backends/configurations via CMake/CTest.
- Add a `.f90` program under `integration_tests/` and wire it through the existing CMake/test macros.
  - Prefer using the existing `RUN_UTIL` macro in `integration_tests/CMakeLists.txt` rather than ad-hoc commands.
  - Avoid custom test generation in CMake; place real sources in the tree and check them in.
- Run integration tests locally: `cd integration_tests && ./run_tests.py -j16` or using CTest targets created by CMake.

### Common Commands
- Run all tests: `ctest` and `./run_tests.py -j16`
- Run a specific test: `./run_tests.py -t pattern -s`
- Update references: `./run_tests.py -u` (or `-t pattern -u`)

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
