# Repository Guidelines

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
- Build tools: CMake (>=3.10), Ninja, Git, Python (>=3.8).
- Compilers: GCC or Clang (C/C++).
- Generators (required): re2c, bison.
- Libraries: zlib (static available).
- Optional (feature flags): LLVM dev (WITH_LLVM), libunwind (stacktrace), RapidJSON (WITH_JSON), fmt (WITH_FMT), xeus + xeus-zmq (WITH_XEUS), Pandoc (docs).

## Build, Test, and Development Commands
- Configure + build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
- Enable LLVM (example): `cmake -S . -B build -DWITH_LLVM=ON`
- Tests: `./run_tests.py -j16` (compiler) and `cd integration_tests && ./run_tests.py -j16`
 
## Architecture & Scope
- AST (syntax only) ↔ ASR (semantic, valid-only). See `doc/src/design.md`.
- Pipeline: parse → semantics → ASR passes → codegen (LLVM/C/C++/x86/WASM).
- ASR is immutable; verified in Debug. Use builders and existing utils.
- Add or tweak passes in `src/libasr/pass`; avoid duplicate helpers/APIs.

## Git Remotes & Issues
- Upstream: `lfortran/lfortran` on GitHub (canonical repo and issues).
- Fork workflow: `origin` is your fork; submit PRs to upstream.
- Setup:
  - `git remote add upstream git@github.com:lfortran/lfortran.git`
  - `git fetch upstream --tags`
  - `git checkout -b feature/xyz && git push -u origin feature/xyz`
 - PRs: always target `upstream/main`. Keep feature branches rebased on `upstream/main`:
   - `git fetch upstream && git rebase upstream/main && git push --force-with-lease`
   - Ensure the `upstream` remote exists (see setup) when working from a fork.

## Coding Style & Naming Conventions
- C/C++: C++17; format via `.clang-format` (pre-commit hooks).
- Pre-commit: `pre-commit install` (once) → `pre-commit run -a`.
- Names: lower_snake_case files; concise CMake target names.
- Keep functions small; no commented-out code.

## Testing Guidelines
- Add focused tests in `tests/` (e.g., `feature_x.f90`).
- Ensure full pass: `./run_tests.py -j16`.
- Multi-file: use `extrafiles` like existing cases.
- Use `integration_tests/` for cross-component flows.

## Commit & Pull Request Guidelines
- Commits: small, single-topic, imperative (e.g., "fix: handle BOZ constants").
- PRs target `upstream/main`; reference issues (`fixes #123`), explain rationale.
- Include test evidence (commands + summary); ensure CI passes.
- Do not commit generated artifacts, large binaries, or local configs.
