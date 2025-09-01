# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core sources
  - `libasr/`: ASR and compiler infrastructure
  - `lfortran/`: frontend, drivers, backends
  - `runtime/`: Fortran runtime (configured and built via CMake)
  - `server/`: language-server and related services
- `tests/`: unit/compiler tests (.f/.f90 with reference outputs)
- `integration_tests/`: end-to-end scenarios and larger suites
- `doc/`: docs, manpages; website docs generated from here
- `examples/`, `grammar/`, `cmake/`, `ci/`, `share/`: supporting assets

## Build, Test, and Development Commands
- Configure + build (Release):
  - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
  - `cmake --build build --parallel`
- Enable LLVM/JIT (example): `cmake -S . -B build -DWITH_LLVM=ON`
- Test suites:
  - Compiler tests: `./run_tests.py` (add `-j16` for parallel)
  - CTest from build dir: `ctest --output-on-failure`
  - Integration: `cd integration_tests && ./run_tests.py -j16`
- Pixi (optional): `pixi run build`, `pixi run tests`

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
- C/C++: C++17, enforced by `.clang-format`. Run pre-commit hooks before pushing.
- Pre-commit: `pre-commit install` once; then `pre-commit run -a`.
- Filenames: lower_snake_case for sources/tests; CMake targets use concise, descriptive names.
- Keep functions small and focused; avoid commented-out code.

## Testing Guidelines
- Add focused tests under `tests/` mirroring existing patterns (e.g., `feature_x.f90`).
- Run `./run_tests.py` locally and ensure a full pass. Do not disable or skip tests.
- For tests needing multiple files, use `extrafiles` as in existing cases.
- Use integration tests for cross-component flows.

## Commit & Pull Request Guidelines
- Commits: small, single-topic, imperative mood (e.g., "fix: handle BOZ constants").
- Reference issues in the description (`fixes #123`) and explain rationale.
- PRs: include what/why, key paths, test evidence (command + summary), and any flags used (e.g., `WITH_LLVM`). Ensure CI passes.
- Do not commit generated artifacts, large binaries, or local configs.

## Security & Configuration Tips
- Prefer CMake options (e.g., `WITH_LLVM`, `WITH_LSP`) over ad‑hoc patches.
- Avoid hardcoded paths/secrets; use env vars or CMake cache vars.
