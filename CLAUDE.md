# fhrs - agent development reference

## about

This project provides a rust extension module for fast python
histogramming routines. It's built primarily with PyO3, rust-numpy,
ndarray, and rayon.

## general

- Never run test commands automatically unless explicitly asked.
- Avoid useless comments; write brief comments if necessary.
- Keep performance in mind; this projects wants to provide very fast
  histogramming routines with low overhead. Avoid costly copies, avoid
  unnecessary work.
- Use _modern_ PyO3 (never write code that isn't compatible with the
  recent `Bound` API)
- Use _modern_ Python and the `uv` package manager.

## build commands

- Run `uv run maturin dev --uv` to build the project.
- Run `uv run --config-setting 'build-args=--profile=dev' pytest` to
  test the project.

## format commands

- `cargo +nightly fmt && uv run ruff format`: run the rust formatter
  and python formatter

## style

- In Python use strict typing compatible with 3.11+ (that is, use
  `dict[K, V]` over `Dict[K, V]` and use `collections.abc` over the
  `typing` module when possible).
