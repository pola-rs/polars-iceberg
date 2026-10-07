# polars-iceberg

Native Iceberg scan planning plugin for [Polars](https://github.com/pola-rs/polars), released to
PyPI as `polars-iceberg` (Python module `polars_iceberg`).

Polars loads the plugin through the versioned FFI contracts in
[`polars-io-ext-ffi`](https://github.com/pola-rs/polars/tree/main/crates/polars-io-ext-ffi);
compatibility is decided by the plugin IDs the module exports, not by the package version.

## Development

This crate is built as part of the Polars workspace: clone it (or add it as a submodule) to
`crates/polars-iceberg` in the [Polars repository](https://github.com/pola-rs/polars). Then, from
the Polars root:

```bash
make -C py-polars build-polars-iceberg  # build and install into the Polars venv
make -C py-polars test                  # also builds the plugin
```

## Branches and CI

- `dev`: development branch. Polars pull request and push CI clones it.
- `main`: release branch. Releases are tagged `py-<version>` (matching the version in `Cargo.toml`).

Releases are done from the Polars repository: the `polars_iceberg` optional dependency in
`py-polars/pyproject.toml` (`polars-iceberg >= <version>`) selects the tag. Polars'
`release-polars-iceberg` workflow builds and publishes that tag to PyPI, and its Python release
workflow runs the test suite against that version installed from PyPI.
