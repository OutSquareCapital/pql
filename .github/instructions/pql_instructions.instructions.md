---
description: Instructions for the PQL project agents.
applyTo: '*'
---
# AGENTS Instructions for `pql`

## Project mission

`pql` aims to expose DuckDB capabilities through a Polars-like API (close to Narwhals semantics when practical, but not constrained by multi-backend compromises).

Primary objective:

- Provide a high-level public API (`pql.Expr`, `pql.LazyFrame`) that compiles to efficient DuckDB-native expressions/relations.

Secondary objective:

- Keep parity visibility against Narwhals/Polars via generated coverage reports.

---

## Architecture (must understand before changing code)

### 1) Public API layer

- `src/pql/_expr.py`: public `Expr` API + namespaces (`.str`, `.list`, `.struct`).
- `src/pql/_frame.py`: public `LazyFrame` API.
- `src/pql/__init__.py`: exported user-facing symbols.

This layer should remain Polars-like and user ergonomic.

### 2) SQL core layer (internal wrapper)

#### Core concepts

The SQL core layer provides two main abstractions that wrap DuckDB objects:

1. **`SqlExpr`**: Wraps `duckdb.Expression` to provide a method chained interface, and a signifcant amount of additionals methods from code generation (e.g., `.str`, `.list`, `.dt` namespaces, and all DuckDB functions as methods). This is the core building block for expression manipulation.
2. **`Relation`**: Wraps `duckdb.DuckDBPyRelation`. Don't provide additional methods "yet" but internally handles SqlExpr inputs and fix some of the stubs errors. This is the core building block for relation manipulation.

Both use a **composition-based design** (wrapper pattern): they store an underlying DuckDB object (`.inner()`) and delegate operations to it, managing automatic conversion between pql wrappers and DuckDB native types.

#### Handler hierarchy (`src/pql/sql/_core.py`)

The handler system is built on three base classes:

1. **`CoreHandler[T]`** (generic base)
  - Generic wrapper for any value of type `T`
  - Provides interface: `.pipe(func, ...)`, `.inner()`, `._new(T) -> Self`
  - `._new(expr)` is critical for method chaining: internally creates a new instance of the current subclass with a transformed expression
  - Used everywhere to maintain fluent style

2. **`DuckHandler(CoreHandler[duckdb.Expression])`**
  - Specialization for `duckdb.Expression`
  - Base for all expression-based functionality
  - No functionality of its own; exists for type clarity

3. **`RelHandler(CoreHandler[duckdb.DuckDBPyRelation])`**
  - Specialization for `duckdb.DuckDBPyRelation`
  - Handles initialization logic: converts various input types (Polars DF/LazyFrame, SQL strings, table functions) to DuckDB relations
  - `__init__` performs matching on input type (see `FrameInit` union)

4. **`NameSpaceHandler[T]`** (for namespaces like `.str`, `.list`, etc.)
  - Wraps a parent expression namespace handler
  - `._parent: T` stores the parent
  - `._new(expr)` creates a new parent instance with transformed expression

#### `SqlExpr` class hierarchy (`src/pql/sql/_expr.py`)

```
CoreHandler[duckdb.Expression]
  ↓
DuckHandler
  ↓
Fns (auto-generated, from _code_gen/_fns.py) + Expression (auto-generated, from _code_gen/_core.py)
  ↓
SqlExpr(Expression, Fns)
  + namespaces: SqlExprStringNameSpace, SqlExprListNameSpace, etc.
  + window functions
  + a few other methods with special casing (see scripts/fns_generator/_rules.py for details)
```

```

#### `Relation` class hierarchy (`src/pql/sql/_code_gen/_core.py`)

```
CoreHandler[duckdb.DuckDBPyRelation]
  ↓
RelHandler
  ↓
Relation (auto-generated wrapper methods)
  + aggregate, project, filter, order_by, etc.
```

#### Converter functions (`src/pql/sql/_expr.py` + `_core.py`)

- **`into_expr(value, as_col=False) -> SqlExpr`**
  - Converts: `SqlExpr` (passthrough), `Expr` (public API), `str` (optional column name), any other value (literal)
  - Used everywhere to normalize inputs
  
- **`into_duckdb(value) -> T | duckdb.Expression`**
  - Inverse: `DuckHandler` → `.inner()`, otherwise passthrough
  - Used at boundary when passing to DuckDB methods

- **`args_into_exprs(exprs, named_exprs=None) -> pc.Iter[SqlExpr]`**
  - Multi-arg converter: flattens iterables, converts all to `SqlExpr`
  - Handles positional + keyword arguments (adds aliases to keyword args)
  - Returns `pc.Iter[SqlExpr]` for chaining

- **`func(name, *args) -> duckdb.Expression`**
  - Creates `duckdb.FunctionExpression(name, *args)`
  - Filters `None` args
  - Converts all args via `into_duckdb()` before passing to DuckDB
  - Is used in generated function wrappers, and anywhere else a raw function call is needed (note that this shouldn't be the case usually, as most functions already exist as `SqlExpr` methods via codegen)

#### Helper utilities

- **`try_iter(val) -> pc.Iter[T]`**: Convert any value to `pc.Iter` (handles strings/bytes as single items)
- **`try_flatten(vals) -> pc.Iter[T]`**: Flatten one or two levels of iterables

#### File manifest (`src/pql/sql/`)

- `src/pql/sql/_expr.py`: `SqlExpr` wrapper + converters (`into_expr`, `lit`, `col`, `when`, `coalesce`, etc.).
- `src/pql/sql/_core.py`: shared handlers (`CoreHandler`, `RelHandler`, `DuckHandler`, `NameSpaceHandler`), `try_iter`, `try_flatten`, `func`, `into_duckdb`.
- `src/pql/sql/_window.py`: window SQL builder (`over_expr`, ordering/null handling, bounds).
- `src/pql/sql/_raw.py`: SQL keyword helpers + query prettify/sanitize (`QueryHolder`).
- `src/pql/sql/datatypes.py`: dtype aliases and precision mapping.

### 3) Auto-generated wrappers (do not edit manually)

- `src/pql/sql/_code_gen/_fns.py`: DuckDB function wrappers.
- `src/pql/sql/_code_gen/_core.py`: `Relation` wrapper methods.

Generated from scripts in `scripts/`.

### 4) Code generation + analysis scripts

- `scripts/fn_generator/*`: generate function wrappers from DuckDB catalog.
- `scripts/core_generator/*`: generate relation wrapper from DuckDB stubs.
- `scripts/comparator/*`: build API comparison report.
- `scripts/__main__.py`: CLI entrypoint.

---

## Non-negotiable implementation rules

1. Prefer native wrappers over raw SQL:

- Always attempt `SqlExpr`/`Relation` methods first.
- Use raw SQL only if impossible with wrappers, and keep it minimal and justified.

1. Do not patch generated files directly:

- Never hand-edit `src/pql/sql/_code_gen/*`.
- Modify generator logic in `scripts/*` and regenerate.

1. Preserve DuckDB semantics:

- Do not “hack” DuckDB behavior to mimic Polars exactly when semantics differ.
- Null ordering/handling differences are acceptable if explicit and consistent.

1. Keep generated SQL/relations efficient:

- Avoid unnecessary projections/materialization.
- Keep expression composition compact.

1. Maintain fluent style:

- Prefer method chaining.
- Reuse existing helpers (`args_into_exprs`, `try_iter`, `try_flatten`, `into_expr`).

---

## Required coding style

### General Python style

- Python version target: `>=3.13`.
- Full typing is required (params, returns, key variables, generics).
- Use `match` where it improves branch clarity.
- Avoid broad/naked exceptions.

### Pyochain style (mandatory in this repo)

- Prefer `pyochain` pipelines over imperative loops.
- Avoid ad-hoc Python container churn when `pc.Iter/Seq/Vec/Dict/Set` fits.
- Prefer `Option`/`Result`-oriented handling over manual `None` and ad-hoc checks.
- Keep iterable transformations chain-based (`map/filter/fold/filter_map/map_star/...`).

---

## Testing protocol (critical)

Goal:

- 100% coverage target for public API behavior.

Current testing layout:

- `tests/test_exprs.py`
- `tests/test_lazyframe.py`
- `tests/test_str_namespace.py`
- `tests/test_list_namespace.py`
- `tests/test_struct_namespace.py`
- `tests/test_files.py`

Rules for any new/updated tests:

1. Comparison-first strategy:

- Prefer comparison helpers (`assert_frame_equal`-based helpers) for behavior checks.
- Avoid naked `assert` for dataframe behavior when helper-based comparison is feasible.

1. Identical call chains:

- pql and reference backend (Narwhals/Polars) chains must be structurally identical.
- No parameter/method-call divergence unless impossible.

1. If identical chains are impossible:

- Do not silently force a divergent implementation.
- Document why parity cannot hold (semantic/API gap), with concrete examples and options.

1. If you notice pre-existing violations while editing nearby tests:

- Fix them immediately as part of the same change scope.

---

## API parity workflow

Use `API_COVERAGE.md` as tracking input, not as a strict blocker.

When implementing a missing/mismatched method:

1. Check if a `SqlExpr`/`Relation` capability already exists (including generated mixins).
2. Validate naming and signature alignment against project intent (Polars-like + DuckDB-centric).
3. Add/adjust tests with identical pql vs reference chains.
4. Regenerate coverage report if API surface changed.

---

## Generator workflow

Use `uv` commands:

- Generate relation wrapper:
  - `uv run -m scripts gen-rel`
- Generate function wrappers:
  - `uv run -m scripts gen-fns`
- Rebuild API coverage:
  - `uv run -m scripts compare`

After generation, run Ruff on touched files.

---

## Validation checklist before opening/merging changes

1. Did you avoid editing `_code_gen` manually?
2. Did you use `SqlExpr`/`Relation` before raw SQL?
3. Did you preserve DuckDB semantics (especially null/order behavior)?
4. Are tests using comparison helpers and identical call chains where required?
5. Did you run Ruff and tests relevant to your change?
6. If API changed, did you refresh/report coverage implications?

---

## Inspirations and reference points

### Narwhals

Installed Narwhals implementation in `.venv` (notably `narwhals/sql.py`, `narwhals/_sql/*`, `narwhals/_duckdb/*`).
<https://narwhals-dev.github.io/narwhals/generating_sql/>
<https://narwhals-dev.github.io/narwhals/api-completeness/>

### SqlFrame

SQLFrame as conceptual reference for SQL/DataFrame translation patterns.
<https://github.com/eakmanrq/sqlframe>

### Duckdb Experimental Spark API

DuckDB experimental Spark API as DuckDB-centric interoperability reference.
<https://github.com/duckdb/duckdb-python/tree/main/duckdb/experimental/spark>
Use these as guidance, but keep `pql` decisions aligned with this repository’s own architecture and constraints.

### Polars API

Polars API as user ergonomics reference, but not a strict template.
venv available for search, and MCP server tool.
