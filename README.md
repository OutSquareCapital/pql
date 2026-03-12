
# PQL, write polars syntax, create SQL queries for DuckDB

This package is **WIP** and ultimately aim to provide a polars like API for [DuckDB Python API](https://duckdb.org/docs/stable/clients/python/overview), as well as full support of the [DuckDB functions](https://duckdb.org/docs/stable/sql/functions/overview).

## Comparison to other tools

### Narwhals

[narwhals](https://github.com/narwhals-dev/narwhals) aims to support more functionnality from polars AND all of those from duckdb, as pql is not limited by multiple backend compatibility.
Furthermore, `narwhals` is primarly designed for library developpers who want integration with multiple dataframe libraries, not for end users.
Narwhals support a **subset** of polars API, hence necessarily a **subset** of DuckDB API, while pql aims to support the **full** API of both.

### SQLFrame

[SQLFrame](https://github.com/eakmanrq/sqlframe) is fundamentally a PySpark oriented library API-wise.

### Ibis

[Ibis](https://ibis-project.org/) is probably the closest serious comparison point, but the overlap is easy to overstate if you only read the top-level docs.

The short version is this:

- `pql` is a DuckDB-first wrapper layer with a Polars-like public API, built directly around DuckDB Python expressions and relations.
- Ibis is a backend-portable expression system whose DuckDB support is one backend implementation inside a much larger framework.

Those are not just different slogans. They show up very clearly in the code.

#### 1. Core architecture is different

`pql`'s public API is thinly layered on top of a DuckDB-native core:

- [`Expr`](src/pql/_expr.py) is literally declared as `class Expr(sql.CoreHandler[sql.SqlExpr])` and described as a "Polars-like API over DuckDB expressions".
- [`LazyFrame`](src/pql/_frame.py) is `class LazyFrame(sql.CoreHandler[sql.SqlFrame])` and described as a "Polars-like API over DuckDB relations".
- [`SqlExpr`](src/pql/sql/_expr.py) is a wrapper around DuckDB expressions.
- [`CoreHandler`](src/pql/sql/_core.py) and [`DuckHandler`](src/pql/sql/_core.py) wrap inner values directly, and [`func`](src/pql/sql/_core.py) constructs `duckdb.FunctionExpression(...)` objects.

So `pql` is not "an abstract query system that happens to target DuckDB". It is much closer to "a user-facing Polars-like facade over DuckDB's Python Expression and Relational APIs".

Ibis is architected very differently:

- the generic SQL backend base is [`SQLBackend`](https://github.com/ibis-project/ibis/blob/main/ibis/backends/sql/__init__.py), whose [`compile`](https://github.com/ibis-project/ibis/blob/main/ibis/backends/sql/__init__.py) method goes through SQL compilation,
- the compilation path is implemented through [`SQLGlotCompiler`](https://github.com/ibis-project/ibis/blob/main/ibis/backends/sql/compilers/base.py),
- and the DuckDB backend is one subclass, [`ibis.backends.duckdb.Backend`](https://github.com/ibis-project/ibis/blob/main/ibis/backends/duckdb/__init__.py).

That matches the Ibis docs exactly: [Why Ibis?](https://ibis-project.org/why) explains that Ibis provides a common Python API and compiles it into the backend's native language, usually SQL.

#### 2. SQL is a fallback in `pql`, but a central compilation target in Ibis

The repository's own architecture section already says that `pql` tries to use DuckDB's [Expression API](https://duckdb.org/docs/stable/clients/python/expression) and [Relational API](https://duckdb.org/docs/stable/clients/python/relational_api) as much as possible, with raw SQL only when needed.

That is also visible in code:

- [`SqlExpr`](src/pql/sql/_expr.py) is built from DuckDB expressions and namespaces,
- [`func`](src/pql/sql/_core.py) builds `duckdb.FunctionExpression` objects,
- and the public wrappers mostly transform those wrapper objects instead of compiling an independent IR.

Ibis does the opposite trade-off by design. The docs for [Table expressions](https://ibis-project.org/reference/expression-tables) describe `Table` as a symbolic, immutable, lazy expression object that is typically translated into SQL. The [Why Ibis?](https://ibis-project.org/why) page states that most backends generate SQL and that Ibis uses SQLGlot. The actual implementation of [`SQLBackend.compile`](https://github.com/ibis-project/ibis/blob/main/ibis/backends/sql/__init__.py) and [`SQLGlotCompiler`](https://github.com/ibis-project/ibis/blob/main/ibis/backends/sql/compilers/base.py) confirms that this is not marketing language but the central execution model.

Even for DuckDB specifically, Ibis still follows that architecture: the DuckDB backend compiles an Ibis expression to SQL and then executes it through DuckDB. For example, its DuckDB backend [`to_pyarrow`](https://github.com/ibis-project/ibis/blob/main/ibis/backends/duckdb/__init__.py) and [`execute`](https://github.com/ibis-project/ibis/blob/main/ibis/backends/duckdb/__init__.py) methods go through compiled expressions and DuckDB execution rather than exposing DuckDB Python expression objects as the primary user abstraction.

#### 3. Where `pql` is genuinely different

The strongest case for `pql` is not breadth. It is alignment.

`pql` is more tightly aligned with:

- DuckDB-native expression and relation capabilities,
- Polars-like ergonomics at the public API layer,
- and a design that does not need to normalize behavior across 22 backends or across a portable backend matrix.

That has concrete consequences.

Because Ibis is portable, it has to maintain a common language across backends. The docs make that explicit in several places:

- [Why Ibis?](https://ibis-project.org/why) emphasizes backend switching and cross-engine portability,
- [Backend Table Hierarchy](https://ibis-project.org/concepts/backend-table-hierarchy) shows that Ibis normalizes concepts like `catalog` and `database` across engines,
- and the [support matrix](https://ibis-project.org/backends/support/matrix) exists precisely because backend support varies across operations.

`pql` does not pay that abstraction cost, because it simply does not try to be portable. If the target is always DuckDB, this is a legitimate advantage.

It also means `pql` can stay closer to DuckDB's Python APIs and their behavior. The internal SQL layer is explicitly about wrapping DuckDB objects, not translating a backend-agnostic IR.

#### 4. Where Ibis is still stronger even for DuckDB users

Even if DuckDB is your only backend, Ibis still has some real strengths:

- the DuckDB backend is already extensive, with documented support for `read_csv`, `read_json`, `read_parquet`, `read_delta`, `read_postgres`, `read_mysql`, `read_sqlite`, `attach_sqlite`, `load_extension`, `register_filesystem`, `to_polars`, `to_pyarrow`, `to_torch`, `to_parquet`, `to_json`, and more in the [DuckDB backend docs](https://ibis-project.org/backends/duckdb),
- the expression API is richer and more mature today,
- and the generic relational API is much broader than `pql`'s current WIP surface.

There is also an important architectural nuance: Ibis's DuckDB backend is not a thin wrapper around SQL strings only. It has DuckDB-specific execution paths and backend-specific logic, such as DuckDB-native connection management, relation execution, export methods, extension loading, geospatial handling, and file registration in [`ibis.backends.duckdb.Backend`](https://github.com/ibis-project/ibis/blob/main/ibis/backends/duckdb/__init__.py). So the comparison is not "direct DuckDB wrappers" versus "purely generic SQL everywhere". It is more precisely:

- `pql`: DuckDB-native wrappers first, portable abstraction not attempted,
- Ibis: portable expression system first, with a substantial DuckDB backend implementation underneath.

#### Bottom line

If the goal is portability, backend switching, a large mature table-expression API, and a framework that already spans many engines, Ibis is the stronger project today.

If the goal is specifically "Polars-like ergonomics over DuckDB, with as little multi-backend abstraction tax as possible", then `pql` has a real and distinct design point.

So the honest comparison is:

- `Ibis` is broader, more mature, and architected as a portable expression framework.
- `pql` is narrower, earlier-stage, and architected as a DuckDB-first Polars-like wrapper layer.

## Architecture

The goal is to use as much as possible of the existing [Expression](https://duckdb.org/docs/stable/clients/python/expression) and [Relational](https://duckdb.org/docs/stable/clients/python/relational_api) API's.
raw SQL is only used when no other choice exist.

The `sql` folder is the internal API to create duckdb Expressions.
It handles data conversions, chainable API wrapping, and datatypes parsing and typing.
Then it is used in the top level modules who wraps this sql API to provide the user-facing API.

## Scripts

Scripts are used for code generation and API comparison at dev time.
They are not meant to be used by end users, and are not part of the public API.

More infos with the following command:

```shell
uv run -m scripts --help
```

### Comparator

The **compare** command will create the [coverage](API_COVERAGE.md) report to compare `pql` vs `polar`s and `narwhals` API's.

### Generators

The **gen-{fns, core, themes}** commands will respectively generate python code for:

- [The functions from the `table_functions` DuckDB table](src/pql/sql/_code_gen/_fns.py)
- [A core API wrapper around `DuckDBPyRelation` and `Expression`](src/pql/sql/_code_gen/_core.py)
- [A `Literal` for SQL display theming](src/pql/_typing.py) (see `Theme` type)

## References

- [DuckDB functions](https://duckdb.org/docs/stable/sql/functions/overview)
