"""Generate typed SQL function wrappers from DuckDB introspection.

This script queries DuckDB's `duckdb_functions()` to extract all available
functions with their signatures, then generates a properly typed `fns.py`.

Usage:
    uv run scripts/generate_fns.py
    uv run scripts/generate_fns.py --output src/pql/sql/fns.py
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from textwrap import dedent
from typing import Annotated

import duckdb
import pyochain as pc
import typer


class PyTypes(StrEnum):
    """Python type names for DuckDB type mapping."""

    STR = auto()
    BOOL = auto()
    INT = auto()
    DATE = auto()
    FLOAT = auto()
    BYTES = auto()
    TIME = auto()
    DATETIME = auto()
    TIMEDELTA = auto()
    EXPR = "SqlExpr"


# DuckDB type -> Python type hint mapping
DUCKDB_TYPE_MAP = pc.Dict.from_kwargs(
    VARCHAR=PyTypes.STR,
    INTEGER=PyTypes.INT,
    BIGINT=PyTypes.INT,
    SMALLINT=PyTypes.INT,
    TINYINT=PyTypes.INT,
    HUGEINT=PyTypes.INT,
    UINTEGER=PyTypes.INT,
    UBIGINT=PyTypes.INT,
    USMALLINT=PyTypes.INT,
    UTINYINT=PyTypes.INT,
    UHUGEINT=PyTypes.INT,
    DOUBLE=PyTypes.FLOAT,
    FLOAT=PyTypes.FLOAT,
    REAL=PyTypes.FLOAT,
    DECIMAL=PyTypes.FLOAT,
    BOOLEAN=PyTypes.BOOL,
    DATE=PyTypes.DATE,
    TIME=PyTypes.TIME,
    TIMESTAMP=PyTypes.DATETIME,
    INTERVAL=PyTypes.TIMEDELTA,
    BLOB=PyTypes.BYTES,
    BIT=PyTypes.BYTES,
    UUID=PyTypes.STR,
    JSON=PyTypes.STR,
    # Generic/complex types - default to SqlExpr
    ANY=PyTypes.EXPR,
    LIST=PyTypes.EXPR,
    MAP=PyTypes.EXPR,
    STRUCT=PyTypes.EXPR,
    ARRAY=PyTypes.EXPR,
    UNION=PyTypes.EXPR,
)

FN_CATEGORY = pc.Dict.from_kwargs(
    scalar="Scalar Functions",
    aggregate="Aggregate Functions",
    table="Table Functions",
    macro="Macro Functions",
)

RESERVED = pc.Set(
    {
        "list",
        "map",
        "filter",
        "from",
        "lambda",
        "type",
        "format",
        "input",
        "min",
        "max",
        "str",
        "any",
        "all",
    }
)
# Python reserved keywords that need renaming
RESERVED_NAMES: pc.Set[str] = pc.Set(
    {
        "all",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
        # Builtins we don't want to shadow
        "sum",
        "min",
        "max",
        "abs",
        "len",
        "map",
        "range",
        "round",
        "format",
        "hash",
        "bin",
        "hex",
        "chr",
        "ascii",
        "pow",
        "any",
    }
)

# Functions to skip (internal, deprecated, or not useful via Python API)
SKIP_FUNCTIONS: pc.Set[str] = pc.Set(
    {
        # Internal functions
        "~~",
        "!~~",
        "~~~",
        "!~~~",
        "^@",
        "@",
        "||",
        "**",
        "!",
        # Already exposed differently
        "main.coalesce",
        "main.if",
    }
)
CATEGORY_PATTERNS: pc.Seq[tuple[str, str]] = pc.Seq(
    (
        ("list_", "List Functions"),
        ("array_", "Array Functions"),
        ("map_", "Map Functions"),
        ("struct_", "Struct Functions"),
        ("regexp_", "Regular Expression Functions"),
        ("string_", "Text Functions"),
        ("date_", "Date Functions"),
        ("time_", "Time Functions"),
        ("timestamp_", "Timestamp Functions"),
        ("enum_", "Enum Functions"),
        ("union_", "Union Functions"),
        ("json_", "JSON Functions"),
        ("to_", "Conversion Functions"),
        ("from_", "Conversion Functions"),
        ("is", "Predicate Functions"),
        ("bit_", "Bitwise Functions"),
    )
)


@dataclass(slots=True)
class FunctionInfo:
    """Metadata for a DuckDB function."""

    name: str
    function_type: str
    return_type: str
    python_name: str
    category: str
    parameters: pc.Seq[str]
    parameter_types: pc.Seq[str]
    description: str | None
    min_param_count: int
    has_varargs: bool = False

    def _docstring(self) -> str:
        """Generate docstring for function."""
        desc = self.description or f"SQL {self.name} function."

        args_doc = (
            self.parameters.then_some()
            .map(
                lambda p: _deduplicate_param_names(p)
                .iter()
                .zip(self.parameter_types)
                .filter_star(lambda name, _: bool(name))
                .map_star(lambda name, typ: f"        {name}: {typ} expression")
                .join("\n")
            )
            .map(lambda lines: f"\n\n    Args:\n{lines}" if lines else "")
            .unwrap_or("")
        )

        return f'    """{desc}{args_doc}\n\n    Returns:\n        SqlExpr: Result expression.\n    """'

    def _body(self) -> str:
        """Generate function body."""
        return (
            self.parameters.then_some()
            .map(lambda p: _deduplicate_param_names(p).iter().join(", "))
            .map(lambda args: f'    return func("{self.name}", {args})')
            .unwrap_or(f'    return func("{self.name}")')
        )

    def _signature(self) -> str:
        """Generate function signature with type hints."""
        return (
            self.parameters.then_some()
            .map(
                lambda p: _deduplicate_param_names(p)
                .iter()
                .enumerate()
                .map_star(
                    lambda idx, name: (
                        f"{name}: SqlExpr"
                        if idx < self.min_param_count
                        else f"{name}: SqlExpr | None = None"
                    )
                )
                .join(", ")
            )
            .map(lambda args: f"def {self.python_name}({args}, /) -> SqlExpr:")
            .unwrap_or(f"def {self.python_name}() -> SqlExpr:")
        )

    def generate_function(self) -> str:
        """Generate complete function definition."""
        return f"{self._signature()}\n{self._docstring()}\n{self._body()}"


@dataclass(slots=True)
class _ParamNameState:
    """Accumulator for parameter name de-duplication."""

    seen: pc.Dict[str, int] = field(default_factory=lambda: pc.Dict[str, int].new())
    names: pc.Vec[str] = field(default_factory=lambda: pc.Vec[str].new())


def _sanitize_param_name(name: str, idx: int) -> str:
    """Sanitize parameter name to be a valid Python identifier."""
    if not name:
        return f"arg{idx}"
    # Remove quotes, parentheses, and other problematic chars
    clean = name.strip("'\"").split("(")[0].replace("...", "")
    match clean:
        case kword if keyword.iskeyword(clean) or clean in RESERVED:
            return f"{kword}_"
        case identifier if identifier.isidentifier():
            return clean
        case _:
            return f"arg{idx}"


def _deduplicate_param_names(params: pc.Seq[str]) -> pc.Vec[str]:
    """Ensure all parameter names are unique by appending index if needed."""
    return (
        params.iter()
        .enumerate()
        .fold(_ParamNameState(), _deduplicate_param_names_step)
        .names
    )


def _deduplicate_param_names_step(
    state: _ParamNameState,
    idx_param: tuple[int, str],
) -> _ParamNameState:
    """Fold step for parameter name de-duplication."""
    idx, param = idx_param
    clean = _sanitize_param_name(param, idx)
    count = state.seen.get_item(clean).unwrap_or(-1) + 1
    state.seen.insert(clean, count)
    state.names.insert(state.names.length(), f"{clean}{count}" if count > 0 else clean)
    return state


def _fetch_functions() -> pc.Seq[FunctionInfo]:
    """Fetch all functions from DuckDB."""
    category_cases = (
        CATEGORY_PATTERNS.iter()
        .map_star(
            lambda prefix, label: f"WHEN function_name LIKE '{prefix}%' THEN '{label}'"
        )
        .join("\n")
    )
    type_cases = (
        FN_CATEGORY.items()
        .iter()
        .map_star(lambda ftype, label: f"WHEN function_type = '{ftype}' THEN '{label}'")
        .join("\n")
    )
    reserved_names = RESERVED_NAMES.iter().map(lambda name: f"'{name}'").join(", ")

    query = f"""
        WITH raw AS (
            SELECT
                function_name,
                function_type,
                return_type,
                CASE
                    WHEN function_name IN ({reserved_names})
                        THEN function_name || '_func'
                    ELSE function_name
                END AS python_name,
                coalesce(parameters, []) AS parameters,
                coalesce(parameter_types, []) AS parameter_types,
                description,
                coalesce(list_count(parameters), 0) AS param_len
            FROM duckdb_functions()
            WHERE function_type IN ('scalar', 'aggregate')
              AND function_name NOT LIKE '%##%'
              AND function_name NOT LIKE '\\_%' ESCAPE '\\'
              AND function_name ~ '^[A-Za-z_][A-Za-z0-9_]*$'
              AND function_name NOT IN ({SKIP_FUNCTIONS.iter().map(lambda name: f"'{name}'").join(", ")})
        ), ranked AS (
            SELECT
                function_name,
                function_type,
                return_type,
                python_name,
                parameters,
                parameter_types,
                description,
                min(param_len) OVER (PARTITION BY function_name) AS min_param_len,
                row_number() OVER (
                    PARTITION BY function_name
                    ORDER BY param_len DESC, parameter_types
                ) AS rn
            FROM raw
        )
        SELECT
            function_name,
            function_type,
            return_type,
            python_name,
            CASE
                {category_cases}
                {type_cases}
                ELSE 'Other Functions'
            END AS category,
            parameters,
            parameter_types,
            description,
            min_param_len
        FROM ranked
        WHERE rn = 1
        ORDER BY function_name, parameter_types, function_type
    """

    return (
        pc.Iter(duckdb.sql(query).fetchall())
        .map(
            lambda row: FunctionInfo(
                name=row[0],
                function_type=row[1],
                return_type=row[2] or "ANY",
                python_name=row[3],
                category=row[4],
                parameters=pc.Seq(row[5]),
                parameter_types=pc.Seq(row[6]),
                description=row[7],
                min_param_count=row[8],
            )
        )
        .collect()
    )


def _group_by_category(
    functions: pc.Seq[FunctionInfo],
) -> pc.Dict[str, pc.Seq[FunctionInfo]]:
    """Group functions by category."""
    return functions.iter().fold(
        pc.Dict[str, pc.Seq[FunctionInfo]].new(),
        _group_by_category_step,
    )


def _group_by_category_step(
    grouped: pc.Dict[str, pc.Seq[FunctionInfo]],
    func: FunctionInfo,
) -> pc.Dict[str, pc.Seq[FunctionInfo]]:
    """Fold step for grouping functions by category."""
    return grouped.insert(
        func.category,
        grouped.get_item(func.category)
        .map(lambda seq: pc.Seq((*seq, func)))
        .unwrap_or(pc.Seq((func,))),
    ).into(lambda _: grouped)


def _generate_file_content(functions: pc.Seq[FunctionInfo]) -> str:
    """Generate the complete fns.py file content."""
    header = dedent('''\
        """DuckDB SQL function wrappers with type hints.

        This file is AUTO-GENERATED by scripts/generate_fns.py
        Do not edit manually - regenerate with:
            uv run scripts/generate_fns.py

        Functions are extracted from DuckDB's duckdb_functions() introspection.
        """

        from __future__ import annotations

        from ._exprs import SqlExpr, func

        __all__ = [
    ''')

    # Generate __all__ list
    all_names = functions.iter().map(lambda f: f'    "{f.python_name}",').join("\n")

    # Group and generate functions
    sections = (
        functions.into(_group_by_category)
        .items()
        .iter()
        .sort(key=lambda kv: kv[0])
        .iter()
        .map_star(
            lambda category, funcs: f"\n\n# {'=' * 60}\n# {category}\n# {'=' * 60}\n\n"
            + funcs.iter().map(lambda f: f.generate_function()).join("\n\n\n")
        )
        .join("")
    )

    return f"{header}{all_names}\n]{sections}\n"


DEFAULT_OUTPUT = Path("src/pql/sql/_generated_fns.py")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    *,
    output: Annotated[Path, typer.Option("--output", "-o")] = DEFAULT_OUTPUT,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n")] = False,
) -> None:
    """Generate typed DuckDB function wrappers."""
    typer.echo("Fetching functions from DuckDB...")
    functions = _fetch_functions()
    typer.echo(f"Found {functions.length()} function signatures")
    content = _generate_file_content(functions)

    if dry_run:
        typer.echo("\n" + "=" * 60)
        typer.echo(content)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        typer.echo(f"Generated {output}")


if __name__ == "__main__":
    app()
