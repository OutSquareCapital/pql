"""Generate typed SQL function wrappers from DuckDB introspection."""

from __future__ import annotations

import keyword
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import partial
from pathlib import Path
from textwrap import dedent
from typing import Annotated, NamedTuple, Self

import duckdb
import pyochain as pc
import typer

type Grouped = pc.Dict[str, pc.Seq[FunctionInfo]]
"""Grouped functions by category."""


DEFAULT_OUTPUT = Path("src", "pql", "sql", "fns.py")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output: Annotated[Path, typer.Option("--output", "-o")] = DEFAULT_OUTPUT,
) -> None:
    """Generate typed DuckDB function wrappers."""
    typer.echo("Fetching functions from DuckDB...")
    content = _run_pipeline()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    typer.echo(f"Generated {output}")
    _run_ruff(output)
    typer.echo("Done!")


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
    LIST = "list[object]"
    DICT = "dict[object, object]"
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
    ANY=PyTypes.EXPR,
    LIST=PyTypes.LIST,
    MAP=PyTypes.DICT,
    STRUCT=PyTypes.DICT,
    ARRAY=PyTypes.LIST,
    UNION=PyTypes.EXPR,
)


FN_CATEGORY = pc.Dict.from_kwargs(
    scalar="Scalar Functions",
    aggregate="Aggregate Functions",
    table="Table Functions",
    macro="Macro Functions",
)
"""Mapping of DuckDB function types to categories."""
KWORDS = pc.Set(keyword.kwlist)
"""Python reserved keywords that need renaming when generating function names."""
BUILTINS = pc.Set(dir(__builtins__))
AMBIGUOUS = pc.Set("l")
SHADOWERS = KWORDS.union(BUILTINS).union(AMBIGUOUS)
"""Python built-in names."""
SKIP_FUNCTIONS = pc.Set(
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
        "main.if",
    }
)
"""Functions to skip (internal, deprecated, or not useful via Python API)."""
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
"""Patterns to categorize functions based on their name prefixes."""


def _run_ruff(output: Path) -> None:
    typer.echo("Running Ruff checks and format...")
    uv_args = ("uv", "run", "ruff")
    run_ruff = partial(subprocess.run, check=False)
    run_ruff((*uv_args, "check", "--fix", "--unsafe-fixes", str(output)))
    run_ruff((*uv_args, "format", str(output)))


def _run_pipeline() -> str:
    return (
        pc.Iter(duckdb.sql(_get_query()).fetchall())
        .map(
            lambda row: FunctionInfo(
                name=row[0],
                function_type=row[1],
                return_type=row[2] or "ANY",
                python_name=row[3],
                category=row[4],
                parameters=pc.Seq(row[5]),
                parameter_types=pc.Iter(row[6])
                .map(lambda dtype: pc.Option(dtype).unwrap_or("ANY"))
                .collect(),
                varargs=pc.Option(row[7]),
                description=pc.Option(row[8]),
                min_param_count=row[9],
            )
        )
        .collect()
        .inspect(lambda fns: typer.echo(f"Found {fns.length()} function signatures"))
        .into(lambda fns: f"{_header()}{_all_names(fns)}\n]{_sections(fns)}\n")
    )


def _all_names(functions: pc.Seq[FunctionInfo]) -> str:
    return functions.iter().map(lambda f: f'    "{f.python_name}",').join("\n")


def _sections(functions: pc.Seq[FunctionInfo]) -> str:
    def _group_by_category_step(grouped: Grouped, func: FunctionInfo) -> Grouped:
        return grouped.insert(
            func.category,
            grouped.get_item(func.category)
            .map(lambda seq: pc.Seq((*seq, func)))
            .unwrap_or(pc.Seq((func,))),
        ).into(lambda _: grouped)

    return (
        functions.iter()
        .fold(pc.Dict[str, pc.Seq[FunctionInfo]].new(), _group_by_category_step)
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
    varargs: pc.Option[str]
    description: pc.Option[str]
    min_param_count: int

    def _docstring(self) -> str:
        """Generate docstring for function."""
        desc = (
            self.description.map(lambda d: d.strip().replace("\u2019", "'").split(". "))
            .map(lambda s: pc.Iter(s).join(".\n\n    ").rstrip("."))
            .map(lambda spaced: f"{spaced}.")
            .unwrap_or(f"SQL {self.name} function.")
        )

        args_lines = self.parameters.then(
            lambda p: p.into(_deduplicate_param_names)
        ).map(
            lambda seq: seq.iter()
            .zip(self.parameter_types)
            .filter_star(lambda name, _: bool(name))
            .map_star(_format_arg_doc)
            .join("\n")
        )
        varargs_line = self.varargs.map(lambda dtype: _format_arg_doc("*args", dtype))
        args_doc = (
            pc.Iter((args_lines, varargs_line))
            .filter_map(lambda x: x)
            .then(lambda x: x.join("\n"))
            .map(lambda lines: f"\n\n    Args:\n{lines}")
            .unwrap_or("")
        )

        return f'    """{desc}{args_doc}\n\n    Returns:\n        SqlExpr: Result expression.\n    """'

    def _body(self) -> str:
        """Generate function body."""
        varargs = self.varargs.map(lambda _: "*args")
        base_args = self.parameters.then_some().map(
            lambda p: p.into(_deduplicate_param_names).join(", ")
        )
        args = pc.Iter((base_args, varargs)).filter_map(lambda x: x).join(", ")
        return (
            f'    return func("{self.name}", {args})'
            if args
            else f'    return func("{self.name}")'
        )

    def _signature(self) -> str:
        """Generate function signature with type hints."""
        has_varargs = self.varargs.map(lambda _: 1).unwrap_or(0) == 1
        varargs = self.varargs.map(
            lambda dtype: f"*args: {_format_varargs_type(dtype)}"
        )
        base_args = self.parameters.then_some().map(
            lambda p: p.into(_deduplicate_param_names)
            .iter()
            .enumerate()
            .zip(self.parameter_types)
            .map_star(
                lambda idx_name, dtype: ParamInfos(
                    idx_name[1],
                    dtype,
                    idx_name[0] >= self.min_param_count,
                    (not has_varargs) and _duckdb_param_py_type(dtype) == PyTypes.BOOL,
                )
            )
            .fold(_SignatureState(), _signature_step)
            .parts.join(", ")
        )
        return (
            pc.Iter((base_args, varargs))
            .filter_map(lambda x: x)
            .then(lambda x: x.join(", "))
            .map(lambda args: f"def {self.python_name}({args}) -> SqlExpr:")
            .unwrap_or(f"def {self.python_name}() -> SqlExpr:")
        )

    def generate_function(self) -> str:
        """Generate complete function definition."""
        return f"{self._signature()}\n{self._docstring()}\n{self._body()}"


def _deduplicate_param_names(params: pc.Seq[str]) -> pc.Vec[str]:
    """Ensure all parameter names are unique by appending index if needed."""
    return params.iter().enumerate().fold(_ParamNameState(), _ParamNameState.step).names


def _signature_step(state: _SignatureState, param: ParamInfos) -> _SignatureState:
    """Fold step for signature assembly with keyword-only bools."""
    if param.is_bool and not state.has_kw_marker:
        state.parts.insert(state.parts.length(), "*")
        state.has_kw_marker = True
    state.parts.insert(state.parts.length(), param.to_formatted())
    return state


def _format_arg_doc(name: str, dtype: str) -> str:
    py_type = _duckdb_param_py_type(dtype)
    match py_type:
        case PyTypes.EXPR:
            return f"        {name} (SqlExpr): `{dtype}` expression"
        case _:
            return f"        {name} (SqlExpr | {py_type}): `{dtype}` expression"


def _format_varargs_type(dtype: str) -> str:
    py_type = _duckdb_param_py_type(dtype)
    match py_type:
        case PyTypes.EXPR:
            return py_type
        case _:
            return f"SqlExpr | {py_type}"


def _duckdb_param_py_type(dtype: str) -> PyTypes:
    """Map a DuckDB type string to a Python type name."""
    return DUCKDB_TYPE_MAP.get_item(
        dtype.split("(")[0].split("[")[0].strip().upper()
    ).unwrap_or(PyTypes.EXPR)


@dataclass(slots=True)
class _ParamNameState:
    """Accumulator for parameter name de-duplication."""

    seen: pc.Dict[str, int] = field(default_factory=lambda: pc.Dict[str, int].new())
    names: pc.Vec[str] = field(default_factory=lambda: pc.Vec[str].new())

    def step(self, idx_param: tuple[int, str]) -> Self:
        """Fold step for parameter name de-duplication."""
        idx, param = idx_param
        clean = _sanitize_param_name(param, idx)
        count = self.seen.get_item(clean).unwrap_or(-1) + 1
        self.seen.insert(clean, count)
        self.names.insert(
            self.names.length(), f"{clean}{count}" if count > 0 else clean
        )
        return self


@dataclass(slots=True)
class _SignatureState:
    """Accumulator for signature building."""

    parts: pc.Vec[str] = field(default_factory=lambda: pc.Vec[str].new())
    has_kw_marker: bool = False


def _sanitize_param_name(name: str, idx: int) -> str:
    """Sanitize parameter name to be a valid Python identifier."""
    if not name:
        return f"arg{idx}"
    clean = name.strip("'\"").split("(")[0].replace("...", "")
    match clean:
        case shadower if clean in SHADOWERS:
            return f"{shadower}_arg"
        case identifier if identifier.isidentifier():
            return clean
        case _:
            return f"arg{idx}"


class ParamInfos(NamedTuple):
    """Signature metadata for a parameter."""

    name: str
    dtype: str
    optional: bool
    is_bool: bool

    def to_formatted(self) -> str:
        """Format a single parameter signature entry."""
        py_type = _duckdb_param_py_type(self.dtype)
        match py_type:
            case PyTypes.EXPR:
                base_type = py_type
            case _:
                base_type = f"SqlExpr | {py_type}"
        return (
            f"{self.name}: {base_type} | None = None"
            if self.optional
            else f"{self.name}: {base_type}"
        )


def _get_query() -> str:
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
    reserved_names = KWORDS.iter().map(lambda name: f"'{name}'").join(", ")
    skipped_funcs = SKIP_FUNCTIONS.iter().map(lambda name: f"'{name}'").join(", ")

    return f"""
        WITH raw AS (
            SELECT
                function_name,
                function_type,
                return_type,
                lower(
                    regexp_replace(
                        CASE
                            WHEN function_name IN ({reserved_names})
                                THEN function_name || '_func'
                            ELSE function_name
                        END,
                        '([a-z0-9])([A-Z])',
                        '\\1_\\2',
                        'g'
                    )
                ) AS python_name,
                coalesce(parameters, []) AS parameters,
                coalesce(parameter_types, []) AS parameter_types,
                varargs,
                description,
                coalesce(list_count(parameters), 0) AS param_len
            FROM duckdb_functions()
            WHERE function_type IN ('scalar', 'aggregate', 'macro')
              AND function_name NOT LIKE '%##%'
              AND function_name NOT LIKE '\\_%' ESCAPE '\\'
              AND function_name ~ '^[A-Za-z_][A-Za-z0-9_]*$'
              AND function_name NOT IN ({skipped_funcs})
        ), ranked AS (
            SELECT
                function_name,
                function_type,
                return_type,
                python_name,
                parameters,
                parameter_types,
                varargs,
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
            varargs,
            description,
            min_param_len
        FROM ranked
        WHERE rn = 1
        ORDER BY function_name, parameter_types, function_type
    """


def _header() -> str:
    return dedent('''\
        """DuckDB SQL function wrappers with type hints.

        This file is AUTO-GENERATED by scripts/generate_fns.py
        Do not edit manually - regenerate with:
            uv run scripts/generate_fns.py

        Functions are extracted from DuckDB's duckdb_functions() introspection.
        """

        from __future__ import annotations

        from datetime import date, datetime, time, timedelta

        from ._exprs import SqlExpr, func

        __all__ = [
    ''')


if __name__ == "__main__":
    app()
