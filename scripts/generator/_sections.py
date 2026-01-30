from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Self

import pyochain as pc

from ._models import DUCKDB_TYPE_MAP, SHADOWERS, PyTypes

type Grouped = pc.Dict[str, pc.Seq[FunctionInfo]]
"""Grouped functions by category."""


def sections(functions: pc.Iter[FunctionInfo]) -> str:
    def _group_by_category_step(grouped: Grouped, func: FunctionInfo) -> Grouped:
        return grouped.insert(
            func.category,
            grouped.get_item(func.category)
            .map(lambda seq: pc.Seq((*seq, func)))
            .unwrap_or(pc.Seq((func,))),
        ).into(lambda _: grouped)

    return (
        functions.fold(
            pc.Dict[str, pc.Seq[FunctionInfo]].new(), _group_by_category_step
        )
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
    parameters: pc.Seq[pc.Option[str]]
    parameter_types: pc.Seq[str]
    varargs: pc.Option[str]
    description: pc.Option[str]
    min_param_count: int

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> Self:
        return cls(
            name=row[0],
            function_type=row[1],
            return_type=row[2] or "ANY",
            python_name=row[3],
            category=row[4],
            parameters=pc.Iter(row[5]).map(pc.Option.if_some).collect(),
            parameter_types=pc.Iter(row[6])
            .map(lambda dtype: pc.Option(dtype).unwrap_or("ANY"))
            .collect(),
            varargs=pc.Option(row[7]),
            description=pc.Option(row[8]),
            min_param_count=row[9],
        )

    def _docstring(self) -> str:
        desc = (
            self.description.map(lambda d: d.strip().replace("\u2019", "'").split(". "))
            .map(lambda s: pc.Iter(s).join(".\n\n    ").rstrip("."))
            .map(lambda spaced: f"{spaced}.")
            .unwrap_or(f"SQL {self.name} function.")
        )

        args_doc = (
            self.parameters.then(
                lambda p: p.into(_deduplicate_param_names)
                .iter()
                .zip(self.parameter_types)
                .filter_star(lambda name, _: bool(name))
                .map_star(_format_arg_doc)
                .join("\n")
            )
            .zip(self.varargs.map(lambda dtype: _format_arg_doc("*args", dtype)))
            .map(lambda lines: pc.Iter(lines).join("\n"))
            .map(lambda lines: f"\n\n    Args:\n{lines}")
            .unwrap_or("")
        )

        return f'    """{desc}{args_doc}\n\n    Returns:\n        SqlExpr: Result expression.\n    """'

    def _body(self) -> str:
        return (
            self.varargs.map(lambda _: "*args")
            .zip(
                self.parameters.then(
                    lambda p: p.into(_deduplicate_param_names).join(", ")
                )
            )
            .map(lambda pair: pc.Iter(pair).join(", "))
            .map(lambda args: f'    return func("{self.name}", {args})')
            .unwrap_or(f'    return func("{self.name}")')
        )

    def _signature(self) -> str:
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
                    (
                        self.varargs.is_none()
                        and _duckdb_param_py_type(dtype) == PyTypes.BOOL
                    ),
                )
            )
            .fold(_SignatureState(), _SignatureState.step)
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


def _deduplicate_param_names(params: pc.Seq[pc.Option[str]]) -> pc.Vec[str]:
    """Ensure all parameter names are unique by appending index if needed."""
    return params.iter().enumerate().fold(_ParamNameState(), _ParamNameState.step).names


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


@dataclass(slots=True)
class _ParamNameState:
    """Accumulator for parameter name de-duplication."""

    seen: pc.Dict[str, int] = field(default_factory=lambda: pc.Dict[str, int].new())
    names: pc.Vec[str] = field(default_factory=lambda: pc.Vec[str].new())

    def step(self, idx_param: tuple[int, pc.Option[str]]) -> Self:
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

    def step(self, param: ParamInfos) -> Self:
        """Fold step for signature assembly with keyword-only bools."""
        if param.is_bool and not self.has_kw_marker:
            self.parts.insert(self.parts.length(), "*")
            self.has_kw_marker = True
        self.parts.insert(self.parts.length(), param.to_formatted())
        return self


def _sanitize_param_name(name: pc.Option[str], idx: int) -> str:
    """Sanitize parameter name to be a valid Python identifier."""
    match name.map(lambda n: n.strip("'\"").split("(")[0].replace("...", "")):
        case pc.NONE:
            return f"arg{idx}"
        case pc.Some(shadower) if shadower in SHADOWERS:
            return f"{shadower}_arg"
        case pc.Some(identifier) if identifier.isidentifier():
            return identifier
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


def _duckdb_param_py_type(dtype: str) -> PyTypes:
    """Map a DuckDB type string to a Python type name."""
    return DUCKDB_TYPE_MAP.get_item(
        dtype.split("(")[0].split("[")[0].strip().upper()
    ).unwrap_or(PyTypes.EXPR)
