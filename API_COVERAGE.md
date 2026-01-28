# pql vs Polars API Comparison Report

This report shows the API coverage of pql compared to Polars.

## Summary

| Class | Coverage | Matched | Missing | Mismatched | Extra |
|-------|----------|---------|---------|------------|-------|
| LazyFrame | 26.9% | 7 | 8 | 11 | 27 |
| Expr | 24.6% | 17 | 48 | 4 | 39 |
| Expr.str | 58.8% | 10 | 1 | 6 | 10 |

## LazyFrame

### [v] Matched Methods (7)

- `collect_schema`
- `columns`
- `gather_every`
- `head`
- `pipe`
- `schema`
- `with_row_index`

### [x] Missing Methods (8)

- `explode` (columns: str | Sequence[str], more_columns: str) -> Self
- `group_by` (keys: IntoExpr | Iterable[IntoExpr], drop_null_keys: bool) -> LazyGroupBy[Self]
- `join` (other: Self, on: str | list[str] | None, how: JoinStrategy, left_on: str | list[str] | None, right_on: str | list[str] | None, suffix: str) -> Self
- `join_asof` (other: Self, left_on: str | None, right_on: str | None, on: str | None, by_left: str | list[str] | None, by_right: str | list[str] | None, by: str | list[str] | None, strategy: AsofJoinStrategy, suffix: str) -> Self
- `lazy` () -> Self
- `tail` (n: int) -> Self
- `to_native` () -> LazyFrameT
- `unpivot` (on: str | list[str] | None, index: str | list[str] | None, variable_name: str, value_name: str) -> Self

### [!] Signature Mismatches (11)

- `collect` (nw)
  - Narwhals: (backend: IntoBackend[Polars | Pandas | Arrow] | None, kwargs: Any) -> DataFrame[Any]
  - Polars: (type_coercion: bool, predicate_pushdown: bool, projection_pushdown: bool, simplify_expression: bool, slice_pushdown: bool, comm_subplan_elim: bool, comm_subexpr_elim: bool, cluster_with_columns: bool, collapse_joins: bool, no_optimization: bool, engine: EngineType, background: bool, optimizations: QueryOptFlags,_kwargs: Any) -> DataFrame | InProcessQuery
  - pql: () -> pl.DataFrame
- `drop` (nw)
  - Narwhals: (columns: str | Iterable[str], strict: bool) -> Self
  - Polars: (columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector], strict: bool) -> LazyFrame
  - pql: (columns: str) -> Self
- `drop_nulls` (nw)
  - Narwhals: (subset: str | list[str] | None) -> Self
  - Polars: (subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None) -> LazyFrame
  - pql: (subset: str | Iterable[str] | None) -> Self
- `filter` (nw)
  - Narwhals: (predicates: IntoExpr | Iterable[IntoExpr], constraints: Any) -> Self
  - Polars: (predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool], constraints: Any) -> LazyFrame
  - pql: (predicates: Expr) -> Self
- `rename` (nw)
  - Narwhals: (mapping: dict[str, str]) -> Self
  - Polars: (mapping: Mapping[str, str] | Callable[[str], str], strict: bool) -> LazyFrame
  - pql: (mapping: Mapping[str, str]) -> Self
- `select` (nw)
  - Narwhals: (exprs: IntoExpr | Iterable[IntoExpr], named_exprs: IntoExpr) -> Self
  - Polars: (exprs: IntoExpr | Iterable[IntoExpr], named_exprs: IntoExpr) -> LazyFrame
  - pql: (exprs: IntoExpr | Iterable[IntoExpr]) -> Self
- `sink_parquet` (nw)
  - Narwhals: (file: str | Path | BytesIO) -> None
  - Polars: (path: str | Path | IO[bytes] | _SinkDirectory | PartitionBy, compression: str, compression_level: int | None, statistics: bool | str | dict[str, bool], row_group_size: int | None, data_page_size: int | None, maintain_order: bool, storage_options: dict[str, Any] | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int, sync_on_close: SyncOnCloseMethod | None, metadata: ParquetMetadata | None, mkdir: bool, lazy: bool, field_overwrites: ParquetFieldOverwrites | Sequence[ParquetFieldOverwrites] | Mapping[str, ParquetFieldOverwrites] | None, engine: EngineType, optimizations: QueryOptFlags) -> LazyFrame | None
  - pql: (path: str | Path, compression: str) -> None
- `sort` (nw)
  - Narwhals: (by: str | Iterable[str], more_by: str, descending: bool | Sequence[bool], nulls_last: bool) -> Self
  - Polars: (by: IntoExpr | Iterable[IntoExpr], more_by: IntoExpr, descending: bool | Sequence[bool], nulls_last: bool | Sequence[bool], maintain_order: bool, multithreaded: bool) -> LazyFrame
  - pql: (by: IntoExpr | Iterable[IntoExpr], descending: bool | Iterable[bool], nulls_last: bool | Iterable[bool]) -> Self
- `top_k` (nw)
  - Narwhals: (k: int, by: str | Iterable[str], reverse: bool | Sequence[bool]) -> Self
  - Polars: (k: int, by: IntoExpr | Iterable[IntoExpr], reverse: bool | Sequence[bool]) -> LazyFrame
  - pql: (k: int, by: IntoExpr | Iterable[IntoExpr], reverse: bool) -> Self
- `unique` (nw)
  - Narwhals: (subset: str | list[str] | None, keep: UniqueKeepStrategy, order_by: str | Sequence[str] | None) -> Self
  - Polars: (subset: IntoExpr | Collection[IntoExpr] | None, keep: UniqueKeepStrategy, maintain_order: bool) -> LazyFrame
  - pql: (subset: str | Iterable[str] | None) -> Self
- `with_columns` (nw)
  - Narwhals: (exprs: IntoExpr | Iterable[IntoExpr], named_exprs: IntoExpr) -> Self
  - Polars: (exprs: IntoExpr | Iterable[IntoExpr], named_exprs: IntoExpr) -> LazyFrame
  - pql: (exprs: IntoExpr | Iterable[IntoExpr]) -> Self

### [+] Extra Methods (pql-only) (27)

- `bottom_k`
- `cast`
- `clone`
- `count`
- `drop_nans`
- `dtypes`
- `explain`
- `fill_nan`
- `fill_null`
- `first`
- `limit`
- `max`
- `mean`
- `median`
- `min`
- `null_count`
- `quantile`
- `relation`
- `reverse`
- `shift`
- `sink_csv`
- `sink_ndjson`
- `std`
- `sum`
- `var`
- `width`
- `with_row_count`

## Expr

### [v] Matched Methods (17)

- `abs`
- `alias`
- `ceil`
- `exp`
- `floor`
- `is_duplicated`
- `is_finite`
- `is_first_distinct`
- `is_last_distinct`
- `is_nan`
- `is_null`
- `is_unique`
- `log`
- `pipe`
- `sin`
- `sqrt`
- `str`

### [x] Missing Methods (48)

- `all` () -> Self
- `any` () -> Self
- `any_value` (ignore_nulls: bool) -> Self
- `cat` ()
- `clip` (lower_bound: IntoExpr | NumericLiteral | TemporalLiteral | None, upper_bound: IntoExpr | NumericLiteral | TemporalLiteral | None) -> Self
- `count` () -> Self
- `cum_count` (reverse: bool) -> Self
- `cum_max` (reverse: bool) -> Self
- `cum_min` (reverse: bool) -> Self
- `cum_prod` (reverse: bool) -> Self
- `cum_sum` (reverse: bool) -> Self
- `diff` () -> Self
- `drop_nulls` () -> Self
- `dt` ()
- `ewm_mean` (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, min_samples: int, ignore_nulls: bool) -> Self
- `fill_null` (value: Expr | NonNestedLiteral, strategy: FillNullStrategy | None, limit: int | None) -> Self
- `filter` (predicates: Any) -> Self
- `first` () -> Self
- `is_between` (lower_bound: Any | IntoExpr, upper_bound: Any | IntoExpr, closed: ClosedInterval) -> Self
- `is_close` (other: Expr | Series[Any] | NumericLiteral, abs_tol: float, rel_tol: float, nans_equal: bool) -> Self
- `kurtosis` () -> Self
- `last` () -> Self
- `len` () -> Self
- `list` ()
- `map_batches` (function: Callable[[Any], CompliantExpr[Any, Any]], return_dtype: DType | None, returns_scalar: bool) -> Self
- `max` () -> Self
- `mean` () -> Self
- `median` () -> Self
- `min` () -> Self
- `mode` (keep: ModeKeepStrategy) -> Self
- `n_unique` () -> Self
- `name` ()
- `null_count` () -> Self
- `over` (partition_by: str | Sequence[str], order_by: str | Sequence[str] | None) -> Self
- `quantile` (quantile: float, interpolation: RollingInterpolationMethod) -> Self
- `rank` (method: RankMethod, descending: bool) -> Self
- `replace_strict` (old: Sequence[Any] | Mapping[Any, Any], new: Sequence[Any] | None, default: Any | NoDefault, return_dtype: IntoDType | None) -> Self
- `rolling_mean` (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_std` (window_size: int, min_samples: int | None, center: bool, ddof: int) -> Self
- `rolling_sum` (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_var` (window_size: int, min_samples: int | None, center: bool, ddof: int) -> Self
- `shift` (n: int) -> Self
- `skew` () -> Self
- `std` (ddof: int) -> Self
- `struct` ()
- `sum` () -> Self
- `unique` () -> Self
- `var` (ddof: int) -> Self

### [!] Signature Mismatches (4)

- `cast` (nw)
  - Narwhals: (dtype: IntoDType) -> Self
  - Polars: (dtype: PolarsDataType | pl.DataTypeExpr | type[Any], strict: bool, wrap_numerical: bool) -> Expr
  - pql: (dtype: datatypes.DataType) -> Self
- `fill_nan` (nw)
  - Narwhals: (value: float | None) -> Self
  - Polars: (value: int | float | Expr | None) -> Expr
  - pql: (value: IntoExpr) -> Self
- `is_in` (nw)
  - Narwhals: (other: Any) -> Self
  - Polars: (other: Expr | Collection[Any] | Series, nulls_equal: bool) -> Expr
  - pql: (other: Collection[IntoExpr] | IntoExpr) -> Self
- `round` (nw)
  - Narwhals: (decimals: int) -> Self
  - Polars: (decimals: int, mode: RoundMode) -> Expr
  - pql: (decimals: int, mode: Literal['half_to_even', 'half_away_from_zero']) -> Self

### [+] Extra Methods (pql-only) (39)

- `add`
- `and_`
- `arctan`
- `backward_fill`
- `cbrt`
- `cos`
- `cosh`
- `degrees`
- `eq`
- `expr`
- `floordiv`
- `forward_fill`
- `ge`
- `gt`
- `hash`
- `interpolate`
- `is_infinite`
- `is_not_nan`
- `is_not_null`
- `le`
- `log10`
- `log1p`
- `lt`
- `mod`
- `mul`
- `ne`
- `neg`
- `not_`
- `or_`
- `pow`
- `radians`
- `repeat_by`
- `replace`
- `sign`
- `sinh`
- `sub`
- `tan`
- `tanh`
- `truediv`

## Expr.str

### [v] Matched Methods (10)

- `contains`
- `ends_with`
- `head`
- `len_chars`
- `slice`
- `starts_with`
- `tail`
- `to_lowercase`
- `to_titlecase`
- `to_uppercase`

### [x] Missing Methods (1)

- `zfill` (width: int) -> ExprT

### [!] Signature Mismatches (6)

- `replace` (nw)
  - Narwhals: (pattern: str, value: str | IntoExpr, literal: bool, n: int) -> ExprT
  - Polars: (pattern: str | Expr, value: str | Expr, literal: bool, n: int) -> Expr
  - pql: (pattern: str, replacement: str) -> Expr
- `replace_all` (nw)
  - Narwhals: (pattern: str, value: IntoExpr, literal: bool) -> ExprT
  - Polars: (pattern: str | Expr, value: str | Expr, literal: bool) -> Expr
  - pql: (pattern: str, value: str, literal: bool) -> Expr
- `split` (nw)
  - Narwhals: (by: str) -> ExprT
  - Polars: (by: IntoExpr, inclusive: bool) -> Expr
  - pql: (by: str, inclusive: bool) -> Expr
- `strip_chars` (nw)
  - Narwhals: (characters: str | None) -> ExprT
  - Polars: (characters: IntoExpr) -> Expr
  - pql: () -> Expr
- `to_date` (nw)
  - Narwhals: (format: str | None) -> ExprT
  - Polars: (format: str | None, strict: bool, exact: bool, cache: bool) -> Expr
  - pql: (fmt: str | None) -> Expr
- `to_datetime` (nw)
  - Narwhals: (format: str | None) -> ExprT
  - Polars: (format: str | None, time_unit: TimeUnit | None, time_zone: str | None, strict: bool, exact: bool, cache: bool, ambiguous: Ambiguous | Expr) -> Expr
  - pql: (fmt: str | None, time_unit: str) -> Expr

### [+] Extra Methods (pql-only) (10)

- `count_matches`
- `extract_all`
- `len_bytes`
- `reverse`
- `strip_chars_end`
- `strip_chars_start`
- `strip_prefix`
- `strip_suffix`
- `to_decimal`
- `to_time`
