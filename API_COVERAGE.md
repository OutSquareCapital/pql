# pql vs Polars API Comparison Report

This report shows the API coverage of pql compared to Polars.

## Summary

| Class | Coverage | Matched | Missing | Mismatched | Extra |
|-------|----------|---------|---------|------------|-------|
| LazyFrame | 46.2% | 24 | 7 | 21 | 1 |
| Expr | 48.6% | 52 | 46 | 9 | 1 |
| Expr.str | 88.9% | 24 | 1 | 2 | 0 |
| Expr.list | 100.0% | 10 | 0 | 0 | 0 |
| Expr.struct | 100.0% | 1 | 0 | 0 | 0 |

## LazyFrame

### [v] Matched Methods (24)

- `clone`
- `collect_schema`
- `columns`
- `count`
- `dtypes`
- `first`
- `gather_every`
- `head`
- `lazy`
- `limit`
- `max`
- `mean`
- `median`
- `min`
- `null_count`
- `pipe`
- `reverse`
- `schema`
- `std`
- `sum`
- `var`
- `width`
- `with_row_count`
- `with_row_index`

### [x] Missing Methods (7)

- `explode` (columns: str | Sequence[str], more_columns: str) -> Self
- `group_by` (keys: IntoExpr | Iterable[IntoExpr], drop_null_keys: bool) -> LazyGroupBy[Self]
- `join` (other: Self, on: str | list[str] | None, how: JoinStrategy, left_on: str | list[str] | None, right_on: str | list[str] | None, suffix: str) -> Self
- `join_asof` (other: Self, left_on: str | None, right_on: str | None, on: str | None, by_left: str | list[str] | None, by_right: str | list[str] | None, by: str | list[str] | None, strategy: AsofJoinStrategy, suffix: str) -> Self
- `tail` (n: int) -> Self
- `to_native` () -> LazyFrameT
- `unpivot` (on: str | list[str] | None, index: str | list[str] | None, variable_name: str, value_name: str) -> Self

### [!] Signature Mismatches (21)

- `bottom_k` (pl)
  - Polars: (k: int, by: IntoExpr | Iterable[IntoExpr], **reverse: bool | Sequence[bool]**) -> LazyFrame
  - pql: (k: int, by: IntoExpr | Iterable[IntoExpr], **reverse: bool**) -> Self
- `cast` (pl)
  - Polars: (**dtypes: Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType | PythonDataType] | PolarsDataType | pl.DataTypeExpr | Schema**, **strict: bool**) -> LazyFrame
  - pql: (**dtypes: Mapping[str, datatypes.DataType] | datatypes.DataType**) -> Self
- `collect` (nw)
  - Narwhals: (**backend: IntoBackend[Polars | Pandas | Arrow] | None**, **kwargs: Any**) -> DataFrame[Any]
  - Polars: (**type_coercion: bool**, **predicate_pushdown: bool**, **projection_pushdown: bool**, **simplify_expression: bool**, **slice_pushdown: bool**, **comm_subplan_elim: bool**, **comm_subexpr_elim: bool**, **cluster_with_columns: bool**, **collapse_joins: bool**, **no_optimization: bool**, **engine: EngineType**, **background: bool**, **optimizations: QueryOptFlags**, **_kwargs: Any**) -> DataFrame | InProcessQuery
  - pql: () -> pl.DataFrame
- `drop` (nw)
  - Narwhals: (**columns: str | Iterable[str]**, **strict: bool**) -> Self
  - Polars: (**columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector]**, **strict: bool**) -> LazyFrame
  - pql: (**columns: str**) -> Self
- `drop_nans` (pl)
  - Polars: (**subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None**) -> LazyFrame
  - pql: (**subset: str | Iterable[str] | None**) -> Self
- `drop_nulls` (nw)
  - Narwhals: (**subset: str | list[str] | None**) -> Self
  - Polars: (**subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None**) -> LazyFrame
  - pql: (**subset: str | Iterable[str] | None**) -> Self
- `explain` (pl)
  - Polars: (**format: ExplainFormat**, **optimized: bool**, **type_coercion: bool**, **predicate_pushdown: bool**, **projection_pushdown: bool**, **simplify_expression: bool**, **slice_pushdown: bool**, **comm_subplan_elim: bool**, **comm_subexpr_elim: bool**, **cluster_with_columns: bool**, **collapse_joins: bool**, **streaming: bool**, **engine: EngineType**, **tree_format: bool | None**, **optimizations: QueryOptFlags**) -> str
  - pql: () -> str
- `fill_nan` (pl)
  - Polars: (**value: int | float | Expr | None**) -> LazyFrame
  - pql: (**value: float | Expr | None**) -> Self
- `fill_null` (pl)
  - Polars: (**value: Any | Expr | None**, **strategy: FillNullStrategy | None**, **limit: int | None**, **matches_supertype: bool**) -> LazyFrame
  - pql: (**value: IntoExpr**, **strategy: Literal['forward', 'backward'] | None**) -> Self
- `filter` (nw)
  - Narwhals: (**predicates: IntoExpr | Iterable[IntoExpr]**, **constraints: Any**) -> Self
  - Polars: (**predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool]**, **constraints: Any**) -> LazyFrame
  - pql: (**predicates: Expr**) -> Self
- `quantile` (pl)
  - Polars: (**quantile: float | Expr**, **interpolation: QuantileMethod**) -> LazyFrame
  - pql: (**quantile: float**) -> Self
- `rename` (nw)
  - Narwhals: (**mapping: dict[str, str]**) -> Self
  - Polars: (**mapping: Mapping[str, str] | Callable[[str], str]**, **strict: bool**) -> LazyFrame
  - pql: (**mapping: Mapping[str, str]**) -> Self
- `select` (nw)
  - Narwhals: (exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr**) -> Self
  - Polars: (exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr**) -> LazyFrame
  - pql: (exprs: IntoExpr | Iterable[IntoExpr]) -> Self
- `shift` (pl)
  - Polars: (**n: int | IntoExprColumn**, fill_value: IntoExpr | None) -> LazyFrame
  - pql: (**n: int**, fill_value: IntoExpr | None) -> Self
- `sink_csv` (pl)
  - Polars: (**path: str | Path | IO[bytes] | IO[str] | _SinkDirectory | PartitionBy**, **include_bom: bool**, include_header: bool, separator: str, **line_terminator: str**, **quote_char: str**, **batch_size: int**, **datetime_format: str | None**, **date_format: str | None**, **time_format: str | None**, **float_scientific: bool | None**, **float_precision: int | None**, **decimal_comma: bool**, **null_value: str | None**, **quote_style: CsvQuoteStyle | None**, **maintain_order: bool**, **storage_options: dict[str, Any] | None**, **credential_provider: CredentialProviderFunction | Literal['auto'] | None**, **retries: int**, **sync_on_close: SyncOnCloseMethod | None**, **mkdir: bool**, **lazy: bool**, **engine: EngineType**, **optimizations: QueryOptFlags**) -> LazyFrame | None
  - pql: (**path: str | Path**, separator: str, include_header: bool) -> None
- `sink_ndjson` (pl)
  - Polars: (**path: str | Path | IO[bytes] | IO[str] | _SinkDirectory | PartitionBy**, **maintain_order: bool**, **storage_options: dict[str, Any] | None**, **credential_provider: CredentialProviderFunction | Literal['auto'] | None**, **retries: int**, **sync_on_close: SyncOnCloseMethod | None**, **mkdir: bool**, **lazy: bool**, **engine: EngineType**, **optimizations: QueryOptFlags**) -> LazyFrame | None
  - pql: (**path: str | Path**) -> None
- `sink_parquet` (nw)
  - Narwhals: (**file: str | Path | BytesIO**) -> None
  - Polars: (**path: str | Path | IO[bytes] | _SinkDirectory | PartitionBy**, compression: str, **compression_level: int | None**, **statistics: bool | str | dict[str, bool]**, **row_group_size: int | None**, **data_page_size: int | None**, **maintain_order: bool**, **storage_options: dict[str, Any] | None**, **credential_provider: CredentialProviderFunction | Literal['auto'] | None**, **retries: int**, **sync_on_close: SyncOnCloseMethod | None**, **metadata: ParquetMetadata | None**, **mkdir: bool**, **lazy: bool**, **field_overwrites: ParquetFieldOverwrites | Sequence[ParquetFieldOverwrites] | Mapping[str, ParquetFieldOverwrites] | None**, **engine: EngineType**, **optimizations: QueryOptFlags**) -> LazyFrame | None
  - pql: (**path: str | Path**, **compression: str**) -> None
- `sort` (nw)
  - Narwhals: (**by: str | Iterable[str]**, **more_by: str**, **descending: bool | Sequence[bool]**, **nulls_last: bool**) -> Self
  - Polars: (by: IntoExpr | Iterable[IntoExpr], **more_by: IntoExpr**, **descending: bool | Sequence[bool]**, **nulls_last: bool | Sequence[bool]**, **maintain_order: bool**, **multithreaded: bool**) -> LazyFrame
  - pql: (**by: IntoExpr | Iterable[IntoExpr]**, **descending: bool | Iterable[bool]**, **nulls_last: bool | Iterable[bool]**) -> Self
- `top_k` (nw)
  - Narwhals: (k: int, **by: str | Iterable[str]**, **reverse: bool | Sequence[bool]**) -> Self
  - Polars: (k: int, by: IntoExpr | Iterable[IntoExpr], **reverse: bool | Sequence[bool]**) -> LazyFrame
  - pql: (k: int, **by: IntoExpr | Iterable[IntoExpr]**, **reverse: bool**) -> Self
- `unique` (nw)
  - Narwhals: (**subset: str | list[str] | None**, **keep: UniqueKeepStrategy**, **order_by: str | Sequence[str] | None**) -> Self
  - Polars: (**subset: IntoExpr | Collection[IntoExpr] | None**, **keep: UniqueKeepStrategy**, **maintain_order: bool**) -> LazyFrame
  - pql: (**subset: str | Iterable[str] | None**) -> Self
- `with_columns` (nw)
  - Narwhals: (exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr**) -> Self
  - Polars: (exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr**) -> LazyFrame
  - pql: (exprs: IntoExpr | Iterable[IntoExpr]) -> Self

### [+] Extra Methods (pql-only) (1)

- `relation`

## Expr

### [v] Matched Methods (52)

- `abs`
- `add`
- `alias`
- `and_`
- `arctan`
- `cbrt`
- `ceil`
- `cos`
- `cosh`
- `degrees`
- `eq`
- `exp`
- `fill_nan`
- `floor`
- `floordiv`
- `ge`
- `gt`
- `is_duplicated`
- `is_finite`
- `is_first_distinct`
- `is_infinite`
- `is_last_distinct`
- `is_nan`
- `is_not_nan`
- `is_not_null`
- `is_null`
- `is_unique`
- `le`
- `list`
- `log`
- `log10`
- `log1p`
- `lt`
- `mod`
- `mul`
- `ne`
- `neg`
- `not_`
- `or_`
- `pipe`
- `radians`
- `round`
- `sign`
- `sin`
- `sinh`
- `sqrt`
- `str`
- `struct`
- `sub`
- `tan`
- `tanh`
- `truediv`

### [x] Missing Methods (46)

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
- `sum` () -> Self
- `unique` () -> Self
- `var` (ddof: int) -> Self

### [!] Signature Mismatches (9)

- `backward_fill` (pl)
  - Polars: (**limit: int | None**) -> Expr
  - pql: () -> Self
- `cast` (nw)
  - Narwhals: (**dtype: IntoDType**) -> Self
  - Polars: (**dtype: PolarsDataType | pl.DataTypeExpr | type[Any]**, **strict: bool**, **wrap_numerical: bool**) -> Expr
  - pql: (**dtype: datatypes.DataType**) -> Self
- `forward_fill` (pl)
  - Polars: (**limit: int | None**) -> Expr
  - pql: () -> Self
- `hash` (pl)
  - Polars: (seed: int, **seed_1: int | None**, **seed_2: int | None**, **seed_3: int | None**) -> Expr
  - pql: (seed: int) -> Self
- `interpolate` (pl)
  - Polars: (**method: InterpolationMethod**) -> Expr
  - pql: () -> Self
- `is_in` (nw)
  - Narwhals: (**other: Any**) -> Self
  - Polars: (**other: Expr | Collection[Any] | Series**, **nulls_equal: bool**) -> Expr
  - pql: (**other: Collection[IntoExpr] | IntoExpr**) -> Self
- `pow` (pl)
  - Polars: (**exponent: IntoExprColumn | int | float**) -> Expr
  - pql: (**other: Any**) -> Self
- `repeat_by` (pl)
  - Polars: (**by: pl.Series | Expr | str | int**) -> Expr
  - pql: (**by: Expr | int**) -> Self
- `replace` (pl)
  - Polars: (**old: IntoExpr | Sequence[Any] | Mapping[Any, Any]**, **new: IntoExpr | Sequence[Any] | NoDefault**, **default: IntoExpr | NoDefault**, **return_dtype: PolarsDataType | None**) -> Expr
  - pql: (**old: IntoExpr**, **new: IntoExpr**) -> Self

### [+] Extra Methods (pql-only) (1)

- `expr`

## Expr.str

### [v] Matched Methods (24)

- `contains`
- `count_matches`
- `ends_with`
- `extract_all`
- `head`
- `len_bytes`
- `len_chars`
- `replace`
- `replace_all`
- `reverse`
- `slice`
- `split`
- `starts_with`
- `strip_chars`
- `strip_chars_end`
- `strip_chars_start`
- `strip_prefix`
- `strip_suffix`
- `tail`
- `to_date`
- `to_decimal`
- `to_lowercase`
- `to_titlecase`
- `to_uppercase`

### [x] Missing Methods (1)

- `zfill` (width: int) -> ExprT

### [!] Signature Mismatches (2)

- `to_datetime` (nw)
  - Narwhals: (format: str | None) -> ExprT
  - Polars: (format: str | None, **time_unit: TimeUnit | None**, **time_zone: str | None**, **strict: bool**, **exact: bool**, **cache: bool**, **ambiguous: Ambiguous | Expr**) -> Expr
  - pql: (format: str | None, **time_unit: str**) -> Expr
- `to_time` (pl)
  - Polars: (format: str | None, **strict: bool**, **cache: bool**) -> Expr
  - pql: (format: str | None) -> Expr

## Expr.list

### [v] Matched Methods (10)

- `contains`
- `get`
- `len`
- `max`
- `mean`
- `median`
- `min`
- `sort`
- `sum`
- `unique`

## Expr.struct

### [v] Matched Methods (1)

- `field`
