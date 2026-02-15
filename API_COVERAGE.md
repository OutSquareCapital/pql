# pql vs Polars API Comparison Report

This report shows the API coverage of pql compared to Polars.

## Summary

| Class       | Coverage vs Narwhals | Total | Matched | Missing | Mismatched | Extra | Extra vs Narwhals |
| ----------- | -------------------- | ----- | ------- | ------- | ---------- | ----- | ----------------- |
| LazyFrame   | 56.2%                | 48    | 27      | 11      | 10         | 2     | 22                |
| Expr        | 53.8%                | 106   | 57      | 41      | 8          | 2     | 36                |
| Expr.str    | 82.8%                | 29    | 24      | 3       | 2          | 0     | 10                |
| Expr.list   | 100.0%               | 10    | 10      | 0       | 0          | 2     | 0                 |
| Expr.struct | 100.0%               | 1     | 1       | 0       | 0          | 2     | 0                 |

## LazyFrame

### [x] Missing Methods (11)

- `drop_nulls` (subset: str | list[str] | None) -> Self
- `explode` (columns: str | Sequence[str], *more_columns: str) -> Self
- `gather_every` (n: int, offset: int) -> Self
- `group_by` (*keys: IntoExpr | Iterable[IntoExpr], drop_null_keys: bool) -> LazyGroupBy[Self]
- `join` (other: Self, on: str | list[str] | None, how: JoinStrategy, left_on: str | list[str] | None, right_on: str | list[str] | None, suffix: str) -> Self
- `join_asof` (other: Self, left_on: str | None, right_on: str | None, on: str | None, by_left: str | list[str] | None, by_right: str | list[str] | None, by: str | list[str] | None, strategy: AsofJoinStrategy, suffix: str) -> Self
- `tail` (n: int) -> Self
- `to_native` () -> LazyFrameT
- `unique` (subset: str | list[str] | None, keep: UniqueKeepStrategy, order_by: str | Sequence[str] | None) -> Self
- `unpivot` (on: str | list[str] | None, index: str | list[str] | None, variable_name: str, value_name: str) -> Self
- `with_row_index` (name: str, order_by: str | Sequence[str]) -> Self

### [!] Signature Mismatches (10)

- `cast` (pl)
  - Polars: (`dtypes: Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType | PythonDataType] | PolarsDataType | pl.DataTypeExpr | Schema`, `strict: bool`) -> LazyFrame
  - pql: (`dtypes: Mapping[str, sql.datatypes.DataType] | sql.datatypes.DataType`) -> Self
- `collect` (nw)
  - Narwhals: (`backend: IntoBackend[Polars | Pandas | Arrow] | None`, `**kwargs: Any`) -> DataFrame[Any]
  - Polars: (`type_coercion: bool`, `predicate_pushdown: bool`, `projection_pushdown: bool`, `simplify_expression: bool`, `slice_pushdown: bool`, `comm_subplan_elim: bool`, `comm_subexpr_elim: bool`, `cluster_with_columns: bool`, `collapse_joins: bool`, `no_optimization: bool`, `engine: EngineType`, `background: bool`, `optimizations: QueryOptFlags`, `**_kwargs: Any`) -> DataFrame | InProcessQuery
  - pql: () -> DataFrame
- `drop` (nw)
  - Narwhals: (`*columns: str | Iterable[str]`, `strict: bool`) -> Self
  - Polars: (`*columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector]`, `strict: bool`) -> LazyFrame
  - pql: (`*columns: str`) -> Self
- `explain` (pl)
  - Polars: (`format: ExplainFormat`, `optimized: bool`, `type_coercion: bool`, `predicate_pushdown: bool`, `projection_pushdown: bool`, `simplify_expression: bool`, `slice_pushdown: bool`, `comm_subplan_elim: bool`, `comm_subexpr_elim: bool`, `cluster_with_columns: bool`, `collapse_joins: bool`, `streaming: bool`, `engine: EngineType`, `tree_format: bool | None`, `optimizations: QueryOptFlags`) -> str
  - pql: () -> str
- `quantile` (pl)
  - Polars: (`quantile: float | Expr`, `interpolation: QuantileMethod`) -> LazyFrame
  - pql: (`quantile: float`) -> Self
- `rename` (nw)
  - Narwhals: (`mapping: dict[str, str]`) -> Self
  - Polars: (`mapping: Mapping[str, str] | Callable[[str], str]`, `strict: bool`) -> LazyFrame
  - pql: (`mapping: Mapping[str, str]`) -> Self
- `shift` (pl)
  - Polars: (`n: int | IntoExprColumn`, fill_value: IntoExpr | None) -> LazyFrame
  - pql: (`n: int`, fill_value: IntoExpr | None) -> Self
- `sink_csv` (pl)
  - Polars: (`path: str | Path | IO[bytes] | IO[str] | PartitionBy`, `include_bom: bool`, `compression: Literal['uncompressed', 'gzip', 'zstd']`, `compression_level: int | None`, `check_extension: bool`, include_header: bool, separator: str, `line_terminator: str`, `quote_char: str`, `batch_size: int`, `datetime_format: str | None`, `date_format: str | None`, `time_format: str | None`, `float_scientific: bool | None`, `float_precision: int | None`, `decimal_comma: bool`, `null_value: str | None`, `quote_style: CsvQuoteStyle | None`, `maintain_order: bool`, `storage_options: StorageOptionsDict | None`, `credential_provider: CredentialProviderFunction | Literal['auto'] | None`, `retries: int | None`, `sync_on_close: SyncOnCloseMethod | None`, `mkdir: bool`, `lazy: bool`, `engine: EngineType`, `optimizations: QueryOptFlags`) -> LazyFrame | None
  - pql: (`path: str | Path`, separator: str, include_header: bool) -> None
- `sink_ndjson` (pl)
  - Polars: (`path: str | Path | IO[bytes] | IO[str] | PartitionBy`, `compression: Literal['uncompressed', 'gzip', 'zstd']`, `compression_level: int | None`, `check_extension: bool`, `maintain_order: bool`, `storage_options: StorageOptionsDict | None`, `credential_provider: CredentialProviderFunction | Literal['auto'] | None`, `retries: int | None`, `sync_on_close: SyncOnCloseMethod | None`, `mkdir: bool`, `lazy: bool`, `engine: EngineType`, `optimizations: QueryOptFlags`) -> LazyFrame | None
  - pql: (`path: str | Path`) -> None
- `sink_parquet` (nw)
  - Narwhals: (`file: str | Path | BytesIO`) -> None
  - Polars: (`path: str | Path | IO[bytes] | PartitionBy`, compression: str, `compression_level: int | None`, `statistics: bool | str | dict[str, bool]`, `row_group_size: int | None`, `data_page_size: int | None`, `maintain_order: bool`, `storage_options: StorageOptionsDict | None`, `credential_provider: CredentialProviderFunction | Literal['auto'] | None`, `retries: int | None`, `sync_on_close: SyncOnCloseMethod | None`, `metadata: ParquetMetadata | None`, `arrow_schema: ArrowSchemaExportable | None`, `mkdir: bool`, `lazy: bool`, `engine: EngineType`, `optimizations: QueryOptFlags`) -> LazyFrame | None
  - pql: (`path: str | Path`, `compression: str`) -> None

### [+] Extra Methods (pql-only) (2)

- `inner`
- `relation`

## Expr

### [x] Missing Methods (41)

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
- `dt` ()
- `ewm_mean` (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, min_samples: int, ignore_nulls: bool) -> Self
- `fill_null` (value: Expr | NonNestedLiteral, strategy: FillNullStrategy | None, limit: int | None) -> Self
- `first` (order_by: str | Iterable[str] | None) -> Self
- `is_close` (other: Expr | Series[Any] | NumericLiteral, abs_tol: float, rel_tol: float, nans_equal: bool) -> Self
- `kurtosis` () -> Self
- `last` (order_by: str | Iterable[str] | None) -> Self
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
- `over` (*partition_by: str | Sequence[str], order_by: str | Sequence[str] | None) -> Self
- `quantile` (quantile: float, interpolation: RollingInterpolationMethod) -> Self
- `rank` (method: RankMethod, descending: bool) -> Self
- `replace_strict` (old: Sequence[Any] | Mapping[Any, Any], new: Sequence[Any] | None, default: Any | NoDefault, return_dtype: IntoDType | None) -> Self
- `rolling_mean` (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_std` (window_size: int, min_samples: int | None, center: bool, ddof: int) -> Self
- `rolling_sum` (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_var` (window_size: int, min_samples: int | None, center: bool, ddof: int) -> Self
- `skew` () -> Self
- `std` (ddof: int) -> Self
- `sum` () -> Self
- `unique` () -> Self
- `var` (ddof: int) -> Self

### [!] Signature Mismatches (8)

- `backward_fill` (pl)
  - Polars: (`limit: int | None`) -> Expr
  - pql: () -> Self
- `cast` (nw)
  - Narwhals: (`dtype: IntoDType`) -> Self
  - Polars: (`dtype: DataTypeExpr | type[Any]`, `strict: bool`, `wrap_numerical: bool`) -> Expr
  - pql: (`dtype: DataType`) -> Self
- `forward_fill` (pl)
  - Polars: (`limit: int | None`) -> Expr
  - pql: () -> Self
- `hash` (pl)
  - Polars: (seed: int, `seed_1: int | None`, `seed_2: int | None`, `seed_3: int | None`) -> Expr
  - pql: (seed: int) -> Self
- `is_in` (nw)
  - Narwhals: (`other: Any`) -> Self
  - Polars: (`other: Expr | Collection[Any] | Series`, `nulls_equal: bool`) -> Expr
  - pql: (`other: Collection[IntoExpr] | IntoExpr`) -> Self
- `pow` (pl)
  - Polars: (`exponent: IntoExprColumn | int | float`) -> Expr
  - pql: (`other: Any`) -> Self
- `repeat_by` (pl)
  - Polars: (`by: Series | Expr | str_ | int`) -> Expr
  - pql: (`by: Expr | int`) -> Self
- `replace` (pl)
  - Polars: (`old: IntoExpr | Sequence[Any] | Mapping[Any, Any]`, `new: IntoExpr | Sequence[Any] | NoDefault`, `default: IntoExpr | NoDefault`, `return_dtype: PolarsDataType | None`) -> Expr
  - pql: (`old: IntoExpr`, `new: IntoExpr`) -> Self

### [+] Extra Methods (pql-only) (2)

- `expr`
- `inner`

## Expr.str

### [x] Missing Methods (3)

- `pad_end` (length: int, fill_char: str) -> ExprT
- `pad_start` (length: int, fill_char: str) -> ExprT
- `zfill` (width: int) -> ExprT

### [!] Signature Mismatches (2)

- `to_datetime` (nw)
  - Narwhals: (format: str | None) -> ExprT
  - Polars: (format: str | None, `time_unit: TimeUnit | None`, `time_zone: str | None`, `strict: bool`, `exact: bool`, `cache: bool`, `ambiguous: Ambiguous | Expr`) -> Expr
  - pql: (format: str | None, `time_unit: TimeUnit`) -> Expr
- `to_time` (pl)
  - Polars: (format: str | None, `strict: bool`, `cache: bool`) -> Expr
  - pql: (format: str | None) -> Expr

## Expr.list

### [+] Extra Methods (pql-only) (2)

- `inner`
- `pipe`

## Expr.struct

### [+] Extra Methods (pql-only) (2)

- `inner`
- `pipe`
