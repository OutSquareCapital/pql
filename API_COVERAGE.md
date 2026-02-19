# pql vs Polars API Comparison Report

This report shows the API coverage of pql compared to Polars.

## Summary

| Class       | Coverage vs Narwhals | Total | Matched | Missing | Mismatched | Extra | Extra vs Narwhals |
| ----------- | -------------------- | ----- | ------- | ------- | ---------- | ----- | ----------------- |
| LazyFrame   | 68.1%                | 47    | 32      | 2       | 13         | 1     | 21                |
| Expr        | 82.1%                | 106   | 87      | 7       | 12         | 2     | 36                |
| LazyGroupBy | 100.0%               | 1     | 1       | 0       | 0          | 0     | 0                 |
| Expr.str    | 92.6%                | 27    | 25      | 2       | 0          | 2     | 8                 |
| Expr.list   | 100.0%               | 10    | 10      | 0       | 0          | 2     | 0                 |
| Expr.struct | 100.0%               | 1     | 1       | 0       | 0          | 2     | 0                 |

## LazyFrame

### [x] Missing Methods (2)

- `tail` (n: int) -> Self
- `to_native` () -> LazyFrameT

### [!] Signature Mismatches (13)

- `cast` (pl)
  - Polars: (`dtypes: Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType | PythonDataType] | PolarsDataType | pl.DataTypeExpr | Schema`, `strict: bool`) -> LazyFrame
  - pql: (`dtypes: Mapping[str, DataType] | DataType`) -> Self
- `collect` (nw)
  - Narwhals: (`backend: IntoBackend[Polars | Pandas | Arrow] | None`, `**kwargs: Any`) -> DataFrame[Any]
  - Polars: (`type_coercion: bool`, `predicate_pushdown: bool`, `projection_pushdown: bool`, `simplify_expression: bool`, `slice_pushdown: bool`, `comm_subplan_elim: bool`, `comm_subexpr_elim: bool`, `cluster_with_columns: bool`, `collapse_joins: bool`, `no_optimization: bool`, `engine: EngineType`, `background: bool`, `optimizations: QueryOptFlags`, `**_kwargs: Any`) -> DataFrame | InProcessQuery
  - pql: () -> DataFrame
- `drop` (nw)
  - Narwhals: (`*columns: str | Iterable[str]`, `strict: bool`) -> Self
  - Polars: (`*columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector]`, `strict: bool`) -> LazyFrame
  - pql: (`*columns: str`) -> Self
- `drop_nulls` (nw)
  - Narwhals: (`subset: str | list[str] | None`) -> Self
  - Polars: (`subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None`) -> LazyFrame
  - pql: (`subset: str | Iterable[str] | None`) -> Self
- `explain` (pl)
  - Polars: (`format: ExplainFormat`, `optimized: bool`, `type_coercion: bool`, `predicate_pushdown: bool`, `projection_pushdown: bool`, `simplify_expression: bool`, `slice_pushdown: bool`, `comm_subplan_elim: bool`, `comm_subexpr_elim: bool`, `cluster_with_columns: bool`, `collapse_joins: bool`, `streaming: bool`, `engine: EngineType`, `tree_format: bool | None`, `optimizations: QueryOptFlags`) -> str
  - pql: () -> str
- `join` (nw)
  - Narwhals: (other: Self, `on: str | list[str] | None`, how: JoinStrategy, `left_on: str | list[str] | None`, `right_on: str | list[str] | None`, suffix: str) -> Self
  - Polars: (other: LazyFrame, `on: str | Expr | Sequence[str | Expr] | None`, how: JoinStrategy, `left_on: str | Expr | Sequence[str | Expr] | None`, `right_on: str | Expr | Sequence[str | Expr] | None`, suffix: str, `validate: JoinValidation`, `nulls_equal: bool`, `coalesce: bool | None`, `maintain_order: MaintainOrderJoin | None`, `allow_parallel: bool`, `force_parallel: bool`) -> LazyFrame
  - pql: (other: Self, `on: str | Iterable[str] | None`, how: JoinStrategy, `left_on: str | Iterable[str] | None`, `right_on: str | Iterable[str] | None`, suffix: str) -> Self
- `join_asof` (nw)
  - Narwhals: (other: Self, left_on: str | None, right_on: str | None, on: str | None, `by_left: str | list[str] | None`, `by_right: str | list[str] | None`, `by: str | list[str] | None`, strategy: AsofJoinStrategy, suffix: str) -> Self
  - Polars: (other: LazyFrame, `left_on: str | None | Expr`, `right_on: str | None | Expr`, `on: str | None | Expr`, `by_left: str | Sequence[str] | None`, `by_right: str | Sequence[str] | None`, `by: str | Sequence[str] | None`, strategy: AsofJoinStrategy, suffix: str, `tolerance: str | int | float | timedelta | None`, `allow_parallel: bool`, `force_parallel: bool`, `coalesce: bool`, `allow_exact_matches: bool`, `check_sortedness: bool`) -> LazyFrame
  - pql: (other: Self, left_on: str | None, right_on: str | None, on: str | None, `by_left: str | Iterable[str] | None`, `by_right: str | Iterable[str] | None`, `by: str | Iterable[str] | None`, strategy: AsofJoinStrategy, suffix: str) -> Self
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

### [+] Extra Methods (pql-only) (1)

- `inner`

## Expr

### [x] Missing Methods (7)

- `any_value` (ignore_nulls: bool) -> Self
- `cat` ()
- `dt` ()
- `ewm_mean` (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, min_samples: int, ignore_nulls: bool) -> Self
- `map_batches` (function: Callable[[Any], CompliantExpr[Any, Any]], return_dtype: DType | None, returns_scalar: bool) -> Self
- `name` ()
- `replace_strict` (old: Sequence[Any] | Mapping[Any, Any], new: Sequence[Any] | None, default: Any | NoDefault, return_dtype: IntoDType | None) -> Self

### [!] Signature Mismatches (12)

- `backward_fill` (pl)
  - Polars: (`limit: int | None`) -> Expr
  - pql: () -> Self
- `cast` (nw)
  - Narwhals: (`dtype: IntoDType`) -> Self
  - Polars: (`dtype: DataTypeExpr | type[Any]`, `strict: bool`, `wrap_numerical: bool`) -> Expr
  - pql: (`dtype: DataType`) -> Self
- `clip` (nw)
  - Narwhals: (`lower_bound: IntoExpr | NumericLiteral | TemporalLiteral | None`, `upper_bound: IntoExpr | NumericLiteral | TemporalLiteral | None`) -> Self
  - Polars: (`lower_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None`, `upper_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None`) -> Expr
  - pql: (`lower_bound: IntoExpr | None`, `upper_bound: IntoExpr | None`) -> Self
- `fill_null` (nw)
  - Narwhals: (`value: Expr | NonNestedLiteral`, strategy: FillNullStrategy | None, limit: int | None) -> Self
  - Polars: (`value: Any | Expr | None`, strategy: FillNullStrategy | None, limit: int | None) -> Expr
  - pql: (`value: IntoExpr | None`, strategy: FillNullStrategy | None, limit: int | None) -> Self
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
- `mode` (nw)
  - Narwhals: (`keep: ModeKeepStrategy`) -> Self
  - Polars: (`maintain_order: bool`) -> Expr
  - pql: () -> Self
- `over` (nw)
  - Narwhals: (`*partition_by: str | Sequence[str]`, `order_by: str | Sequence[str] | None`) -> Self
  - Polars: (partition_by: IntoExpr | Iterable[IntoExpr] | None, *more_exprs: IntoExpr, order_by: IntoExpr | Iterable[IntoExpr] | None, `descending: bool`, `nulls_last: bool`, `mapping_strategy: WindowMappingStrategy`) -> Expr
  - pql: (`partition_by: IntoExpr | Iterable[IntoExpr] | None`, `*more_exprs: IntoExpr`, `order_by: IntoExpr | Iterable[IntoExpr] | None`) -> Self
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

## LazyGroupBy

## Expr.str

### [x] Missing Methods (2)

- `to_date` (format: str | None) -> ExprT
- `to_datetime` (format: str | None) -> ExprT

### [+] Extra Methods (pql-only) (2)

- `inner`
- `pipe`

## Expr.list

### [+] Extra Methods (pql-only) (2)

- `inner`
- `pipe`

## Expr.struct

### [+] Extra Methods (pql-only) (2)

- `inner`
- `pipe`
