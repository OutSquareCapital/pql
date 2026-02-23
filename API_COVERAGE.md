# pql vs Polars API Comparison Report

This report shows the API coverage of pql compared to Polars.

## Summary

| Class       | Coverage        | Total     | Matched  | Missing | Mismatched | Extra   |
| ----------- | --------------- | --------- | -------- | ------- | ---------- | ------- |
| LazyFrame   | (53.8%, 29.5%)  | (26, 88)  | (14, 26) | (2, 43) | (10, 19)   | (29, 8) |
| Expr        | (70.0%, 41.0%)  | (70, 217) | (49, 89) | (8, 99) | (13, 29)   | (57, 1) |
| LazyGroupBy | (0.0%, 5.9%)    | (1, 17)   | (0, 1)   | (0, 16) | (1, 0)     | (0, 0)  |
| Expr.str    | (89.5%, 24.5%)  | (19, 49)  | (17, 12) | (2, 24) | (0, 13)    | (9, 1)  |
| Expr.list   | (90.0%, 34.9%)  | (10, 43)  | (9, 15)  | (0, 25) | (1, 3)     | (9, 1)  |
| Expr.struct | (100.0%, 20.0%) | (1, 5)    | (1, 1)   | (0, 3)  | (0, 1)     | (2, 1)  |
| Expr.name   | (100.0%, 70.0%) | (6, 10)   | (6, 7)   | (0, 3)  | (0, 0)     | (2, 1)  |

## LazyFrame

### [x] Missing Methods (44)

- `approx_n_unique`
- `cache`
- `clear`
- `collect_async`
- `collect_batches`
- `describe`
- `deserialize`
- `drop_nans`
- `dtypes`
- `fetch`
- `fill_null`
- `group_by_dynamic`
- `inspect`
- `interpolate`
- `join_where`
- `last`
- `map_batches`
- `match_to_schema`
- `melt`
- `merge_sorted`
- `pipe_with_schema`
- `pivot`
- `profile`
- `remote`
- `remove`
- `reverse`
- `rolling`
- `select_seq`
- `serialize`
- `set_sorted`
- `show`
- `show_graph`
- `sink_batches`
- `sink_delta`
- `sink_ipc`
- `slice`
- `sql`
- `tail` (n: int) -> Self
- `to_native` () -> LazyFrameT
- `unnest`
- `update`
- `with_columns_seq`
- `with_context`
- `with_row_count`

### [!] Signature Mismatches (14)

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
- `filter` (nw)
  - Narwhals: (`*predicates: IntoExpr | Iterable[IntoExpr]`, **constraints: Any) -> Self
  - Polars: (`*predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool]`, **constraints: Any) -> LazyFrame
  - pql: (`*predicates: IntoExprColumn | Iterable[IntoExprColumn]`, `**constraints: IntoExpr`) -> Self
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

### [+] Extra Methods (pql-only) (8)

- `from_df`
- `from_mapping`
- `from_numpy`
- `from_query`
- `from_sequence`
- `from_table`
- `from_table_function`
- `inner`

## Expr

### [x] Missing Methods (100)

- `agg_groups`
- `any_value` (ignore_nulls: bool) -> Self
- `append`
- `arg_max`
- `arg_min`
- `arg_sort`
- `arg_true`
- `arg_unique`
- `arr`
- `bin`
- `bitwise_count_ones`
- `bitwise_count_zeros`
- `bitwise_leading_ones`
- `bitwise_leading_zeros`
- `bitwise_trailing_ones`
- `bitwise_trailing_zeros`
- `bottom_k`
- `bottom_k_by`
- `cat` ()
- `cumulative_eval`
- `cut`
- `deserialize`
- `dot`
- `drop_nans`
- `drop_nulls` () -> Self
- `dt` ()
- `entropy`
- `eq_missing`
- `ewm_mean` (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, min_samples: int, ignore_nulls: bool) -> Self
- `ewm_mean_by`
- `ewm_std`
- `ewm_var`
- `exclude`
- `explode`
- `ext`
- `extend_constant`
- `filter` (*predicates: Any) -> Self
- `flatten`
- `from_json`
- `gather`
- `gather_every`
- `get`
- `head`
- `hist`
- `index_of`
- `inspect`
- `interpolate`
- `interpolate_by`
- `item`
- `limit`
- `lower_bound`
- `map_batches` (function: Callable[[Any], CompliantExpr[Any, Any]], return_dtype: DType | None, returns_scalar: bool) -> Self
- `map_elements`
- `meta`
- `nan_max`
- `nan_min`
- `ne_missing`
- `peak_max`
- `peak_min`
- `qcut`
- `rechunk`
- `register_plugin`
- `reinterpret`
- `replace_strict` (old: Sequence[Any] | Mapping[Any, Any], new: Sequence[Any] | None, default: Any | NoDefault, return_dtype: IntoDType | None) -> Self
- `reshape`
- `reverse`
- `rle`
- `rle_id`
- `rolling`
- `rolling_kurtosis`
- `rolling_map`
- `rolling_max_by`
- `rolling_mean_by`
- `rolling_median_by`
- `rolling_min_by`
- `rolling_quantile`
- `rolling_quantile_by`
- `rolling_rank`
- `rolling_rank_by`
- `rolling_skew`
- `rolling_std_by`
- `rolling_sum_by`
- `rolling_var_by`
- `round_sig_figs`
- `sample`
- `search_sorted`
- `set_sorted`
- `shrink_dtype`
- `shuffle`
- `slice`
- `sort`
- `sort_by`
- `tail`
- `to_physical`
- `top_k`
- `top_k_by`
- `unique_counts`
- `upper_bound`
- `value_counts`
- `where`

### [!] Signature Mismatches (15)

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
- `last` (nw)
  - Narwhals: (`order_by: str | Iterable[str] | None`) -> Self
  - Polars: (`ignore_nulls: bool`) -> Expr
  - pql: () -> Self
- `mode` (nw)
  - Narwhals: (`keep: ModeKeepStrategy`) -> Self
  - Polars: (`maintain_order: bool`) -> Expr
  - pql: () -> Self
- `over` (nw)
  - Narwhals: (`*partition_by: str | Sequence[str]`, `order_by: str | Sequence[str] | None`) -> Self
  - Polars: (partition_by: IntoExpr | Iterable[IntoExpr] | None, *more_exprs: IntoExpr, order_by: IntoExpr | Iterable[IntoExpr] | None, descending: bool, nulls_last: bool, `mapping_strategy: WindowMappingStrategy`) -> Expr
  - pql: (`partition_by: IntoExpr | Iterable[IntoExpr] | None`, `*more_exprs: IntoExpr`, `order_by: IntoExpr | Iterable[IntoExpr] | None`, `descending: bool`, `nulls_last: bool`) -> Self
- `pct_change` (pl)
  - Polars: (`n: int | IntoExprColumn`) -> Expr
  - pql: (`n: int`) -> Self
- `pow` (pl)
  - Polars: (`exponent: IntoExprColumn | int | float`) -> Expr
  - pql: (`other: Any`) -> Self
- `repeat_by` (pl)
  - Polars: (`by: Series | Expr | str_ | int`) -> Expr
  - pql: (`by: Expr | int`) -> Self
- `replace` (pl)
  - Polars: (`old: IntoExpr | Sequence[Any] | Mapping[Any, Any]`, `new: IntoExpr | Sequence[Any] | NoDefault`, `default: IntoExpr | NoDefault`, `return_dtype: PolarsDataType | None`) -> Expr
  - pql: (`old: IntoExpr`, `new: IntoExpr`) -> Self
- `rolling_max` (pl)
  - Polars: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool) -> Expr
  - pql: (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_median` (pl)
  - Polars: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool) -> Expr
  - pql: (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_min` (pl)
  - Polars: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool) -> Expr
  - pql: (window_size: int, min_samples: int | None, center: bool) -> Self

### [+] Extra Methods (pql-only) (1)

- `inner`

## LazyGroupBy

### [x] Missing Methods (16)

- `all`
- `count`
- `first`
- `having`
- `head`
- `last`
- `len`
- `map_groups`
- `max`
- `mean`
- `median`
- `min`
- `n_unique`
- `quantile`
- `sum`
- `tail`

## Expr.str

### [x] Missing Methods (24)

- `concat`
- `contains_any`
- `decode`
- `encode`
- `escape_regex`
- `explode`
- `extract`
- `extract_groups`
- `extract_many`
- `find`
- `find_many`
- `join`
- `json_decode`
- `json_path_match`
- `normalize`
- `replace_many`
- `split_exact`
- `splitn`
- `strptime`
- `to_date` (format: str | None) -> ExprT
- `to_datetime` (format: str | None) -> ExprT
- `to_decimal`
- `to_integer`
- `to_time`

### [+] Extra Methods (pql-only) (1)

- `inner`

## Expr.list

### [x] Missing Methods (25)

- `agg`
- `arg_max`
- `arg_min`
- `concat`
- `count_matches`
- `diff`
- `drop_nulls`
- `explode`
- `filter`
- `gather`
- `gather_every`
- `head`
- `item`
- `join`
- `n_unique`
- `sample`
- `set_difference`
- `set_intersection`
- `set_symmetric_difference`
- `set_union`
- `shift`
- `slice`
- `tail`
- `to_array`
- `to_struct`

### [!] Signature Mismatches (1)

- `eval` (pl)
  - Polars: (expr: Expr, `parallel: bool`) -> Expr
  - pql: (expr: Expr) -> Expr

### [+] Extra Methods (pql-only) (1)

- `inner`

## Expr.struct

### [x] Missing Methods (3)

- `json_encode`
- `rename_fields`
- `unnest`

### [+] Extra Methods (pql-only) (1)

- `inner`

## Expr.name

### [x] Missing Methods (3)

- `map_fields`
- `prefix_fields`
- `suffix_fields`

### [+] Extra Methods (pql-only) (1)

- `inner`
