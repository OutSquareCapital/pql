# pql vs Polars API Comparison Report

This report shows the API coverage of pql compared to Polars.

## Summary

| Class       | Coverage        | Total     | Matched  | Missing | Mismatched | Extra   |
| ----------- | --------------- | --------- | -------- | ------- | ---------- | ------- |
| LazyFrame   | (57.7%, 34.9%)  | (26, 83)  | (15, 29) | (1, 33) | (10, 21)   | (33, 8) |
| Expr        | (70.0%, 42.3%)  | (70, 213) | (49, 90) | (8, 94) | (13, 29)   | (58, 1) |
| LazyGroupBy | (0.0%, 50.0%)   | (1, 16)   | (0, 8)   | (0, 4)  | (1, 4)     | (11, 0) |
| Expr.str    | (100.0%, 38.3%) | (19, 47)  | (19, 18) | (0, 10) | (0, 19)    | (19, 1) |
| Expr.list   | (90.0%, 41.9%)  | (10, 43)  | (9, 18)  | (0, 21) | (1, 4)     | (13, 1) |
| Expr.struct | (100.0%, 40.0%) | (1, 5)    | (1, 2)   | (0, 2)  | (0, 1)     | (3, 1)  |
| Expr.name   | (100.0%, 70.0%) | (6, 10)   | (6, 7)   | (0, 3)  | (0, 0)     | (2, 1)  |
| Expr.arr    | (100.0%, 54.8%) | (0, 31)   | (0, 17)  | (0, 10) | (0, 4)     | (23, 2) |

## LazyFrame

### [x] Missing Methods (34)

- `cache`
  - **Polars**: () -> LazyFrame
- `clear`
  - **Polars**: (n: int) -> LazyFrame
- `collect_async`
  - **Polars**: (gevent: bool, engine: EngineType, optimizations: QueryOptFlags) -> Awaitable[DataFrame] | _GeventDataFrameResult[DataFrame]
- `collect_batches`
  - **Polars**: (chunk_size: int | None, maintain_order: bool, lazy: bool, engine: EngineType, optimizations: QueryOptFlags) -> Iterator[DataFrame]
- `deserialize`
  - **Polars**: (source: str | bytes | Path | IOBase, format: SerializationFormat) -> LazyFrame
- `drop_nans`
  - **Polars**: (subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None) -> LazyFrame
- `dtypes`
  - **Polars**: ()
- `group_by_dynamic`
  - **Polars**: (index_column: IntoExpr, every: str | timedelta, period: str | timedelta | None, offset: str | timedelta | None, include_boundaries: bool, closed: ClosedInterval, label: Label, group_by: IntoExpr | Iterable[IntoExpr] | None, start_by: StartBy) -> LazyGroupBy
- `inspect`
  - **Polars**: (fmt: str) -> LazyFrame
- `interpolate`
  - **Polars**: () -> LazyFrame
- `join_where`
  - **Polars**: (other: LazyFrame, *predicates: Expr | Iterable[Expr], suffix: str) -> LazyFrame
- `map_batches`
  - **Polars**: (function: Callable[[DataFrame], DataFrame], predicate_pushdown: bool, projection_pushdown: bool, slice_pushdown: bool, no_optimizations: bool, schema: None | SchemaDict, validate_output_schema: bool, streamable: bool) -> LazyFrame
- `match_to_schema`
  - **Polars**: (schema: SchemaDict | Schema, missing_columns: Literal['insert', 'raise'] | Mapping[str, Literal['insert', 'raise'] | Expr], missing_struct_fields: Literal['insert', 'raise'] | Mapping[str, Literal['insert', 'raise']], extra_columns: Literal['ignore', 'raise'], extra_struct_fields: Literal['ignore', 'raise'] | Mapping[str, Literal['ignore', 'raise']], integer_cast: Literal['upcast', 'forbid'] | Mapping[str, Literal['upcast', 'forbid']], float_cast: Literal['upcast', 'forbid'] | Mapping[str, Literal['upcast', 'forbid']]) -> LazyFrame
- `merge_sorted`
  - **Polars**: (other: LazyFrame, key: str) -> LazyFrame
- `pipe_with_schema`
  - **Polars**: (function: Callable[[LazyFrame, Schema], LazyFrame]) -> LazyFrame
- `pivot`
  - **Polars**: (on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector], on_columns: Sequence[Any] | pl.Series | pl.DataFrame, index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None, values: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None, aggregate_function: PivotAgg | Expr | None, maintain_order: bool, separator: str) -> LazyFrame
- `profile`
  - **Polars**: (type_coercion: bool, predicate_pushdown: bool, projection_pushdown: bool, simplify_expression: bool, no_optimization: bool, slice_pushdown: bool, comm_subplan_elim: bool, comm_subexpr_elim: bool, cluster_with_columns: bool, collapse_joins: bool, show_plot: bool, truncate_nodes: int, figsize: tuple[int, int], engine: EngineType, optimizations: QueryOptFlags, **_kwargs: Any) -> tuple[DataFrame, DataFrame]
- `remote`
  - **Polars**: (context: ComputeContext | None, plan_type: PlanTypePreference, n_retries: int, engine: Engine, scaling_mode: ScalingMode) -> LazyFrameRemote
- `remove`
  - **Polars**: (*predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool] | np.ndarray[Any, Any], **constraints: Any) -> LazyFrame
- `reverse`
  - **Polars**: () -> LazyFrame
- `rolling`
  - **Polars**: (index_column: IntoExpr, period: str | timedelta, offset: str | timedelta | None, closed: ClosedInterval, group_by: IntoExpr | Iterable[IntoExpr] | None) -> LazyGroupBy
- `select_seq`
  - **Polars**: (*exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> LazyFrame
- `serialize`
  - **Polars**: (file: IOBase | str | Path | None, format: SerializationFormat) -> bytes | str | None
- `set_sorted`
  - **Polars**: (column: str | list[str], *more_columns: str, descending: bool | list[bool], nulls_last: bool | list[bool]) -> LazyFrame
- `show`
  - **Polars**: (limit: int | None, ascii_tables: bool | None, decimal_separator: str | None, thousands_separator: str | bool | None, float_precision: int | None, fmt_float: FloatFmt | None, fmt_str_lengths: int | None, fmt_table_cell_list_len: int | None, tbl_cell_alignment: Literal['LEFT', 'CENTER', 'RIGHT'] | None, tbl_cell_numeric_alignment: Literal['LEFT', 'CENTER', 'RIGHT'] | None, tbl_cols: int | None, tbl_column_data_type_inline: bool | None, tbl_dataframe_shape_below: bool | None, tbl_formatting: TableFormatNames | None, tbl_hide_column_data_types: bool | None, tbl_hide_column_names: bool | None, tbl_hide_dtype_separator: bool | None, tbl_hide_dataframe_shape: bool | None, tbl_width_chars: int | None, trim_decimal_zeros: bool | None) -> None
- `show_graph`
  - **Polars**: (optimized: bool, show: bool, output_path: str | Path | None, raw_output: bool, figsize: tuple[float, float], type_coercion: bool,_type_check: bool, predicate_pushdown: bool, projection_pushdown: bool, simplify_expression: bool, slice_pushdown: bool, comm_subplan_elim: bool, comm_subexpr_elim: bool, cluster_with_columns: bool, collapse_joins: bool, engine: EngineType, plan_stage: PlanStage,_check_order: bool, optimizations: QueryOptFlags) -> str | None
- `sink_batches`
  - **Polars**: (function: Callable[[DataFrame], bool | None], chunk_size: int | None, maintain_order: bool, lazy: bool, engine: EngineType, optimizations: QueryOptFlags) -> LazyFrame | None
- `sink_delta`
  - **Polars**: (target: DeltaTable, mode: Literal['error', 'append', 'overwrite', 'ignore', 'merge'], storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, delta_write_options: dict[str, Any] | None, delta_merge_options: dict[str, Any] | None, optimizations: QueryOptFlags) -> TableMerger | None
- `sink_ipc`
  - **Polars**: (path: str | Path | IO[bytes] | PartitionBy, compression: IpcCompression | None, compat_level: CompatLevel | None, record_batch_size: int | None, maintain_order: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, sync_on_close: SyncOnCloseMethod | None, mkdir: bool, lazy: bool, engine: EngineType, optimizations: QueryOptFlags, _record_batch_statistics: bool) -> LazyFrame | None
- `sql`
  - **Polars**: (query: str, table_name: str) -> LazyFrame
- `to_native`
  - **Narwhals**: () -> LazyFrameT
- `unnest`
  - **Polars**: (columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector], *more_columns: ColumnNameOrSelector, separator: str | None) -> LazyFrame
- `update`
  - **Polars**: (other: LazyFrame, on: str | Sequence[str] | None, how: Literal['left', 'inner', 'full'], left_on: str | Sequence[str] | None, right_on: str | Sequence[str] | None, include_nulls: bool, maintain_order: MaintainOrderJoin | None) -> LazyFrame
- `with_columns_seq`
  - **Polars**: (*exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> LazyFrame

### [!] Signature Mismatches (16)

- `cast` (pl)
  - **Polars***: (`dtypes: Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType | PythonDataType] | PolarsDataType | pl.DataTypeExpr | Schema`, `strict: bool`) -> LazyFrame
  - **pql**: (`dtypes: Mapping[str, DataType] | DataType`) -> Self
- `collect` (nw)
  - **Narwhals**: (`backend: IntoBackend[Polars | Pandas | Arrow] | None`, `**kwargs: Any`) -> DataFrame[Any]
  - **Polars**: (`type_coercion: bool`, `predicate_pushdown: bool`, `projection_pushdown: bool`, `simplify_expression: bool`, `slice_pushdown: bool`, `comm_subplan_elim: bool`, `comm_subexpr_elim: bool`, `cluster_with_columns: bool`, `collapse_joins: bool`, `no_optimization: bool`, `engine: EngineType`, `background: bool`, `optimizations: QueryOptFlags`, `**_kwargs: Any`) -> DataFrame | InProcessQuery
  - **pql**: () -> DataFrame
- `describe` (pl)
  - **Polars***: (`percentiles: Sequence[float] | float | None`, `interpolation: QuantileMethod`) -> DataFrame
  - **pql**: () -> Self
- `drop` (nw)
  - **Narwhals**: (`*columns: str | Iterable[str]`, `strict: bool`) -> Self
  - **Polars**: (`*columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector]`, `strict: bool`) -> LazyFrame
  - **pql**: (`*columns: str`) -> Self
- `drop_nulls` (nw)
  - **Narwhals**: (`subset: str | list[str] | None`) -> Self
  - **Polars**: (`subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None`) -> LazyFrame
  - **pql**: (`subset: str | Iterable[str] | None`) -> Self
- `explain` (pl)
  - **Polars***: (`format: ExplainFormat`, `optimized: bool`, `type_coercion: bool`, `predicate_pushdown: bool`, `projection_pushdown: bool`, `simplify_expression: bool`, `slice_pushdown: bool`, `comm_subplan_elim: bool`, `comm_subexpr_elim: bool`, `cluster_with_columns: bool`, `collapse_joins: bool`, `streaming: bool`, `engine: EngineType`, `tree_format: bool | None`, `optimizations: QueryOptFlags`) -> str
  - **pql**: () -> str
- `fill_null` (pl)
  - **Polars***: (`value: Any | Expr | None`, strategy: FillNullStrategy | None, limit: int | None, `matches_supertype: bool`) -> LazyFrame
  - **pql**: (`value: IntoExpr | None`, strategy: FillNullStrategy | None, limit: int | None) -> Self
- `filter` (nw)
  - **Narwhals**: (`*predicates: IntoExpr | Iterable[IntoExpr]`, **constraints: Any) -> Self
  - **Polars**: (`*predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool]`, **constraints: Any) -> LazyFrame
  - **pql**: (`*predicates: IntoExprColumn | Iterable[IntoExprColumn]`, `**constraints: IntoExpr`) -> Self
- `join` (nw)
  - **Narwhals**: (other: Self, `on: str | list[str] | None`, how: JoinStrategy, `left_on: str | list[str] | None`, `right_on: str | list[str] | None`, suffix: str) -> Self
  - **Polars**: (other: LazyFrame, `on: str | Expr | Sequence[str | Expr] | None`, how: JoinStrategy, `left_on: str | Expr | Sequence[str | Expr] | None`, `right_on: str | Expr | Sequence[str | Expr] | None`, suffix: str, `validate: JoinValidation`, `nulls_equal: bool`, `coalesce: bool | None`, `maintain_order: MaintainOrderJoin | None`, `allow_parallel: bool`, `force_parallel: bool`) -> LazyFrame
  - **pql**: (other: Self, `on: str | Iterable[str] | None`, how: JoinStrategy, `left_on: str | Iterable[str] | None`, `right_on: str | Iterable[str] | None`, suffix: str) -> Self
- `join_asof` (nw)
  - **Narwhals**: (other: Self, left_on: str | None, right_on: str | None, on: str | None, `by_left: str | list[str] | None`, `by_right: str | list[str] | None`, `by: str | list[str] | None`, strategy: AsofJoinStrategy, suffix: str) -> Self
  - **Polars**: (other: LazyFrame, `left_on: str | None | Expr`, `right_on: str | None | Expr`, `on: str | None | Expr`, `by_left: str | Sequence[str] | None`, `by_right: str | Sequence[str] | None`, `by: str | Sequence[str] | None`, strategy: AsofJoinStrategy, suffix: str, `tolerance: str | int | float | timedelta | None`, `allow_parallel: bool`, `force_parallel: bool`, `coalesce: bool`, `allow_exact_matches: bool`, `check_sortedness: bool`) -> LazyFrame
  - **pql**: (other: Self, left_on: str | None, right_on: str | None, on: str | None, `by_left: str | Iterable[str] | None`, `by_right: str | Iterable[str] | None`, `by: str | Iterable[str] | None`, strategy: AsofJoinStrategy, suffix: str) -> Self
- `quantile` (pl)
  - **Polars***: (`quantile: float | Expr`, `interpolation: QuantileMethod`) -> LazyFrame
  - **pql**: (`quantile: float`) -> Self
- `rename` (nw)
  - **Narwhals**: (`mapping: dict[str, str]`) -> Self
  - **Polars**: (`mapping: Mapping[str, str] | Callable[[str], str]`, `strict: bool`) -> LazyFrame
  - **pql**: (`mapping: Mapping[str, str]`) -> Self
- `shift` (pl)
  - **Polars***: (`n: int | IntoExprColumn`, fill_value: IntoExpr | None) -> LazyFrame
  - **pql**: (`n: int`, fill_value: IntoExpr | None) -> Self
- `sink_csv` (pl)
  - **Polars***: (`path: str | Path | IO[bytes] | IO[str] | PartitionBy`, `include_bom: bool`, `compression: Literal['uncompressed', 'gzip', 'zstd']`, `compression_level: int | None`, `check_extension: bool`, include_header: bool, separator: str, `line_terminator: str`, `quote_char: str`, `batch_size: int`, `datetime_format: str | None`, `date_format: str | None`, `time_format: str | None`, `float_scientific: bool | None`, `float_precision: int | None`, `decimal_comma: bool`, `null_value: str | None`, `quote_style: CsvQuoteStyle | None`, `maintain_order: bool`, `storage_options: StorageOptionsDict | None`, `credential_provider: CredentialProviderFunction | Literal['auto'] | None`, `retries: int | None`, `sync_on_close: SyncOnCloseMethod | None`, `mkdir: bool`, `lazy: bool`, `engine: EngineType`, `optimizations: QueryOptFlags`) -> LazyFrame | None
  - **pql**: (`path: str | Path`, separator: str, include_header: bool) -> None
- `sink_ndjson` (pl)
  - **Polars***: (`path: str | Path | IO[bytes] | IO[str] | PartitionBy`, `compression: Literal['uncompressed', 'gzip', 'zstd']`, `compression_level: int | None`, `check_extension: bool`, `maintain_order: bool`, `storage_options: StorageOptionsDict | None`, `credential_provider: CredentialProviderFunction | Literal['auto'] | None`, `retries: int | None`, `sync_on_close: SyncOnCloseMethod | None`, `mkdir: bool`, `lazy: bool`, `engine: EngineType`, `optimizations: QueryOptFlags`) -> LazyFrame | None
  - **pql**: (`path: str | Path`) -> None
- `sink_parquet` (nw)
  - **Narwhals**: (`file: str | Path | BytesIO`) -> None
  - **Polars**: (`path: str | Path | IO[bytes] | PartitionBy`, compression: str, `compression_level: int | None`, `statistics: bool | str | dict[str, bool]`, `row_group_size: int | None`, `data_page_size: int | None`, `maintain_order: bool`, `storage_options: StorageOptionsDict | None`, `credential_provider: CredentialProviderFunction | Literal['auto'] | None`, `retries: int | None`, `sync_on_close: SyncOnCloseMethod | None`, `metadata: ParquetMetadata | None`, `arrow_schema: ArrowSchemaExportable | None`, `mkdir: bool`, `lazy: bool`, `engine: EngineType`, `optimizations: QueryOptFlags`) -> LazyFrame | None
  - **pql**: (`path: str | Path`, `compression: str`) -> None

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

### [x] Missing Methods (95)

- `agg_groups`
  - **Polars**: () -> Expr
- `any_value`
  - **Narwhals**: (ignore_nulls: bool) -> Self
- `append`
  - **Polars**: (other: IntoExpr, upcast: bool) -> Expr
- `arg_max`
  - **Polars**: () -> Expr
- `arg_min`
  - **Polars**: () -> Expr
- `arg_sort`
  - **Polars**: (descending: bool, nulls_last: bool) -> Expr
- `arg_true`
  - **Polars**: () -> Expr
- `arg_unique`
  - **Polars**: () -> Expr
- `bin`
  - **Polars**: ()
- `bitwise_count_ones`
  - **Polars**: () -> Expr
- `bitwise_count_zeros`
  - **Polars**: () -> Expr
- `bitwise_leading_ones`
  - **Polars**: () -> Expr
- `bitwise_leading_zeros`
  - **Polars**: () -> Expr
- `bitwise_trailing_ones`
  - **Polars**: () -> Expr
- `bitwise_trailing_zeros`
  - **Polars**: () -> Expr
- `bottom_k`
  - **Polars**: (k: int | IntoExprColumn) -> Expr
- `bottom_k_by`
  - **Polars**: (by: IntoExpr | Iterable[IntoExpr], k: int | IntoExprColumn, reverse: bool | Sequence[bool]) -> Expr
- `cat`
  - **Narwhals**: ()
  - **Polars**: ()
- `cumulative_eval`
  - **Polars**: (expr: Expr, min_samples: int) -> Expr
- `cut`
  - **Polars**: (breaks: Sequence[float], labels: Sequence[str_] | None, left_closed: bool, include_breaks: bool) -> Expr
- `deserialize`
  - **Polars**: (source: str_ | Path | IOBase | bytes, format: SerializationFormat) -> Expr
- `dot`
  - **Polars**: (other: Expr | str_) -> Expr
- `drop_nans`
  - **Polars**: () -> Expr
- `drop_nulls`
  - **Narwhals**: () -> Self
  - **Polars**: () -> Expr
- `dt`
  - **Narwhals**: ()
  - **Polars**: ()
- `entropy`
  - **Polars**: (base: float, normalize: bool) -> Expr
- `eq_missing`
  - **Polars**: (other: Any) -> Expr
- `ewm_mean`
  - **Narwhals**: (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, min_samples: int, ignore_nulls: bool) -> Self
  - **Polars**: (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, min_samples: int, ignore_nulls: bool) -> Expr
- `ewm_mean_by`
  - **Polars**: (by: str_| IntoExpr, half_life: str_ | timedelta) -> Expr
- `ewm_std`
  - **Polars**: (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, bias: bool, min_samples: int, ignore_nulls: bool) -> Expr
- `ewm_var`
  - **Polars**: (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, bias: bool, min_samples: int, ignore_nulls: bool) -> Expr
- `exclude`
  - **Polars**: (columns: str_| PolarsDataType | Collection[str_ | PolarsDataType], *more_columns: str_ | PolarsDataType) -> Expr
- `explode`
  - **Polars**: (empty_as_null: bool, keep_nulls: bool) -> Expr
- `ext`
  - **Polars**: ()
- `extend_constant`
  - **Polars**: (value: IntoExpr, n: int | IntoExprColumn) -> Expr
- `filter`
  - **Narwhals**: (*predicates: Any) -> Self
  - **Polars**: (*predicates: IntoExprColumn | Iterable[IntoExprColumn], **constraints: Any) -> Expr
- `from_json`
  - **Polars**: (value: str_) -> Expr
- `gather`
  - **Polars**: (indices: int | Sequence[int] | IntoExpr | Series | np.ndarray[Any, Any]) -> Expr
- `gather_every`
  - **Polars**: (n: int, offset: int) -> Expr
- `get`
  - **Polars**: (index: int | Expr, null_on_oob: bool) -> Expr
- `head`
  - **Polars**: (n: int | Expr) -> Expr
- `hist`
  - **Polars**: (bins: IntoExpr | None, bin_count: int | None, include_category: bool, include_breakpoint: bool) -> Expr
- `index_of`
  - **Polars**: (element: IntoExpr) -> Expr
- `inspect`
  - **Polars**: (fmt: str_) -> Expr
- `interpolate`
  - **Polars**: (method: InterpolationMethod) -> Expr
- `interpolate_by`
  - **Polars**: (by: IntoExpr) -> Expr
- `item`
  - **Polars**: (allow_empty: bool) -> Expr
- `limit`
  - **Polars**: (n: int | Expr) -> Expr
- `lower_bound`
  - **Polars**: () -> Expr
- `map_batches`
  - **Narwhals**: (function: Callable[[Any], CompliantExpr[Any, Any]], return_dtype: DType | None, returns_scalar: bool) -> Self
  - **Polars**: (function: Callable[[Series], Series | Any], return_dtype: DataTypeExpr | None, agg_list: bool, is_elementwise: bool, returns_scalar: bool) -> Expr
- `map_elements`
  - **Polars**: (function: Callable[[Any], Any], return_dtype: DataTypeExpr | None, skip_nulls: bool, pass_name: bool, strategy: MapElementsStrategy, returns_scalar: bool) -> Expr
- `meta`
  - **Polars**: ()
- `nan_max`
  - **Polars**: () -> Expr
- `nan_min`
  - **Polars**: () -> Expr
- `ne_missing`
  - **Polars**: (other: Any) -> Expr
- `peak_max`
  - **Polars**: () -> Expr
- `peak_min`
  - **Polars**: () -> Expr
- `qcut`
  - **Polars**: (quantiles: Sequence[float] | int, labels: Sequence[str_] | None, left_closed: bool, allow_duplicates: bool, include_breaks: bool) -> Expr
- `rechunk`
  - **Polars**: () -> Expr
- `reinterpret`
  - **Polars**: (signed: bool) -> Expr
- `replace_strict`
  - **Narwhals**: (old: Sequence[Any] | Mapping[Any, Any], new: Sequence[Any] | None, default: Any | NoDefault, return_dtype: IntoDType | None) -> Self
  - **Polars**: (old: IntoExpr | Sequence[Any] | Mapping[Any, Any], new: IntoExpr | Sequence[Any] | NoDefault, default: IntoExpr | NoDefault, return_dtype: DataTypeExpr | None) -> Expr
- `reshape`
  - **Polars**: (dimensions: tuple[int, ...]) -> Expr
- `reverse`
  - **Polars**: () -> Expr
- `rle`
  - **Polars**: () -> Expr
- `rle_id`
  - **Polars**: () -> Expr
- `rolling`
  - **Polars**: (index_column: IntoExprColumn, period: str_ | timedelta, offset: str_ | timedelta | None, closed: ClosedInterval) -> Expr
- `rolling_kurtosis`
  - **Polars**: (window_size: int, fisher: bool, bias: bool, min_samples: int | None, center: bool) -> Expr
- `rolling_map`
  - **Polars**: (function: Callable[[Series], Any], window_size: int, weights: list_[float] | None, min_samples: int | None, center: bool) -> Expr
- `rolling_max_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_mean_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_median_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_min_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_quantile`
  - **Polars**: (quantile: float, interpolation: QuantileMethod, window_size: int, weights: list_[float] | None, min_samples: int | None, center: bool) -> Expr
- `rolling_quantile_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, quantile: float, interpolation: QuantileMethod, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_rank`
  - **Polars**: (window_size: int, method: RankMethod, seed: int | None, min_samples: int | None, center: bool) -> Expr
- `rolling_rank_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, method: RankMethod, seed: int | None, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_skew`
  - **Polars**: (window_size: int, bias: bool, min_samples: int | None, center: bool) -> Expr
- `rolling_std_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval, ddof: int) -> Expr
- `rolling_sum_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_var_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval, ddof: int) -> Expr
- `round_sig_figs`
  - **Polars**: (digits: int) -> Expr
- `sample`
  - **Polars**: (n: int | IntoExprColumn | None, fraction: float | IntoExprColumn | None, with_replacement: bool, shuffle: bool, seed: int | None) -> Expr
- `search_sorted`
  - **Polars**: (element: ndarray[Any, Any], side: SearchSortedSide, descending: bool) -> Expr
- `set_sorted`
  - **Polars**: (descending: bool) -> Expr
- `shuffle`
  - **Polars**: (seed: int | None) -> Expr
- `slice`
  - **Polars**: (offset: int | Expr, length: int | Expr | None) -> Expr
- `sort`
  - **Polars**: (descending: bool, nulls_last: bool) -> Expr
- `sort_by`
  - **Polars**: (by: IntoExpr | Iterable[IntoExpr], *more_by: IntoExpr, descending: bool | Sequence[bool], nulls_last: bool | Sequence[bool], multithreaded: bool, maintain_order: bool) -> Expr
- `tail`
  - **Polars**: (n: int | Expr) -> Expr
- `to_physical`
  - **Polars**: () -> Expr
- `top_k`
  - **Polars**: (k: int | IntoExprColumn) -> Expr
- `top_k_by`
  - **Polars**: (by: IntoExpr | Iterable[IntoExpr], k: int | IntoExprColumn, reverse: bool | Sequence[bool]) -> Expr
- `unique_counts`
  - **Polars**: () -> Expr
- `upper_bound`
  - **Polars**: () -> Expr
- `value_counts`
  - **Polars**: (sort: bool, parallel: bool, name: str_ | None, normalize: bool) -> Expr

### [!] Signature Mismatches (15)

- `cast` (nw)
  - **Narwhals**: (`dtype: IntoDType`) -> Self
  - **Polars**: (`dtype: DataTypeExpr | type[Any]`, `strict: bool`, `wrap_numerical: bool`) -> Expr
  - **pql**: (`dtype: DataType`) -> Self
- `clip` (nw)
  - **Narwhals**: (`lower_bound: IntoExpr | NumericLiteral | TemporalLiteral | None`, `upper_bound: IntoExpr | NumericLiteral | TemporalLiteral | None`) -> Self
  - **Polars**: (`lower_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None`, `upper_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None`) -> Expr
  - **pql**: (`lower_bound: IntoExpr | None`, `upper_bound: IntoExpr | None`) -> Self
- `fill_null` (nw)
  - **Narwhals**: (`value: Expr | NonNestedLiteral`, strategy: FillNullStrategy | None, limit: int | None) -> Self
  - **Polars**: (`value: Any | Expr | None`, strategy: FillNullStrategy | None, limit: int | None) -> Expr
  - **pql**: (`value: IntoExpr | None`, strategy: FillNullStrategy | None, limit: int | None) -> Self
- `forward_fill` (pl)
  - **Polars***: (`limit: int | None`) -> Expr
  - **pql**: () -> Self
- `hash` (pl)
  - **Polars***: (seed: int, `seed_1: int | None`, `seed_2: int | None`, `seed_3: int | None`) -> Expr
  - **pql**: (seed: int) -> Self
- `last` (nw)
  - **Narwhals**: (`order_by: str | Iterable[str] | None`) -> Self
  - **Polars**: (`ignore_nulls: bool`) -> Expr
  - **pql**: () -> Self
- `mode` (nw)
  - **Narwhals**: (`keep: ModeKeepStrategy`) -> Self
  - **Polars**: (`maintain_order: bool`) -> Expr
  - **pql**: () -> Self
- `over` (nw)
  - **Narwhals**: (`*partition_by: str | Sequence[str]`, `order_by: str | Sequence[str] | None`) -> Self
  - **Polars**: (partition_by: IntoExpr | Iterable[IntoExpr] | None, *more_exprs: IntoExpr, order_by: IntoExpr | Iterable[IntoExpr] | None, descending: bool, nulls_last: bool, `mapping_strategy: WindowMappingStrategy`) -> Expr
  - **pql**: (`partition_by: IntoExpr | Iterable[IntoExpr] | None`, `*more_exprs: IntoExpr`, `order_by: IntoExpr | Iterable[IntoExpr] | None`, `descending: bool`, `nulls_last: bool`) -> Self
- `pct_change` (pl)
  - **Polars***: (`n: int | IntoExprColumn`) -> Expr
  - **pql**: (`n: int`) -> Self
- `pow` (pl)
  - **Polars***: (`exponent: IntoExprColumn | int | float`) -> Expr
  - **pql**: (`other: Any`) -> Self
- `repeat_by` (pl)
  - **Polars***: (`by: Series | Expr | str_ | int`) -> Expr
  - **pql**: (`by: Expr | int`) -> Self
- `replace` (pl)
  - **Polars***: (`old: IntoExpr | Sequence[Any] | Mapping[Any, Any]`, `new: IntoExpr | Sequence[Any] | NoDefault`, `default: IntoExpr | NoDefault`, `return_dtype: PolarsDataType | None`) -> Expr
  - **pql**: (`old: IntoExpr`, `new: IntoExpr`) -> Self
- `rolling_max` (pl)
  - **Polars***: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool) -> Expr
  - **pql**: (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_median` (pl)
  - **Polars***: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool) -> Expr
  - **pql**: (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_min` (pl)
  - **Polars***: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool) -> Expr
  - **pql**: (window_size: int, min_samples: int | None, center: bool) -> Self

### [+] Extra Methods (pql-only) (1)

- `inner`

## LazyGroupBy

### [x] Missing Methods (4)

- `having`
  - **Polars**: (*predicates: IntoExpr | Iterable[IntoExpr]) -> LazyGroupBy
- `head`
  - **Polars**: (n: int) -> LazyFrame
- `map_groups`
  - **Polars**: (function: Callable[[DataFrame], DataFrame], schema: SchemaDict | None) -> LazyFrame
- `tail`
  - **Polars**: (n: int) -> LazyFrame

### [!] Signature Mismatches (4)

- `first` (pl)
  - **Polars***: (`ignore_nulls: bool`) -> LazyFrame
  - **pql**: () -> LazyFrame
- `last` (pl)
  - **Polars***: (`ignore_nulls: bool`) -> LazyFrame
  - **pql**: () -> LazyFrame
- `len` (pl)
  - **Polars***: (`name: str | None`) -> LazyFrame
  - **pql**: (`name: str`) -> LazyFrame
- `quantile` (pl)
  - **Polars***: (quantile: float, `interpolation: QuantileMethod`) -> LazyFrame
  - **pql**: (quantile: float, `interpolation: RollingInterpolationMethod`) -> LazyFrame

## Expr.str

### [x] Missing Methods (10)

- `contains_any`
  - **Polars**: (patterns: IntoExpr, ascii_case_insensitive: bool) -> Expr
- `decode`
  - **Polars**: (encoding: TransferEncoding, strict: bool) -> Expr
- `extract_groups`
  - **Polars**: (pattern: str) -> Expr
- `extract_many`
  - **Polars**: (patterns: IntoExpr, ascii_case_insensitive: bool, overlapping: bool, leftmost: bool) -> Expr
- `find_many`
  - **Polars**: (patterns: IntoExpr, ascii_case_insensitive: bool, overlapping: bool, leftmost: bool) -> Expr
- `json_decode`
  - **Polars**: (dtype: DataTypeExpr, infer_schema_length: int | None) -> Expr
- `replace_many`
  - **Polars**: (patterns: IntoExpr | Mapping[str, str], replace_with: IntoExpr | NoDefault, ascii_case_insensitive: bool, leftmost: bool) -> Expr
- `split_exact`
  - **Polars**: (by: IntoExpr, n: int, inclusive: bool) -> Expr
- `splitn`
  - **Polars**: (by: IntoExpr, n: int) -> Expr
- `to_integer`
  - **Polars**: (base: int | IntoExprColumn, dtype: PolarsIntegerType, strict: bool) -> Expr

### [!] Signature Mismatches (4)

- `find` (pl)
  - **Polars***: (pattern: str | Expr, literal: bool, `strict: bool`) -> Expr
  - **pql**: (pattern: str | Expr, literal: bool) -> Expr
- `normalize` (pl)
  - **Polars***: (`form: UnicodeForm`) -> Expr
  - **pql**: () -> Expr
- `strptime` (pl)
  - **Polars***: (`dtype: PolarsTemporalType`, `format: str | None`, `strict: bool`, `exact: bool`, `cache: bool`, `ambiguous: Ambiguous | Expr`) -> Expr
  - **pql**: (`format: str | Expr`) -> Expr
- `to_time` (pl)
  - **Polars***: (format: str | None, `strict: bool`, `cache: bool`) -> Expr
  - **pql**: (format: str | None) -> Expr

### [+] Extra Methods (pql-only) (1)

- `inner`

## Expr.list

### [x] Missing Methods (21)

- `agg`
  - **Polars**: (expr: Expr) -> Expr
- `arg_max`
  - **Polars**: () -> Expr
- `arg_min`
  - **Polars**: () -> Expr
- `concat`
  - **Polars**: (other: list[Expr | str] | Expr | str | Series | list[Any]) -> Expr
- `diff`
  - **Polars**: (n: int, null_behavior: NullBehavior) -> Expr
- `explode`
  - **Polars**: (empty_as_null: bool, keep_nulls: bool) -> Expr
- `filter`
  - **Polars**: (predicate: Expr) -> Expr
- `gather`
  - **Polars**: (indices: Expr | Series | list[int] | list[list[int]], null_on_oob: bool) -> Expr
- `gather_every`
  - **Polars**: (n: int | IntoExprColumn, offset: int | IntoExprColumn) -> Expr
- `head`
  - **Polars**: (n: int | str | Expr) -> Expr
- `item`
  - **Polars**: (allow_empty: bool) -> Expr
- `sample`
  - **Polars**: (n: int | IntoExprColumn | None, fraction: float | IntoExprColumn | None, with_replacement: bool, shuffle: bool, seed: int | None) -> Expr
- `set_difference`
  - **Polars**: (other: IntoExpr | Collection[Any]) -> Expr
- `set_intersection`
  - **Polars**: (other: IntoExpr | Collection[Any]) -> Expr
- `set_symmetric_difference`
  - **Polars**: (other: IntoExpr | Collection[Any]) -> Expr
- `set_union`
  - **Polars**: (other: IntoExpr | Collection[Any]) -> Expr
- `shift`
  - **Polars**: (n: int | IntoExprColumn) -> Expr
- `slice`
  - **Polars**: (offset: int | str | Expr, length: int | str | Expr | None) -> Expr
- `tail`
  - **Polars**: (n: int | str | Expr) -> Expr
- `to_array`
  - **Polars**: (width: int) -> Expr
- `to_struct`
  - **Polars**: (n_field_strategy: ListToStructWidthStrategy | None, fields: Sequence[str] | Callable[[int], str] | None, upper_bound: int | None) -> Expr

### [!] Signature Mismatches (2)

- `contains` (nw)
  - **Narwhals**: (`item: NonNestedLiteral`) -> ExprT
  - **Polars**: (item: IntoExpr, `nulls_equal: bool`) -> Expr
  - **pql**: (`item: IntoExpr`) -> Expr
- `eval` (pl)
  - **Polars***: (expr: Expr, `parallel: bool`) -> Expr
  - **pql**: (expr: Expr) -> Expr

### [+] Extra Methods (pql-only) (1)

- `inner`

## Expr.struct

### [x] Missing Methods (2)

- `rename_fields`
  - **Polars**: (names: Sequence[str]) -> Expr
- `unnest`
  - **Polars**: () -> Expr

### [+] Extra Methods (pql-only) (1)

- `inner`

## Expr.name

### [x] Missing Methods (3)

- `map_fields`
  - **Polars**: (function: Callable[[str], str]) -> Expr
- `prefix_fields`
  - **Polars**: (prefix: str) -> Expr
- `suffix_fields`
  - **Polars**: (suffix: str) -> Expr

### [+] Extra Methods (pql-only) (1)

- `inner`

## Expr.arr

### [x] Missing Methods (10)

- `agg`
  - **Polars**: (expr: Expr) -> Expr
- `arg_max`
  - **Polars**: () -> Expr
- `arg_min`
  - **Polars**: () -> Expr
- `explode`
  - **Polars**: (empty_as_null: bool, keep_nulls: bool) -> Expr
- `head`
  - **Polars**: (n: int | str | Expr, as_array: bool) -> Expr
- `shift`
  - **Polars**: (n: int | IntoExprColumn) -> Expr
- `slice`
  - **Polars**: (offset: int | str | Expr, length: int | str | Expr | None, as_array: bool) -> Expr
- `tail`
  - **Polars**: (n: int | str | Expr, as_array: bool) -> Expr
- `to_list`
  - **Polars**: () -> Expr
- `to_struct`
  - **Polars**: (fields: Sequence[str] | Callable[[int], str] | None) -> Expr

### [!] Signature Mismatches (4)

- `contains` (pl)
  - **Polars***: (item: IntoExpr, `nulls_equal: bool`) -> Expr
  - **pql**: (item: IntoExpr) -> Expr
- `eval` (pl)
  - **Polars***: (expr: Expr, `as_list: bool`) -> Expr
  - **pql**: (expr: Expr) -> Expr
- `get` (pl)
  - **Polars***: (`index: int | IntoExprColumn`, `null_on_oob: bool`) -> Expr
  - **pql**: (`index: int`) -> Expr
- `unique` (pl)
  - **Polars***: (`maintain_order: bool`) -> Expr
  - **pql**: () -> Expr

### [+] Extra Methods (pql-only) (2)

- `drop_nulls`
- `inner`
