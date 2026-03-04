
# pql vs Polars API Comparison Report

This report shows the API coverage of pql compared to other libraries.

## Summary

Each summary cell is `global (Narwhals, Polars)`.

| Class               | Coverage              | Implemented   | Matched      | Missing       | Mismatched  | Extra      |
| ------------------- | --------------------- | ------------- | ------------ | ------------- | ----------- | ---------- |
| LazyFrame           | 42.5% (64.0%, 35.8%)  | 106 (25, 81)  | 45 (16, 29)  | 30 (0, 30)    | 31 (9, 22)  | 44 (35, 9) |
| Expr                | 49.8% (71.4%, 42.7%)  | 283 (70, 213) | 141 (50, 91) | 100 (7, 93)   | 42 (13, 29) | 59 (58, 1) |
| LazyGroupBy         | 52.9% (0.0%, 56.2%)   | 17 (1, 16)    | 9 (0, 9)     | 4 (0, 4)      | 4 (1, 3)    | 11 (11, 0) |
| ExprStrNameSpace    | 31.8% (42.1%, 27.7%)  | 66 (19, 47)   | 21 (8, 13)   | 10 (0, 10)    | 35 (11, 24) | 20 (19, 1) |
| ExprListNameSpace   | 54.7% (90.0%, 46.5%)  | 53 (10, 43)   | 29 (9, 20)   | 20 (0, 20)    | 4 (1, 3)    | 15 (14, 1) |
| ExprStructNameSpace | 33.3% (100.0%, 20.0%) | 6 (1, 5)      | 2 (1, 1)     | 2 (0, 2)      | 2 (0, 2)    | 4 (3, 1)   |
| ExprNameNameSpace   | 81.2% (100.0%, 70.0%) | 16 (6, 10)    | 13 (6, 7)    | 3 (0, 3)      | 0 (0, 0)    | 3 (2, 1)   |
| ExprArrNameSpace    | 54.8% (100.0%, 54.8%) | 31 (0, 31)    | 17 (0, 17)   | 10 (0, 10)    | 4 (0, 4)    | 27 (24, 3) |
| ExprDtNameSpace     | 51.5% (60.9%, 46.7%)  | 68 (23, 45)   | 35 (14, 21)  | 25 (7, 18)    | 8 (2, 6)    | 13 (12, 1) |
| ModuleFunctions     | 21.1% (38.4%, 14.2%)  | 256 (73, 183) | 54 (28, 26)  | 170 (31, 139) | 32 (14, 18) | 18 (10, 8) |

## LazyFrame

### [x] Missing Methods (30)

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
- `update`
  - **Polars**: (other: LazyFrame, on: str | Sequence[str] | None, how: Literal['left', 'inner', 'full'], left_on: str | Sequence[str] | None, right_on: str | Sequence[str] | None, include_nulls: bool, maintain_order: MaintainOrderJoin | None) -> LazyFrame
- `with_columns_seq`
  - **Polars**: (*exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> LazyFrame

### [!] Signature Mismatches (16)

- `cast` (pl)
  - **Polars***: (`dtypes: Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType | PythonDataType] | PolarsDataType | pl.DataTypeExpr | Schema`, `strict: bool`) -> LazyFrame
  - **pql**: (`dtypes: Mapping[str, DataType] | DataType`) -> Self
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
  - **pql**: (`subset: TryIter[str] | None`) -> Self
- `explain` (pl)
  - **Polars***: (`format: ExplainFormat`, `optimized: bool`, `type_coercion: bool`, `predicate_pushdown: bool`, `projection_pushdown: bool`, `simplify_expression: bool`, `slice_pushdown: bool`, `comm_subplan_elim: bool`, `comm_subexpr_elim: bool`, `cluster_with_columns: bool`, `collapse_joins: bool`, `streaming: bool`, `engine: EngineType`, `tree_format: bool | None`, `optimizations: QueryOptFlags`) -> str
  - **pql**: (`kind: Literal['standard', 'analyze']`) -> str
- `fill_null` (pl)
  - **Polars***: (`value: Any | Expr | None`, strategy: FillNullStrategy | None, limit: int | None, `matches_supertype: bool`) -> LazyFrame
  - **pql**: (`value: IntoExpr | None`, strategy: FillNullStrategy | None, limit: int | None) -> Self
- `filter` (nw)
  - **Narwhals**: (`*predicates: IntoExpr | Iterable[IntoExpr]`, **constraints: Any) -> Self
  - **Polars**: (`*predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool]`, **constraints: Any) -> LazyFrame
  - **pql**: (`predicates: TryIter[IntoExprColumn]`, *more_predicates: IntoExprColumn, `**constraints: IntoExpr`) -> Self
- `join` (nw)
  - **Narwhals**: (other: Self, `on: str | list[str] | None`, how: JoinStrategy, `left_on: str | list[str] | None`, `right_on: str | list[str] | None`, suffix: str) -> Self
  - **Polars**: (other: LazyFrame, `on: str | Expr | Sequence[str | Expr] | None`, how: JoinStrategy, `left_on: str | Expr | Sequence[str | Expr] | None`, `right_on: str | Expr | Sequence[str | Expr] | None`, suffix: str, `validate: JoinValidation`, `nulls_equal: bool`, `coalesce: bool | None`, `maintain_order: MaintainOrderJoin | None`, allow_parallel: bool, force_parallel: bool) -> LazyFrame
  - **pql**: (other: Self, `on: TryIter[str] | None`, how: JoinStrategy, `left_on: TryIter[str] | None`, `right_on: TryIter[str] | None`, suffix: str) -> Self
- `join_asof` (nw)
  - **Narwhals**: (other: Self, left_on: str | None, right_on: str | None, on: str | None, `by_left: str | list[str] | None`, `by_right: str | list[str] | None`, `by: str | list[str] | None`, strategy: AsofJoinStrategy, suffix: str) -> Self
  - **Polars**: (other: LazyFrame, `left_on: str | None | Expr`, `right_on: str | None | Expr`, `on: str | None | Expr`, `by_left: str | Sequence[str] | None`, `by_right: str | Sequence[str] | None`, `by: str | Sequence[str] | None`, strategy: AsofJoinStrategy, suffix: str, `tolerance: str | int | float | timedelta | None`, allow_parallel: bool, force_parallel: bool, `coalesce: bool`, `allow_exact_matches: bool`, `check_sortedness: bool`) -> LazyFrame
  - **pql**: (other: Self, left_on: str | None, right_on: str | None, on: str | None, `by_left: TryIter[str] | None`, `by_right: TryIter[str] | None`, `by: TryIter[str] | None`, strategy: AsofJoinStrategy, suffix: str) -> Self
- `quantile` (pl)
  - **Polars***: (`quantile: float | Expr`, interpolation: QuantileMethod) -> LazyFrame
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
  - **Polars**: (`path: str | Path | IO[bytes] | PartitionBy`, `compression: str`, `compression_level: int | None`, `statistics: bool | str | dict[str, bool]`, `row_group_size: int | None`, `data_page_size: int | None`, `maintain_order: bool`, `storage_options: StorageOptionsDict | None`, `credential_provider: CredentialProviderFunction | Literal['auto'] | None`, `retries: int | None`, `sync_on_close: SyncOnCloseMethod | None`, `metadata: ParquetMetadata | None`, `arrow_schema: ArrowSchemaExportable | None`, `mkdir: bool`, `lazy: bool`, `engine: EngineType`, `optimizations: QueryOptFlags`) -> LazyFrame | None
  - **pql**: (`path: str | Path`, `compression: ParquetCompression`) -> None
- `unnest` (pl)
  - **Polars***: (`columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector]`, `*more_columns: ColumnNameOrSelector`, `separator: str | None`) -> LazyFrame
  - **pql**: (`columns: TryIter[IntoExprColumn]`, `*more_columns: IntoExprColumn`) -> Self

### [+] Extra Methods (pql-only) (9)

- `from_df`
- `from_mapping`
- `from_numpy`
- `from_query`
- `from_sequence`
- `from_table`
- `from_table_function`
- `inner`
- `sql_query`

## Expr

### [x] Missing Methods (94)

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
  - **pql**: (`other: IntoExpr`) -> Self
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

### [!] Signature Mismatches (3)

- `first` (pl)
  - **Polars***: (`ignore_nulls: bool`) -> LazyFrame
  - **pql**: () -> LazyFrame
- `last` (pl)
  - **Polars***: (`ignore_nulls: bool`) -> LazyFrame
  - **pql**: () -> LazyFrame
- `len` (pl)
  - **Polars***: (`name: str | None`) -> LazyFrame
  - **pql**: (`name: str`) -> LazyFrame

## ExprStrNameSpace

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

### [!] Signature Mismatches (20)

- `contains` (nw)
  - **Narwhals**: (`pattern: str`, literal: bool) -> ExprT
  - **Polars**: (`pattern: str | Expr`, literal: bool, `strict: bool`) -> Expr
  - **pql**: (`pattern: IntoExprColumn`, literal: bool) -> Expr
- `count_matches` (pl)
  - **Polars***: (`pattern: str | Expr`, literal: bool) -> Expr
  - **pql**: (`pattern: IntoExprColumn`, literal: bool) -> Expr
- `ends_with` (nw)
  - **Narwhals**: (`suffix: str`) -> ExprT
  - **Polars**: (`suffix: str | Expr`) -> Expr
  - **pql**: (`suffix: IntoExprColumn`) -> Expr
- `extract_all` (pl)
  - **Polars***: (`pattern: str | Expr`) -> Expr
  - **pql**: (`pattern: IntoExprColumn`) -> Expr
- `find` (pl)
  - **Polars***: (`pattern: str | Expr`, literal: bool, `strict: bool`) -> Expr
  - **pql**: (`pattern: IntoExprColumn`, literal: bool) -> Expr
- `join` (pl)
  - **Polars***: (`delimiter: str`, ignore_nulls: bool) -> Expr
  - **pql**: (`delimiter: IntoExprColumn`, ignore_nulls: bool) -> Expr
- `normalize` (pl)
  - **Polars***: (`form: UnicodeForm`) -> Expr
  - **pql**: () -> Expr
- `pad_end` (nw)
  - **Narwhals**: (length: int, `fill_char: str`) -> ExprT
  - **Polars**: (`length: int | IntoExprColumn`, `fill_char: str`) -> Expr
  - **pql**: (length: int, `fill_char: IntoExprColumn`) -> Expr
- `pad_start` (nw)
  - **Narwhals**: (length: int, `fill_char: str`) -> ExprT
  - **Polars**: (`length: int | IntoExprColumn`, `fill_char: str`) -> Expr
  - **pql**: (length: int, `fill_char: IntoExprColumn`) -> Expr
- `replace` (nw)
  - **Narwhals**: (pattern: str, `value: str | IntoExpr`, literal: bool, n: int) -> ExprT
  - **Polars**: (`pattern: str | Expr`, `value: str | Expr`, literal: bool, n: int) -> Expr
  - **pql**: (pattern: str, `value: IntoExprColumn`, literal: bool, n: int) -> Expr
- `replace_all` (nw)
  - **Narwhals**: (`pattern: str`, `value: IntoExpr`, literal: bool) -> ExprT
  - **Polars**: (`pattern: str | Expr`, `value: str | Expr`, literal: bool) -> Expr
  - **pql**: (`pattern: IntoExprColumn`, `value: IntoExprColumn`, literal: bool) -> Expr
- `split` (nw)
  - **Narwhals**: (`by: str`) -> ExprT
  - **Polars**: (`by: IntoExpr`, `inclusive: bool`, `literal: bool`, `strict: bool`) -> Expr
  - **pql**: (`by: IntoExprColumn`) -> Expr
- `starts_with` (nw)
  - **Narwhals**: (`prefix: str`) -> ExprT
  - **Polars**: (`prefix: str | Expr`) -> Expr
  - **pql**: (`prefix: IntoExprColumn`) -> Expr
- `strip_chars` (nw)
  - **Narwhals**: (`characters: str | None`) -> ExprT
  - **Polars**: (`characters: IntoExpr`) -> Expr
  - **pql**: (`characters: IntoExprColumn | None`) -> Expr
- `strip_chars_end` (pl)
  - **Polars***: (`characters: IntoExpr`) -> Expr
  - **pql**: (`characters: IntoExprColumn | None`) -> Expr
- `strip_chars_start` (pl)
  - **Polars***: (`characters: IntoExpr`) -> Expr
  - **pql**: (`characters: IntoExprColumn | None`) -> Expr
- `strptime` (pl)
  - **Polars***: (`dtype: PolarsTemporalType`, `format: str | None`, `strict: bool`, `exact: bool`, `cache: bool`, `ambiguous: Ambiguous | Expr`) -> Expr
  - **pql**: (`format: IntoExprColumn`) -> Expr
- `to_date` (nw)
  - **Narwhals**: (`format: str | None`) -> ExprT
  - **Polars**: (`format: str | None`, `strict: bool`, `exact: bool`, `cache: bool`) -> Expr
  - **pql**: (`format: IntoExprColumn | None`) -> Expr
- `to_datetime` (nw)
  - **Narwhals**: (`format: str | None`) -> ExprT
  - **Polars**: (`format: str | None`, `time_unit: TimeUnit | None`, `time_zone: str | None`, `strict: bool`, `exact: bool`, `cache: bool`, `ambiguous: Ambiguous | Expr`) -> Expr
  - **pql**: (`format: IntoExprColumn | None`) -> Expr
- `to_time` (pl)
  - **Polars***: (`format: str | None`, `strict: bool`, `cache: bool`) -> Expr
  - **pql**: (`format: IntoExprColumn | None`) -> Expr

### [+] Extra Methods (pql-only) (1)

- `inner`

## ExprListNameSpace

### [x] Missing Methods (20)

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

### [!] Signature Mismatches (1)

- `contains` (nw)
  - **Narwhals**: (`item: NonNestedLiteral`) -> ExprT
  - **Polars**: (item: IntoExpr, `nulls_equal: bool`) -> Expr
  - **pql**: (`item: IntoExpr`) -> Expr

### [+] Extra Methods (pql-only) (1)

- `inner`

## ExprStructNameSpace

### [x] Missing Methods (2)

- `rename_fields`
  - **Polars**: (names: Sequence[str]) -> Expr
- `unnest`
  - **Polars**: () -> Expr

### [!] Signature Mismatches (1)

- `with_fields` (pl)
  - **Polars***: (`*exprs: IntoExpr | Iterable[IntoExpr]`, **named_exprs: IntoExpr) -> Expr
  - **pql**: (`expr: IntoExpr | Iterable[IntoExpr]`, `*more_exprs: IntoExpr`, **named_exprs: IntoExpr) -> Expr

### [+] Extra Methods (pql-only) (1)

- `inner`

## ExprNameNameSpace

### [x] Missing Methods (3)

- `map_fields`
  - **Polars**: (function: Callable[[str], str]) -> Expr
- `prefix_fields`
  - **Polars**: (prefix: str) -> Expr
- `suffix_fields`
  - **Polars**: (suffix: str) -> Expr

### [+] Extra Methods (pql-only) (1)

- `inner`

## ExprArrNameSpace

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

### [+] Extra Methods (pql-only) (3)

- `drop_nulls`
- `filter`
- `inner`

## ExprDtNameSpace

### [x] Missing Methods (18)

- `add_business_days`
  - **Polars**: (n: int | IntoExpr, week_mask: Iterable[bool], holidays: Iterable[dt.date], roll: Roll) -> Expr
- `base_utc_offset`
  - **Polars**: () -> Expr
- `cast_time_unit`
  - **Polars**: (time_unit: TimeUnit) -> Expr
- `combine`
  - **Polars**: (time: time | Expr, time_unit: TimeUnit) -> Expr
- `convert_time_zone`
  - **Narwhals**: (time_zone: str) -> ExprT
  - **Polars**: (time_zone: str) -> Expr
- `days_in_month`
  - **Polars**: () -> Expr
- `dst_offset`
  - **Polars**: () -> Expr
- `is_business_day`
  - **Polars**: (week_mask: Iterable[bool], holidays: Iterable[dt.date]) -> Expr
- `is_leap_year`
  - **Polars**: () -> Expr
- `replace`
  - **Polars**: (year: int | IntoExpr | None, month: int | IntoExpr | None, day: int | IntoExpr | None, hour: int | IntoExpr | None, minute: int | IntoExpr | None, second: int | IntoExpr | None, microsecond: int | IntoExpr | None, ambiguous: Ambiguous | Expr) -> Expr
- `replace_time_zone`
  - **Narwhals**: (time_zone: str | None) -> ExprT
  - **Polars**: (time_zone: str | None, ambiguous: Ambiguous | Expr, non_existent: NonExistent) -> Expr
- `total_days`
  - **Polars**: (fractional: bool) -> Expr
- `total_hours`
  - **Polars**: (fractional: bool) -> Expr
- `total_microseconds`
  - **Narwhals**: () -> ExprT
  - **Polars**: (fractional: bool) -> Expr
- `total_milliseconds`
  - **Narwhals**: () -> ExprT
  - **Polars**: (fractional: bool) -> Expr
- `total_minutes`
  - **Narwhals**: () -> ExprT
  - **Polars**: (fractional: bool) -> Expr
- `total_nanoseconds`
  - **Narwhals**: () -> ExprT
  - **Polars**: (fractional: bool) -> Expr
- `total_seconds`
  - **Narwhals**: () -> ExprT
  - **Polars**: (fractional: bool) -> Expr

### [!] Signature Mismatches (4)

- `offset_by` (nw)
  - **Narwhals**: (`by: str`) -> ExprT
  - **Polars**: (`by: str | Expr`) -> Expr
  - **pql**: (`by: IntoExpr`) -> Expr
- `round` (pl)
  - **Polars***: (`every: timedelta | IntoExprColumn`) -> Expr
  - **pql**: (`every: str`) -> Expr
- `strftime` (pl)
  - **Polars***: (`format: str`) -> Expr
  - **pql**: (`format: IntoExprColumn`) -> Expr
- `to_string` (nw)
  - **Narwhals**: (`format: str`) -> ExprT
  - **Polars**: (`format: str | None`) -> Expr
  - **pql**: (`format: IntoExprColumn`) -> Expr

### [+] Extra Methods (pql-only) (1)

- `inner`

## ModuleFunctions

### [x] Missing Methods (152)

- `Any`
  - **Polars**: (*args, **kwargs)
- `BaseExtension`
  - **Polars**: (name: str, storage: PolarsDataType, metadata: str | None) -> None
- `Catalog`
  - **Polars**: (workspace_url: str, bearer_token: str | None, require_https: bool) -> None
- `Categorical`
  - **Narwhals**: ()
  - **Polars**: (categories: Categories | str | None, ordering: CategoricalOrdering | None) -> None
- `Categories`
  - **Polars**: (name: str | None, namespace: str, physical: PolarsDataType) -> None
- `CompatLevel`
  - **Polars**: () -> None
- `Config`
  - **Polars**: (restore_defaults: bool, apply_on_context_enter: bool, **options: Unpack[ConfigParameters]) -> None
- `DataTypeExpr`
  - **Polars**: ()
- `Extension`
  - **Polars**: (name: str, storage: PolarsDataType, metadata: str | None) -> None
- `Field`
  - **Narwhals**: (name: str, dtype: IntoDType) -> None
  - **Polars**: (name: str, dtype: PolarsDataType) -> None
- `FileProviderArgs`
  - **Polars**: (index_in_partition: int, partition_keys: DataFrame) -> None
- `Float16`
  - **Polars**: ()
- `GPUEngine`
  - **Polars**: (device: int | None, memory_resource: Any | None, raise_on_fail: bool, **kwargs: Any) -> None
- `Implementation`
  - **Narwhals**: (*values)
- `Null`
  - **Polars**: ()
- `Object`
  - **Narwhals**: ()
  - **Polars**: ()
- `PartitionBy`
  - **Polars**: (base_path: str | Path, file_path_provider: Callable[[FileProviderArgs], str | Path | IO[bytes] | IO[str]] | None, key: str | Expr | Sequence[str | Expr] | Mapping[str, Expr] | None, include_key: bool | None, max_rows_per_file: int | None, approximate_bytes_per_file: int | Literal['auto'] | None) -> None
- `QueryOptFlags`
  - **Polars**: (predicate_pushdown: None | bool, projection_pushdown: None | bool, simplify_expression: None | bool, slice_pushdown: None | bool, comm_subplan_elim: None | bool, comm_subexpr_elim: None | bool, cluster_with_columns: None | bool, collapse_joins: None | bool, check_order_observe: None | bool, fast_projection: None | bool) -> None
- `SQLContext`
  - **Polars**: (frames: Mapping[str, CompatibleFrameType | None] | None, register_globals: bool | int, eager: bool, **named_frames: CompatibleFrameType | None) -> None
- `ScanCastOptions`
  - **Polars**: (integer_cast: Literal['upcast', 'forbid'], float_cast: Literal['forbid'] | FloatCastOption | Collection[FloatCastOption], datetime_cast: Literal['forbid'] | DatetimeCastOption | Collection[DatetimeCastOption], missing_struct_fields: Literal['insert', 'raise'], extra_struct_fields: Literal['ignore', 'raise'], categorical_to_string: Literal['allow', 'forbid'],_internal_call: bool) -> None
- `Schema`
  - **Narwhals**: (schema: Mapping[str, DType] | Iterable[tuple[str, DType]] | None) -> None
  - **Polars**: (schema: Mapping[str, SchemaInitDataType] | Iterable[tuple[str, SchemaInitDataType] | ArrowSchemaExportable] | ArrowSchemaExportable | None, check_dtypes: bool) -> None
- `Unknown`
  - **Narwhals**: ()
  - **Polars**: ()
- `align_frames`
  - **Polars**: (*frames: FrameType | Iterable[FrameType], on: str | Expr | Sequence[str] | Sequence[Expr] | Sequence[str | Expr], how: JoinStrategy, select: str | Expr | Sequence[str | Expr] | None, descending: bool | Sequence[bool]) -> list[FrameType]
- `any`
  - **Polars**: (*names: str, ignore_nulls: bool) -> Expr | bool | None
- `approx_n_unique`
  - **Polars**: (*columns: str) -> Expr
- `arange`
  - **Polars**: (start: int | IntoExprColumn, end: int | IntoExprColumn | None, step: int, dtype: PolarsIntegerType | DataTypeExpr, eager: bool) -> Expr | Series
- `arctan2`
  - **Polars**: (y: str | Expr, x: str | Expr) -> Expr
- `arg_sort_by`
  - **Polars**: (exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr, descending: bool | Sequence[bool], nulls_last: bool | Sequence[bool], multithreaded: bool, maintain_order: bool) -> Expr
- `arg_where`
  - **Polars**: (condition: Expr | Series, eager: bool) -> Expr | Series
- `build_info`
  - **Polars**: () -> dict[str, Any]
- `business_day_count`
  - **Polars**: (start: date | IntoExprColumn, end: date | IntoExprColumn, week_mask: Iterable[bool], holidays: Iterable[date]) -> Expr
- `collect_all`
  - **Polars**: (lazy_frames: Iterable[LazyFrame], type_coercion: bool, predicate_pushdown: bool, projection_pushdown: bool, simplify_expression: bool, no_optimization: bool, slice_pushdown: bool, comm_subplan_elim: bool, comm_subexpr_elim: bool, cluster_with_columns: bool, collapse_joins: bool, optimizations: QueryOptFlags, engine: EngineType, lazy: bool) -> list[DataFrame] | LazyFrame
- `collect_all_async`
  - **Polars**: (lazy_frames: Iterable[LazyFrame], gevent: bool, engine: EngineType, optimizations: QueryOptFlags) -> Awaitable[list[DataFrame]] |_GeventDataFrameResult[list[DataFrame]]
- `concat`
  - **Narwhals**: (items: Iterable[FrameT], how: ConcatMethod) -> FrameT
  - **Polars**: (items: Iterable[PolarsType], how: ConcatMethod, rechunk: bool, parallel: bool, strict: bool) -> PolarsType
- `concat_arr`
  - **Polars**: (exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr
- `concat_list`
  - **Polars**: (exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr
- `concat_str`
  - **Narwhals**: (exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr, separator: str, ignore_nulls: bool) -> Expr
  - **Polars**: (exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr, separator: str, ignore_nulls: bool) -> Expr
- `corr`
  - **Polars**: (a: IntoExpr, b: IntoExpr, method: CorrelationMethod, ddof: int | None, propagate_nans: bool, eager: bool) -> Expr | Series
- `count`
  - **Polars**: (*columns: str) -> Expr
- `cov`
  - **Polars**: (a: IntoExpr, b: IntoExpr, ddof: int, eager: bool) -> Expr | Series
- `cum_count`
  - **Polars**: (*columns: str, reverse: bool) -> Expr
- `cum_fold`
  - **Polars**: (acc: IntoExpr, function: Callable[[Series, Series], Series], exprs: Sequence[Expr | str] | Expr, returns_scalar: bool, return_dtype: DataTypeExpr | PolarsDataType | None, include_init: bool) -> Expr
- `cum_reduce`
  - **Polars**: (function: Callable[[Series, Series], Series], exprs: Sequence[Expr | str] | Expr, returns_scalar: bool, return_dtype: DataTypeExpr | PolarsDataType | None) -> Expr
- `cum_sum`
  - **Polars**: (*names: str) -> Expr
- `cum_sum_horizontal`
  - **Polars**: (*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr
- `date`
  - **Polars**: (year: Expr | str | int, month: Expr | str | int, day: Expr | str | int) -> Expr
- `date_range`
  - **Polars**: (start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta, closed: ClosedInterval, eager: bool) -> Series | Expr
- `date_ranges`
  - **Polars**: (start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta, closed: ClosedInterval, eager: bool) -> Series | Expr
- `datetime`
  - **Polars**: (year: int | IntoExpr, month: int | IntoExpr, day: int | IntoExpr, hour: int | IntoExpr | None, minute: int | IntoExpr | None, second: int | IntoExpr | None, microsecond: int | IntoExpr | None, time_unit: TimeUnit, time_zone: str | None, ambiguous: Ambiguous | Expr) -> Expr
- `datetime_range`
  - **Polars**: (start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta, closed: ClosedInterval, time_unit: TimeUnit | None, time_zone: str | None, eager: bool) -> Series | Expr
- `datetime_ranges`
  - **Polars**: (start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta, closed: ClosedInterval, time_unit: TimeUnit | None, time_zone: str | None, eager: bool) -> Series | Expr
- `defer`
  - **Polars**: (function: Callable[[], DataFrame], schema: SchemaDict | Callable[[], SchemaDict], validate_schema: bool) -> LazyFrame
- `disable_string_cache`
  - **Polars**: () -> None
- `dtype_of`
  - **Polars**: (col_or_expr: str | Expr) -> DataTypeExpr
- `duration`
  - **Polars**: (weeks: Expr | str | int | float | None, days: Expr | str | int | float | None, hours: Expr | str | int | float | None, minutes: Expr | str | int | float | None, seconds: Expr | str | int | float | None, milliseconds: Expr | str | int | float | None, microseconds: Expr | str | int | float | None, nanoseconds: Expr | str | int | float | None, time_unit: TimeUnit | None) -> Expr
- `escape_regex`
  - **Polars**: (s: str) -> str
- `exclude`
  - **Narwhals**: (*names: str | Iterable[str]) -> Expr
  - **Polars**: (columns: str | PolarsDataType | Collection[str] | Collection[PolarsDataType], *more_columns: str | PolarsDataType) -> Expr
- `explain_all`
  - **Polars**: (lazy_frames: Iterable[LazyFrame], optimizations: QueryOptFlags) -> str
- `field`
  - **Polars**: (name: str | list[str]) -> Expr
- `first`
  - **Polars**: (*columns: str) -> Expr
- `fold`
  - **Polars**: (acc: IntoExpr, function: Callable[[Series, Series], Series], exprs: Sequence[Expr | str] | Expr, returns_scalar: bool, return_dtype: DataTypeExpr | PolarsDataType | None) -> Expr
- `format`
  - **Narwhals**: (f_string: str, *args: IntoExpr) -> Expr
  - **Polars**: (f_string: str, *args: Expr | str) -> Expr
- `from_arrow`
  - **Narwhals**: (native_frame: IntoArrowTable, backend: IntoBackend[EagerAllowed]) -> DataFrame[Any]
  - **Polars**: (data: RecordBatch | Iterable[pa.RecordBatch | pa.Table] | ArrowArrayExportable | ArrowStreamExportable, schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, rechunk: bool) -> DataFrame | Series
- `from_dataframe`
  - **Polars**: (df: SupportsInterchange | ArrowArrayExportable | ArrowStreamExportable, allow_copy: bool | None, rechunk: bool) -> DataFrame
- `from_dict`
  - **Narwhals**: (data: Mapping[str, Any], schema: IntoSchema | Mapping[str, DType | None] | None, backend: IntoBackend[EagerAllowed] | None, native_namespace: ModuleType | None) -> DataFrame[Any]
  - **Polars**: (data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, strict: bool) -> DataFrame
- `from_dicts`
  - **Narwhals**: (data: Sequence[Mapping[str, Any]], schema: IntoSchema | Mapping[str, DType | None] | None, backend: IntoBackend[EagerAllowed]) -> DataFrame[Any]
  - **Polars**: (data: Iterable[Mapping[str, Any]], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, strict: bool, infer_schema_length: int | None) -> DataFrame
- `from_epoch`
  - **Polars**: (column: str | Expr | Series | Sequence[int], time_unit: EpochTimeUnit) -> Expr | Series
- `from_numpy`
  - **Narwhals**: (data: _2DArray, schema: IntoSchema | Sequence[str] | None, backend: IntoBackend[EagerAllowed]) -> DataFrame[Any]
  - **Polars**: (data: ndarray[Any, Any], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, orient: Orientation | None) -> DataFrame
- `from_pandas`
  - **Polars**: (data: Series[Any] | pd.Index[Any] | pd.DatetimeIndex, schema_overrides: SchemaDict | None, rechunk: bool, nan_to_null: bool, include_index: bool) -> DataFrame | Series
- `from_records`
  - **Polars**: (data: Sequence[Any], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, strict: bool, orient: Orientation | None, infer_schema_length: int | None) -> DataFrame
- `from_repr`
  - **Polars**: (data: str) -> DataFrame | Series
- `from_torch`
  - **Polars**: (tensor: Tensor, schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, orient: Orientation | None, force: bool) -> DataFrame
- `generate_temporary_column_name`
  - **Narwhals**: (n_bytes: int, columns: Container[str], prefix: str) -> str
- `get_extension_type`
  - **Polars**: (ext_name: str) -> type[dt.BaseExtension] | str | None
- `get_index_type`
  - **Polars**: () -> PolarsIntegerType
- `get_native_namespace`
  - **Narwhals**: (*obj: Frame | Series[Any] | IntoFrame | IntoSeries) -> Any
- `groups`
  - **Polars**: (column: str) -> Expr
- `head`
  - **Polars**: (column: str, n: int) -> Expr
- `implode`
  - **Polars**: (*columns: str) -> Expr
- `int_range`
  - **Polars**: (start: int | IntoExprColumn, end: int | IntoExprColumn | None, step: int, dtype: PolarsIntegerType | DataTypeExpr, eager: bool) -> Expr | Series
- `int_ranges`
  - **Polars**: (start: int | IntoExprColumn, end: int | IntoExprColumn | None, step: int | IntoExprColumn, dtype: PolarsIntegerType | DataTypeExpr, eager: bool) -> Expr | Series
- `is_ordered_categorical`
  - **Narwhals**: (series: Series[Any]) -> bool
- `json_normalize`
  - **Polars**: (data: dict[Any, Any] | Sequence[dict[Any, Any] | Any], separator: str, max_level: int | None, schema: Schema | None, strict: bool, infer_schema_length: int | None, encoder: JSONEncoder | None) -> DataFrame
- `last`
  - **Polars**: (*columns: str) -> Expr
- `linear_space`
  - **Polars**: (start: NumericLiteral | TemporalLiteral | IntoExpr, end: NumericLiteral | TemporalLiteral | IntoExpr, num_samples: int | IntoExpr, closed: ClosedInterval, eager: bool) -> Expr | Series
- `linear_spaces`
  - **Polars**: (start: NumericLiteral | TemporalLiteral | IntoExprColumn, end: NumericLiteral | TemporalLiteral | IntoExprColumn, num_samples: int | IntoExprColumn, closed: ClosedInterval, as_array: bool, eager: bool) -> Expr | Series
- `map_batches`
  - **Polars**: (exprs: Sequence[str | Expr], function: Callable[[Sequence[Series]], Series | Any], return_dtype: DataTypeExpr | None, is_elementwise: bool, returns_scalar: bool) -> Expr
- `map_groups`
  - **Polars**: (exprs: Sequence[str | Expr], function: Callable[[Sequence[Series]], Series | Any], return_dtype: DataTypeExpr | None, is_elementwise: bool, returns_scalar: bool) -> Expr
- `maybe_align_index`
  - **Narwhals**: (lhs: FrameOrSeriesT, rhs: Series[Any] | DataFrame[Any] | LazyFrame[Any]) -> FrameOrSeriesT
- `maybe_convert_dtypes`
  - **Narwhals**: (obj: FrameOrSeriesT, *args: bool, **kwargs: bool | str) -> FrameOrSeriesT
- `maybe_get_index`
  - **Narwhals**: (obj: DataFrame[Any] | LazyFrame[Any] | Series[Any]) -> Any | None
- `maybe_reset_index`
  - **Narwhals**: (obj: FrameOrSeriesT) -> FrameOrSeriesT
- `maybe_set_index`
  - **Narwhals**: (obj: FrameOrSeriesT, column_names: str | list[str] | None, index: Series[IntoSeriesT] | list[Series[IntoSeriesT]] | None) -> FrameOrSeriesT
- `n_unique`
  - **Polars**: (*columns: str) -> Expr
- `narwhalify`
  - **Narwhals**: (func: Callable[..., Any] | None, pass_through: bool, eager_only: bool, series_only: bool, allow_series: bool | None) -> Callable[..., Any]
- `new_series`
  - **Narwhals**: (name: str, values: Any, dtype: IntoDType | None, backend: IntoBackend[EagerAllowed]) -> Series[Any]
- `nth`
  - **Narwhals**: (*indices: int | Sequence[int]) -> Expr
  - **Polars**: (*indices: int | Sequence[int], strict: bool) -> Expr
- `ones`
  - **Polars**: (n: int | Expr, dtype: PolarsDataType, eager: bool) -> Expr | Series
- `quantile`
  - **Polars**: (column: str, quantile: float | Expr, interpolation: QuantileMethod) -> Expr
- `read_avro`
  - **Polars**: (source: str | Path | IO[bytes] | bytes, columns: list[int] | list[str] | None, n_rows: int | None) -> DataFrame
- `read_clipboard`
  - **Polars**: (separator: str, **kwargs: Any) -> DataFrame
- `read_csv`
  - **Narwhals**: (source: FileSource, backend: IntoBackend[EagerAllowed], separator: str, **kwargs: Any) -> DataFrame[Any]
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes, has_header: bool, columns: Sequence[int] | Sequence[str] | None, new_columns: Sequence[str] | None, separator: str, comment_prefix: str | None, quote_char: str | None, skip_rows: int, skip_lines: int, schema: SchemaDict | None, schema_overrides: Mapping[str, PolarsDataType] | Sequence[PolarsDataType] | None, null_values: str | Sequence[str] | dict[str, str] | None, missing_utf8_is_empty_string: bool, ignore_errors: bool, try_parse_dates: bool, n_threads: int | None, infer_schema: bool, infer_schema_length: int | None, batch_size: int, n_rows: int | None, encoding: CsvEncoding | str, low_memory: bool, rechunk: bool, use_pyarrow: bool, storage_options: StorageOptionsDict | None, skip_rows_after_header: int, row_index_name: str | None, row_index_offset: int, sample_size: int, eol_char: str, raise_if_empty: bool, truncate_ragged_lines: bool, decimal_comma: bool, glob: bool) -> DataFrame
- `read_database`
  - **Polars**: (query: str | TextClause | Selectable, connection: ConnectionOrCursor | str, iter_batches: bool, batch_size: int | None, schema_overrides: SchemaDict | None, infer_schema_length: int | None, execute_options: dict[str, Any] | None) -> DataFrame | Iterator[DataFrame]
- `read_database_uri`
  - **Polars**: (query: list[str] | str, uri: str, partition_on: str | None, partition_range: tuple[int, int] | None, partition_num: int | None, protocol: str | None, engine: DbReadEngine | None, schema_overrides: SchemaDict | None, execute_options: dict[str, Any] | None, pre_execution_query: str | list[str] | None) -> DataFrame
- `read_delta`
  - **Polars**: (source: str | Path | DeltaTable, version: int | str | datetime | None, columns: list[str] | None, rechunk: bool | None, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, delta_table_options: dict[str, Any] | None, use_pyarrow: bool, pyarrow_options: dict[str, Any] | None) -> DataFrame
- `read_excel`
  - **Polars**: (source: FileSource, sheet_id: int | Sequence[int] | None, sheet_name: str | list[str] | tuple[str, ...] | None, table_name: str | None, engine: ExcelSpreadsheetEngine, engine_options: dict[str, Any] | None, read_options: dict[str, Any] | None, has_header: bool, columns: Sequence[int] | Sequence[str] | str | None, schema_overrides: SchemaDict | None, infer_schema_length: int | None, include_file_paths: str | None, drop_empty_rows: bool, drop_empty_cols: bool, raise_if_empty: bool) -> DataFrame | dict[str, pl.DataFrame]
- `read_ipc`
  - **Polars**: (source: str | Path | IO[bytes] | bytes, columns: list[int] | list[str] | None, n_rows: int | None, use_pyarrow: bool, memory_map: bool, storage_options: StorageOptionsDict | None, row_index_name: str | None, row_index_offset: int, rechunk: bool) -> DataFrame
- `read_ipc_schema`
  - **Polars**: (source: str | Path | IO[bytes] | bytes) -> dict[str, DataType]
- `read_ipc_stream`
  - **Polars**: (source: str | Path | IO[bytes] | bytes, columns: list[int] | list[str] | None, n_rows: int | None, use_pyarrow: bool, storage_options: StorageOptionsDict | None, row_index_name: str | None, row_index_offset: int, rechunk: bool) -> DataFrame
- `read_json`
  - **Polars**: (source: str | Path | IOBase | bytes, schema: SchemaDefinition | None, schema_overrides: SchemaDefinition | None, infer_schema_length: int | None) -> DataFrame
- `read_lines`
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] | list[IO[str]] | list[IO[bytes]], name: str, n_rows: int | None, row_index_name: str | None, row_index_offset: int, glob: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, include_file_paths: str | None) -> DataFrame
- `read_ndjson`
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] | list[IO[str]] | list[IO[bytes]], schema: SchemaDefinition | None, schema_overrides: SchemaDefinition | None, infer_schema_length: int | None, batch_size: int | None, n_rows: int | None, low_memory: bool, rechunk: bool, row_index_name: str | None, row_index_offset: int, ignore_errors: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, file_cache_ttl: int | None, include_file_paths: str | None) -> DataFrame
- `read_ods`
  - **Polars**: (source: FileSource, sheet_id: int | Sequence[int] | None, sheet_name: str | list[str] | tuple[str, ...] | None, has_header: bool, columns: Sequence[int] | Sequence[str] | None, schema_overrides: SchemaDict | None, infer_schema_length: int | None, include_file_paths: str | None, drop_empty_rows: bool, drop_empty_cols: bool, raise_if_empty: bool) -> DataFrame | dict[str, pl.DataFrame]
- `read_parquet`
  - **Narwhals**: (source: FileSource, backend: IntoBackend[EagerAllowed], **kwargs: Any) -> DataFrame[Any]
  - **Polars**: (source: FileSource, columns: list[int] | list[str] | None, n_rows: int | None, row_index_name: str | None, row_index_offset: int, parallel: ParallelStrategy, use_statistics: bool, hive_partitioning: bool | None, glob: bool, schema: SchemaDict | None, hive_schema: SchemaDict | None, try_parse_hive_dates: bool, rechunk: bool, low_memory: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, use_pyarrow: bool, pyarrow_options: dict[str, Any] | None, memory_map: bool, include_file_paths: str | None, missing_columns: Literal['insert', 'raise'], allow_missing_columns: bool | None) -> DataFrame
- `read_parquet_metadata`
  - **Polars**: (source: str | Path | IO[bytes] | bytes, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None) -> dict[str, str]
- `read_parquet_schema`
  - **Polars**: (source: str | Path | IO[bytes] | bytes) -> dict[str, DataType]
- `reduce`
  - **Polars**: (function: Callable[[Series, Series], Series], exprs: Sequence[Expr | str] | Expr, returns_scalar: bool, return_dtype: DataTypeExpr | PolarsDataType | None) -> Expr
- `register_extension_type`
  - **Polars**: (ext_name: str, ext_class: type[dt.BaseExtension] | None, as_storage: bool) -> None
- `repeat`
  - **Polars**: (value: IntoExpr | None, n: int | Expr, dtype: PolarsDataType | None, eager: bool) -> Expr | Series
- `rolling_corr`
  - **Polars**: (a: str | Expr, b: str | Expr, window_size: int, min_samples: int | None, ddof: int) -> Expr
- `rolling_cov`
  - **Polars**: (a: str | Expr, b: str | Expr, window_size: int, min_samples: int | None, ddof: int) -> Expr
- `row_index`
  - **Polars**: (name: str) -> Expr
- `scan_csv`
  - **Narwhals**: (source: FileSource, backend: IntoBackend[Backend], separator: str, **kwargs: Any) -> LazyFrame[Any]
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] | list[IO[str]] | list[IO[bytes]] | list[bytes], has_header: bool, separator: str, comment_prefix: str | None, quote_char: str | None, skip_rows: int, skip_lines: int, schema: SchemaDict | None, schema_overrides: SchemaDict | Sequence[PolarsDataType] | None, null_values: str | Sequence[str] | dict[str, str] | None, missing_utf8_is_empty_string: bool, ignore_errors: bool, cache: bool, with_column_names: Callable[[list[str]], list[str]] | None, infer_schema: bool, infer_schema_length: int | None, n_rows: int | None, encoding: CsvEncoding, low_memory: bool, rechunk: bool, skip_rows_after_header: int, row_index_name: str | None, row_index_offset: int, try_parse_dates: bool, eol_char: str, new_columns: Sequence[str] | None, raise_if_empty: bool, truncate_ragged_lines: bool, decimal_comma: bool, glob: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, file_cache_ttl: int | None, include_file_paths: str | None) -> LazyFrame
- `scan_delta`
  - **Polars**: (source: str | Path | DeltaTable, version: int | str | datetime | None, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, delta_table_options: dict[str, Any] | None, use_pyarrow: bool, pyarrow_options: dict[str, Any] | None, rechunk: bool | None) -> LazyFrame
- `scan_iceberg`
  - **Polars**: (source: str | Table, snapshot_id: int | None, storage_options: StorageOptionsDict | None, reader_override: Literal['native', 'pyiceberg'] | None, use_metadata_statistics: bool, fast_deletion_count: bool | None, use_pyiceberg_filter: bool) -> LazyFrame
- `scan_ipc`
  - **Polars**: (source: str | Path | IO[bytes] | bytes | list[str] | list[Path] | list[IO[bytes]] | list[bytes], n_rows: int | None, cache: bool, rechunk: bool, row_index_name: str | None, row_index_offset: int, glob: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, memory_map: bool, retries: int | None, file_cache_ttl: int | None, hive_partitioning: bool | None, hive_schema: SchemaDict | None, try_parse_hive_dates: bool, include_file_paths: str | None,_record_batch_statistics: bool) -> LazyFrame
- `scan_lines`
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] | list[IO[str]] | list[IO[bytes]], name: str, n_rows: int | None, row_index_name: str | None, row_index_offset: int, glob: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, include_file_paths: str | None) -> LazyFrame
- `scan_ndjson`
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] | list[IO[str]] | list[IO[bytes]], schema: SchemaDefinition | None, schema_overrides: SchemaDefinition | None, infer_schema_length: int | None, batch_size: int | None, n_rows: int | None, low_memory: bool, rechunk: bool, row_index_name: str | None, row_index_offset: int, ignore_errors: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, file_cache_ttl: int | None, include_file_paths: str | None) -> LazyFrame
- `scan_parquet`
  - **Narwhals**: (source: FileSource, backend: IntoBackend[Backend], **kwargs: Any) -> LazyFrame[Any]
  - **Polars**: (source: FileSource, n_rows: int | None, row_index_name: str | None, row_index_offset: int, parallel: ParallelStrategy, use_statistics: bool, hive_partitioning: bool | None, glob: bool, hidden_file_prefix: str | Sequence[str] | None, schema: SchemaDict | None, hive_schema: SchemaDict | None, try_parse_hive_dates: bool, rechunk: bool, low_memory: bool, cache: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, include_file_paths: str | None, missing_columns: Literal['insert', 'raise'], allow_missing_columns: bool | None, extra_columns: Literal['ignore', 'raise'], cast_options: ScanCastOptions | None,_column_mapping: ColumnMapping | None,_default_values: DefaultFieldValues | None,_deletion_files: DeletionFiles | None,_table_statistics: DataFrame | None,_row_count: tuple[int, int] | None) -> LazyFrame
- `scan_pyarrow_dataset`
  - **Polars**: (source: Dataset, allow_pyarrow_filter: bool, batch_size: int | None) -> LazyFrame
- `select`
  - **Polars**: (*exprs: IntoExpr | Iterable[IntoExpr], eager: bool, **named_exprs: IntoExpr) -> DataFrame | LazyFrame
- `self_dtype`
  - **Polars**: () -> DataTypeExpr
- `set_random_seed`
  - **Polars**: (seed: int) -> None
- `sql`
  - **Polars**: (query: str, eager: bool) -> DataFrame | LazyFrame
- `sql_expr`
  - **Polars**: (sql: str | Sequence[str]) -> Expr | list[Expr]
- `std`
  - **Polars**: (column: str, ddof: int) -> Expr
- `struct`
  - **Polars**: (*exprs: IntoExpr | Iterable[IntoExpr], schema: SchemaDict | None, eager: bool, **named_exprs: IntoExpr) -> Expr | Series
- `struct_with_fields`
  - **Polars**: (mapping: Mapping[str, PolarsDataType | pl.DataTypeExpr]) -> DataTypeExpr
- `tail`
  - **Polars**: (column: str, n: int) -> Expr
- `thread_pool_size`
  - **Polars**: () -> int
- `time`
  - **Polars**: (hour: Expr | str | int | None, minute: Expr | str | int | None, second: Expr | str | int | None, microsecond: Expr | str | int | None) -> Expr
- `time_range`
  - **Polars**: (start: time | IntoExprColumn | None, end: time | IntoExprColumn | None, interval: str | timedelta, closed: ClosedInterval, eager: bool) -> Series | Expr
- `time_ranges`
  - **Polars**: (start: time | IntoExprColumn | None, end: time | IntoExprColumn | None, interval: str | timedelta, closed: ClosedInterval, eager: bool) -> Series | Expr
- `to_native`
  - **Narwhals**: (narwhals_object: DataFrame[IntoDataFrameT] | LazyFrame[IntoLazyFrameT] | Series[IntoSeriesT], pass_through: bool) -> IntoDataFrameT | IntoLazyFrameT | IntoSeriesT | Any
- `to_py_scalar`
  - **Narwhals**: (scalar_like: Any) -> Any
- `union`
  - **Polars**: (items: Iterable[PolarsType], how: ConcatMethod, strict: bool) -> PolarsType
- `unregister_extension_type`
  - **Polars**: (ext_name: str) -> None
- `using_string_cache`
  - **Polars**: () -> bool
- `var`
  - **Polars**: (column: str, ddof: int) -> Expr
- `wrap_df`
  - **Polars**: (df: PyDataFrame) -> DataFrame
- `wrap_s`
  - **Polars**: (s: PySeries) -> Series
- `zeros`
  - **Polars**: (n: int | Expr, dtype: PolarsDataType, eager: bool) -> Expr | Series

### [!] Signature Mismatches (12)

- `Array` (nw)
  - **Narwhals**: (`inner: IntoDType`, `shape: int | tuple[int, ...]`) -> None
  - **Polars**: (`inner: PolarsDataType | PythonDataType`, `shape: int | tuple[int, ...] | None`, `width: int | None`) -> None
  - **pql**: (`inner: DataType`, `size: int`) -> None
- `Datetime` (nw)
  - **Narwhals**: (`time_unit: TimeUnit`, `time_zone: str | timezone | None`) -> None
  - **Polars**: (`time_unit: TimeUnit`, `time_zone: str | tzinfo | None`) -> None
  - **pql**: (`time_unit: EpochTimeUnit`) -> None
- `Decimal` (nw)
  - **Narwhals**: (`precision: int | None`, scale: int) -> None
  - **Polars**: (`precision: int | None`, scale: int) -> None
  - **pql**: (`precision: int`, scale: int) -> None
- `Duration` (nw)
  - **Narwhals**: (`time_unit: TimeUnit`) -> None
  - **Polars**: (`time_unit: TimeUnit`) -> None
  - **pql**: () -> None
- `Enum` (nw)
  - **Narwhals**: (`categories: Iterable[str] | type[enum.Enum]`) -> None
  - **Polars**: (`categories: Series | Iterable[str] | type[enum.Enum]`) -> None
  - **pql**: (`categories: Iterable[str] | type[PyEnum]`) -> None
- `Expr` (nw)
  - **Narwhals**: (`*nodes: ExprNode`) -> None
  - **Polars**: ()
  - **pql**: (`inner: SqlExpr`, `meta: Option[ExprMeta]`) -> None
- `LazyFrame` (nw)
  - **Narwhals**: (`df: Any`, `level: Literal['full', 'lazy', 'interchange']`) -> None
  - **Polars**: (`data: FrameInitTypes | None`, `schema: SchemaDefinition | None`, `schema_overrides: SchemaDict | None`, `strict: bool`, `orient: Orientation | None`, `infer_schema_length: int | None`, `nan_to_null: bool`, `height: int | None`) -> None
  - **pql**: (`data: IntoRel`) -> None
- `List` (nw)
  - **Narwhals**: (`inner: IntoDType`) -> None
  - **Polars**: (`inner: PolarsDataType | PythonDataType`) -> None
  - **pql**: (`inner: DataType`) -> None
- `Struct` (nw)
  - **Narwhals**: (`fields: Sequence[Field] | Mapping[str, IntoDType]`) -> None
  - **Polars**: (`fields: Sequence[Field] | SchemaDict`) -> None
  - **pql**: (`fields: IntoDict[str, DataType]`) -> None
- `all` (nw)
  - **Narwhals**: () -> Expr
  - **Polars**: (`*names: str`, `ignore_nulls: bool`) -> Expr
  - **pql**: (`exclude: Iterable[IntoExprColumn] | None`) -> Expr
- `coalesce` (nw)
  - **Narwhals**: (exprs: IntoExpr | Iterable[IntoExpr], `*more_exprs: IntoExpr | NonNestedLiteral`) -> Expr
  - **Polars**: (exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr, `eager: bool`) -> Expr | Series
  - **pql**: (exprs: TryIter[IntoExpr], `*more_exprs: IntoExpr`) -> Expr
- `lit` (nw)
  - **Narwhals**: (value: PythonLiteral, `dtype: IntoDType | None`) -> Expr
  - **Polars**: (value: Any, `dtype: PolarsDataType | None`, `allow_object: bool`) -> Expr
  - **pql**: (value: PythonLiteral) -> Expr

### [+] Extra Methods (pql-only) (8)

- `BitString`
- `DatetimeTZ`
- `Json`
- `Map`
- `Number`
- `TimeTZ`
- `UUID`
- `Union`
