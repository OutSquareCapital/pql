# pql vs Polars API Comparison Report

This report shows the API coverage of pql compared to Polars.

## Summary

| Class | Coverage | Matched | Missing | Mismatched | Extra |
|-------|----------|---------|---------|------------|-------|
| LazyFrame | 3.4% | 3 | 74 | 11 | 2 |
| Expr | 13.8% | 30 | 181 | 6 | 13 |
| Expr.str | 12.2% | 6 | 38 | 5 | 0 |
| Expr.dt | 19.1% | 9 | 37 | 1 | 0 |

## LazyFrame

### [v] Matched Methods (3)

- `head`
- `limit`
- `tail`

### [x] Missing Methods (74)

- `approx_n_unique` () -> LazyFrame
- `bottom_k` (k: int, by: IntoExpr | Iterable[IntoExpr], reverse: bool | Sequence[bool]) -> LazyFrame
- `cache` () -> LazyFrame
- `cast` (dtypes: Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType | PythonDataType] | PolarsDataType | pl.DataTypeExpr | Schema, strict: bool) -> LazyFrame
- `clear` (n: int) -> LazyFrame
- `clone` () -> LazyFrame
- `collect_async` (gevent: bool, engine: EngineType, optimizations: QueryOptFlags) -> Awaitable[DataFrame] | _GeventDataFrameResult[DataFrame]
- `collect_batches` (chunk_size: int | None, maintain_order: bool, lazy: bool, engine: EngineType, optimizations: QueryOptFlags) -> Iterator[DataFrame]
- `collect_schema` () -> Schema
- `columns` ()
- `count` () -> LazyFrame
- `describe` (percentiles: Sequence[float] | float | None, interpolation: QuantileMethod) -> DataFrame
- `deserialize` (source: str | bytes | Path | IOBase, format: SerializationFormat) -> LazyFrame
- `drop_nans` (subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None) -> LazyFrame
- `drop_nulls` (subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None) -> LazyFrame
- `dtypes` ()
- `explain` (format: ExplainFormat, optimized: bool, type_coercion: bool, predicate_pushdown: bool, projection_pushdown: bool, simplify_expression: bool, slice_pushdown: bool, comm_subplan_elim: bool, comm_subexpr_elim: bool, cluster_with_columns: bool, collapse_joins: bool, streaming: bool, engine: EngineType, tree_format: bool | None, optimizations: QueryOptFlags) -> str
- `explode` (columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector], more_columns: ColumnNameOrSelector, empty_as_null: bool, keep_nulls: bool) -> LazyFrame
- `fetch` (n_rows: int, kwargs: Any) -> DataFrame
- `fill_nan` (value: int | float | Expr | None) -> LazyFrame
- `fill_null` (value: Any | Expr | None, strategy: FillNullStrategy | None, limit: int | None, matches_supertype: bool) -> LazyFrame
- `first` () -> LazyFrame
- `gather_every` (n: int, offset: int) -> LazyFrame
- `group_by_dynamic` (index_column: IntoExpr, every: str | timedelta, period: str | timedelta | None, offset: str | timedelta | None, include_boundaries: bool, closed: ClosedInterval, label: Label, group_by: IntoExpr | Iterable[IntoExpr] | None, start_by: StartBy) -> LazyGroupBy
- `inspect` (fmt: str) -> LazyFrame
- `interpolate` () -> LazyFrame
- `join_asof` (other: LazyFrame, left_on: str | None | Expr, right_on: str | None | Expr, on: str | None | Expr, by_left: str | Sequence[str] | None, by_right: str | Sequence[str] | None, by: str | Sequence[str] | None, strategy: AsofJoinStrategy, suffix: str, tolerance: str | int | float | timedelta | None, allow_parallel: bool, force_parallel: bool, coalesce: bool, allow_exact_matches: bool, check_sortedness: bool) -> LazyFrame
- `join_where` (other: LazyFrame, predicates: Expr | Iterable[Expr], suffix: str) -> LazyFrame
- `last` () -> LazyFrame
- `lazy` () -> LazyFrame
- `map_batches` (function: Callable[[DataFrame], DataFrame], predicate_pushdown: bool, projection_pushdown: bool, slice_pushdown: bool, no_optimizations: bool, schema: None | SchemaDict, validate_output_schema: bool, streamable: bool) -> LazyFrame
- `match_to_schema` (schema: SchemaDict | Schema, missing_columns: Literal['insert', 'raise'] | Mapping[str, Literal['insert', 'raise'] | Expr], missing_struct_fields: Literal['insert', 'raise'] | Mapping[str, Literal['insert', 'raise']], extra_columns: Literal['ignore', 'raise'], extra_struct_fields: Literal['ignore', 'raise'] | Mapping[str, Literal['ignore', 'raise']], integer_cast: Literal['upcast', 'forbid'] | Mapping[str, Literal['upcast', 'forbid']], float_cast: Literal['upcast', 'forbid'] | Mapping[str, Literal['upcast', 'forbid']]) -> LazyFrame
- `max` () -> LazyFrame
- `mean` () -> LazyFrame
- `median` () -> LazyFrame
- `melt` (id_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None, value_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None, variable_name: str | None, value_name: str | None, streamable: bool) -> LazyFrame
- `merge_sorted` (other: LazyFrame, key: str) -> LazyFrame
- `min` () -> LazyFrame
- `null_count` () -> LazyFrame
- `pipe` (function: Callable[Concatenate[LazyFrame, P], T], args: P.args, kwargs: P.kwargs) -> T
- `pipe_with_schema` (function: Callable[[LazyFrame, Schema], LazyFrame]) -> LazyFrame
- `pivot` (on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector], on_columns: Sequence[Any] | pl.Series | pl.DataFrame, index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None, values: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None, aggregate_function: PivotAgg | Expr | None, maintain_order: bool, separator: str) -> LazyFrame
- `profile` (type_coercion: bool, predicate_pushdown: bool, projection_pushdown: bool, simplify_expression: bool, no_optimization: bool, slice_pushdown: bool, comm_subplan_elim: bool, comm_subexpr_elim: bool, cluster_with_columns: bool, collapse_joins: bool, show_plot: bool, truncate_nodes: int, figsize: tuple[int, int], engine: EngineType, optimizations: QueryOptFlags,_kwargs: Any) -> tuple[DataFrame, DataFrame]
- `quantile` (quantile: float | Expr, interpolation: QuantileMethod) -> LazyFrame
- `remote` (context: pc.ComputeContext | None, plan_type: pc._typing.PlanTypePreference, n_retries: int, engine: pc._typing.Engine, scaling_mode: pc._typing.ScalingMode) -> pc.LazyFrameRemote
- `remove` (predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool] | np.ndarray[Any, Any], constraints: Any) -> LazyFrame
- `reverse` () -> LazyFrame
- `rolling` (index_column: IntoExpr, period: str | timedelta, offset: str | timedelta | None, closed: ClosedInterval, group_by: IntoExpr | Iterable[IntoExpr] | None) -> LazyGroupBy
- `schema` ()
- `select_seq` (exprs: IntoExpr | Iterable[IntoExpr], named_exprs: IntoExpr) -> LazyFrame
- `serialize` (file: IOBase | str | Path | None, format: SerializationFormat) -> bytes | str | None
- `set_sorted` (column: str | list[str], more_columns: str, descending: bool | list[bool], nulls_last: bool | list[bool]) -> LazyFrame
- `shift` (n: int | IntoExprColumn, fill_value: IntoExpr | None) -> LazyFrame
- `show` (limit: int | None, ascii_tables: bool | None, decimal_separator: str | None, thousands_separator: str | bool | None, float_precision: int | None, fmt_float: FloatFmt | None, fmt_str_lengths: int | None, fmt_table_cell_list_len: int | None, tbl_cell_alignment: Literal['LEFT', 'CENTER', 'RIGHT'] | None, tbl_cell_numeric_alignment: Literal['LEFT', 'CENTER', 'RIGHT'] | None, tbl_cols: int | None, tbl_column_data_type_inline: bool | None, tbl_dataframe_shape_below: bool | None, tbl_formatting: TableFormatNames | None, tbl_hide_column_data_types: bool | None, tbl_hide_column_names: bool | None, tbl_hide_dtype_separator: bool | None, tbl_hide_dataframe_shape: bool | None, tbl_width_chars: int | None, trim_decimal_zeros: bool | None) -> None
- `show_graph` (optimized: bool, show: bool, output_path: str | Path | None, raw_output: bool, figsize: tuple[float, float], type_coercion: bool,_type_check: bool, predicate_pushdown: bool, projection_pushdown: bool, simplify_expression: bool, slice_pushdown: bool, comm_subplan_elim: bool, comm_subexpr_elim: bool, cluster_with_columns: bool, collapse_joins: bool, engine: EngineType, plan_stage: PlanStage,_check_order: bool, optimizations: QueryOptFlags) -> str | None
- `sink_batches` (function: Callable[[DataFrame], bool | None], chunk_size: int | None, maintain_order: bool, lazy: bool, engine: EngineType, optimizations: QueryOptFlags) -> pl.LazyFrame | None
- `sink_csv` (path: str | Path | IO[bytes] | IO[str] | _SinkDirectory | PartitionBy, include_bom: bool, include_header: bool, separator: str, line_terminator: str, quote_char: str, batch_size: int, datetime_format: str | None, date_format: str | None, time_format: str | None, float_scientific: bool | None, float_precision: int | None, decimal_comma: bool, null_value: str | None, quote_style: CsvQuoteStyle | None, maintain_order: bool, storage_options: dict[str, Any] | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int, sync_on_close: SyncOnCloseMethod | None, mkdir: bool, lazy: bool, engine: EngineType, optimizations: QueryOptFlags) -> LazyFrame | None
- `sink_delta` (target: str | Path | deltalake.DeltaTable, mode: Literal['error', 'append', 'overwrite', 'ignore', 'merge'], storage_options: dict[str, str] | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, delta_write_options: dict[str, Any] | None, delta_merge_options: dict[str, Any] | None, optimizations: QueryOptFlags) -> deltalake.table.TableMerger | None
- `sink_ipc` (path: str | Path | IO[bytes] | _SinkDirectory | PartitionBy, compression: IpcCompression | None, compat_level: CompatLevel | None, record_batch_size: int | None, maintain_order: bool, storage_options: dict[str, Any] | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int, sync_on_close: SyncOnCloseMethod | None, mkdir: bool, lazy: bool, engine: EngineType, optimizations: QueryOptFlags) -> LazyFrame | None
- `sink_ndjson` (path: str | Path | IO[bytes] | IO[str] | _SinkDirectory | PartitionBy, maintain_order: bool, storage_options: dict[str, Any] | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int, sync_on_close: SyncOnCloseMethod | None, mkdir: bool, lazy: bool, engine: EngineType, optimizations: QueryOptFlags) -> LazyFrame | None
- `sink_parquet` (path: str | Path | IO[bytes] | _SinkDirectory | PartitionBy, compression: str, compression_level: int | None, statistics: bool | str | dict[str, bool], row_group_size: int | None, data_page_size: int | None, maintain_order: bool, storage_options: dict[str, Any] | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int, sync_on_close: SyncOnCloseMethod | None, metadata: ParquetMetadata | None, mkdir: bool, lazy: bool, field_overwrites: ParquetFieldOverwrites | Sequence[ParquetFieldOverwrites] | Mapping[str, ParquetFieldOverwrites] | None, engine: EngineType, optimizations: QueryOptFlags) -> LazyFrame | None
- `slice` (offset: int, length: int | None) -> LazyFrame
- `std` (ddof: int) -> LazyFrame
- `sum` () -> LazyFrame
- `top_k` (k: int, by: IntoExpr | Iterable[IntoExpr], reverse: bool | Sequence[bool]) -> LazyFrame
- `unnest` (columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector], more_columns: ColumnNameOrSelector, separator: str | None) -> LazyFrame
- `unpivot` (on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None, index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None, variable_name: str | None, value_name: str | None, streamable: bool) -> LazyFrame
- `update` (other: LazyFrame, on: str | Sequence[str] | None, how: Literal['left', 'inner', 'full'], left_on: str | Sequence[str] | None, right_on: str | Sequence[str] | None, include_nulls: bool, maintain_order: MaintainOrderJoin | None) -> LazyFrame
- `var` (ddof: int) -> LazyFrame
- `width` ()
- `with_columns_seq` (exprs: IntoExpr | Iterable[IntoExpr], named_exprs: IntoExpr) -> LazyFrame
- `with_context` (other: Self | list[Self]) -> LazyFrame
- `with_row_count` (name: str, offset: int) -> LazyFrame
- `with_row_index` (name: str, offset: int) -> LazyFrame

### [!] Signature Mismatches (11)

- `collect`: Missing params: _kwargs, background, cluster_with_columns, collapse_joins, comm_subexpr_elim, comm_subplan_elim, engine, no_optimization, optimizations, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, type_coercion
  - Polars: (type_coercion: bool, predicate_pushdown: bool, projection_pushdown: bool, simplify_expression: bool, slice_pushdown: bool, comm_subplan_elim: bool, comm_subexpr_elim: bool, cluster_with_columns: bool, collapse_joins: bool, no_optimization: bool, engine: EngineType, background: bool, optimizations: QueryOptFlags,_kwargs: Any) -> DataFrame | InProcessQuery
  - pql: () -> pl.DataFrame
- `drop`: Missing params: strict
  - Polars: (columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector], strict: bool) -> LazyFrame
  - pql: (columns: str) -> Self
- `filter`: Missing params: constraints
  - Polars: (predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool], constraints: Any) -> LazyFrame
  - pql: (predicates: Expr) -> Self
- `group_by`: Missing params: maintain_order, named_by
  - Polars: (by: IntoExpr | Iterable[IntoExpr], maintain_order: bool, named_by: IntoExpr) -> LazyGroupBy
  - pql: (by: str | Expr) -> GroupBy
- `join`: Missing params: allow_parallel, coalesce, force_parallel, maintain_order, nulls_equal, validate
  - Polars: (other: LazyFrame, on: str | Expr | Sequence[str | Expr] | None, how: JoinStrategy, left_on: str | Expr | Sequence[str | Expr] | None, right_on: str | Expr | Sequence[str | Expr] | None, suffix: str, validate: JoinValidation, nulls_equal: bool, coalesce: bool | None, maintain_order: MaintainOrderJoin | None, allow_parallel: bool, force_parallel: bool) -> LazyFrame
  - pql: (other: LazyFrame, on: str | Expr | Iterable[str] | None, left_on: str | Expr | Iterable[str] | None, right_on: str | Expr | Iterable[str] | None, how: Literal['inner', 'left', 'right', 'outer', 'cross', 'semi', 'anti'], suffix: str) -> Self
- `rename`: Missing params: strict
  - Polars: (mapping: Mapping[str, str] | Callable[[str], str], strict: bool) -> LazyFrame
  - pql: (mapping: Mapping[str, str]) -> Self
- `select`: Missing params: named_exprs
  - Polars: (exprs: IntoExpr | Iterable[IntoExpr], named_exprs: IntoExpr) -> LazyFrame
  - pql: (exprs: Expr | str) -> Self
- `sort`: Missing params: maintain_order, more_by, multithreaded
  - Polars: (by: IntoExpr | Iterable[IntoExpr], more_by: IntoExpr, descending: bool | Sequence[bool], nulls_last: bool | Sequence[bool], maintain_order: bool, multithreaded: bool) -> LazyFrame
  - pql: (by: str | Expr, descending: bool | Iterable[bool], nulls_last: bool | Iterable[bool]) -> Self
- `sql`: Missing params: query, table_name; Extra params: pretty
  - Polars: (query: str, table_name: str) -> LazyFrame
  - pql: (pretty: bool) -> str
- `unique`: Missing params: keep, maintain_order
  - Polars: (subset: IntoExpr | Collection[IntoExpr] | None, keep: UniqueKeepStrategy, maintain_order: bool) -> LazyFrame
  - pql: (subset: str | Iterable[str] | None) -> Self
- `with_columns`: Missing params: named_exprs
  - Polars: (exprs: IntoExpr | Iterable[IntoExpr], named_exprs: IntoExpr) -> LazyFrame
  - pql: (exprs: Expr) -> Self

### [+] Extra Methods (pql-only) (2)

- `distinct`
- `relation`

## Expr

### [v] Matched Methods (30)

- `abs`
- `add`
- `alias`
- `and_`
- `count`
- `dt`
- `eq`
- `floordiv`
- `ge`
- `gt`
- `is_not_null`
- `is_null`
- `le`
- `lt`
- `max`
- `mean`
- `min`
- `mod`
- `mul`
- `n_unique`
- `ne`
- `neg`
- `not_`
- `or_`
- `std`
- `str`
- `sub`
- `sum`
- `truediv`
- `var`

### [x] Missing Methods (181)

- `agg_groups` () -> Expr
- `all` (ignore_nulls: bool) -> Expr
- `any` (ignore_nulls: bool) -> Expr
- `append` (other: IntoExpr, upcast: bool) -> Expr
- `approx_n_unique` () -> Expr
- `arccos` () -> Expr
- `arccosh` () -> Expr
- `arcsin` () -> Expr
- `arcsinh` () -> Expr
- `arctan` () -> Expr
- `arctanh` () -> Expr
- `arg_max` () -> Expr
- `arg_min` () -> Expr
- `arg_sort` (descending: bool, nulls_last: bool) -> Expr
- `arg_true` () -> Expr
- `arg_unique` () -> Expr
- `arr` ()
- `backward_fill` (limit: int | None) -> Expr
- `bin` ()
- `bitwise_and` () -> Expr
- `bitwise_count_ones` () -> Expr
- `bitwise_count_zeros` () -> Expr
- `bitwise_leading_ones` () -> Expr
- `bitwise_leading_zeros` () -> Expr
- `bitwise_or` () -> Expr
- `bitwise_trailing_ones` () -> Expr
- `bitwise_trailing_zeros` () -> Expr
- `bitwise_xor` () -> Expr
- `bottom_k` (k: int | IntoExprColumn) -> Expr
- `bottom_k_by` (by: IntoExpr | Iterable[IntoExpr], k: int | IntoExprColumn, reverse: bool | Sequence[bool]) -> Expr
- `cat` ()
- `cbrt` () -> Expr
- `ceil` () -> Expr
- `clip` (lower_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None, upper_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None) -> Expr
- `cos` () -> Expr
- `cosh` () -> Expr
- `cot` () -> Expr
- `cum_count` (reverse: bool) -> Expr
- `cum_max` (reverse: bool) -> Expr
- `cum_min` (reverse: bool) -> Expr
- `cum_prod` (reverse: bool) -> Expr
- `cum_sum` (reverse: bool) -> Expr
- `cumulative_eval` (expr: Expr, min_samples: int) -> Expr
- `cut` (breaks: Sequence[float], labels: Sequence[str] | None, left_closed: bool, include_breaks: bool) -> Expr
- `degrees` () -> Expr
- `deserialize` (source: str | Path | IOBase | bytes, format: SerializationFormat) -> Expr
- `diff` (n: int | IntoExpr, null_behavior: NullBehavior) -> Expr
- `dot` (other: Expr | str) -> Expr
- `drop_nans` () -> Expr
- `drop_nulls` () -> Expr
- `entropy` (base: float, normalize: bool) -> Expr
- `eq_missing` (other: Any) -> Expr
- `ewm_mean` (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, min_samples: int, ignore_nulls: bool) -> Expr
- `ewm_mean_by` (by: str | IntoExpr, half_life: str | timedelta) -> Expr
- `ewm_std` (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, bias: bool, min_samples: int, ignore_nulls: bool) -> Expr
- `ewm_var` (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, bias: bool, min_samples: int, ignore_nulls: bool) -> Expr
- `exclude` (columns: str | PolarsDataType | Collection[str] | Collection[PolarsDataType], more_columns: str | PolarsDataType) -> Expr
- `exp` () -> Expr
- `explode` (empty_as_null: bool, keep_nulls: bool) -> Expr
- `ext` ()
- `extend_constant` (value: IntoExpr, n: int | IntoExprColumn) -> Expr
- `fill_nan` (value: int | float | Expr | None) -> Expr
- `filter` (predicates: IntoExprColumn | Iterable[IntoExprColumn], constraints: Any) -> Expr
- `flatten` () -> Expr
- `floor` () -> Expr
- `forward_fill` (limit: int | None) -> Expr
- `from_json` (value: str) -> Expr
- `gather` (indices: int | Sequence[int] | IntoExpr | Series | np.ndarray[Any, Any]) -> Expr
- `gather_every` (n: int, offset: int) -> Expr
- `get` (index: int | Expr, null_on_oob: bool) -> Expr
- `has_nulls` () -> Expr
- `hash` (seed: int, seed_1: int | None, seed_2: int | None, seed_3: int | None) -> Expr
- `head` (n: int | Expr) -> Expr
- `hist` (bins: IntoExpr | None, bin_count: int | None, include_category: bool, include_breakpoint: bool) -> Expr
- `implode` () -> Expr
- `index_of` (element: IntoExpr) -> Expr
- `inspect` (fmt: str) -> Expr
- `interpolate` (method: InterpolationMethod) -> Expr
- `interpolate_by` (by: IntoExpr) -> Expr
- `is_between` (lower_bound: IntoExpr, upper_bound: IntoExpr, closed: ClosedInterval) -> Expr
- `is_close` (other: IntoExpr, abs_tol: float, rel_tol: float, nans_equal: bool) -> Expr
- `is_duplicated` () -> Expr
- `is_finite` () -> Expr
- `is_first_distinct` () -> Expr
- `is_infinite` () -> Expr
- `is_last_distinct` () -> Expr
- `is_nan` () -> Expr
- `is_not_nan` () -> Expr
- `is_unique` () -> Expr
- `item` (allow_empty: bool) -> Expr
- `kurtosis` (fisher: bool, bias: bool) -> Expr
- `len` () -> Expr
- `limit` (n: int | Expr) -> Expr
- `list` ()
- `log` (base: float | IntoExpr) -> Expr
- `log10` () -> Expr
- `log1p` () -> Expr
- `lower_bound` () -> Expr
- `map_batches` (function: Callable[[Series], Series | Any], return_dtype: PolarsDataType | pl.DataTypeExpr | None, agg_list: bool, is_elementwise: bool, returns_scalar: bool) -> Expr
- `map_elements` (function: Callable[[Any], Any], return_dtype: PolarsDataType | pl.DataTypeExpr | None, skip_nulls: bool, pass_name: bool, strategy: MapElementsStrategy, returns_scalar: bool) -> Expr
- `max_by` (by: IntoExpr) -> Expr
- `median` () -> Expr
- `meta` ()
- `min_by` (by: IntoExpr) -> Expr
- `mode` (maintain_order: bool) -> Expr
- `name` ()
- `nan_max` () -> Expr
- `nan_min` () -> Expr
- `ne_missing` (other: Any) -> Expr
- `null_count` () -> Expr
- `over` (partition_by: IntoExpr | Iterable[IntoExpr] | None, more_exprs: IntoExpr, order_by: IntoExpr | Iterable[IntoExpr] | None, descending: bool, nulls_last: bool, mapping_strategy: WindowMappingStrategy) -> Expr
- `pct_change` (n: int | IntoExprColumn) -> Expr
- `peak_max` () -> Expr
- `peak_min` () -> Expr
- `pipe` (function: Callable[Concatenate[Expr, P], T], args: P.args, kwargs: P.kwargs) -> T
- `product` () -> Expr
- `qcut` (quantiles: Sequence[float] | int, labels: Sequence[str] | None, left_closed: bool, allow_duplicates: bool, include_breaks: bool) -> Expr
- `quantile` (quantile: float | Expr, interpolation: QuantileMethod) -> Expr
- `radians` () -> Expr
- `rank` (method: RankMethod, descending: bool, seed: int | None) -> Expr
- `rechunk` () -> Expr
- `register_plugin` (lib: str, symbol: str, args: list[IntoExpr] | None, kwargs: dict[Any, Any] | None, is_elementwise: bool, input_wildcard_expansion: bool, returns_scalar: bool, cast_to_supertypes: bool, pass_name_to_apply: bool, changes_length: bool) -> Expr
- `reinterpret` (signed: bool) -> Expr
- `repeat_by` (by: pl.Series | Expr | str | int) -> Expr
- `replace` (old: IntoExpr | Sequence[Any] | Mapping[Any, Any], new: IntoExpr | Sequence[Any] | NoDefault, default: IntoExpr | NoDefault, return_dtype: PolarsDataType | None) -> Expr
- `replace_strict` (old: IntoExpr | Sequence[Any] | Mapping[Any, Any], new: IntoExpr | Sequence[Any] | NoDefault, default: IntoExpr | NoDefault, return_dtype: PolarsDataType | pl.DataTypeExpr | None) -> Expr
- `reshape` (dimensions: tuple[int, ...]) -> Expr
- `reverse` () -> Expr
- `rle` () -> Expr
- `rle_id` () -> Expr
- `rolling` (index_column: IntoExprColumn, period: str | timedelta, offset: str | timedelta | None, closed: ClosedInterval) -> Expr
- `rolling_kurtosis` (window_size: int, fisher: bool, bias: bool, min_samples: int | None, center: bool) -> Expr
- `rolling_map` (function: Callable[[Series], Any], window_size: int, weights: list[float] | None, min_samples: int | None, center: bool) -> Expr
- `rolling_max` (window_size: int, weights: list[float] | None, min_samples: int | None, center: bool) -> Expr
- `rolling_max_by` (by: IntoExpr, window_size: timedelta | str, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_mean` (window_size: int, weights: list[float] | None, min_samples: int | None, center: bool) -> Expr
- `rolling_mean_by` (by: IntoExpr, window_size: timedelta | str, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_median` (window_size: int, weights: list[float] | None, min_samples: int | None, center: bool) -> Expr
- `rolling_median_by` (by: IntoExpr, window_size: timedelta | str, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_min` (window_size: int, weights: list[float] | None, min_samples: int | None, center: bool) -> Expr
- `rolling_min_by` (by: IntoExpr, window_size: timedelta | str, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_quantile` (quantile: float, interpolation: QuantileMethod, window_size: int, weights: list[float] | None, min_samples: int | None, center: bool) -> Expr
- `rolling_quantile_by` (by: IntoExpr, window_size: timedelta | str, quantile: float, interpolation: QuantileMethod, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_rank` (window_size: int, method: RankMethod, seed: int | None, min_samples: int | None, center: bool) -> Expr
- `rolling_rank_by` (by: IntoExpr, window_size: timedelta | str, method: RankMethod, seed: int | None, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_skew` (window_size: int, bias: bool, min_samples: int | None, center: bool) -> Expr
- `rolling_std` (window_size: int, weights: list[float] | None, min_samples: int | None, center: bool, ddof: int) -> Expr
- `rolling_std_by` (by: IntoExpr, window_size: timedelta | str, min_samples: int, closed: ClosedInterval, ddof: int) -> Expr
- `rolling_sum` (window_size: int, weights: list[float] | None, min_samples: int | None, center: bool) -> Expr
- `rolling_sum_by` (by: IntoExpr, window_size: timedelta | str, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_var` (window_size: int, weights: list[float] | None, min_samples: int | None, center: bool, ddof: int) -> Expr
- `rolling_var_by` (by: IntoExpr, window_size: timedelta | str, min_samples: int, closed: ClosedInterval, ddof: int) -> Expr
- `round` (decimals: int, mode: RoundMode) -> Expr
- `round_sig_figs` (digits: int) -> Expr
- `sample` (n: int | IntoExprColumn | None, fraction: float | IntoExprColumn | None, with_replacement: bool, shuffle: bool, seed: int | None) -> Expr
- `search_sorted` (element: IntoExpr | np.ndarray[Any, Any], side: SearchSortedSide, descending: bool) -> Expr
- `set_sorted` (descending: bool) -> Expr
- `shift` (n: int | IntoExprColumn, fill_value: IntoExpr | None) -> Expr
- `shrink_dtype` () -> Expr
- `shuffle` (seed: int | None) -> Expr
- `sign` () -> Expr
- `sin` () -> Expr
- `sinh` () -> Expr
- `skew` (bias: bool) -> Expr
- `slice` (offset: int | Expr, length: int | Expr | None) -> Expr
- `sort` (descending: bool, nulls_last: bool) -> Expr
- `sort_by` (by: IntoExpr | Iterable[IntoExpr], more_by: IntoExpr, descending: bool | Sequence[bool], nulls_last: bool | Sequence[bool], multithreaded: bool, maintain_order: bool) -> Expr
- `sqrt` () -> Expr
- `struct` ()
- `tail` (n: int | Expr) -> Expr
- `tan` () -> Expr
- `tanh` () -> Expr
- `to_physical` () -> Expr
- `top_k` (k: int | IntoExprColumn) -> Expr
- `top_k_by` (by: IntoExpr | Iterable[IntoExpr], k: int | IntoExprColumn, reverse: bool | Sequence[bool]) -> Expr
- `unique` (maintain_order: bool) -> Expr
- `unique_counts` () -> Expr
- `upper_bound` () -> Expr
- `value_counts` (sort: bool, parallel: bool, name: str | None, normalize: bool) -> Expr
- `where` (predicate: Expr) -> Expr
- `xor` (other: Any) -> Expr

### [!] Signature Mismatches (6)

- `cast`: Missing params: strict, wrap_numerical
  - Polars: (dtype: PolarsDataType | pl.DataTypeExpr | type[Any], strict: bool, wrap_numerical: bool) -> Expr
  - pql: (dtype: datatypes.DataType) -> Self
- `fill_null`: Missing params: limit, strategy
  - Polars: (value: Any | Expr | None, strategy: FillNullStrategy | None, limit: int | None) -> Expr
  - pql: (value: object) -> Self
- `first`: Missing params: ignore_nulls
  - Polars: (ignore_nulls: bool) -> Expr
  - pql: () -> Self
- `is_in`: Missing params: nulls_equal, other; Extra params: values
  - Polars: (other: Expr | Collection[Any] | Series, nulls_equal: bool) -> Expr
  - pql: (values: Iterable[object]) -> Self
- `last`: Missing params: ignore_nulls
  - Polars: (ignore_nulls: bool) -> Expr
  - pql: () -> Self
- `pow`: Missing params: exponent; Extra params: other
  - Polars: (exponent: IntoExprColumn | int | float) -> Expr
  - pql: (other: object) -> Self

### [+] Extra Methods (pql-only) (13)

- `between`
- `expr`
- `is_not_in`
- `pos`
- `radd`
- `rand`
- `rfloordiv`
- `rmod`
- `rmul`
- `ror`
- `rpow`
- `rsub`
- `rtruediv`

## Expr.str

### [v] Matched Methods (6)

- `ends_with`
- `len_chars`
- `slice`
- `starts_with`
- `to_lowercase`
- `to_uppercase`

### [x] Missing Methods (38)

- `concat` (delimiter: str | None, ignore_nulls: bool) -> Expr
- `contains_any` (patterns: IntoExpr, ascii_case_insensitive: bool) -> Expr
- `count_matches` (pattern: str | Expr, literal: bool) -> Expr
- `decode` (encoding: TransferEncoding, strict: bool) -> Expr
- `encode` (encoding: TransferEncoding) -> Expr
- `escape_regex` () -> Expr
- `explode` () -> Expr
- `extract` (pattern: IntoExprColumn, group_index: int) -> Expr
- `extract_all` (pattern: str | Expr) -> Expr
- `extract_groups` (pattern: str) -> Expr
- `extract_many` (patterns: IntoExpr, ascii_case_insensitive: bool, overlapping: bool, leftmost: bool) -> Expr
- `find` (pattern: str | Expr, literal: bool, strict: bool) -> Expr
- `find_many` (patterns: IntoExpr, ascii_case_insensitive: bool, overlapping: bool, leftmost: bool) -> Expr
- `head` (n: int | IntoExprColumn) -> Expr
- `join` (delimiter: str, ignore_nulls: bool) -> Expr
- `json_decode` (dtype: PolarsDataType | pl.DataTypeExpr, infer_schema_length: int | None) -> Expr
- `json_path_match` (json_path: IntoExprColumn) -> Expr
- `len_bytes` () -> Expr
- `normalize` (form: UnicodeForm) -> Expr
- `pad_end` (length: int | IntoExprColumn, fill_char: str) -> Expr
- `pad_start` (length: int | IntoExprColumn, fill_char: str) -> Expr
- `replace_all` (pattern: str | Expr, value: str | Expr, literal: bool) -> Expr
- `replace_many` (patterns: IntoExpr | Mapping[str, str], replace_with: IntoExpr | NoDefault, ascii_case_insensitive: bool, leftmost: bool) -> Expr
- `reverse` () -> Expr
- `split` (by: IntoExpr, inclusive: bool) -> Expr
- `split_exact` (by: IntoExpr, n: int, inclusive: bool) -> Expr
- `splitn` (by: IntoExpr, n: int) -> Expr
- `strip_prefix` (prefix: IntoExpr) -> Expr
- `strip_suffix` (suffix: IntoExpr) -> Expr
- `strptime` (dtype: PolarsTemporalType, format: str | None, strict: bool, exact: bool, cache: bool, ambiguous: Ambiguous | Expr) -> Expr
- `tail` (n: int | IntoExprColumn) -> Expr
- `to_date` (format: str | None, strict: bool, exact: bool, cache: bool) -> Expr
- `to_datetime` (format: str | None, time_unit: TimeUnit | None, time_zone: str | None, strict: bool, exact: bool, cache: bool, ambiguous: Ambiguous | Expr) -> Expr
- `to_decimal` (scale: int) -> Expr
- `to_integer` (base: int | IntoExprColumn, dtype: PolarsIntegerType, strict: bool) -> Expr
- `to_time` (format: str | None, strict: bool, cache: bool) -> Expr
- `to_titlecase` () -> Expr
- `zfill` (length: int | IntoExprColumn) -> Expr

### [!] Signature Mismatches (5)

- `contains`: Missing params: strict
  - Polars: (pattern: str | Expr, literal: bool, strict: bool) -> Expr
  - pql: (pattern: str, literal: bool) -> Expr
- `replace`: Missing params: literal, n, value; Extra params: replacement
  - Polars: (pattern: str | Expr, value: str | Expr, literal: bool, n: int) -> Expr
  - pql: (pattern: str, replacement: str) -> Expr
- `strip_chars`: Missing params: characters
  - Polars: (characters: IntoExpr) -> Expr
  - pql: () -> Expr
- `strip_chars_end`: Missing params: characters
  - Polars: (characters: IntoExpr) -> Expr
  - pql: () -> Expr
- `strip_chars_start`: Missing params: characters
  - Polars: (characters: IntoExpr) -> Expr
  - pql: () -> Expr

## Expr.dt

### [v] Matched Methods (9)

- `convert_time_zone`
- `date`
- `day`
- `hour`
- `minute`
- `month`
- `week`
- `weekday`
- `year`

### [x] Missing Methods (37)

- `add_business_days` (n: int | IntoExpr, week_mask: Iterable[bool], holidays: Iterable[dt.date], roll: Roll) -> Expr
- `base_utc_offset` () -> Expr
- `cast_time_unit` (time_unit: TimeUnit) -> Expr
- `century` () -> Expr
- `combine` (time: dt.time | Expr, time_unit: TimeUnit) -> Expr
- `datetime` () -> Expr
- `days_in_month` () -> Expr
- `dst_offset` () -> Expr
- `epoch` (time_unit: EpochTimeUnit) -> Expr
- `is_business_day` (week_mask: Iterable[bool], holidays: Iterable[dt.date]) -> Expr
- `is_leap_year` () -> Expr
- `iso_year` () -> Expr
- `microsecond` () -> Expr
- `millennium` () -> Expr
- `millisecond` () -> Expr
- `month_end` () -> Expr
- `month_start` () -> Expr
- `nanosecond` () -> Expr
- `offset_by` (by: str | Expr) -> Expr
- `ordinal_day` () -> Expr
- `quarter` () -> Expr
- `replace` (year: int | IntoExpr | None, month: int | IntoExpr | None, day: int | IntoExpr | None, hour: int | IntoExpr | None, minute: int | IntoExpr | None, second: int | IntoExpr | None, microsecond: int | IntoExpr | None, ambiguous: Ambiguous | Expr) -> Expr
- `replace_time_zone` (time_zone: str | None, ambiguous: Ambiguous | Expr, non_existent: NonExistent) -> Expr
- `round` (every: str | dt.timedelta | IntoExprColumn) -> Expr
- `strftime` (format: str) -> Expr
- `time` () -> Expr
- `timestamp` (time_unit: TimeUnit) -> Expr
- `to_string` (format: str | None) -> Expr
- `total_days` (fractional: bool) -> Expr
- `total_hours` (fractional: bool) -> Expr
- `total_microseconds` (fractional: bool) -> Expr
- `total_milliseconds` (fractional: bool) -> Expr
- `total_minutes` (fractional: bool) -> Expr
- `total_nanoseconds` (fractional: bool) -> Expr
- `total_seconds` (fractional: bool) -> Expr
- `truncate` (every: str | dt.timedelta | Expr) -> Expr
- `with_time_unit` (time_unit: TimeUnit) -> Expr

### [!] Signature Mismatches (1)

- `second`: Missing params: fractional
  - Polars: (fractional: bool) -> Expr
  - pql: () -> Expr
