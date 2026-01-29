from functools import partial

from ._exprs import func


def _fn(name: str):  # noqa: ANN202
    return partial(func, name)


# SQL Function partials
# Numeric Functions
read_text = _fn("read_text")
extract = _fn("extract")
isfinite_date = _fn("isfinite")
isinf_date = _fn("isinf")
ago = _fn("ago")
extract = _fn("extract")
count_distinct = _fn("count_distinct")
array_first = _fn("array_first")
unnest = _fn("unnest")

read_blob = _fn("read_blob")


# Window Functions
cume_dist = _fn("cume_dist")
dense_rank = _fn("dense_rank")
rank_dense = _fn("rank_dense")
fill = _fn("fill")
first_value = _fn("first_value")
lag = _fn("lag")
last_value = _fn("last_value")
lead = _fn("lead")
nth_value = _fn("nth_value")
ntile = _fn("ntile")
percent_rank = _fn("percent_rank")
rank = _fn("rank")
row_number = _fn("row_number")


# Utility Functions
checkpoint = _fn("checkpoint")
coalesce = _fn("coalesce")
force_checkpoint = _fn("force_checkpoint")
getenv = _fn("getenv")
if_func = _fn("if")
ifnull = _fn("ifnull")
query = _fn("query")
query_table = _fn("query_table")
glob = _fn("glob")
repeat_row = _fn("repeat_row")

# Pattern Matching Functions
like = _fn("like")
ilike = _fn("ilike")
similar_to = _fn("similar_to")
