from functools import partial

from ._exprs import func


def _fn(name: str):  # noqa: ANN202
    return partial(func, name)


# SQL Function partials
# Numeric Functions
extract = _fn("extract")
ago = _fn("ago")
count_distinct = _fn("count_distinct")


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
coalesce = _fn("coalesce")
getenv = _fn("getenv")
if_func = _fn("if")
ifnull = _fn("ifnull")

# Pattern Matching Functions
like = _fn("like")
ilike = _fn("ilike")
similar_to = _fn("similar_to")
