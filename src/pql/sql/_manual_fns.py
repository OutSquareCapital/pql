from ._exprs import SqlExpr, func


def cume_dist() -> SqlExpr:
    """The cumulative distribution: (number of partition rows preceding or peer with current row) / total partition rows.

    If an ORDER BY clause is specified, the distribution is computed within the frame using the provided ordering instead of the frame ordering.

    Returns:
        SqlExpr
    """
    return func("cume_dist")


def dense_rank() -> SqlExpr:
    """The rank of the current row without gaps; this function counts peer groups.

    Returns:
        SqlExpr
    """
    return func("dense_rank")


def fill(expr: SqlExpr) -> SqlExpr:
    """Replaces NULL values of expr with a linear interpolation based on the closest non-NULL values and the sort values.

    Both values must support arithmetic and there must be only one ordering key. For missing values at the ends, linear extrapolation is used. Failure to interpolate results in the NULL value being retained.

    Args:
        expr (SqlExpr): Expression to fill

    Returns:
        SqlExpr
    """
    return func("fill", expr)


def first_value(expr: SqlExpr) -> SqlExpr:
    """Returns expr evaluated at the row that is the first row (with a non-null value of expr if IGNORE NULLS is set) of the window frame.

    If an ORDER BY clause is specified, the first row number is computed within the frame using the provided ordering instead of the frame ordering.

    Args:
        expr (SqlExpr): Expression to evaluate

    Returns:
        SqlExpr
    """
    return func("first_value", expr)


def lag(
    expr: SqlExpr, offset: SqlExpr | int = 1, default: SqlExpr | None = None
) -> SqlExpr:
    """Returns expr evaluated at the row that is offset rows (among rows with a non-null value of expr if IGNORE NULLS is set) before the current row within the window frame; if there is no such row, instead return default (which must be of the Same type as expr).

    Both offset and default are evaluated with respect to the current row. If omitted, offset defaults to 1 and default to NULL. If an ORDER BY clause is specified, the lagged row number is computed within the frame using the provided ordering instead of the frame ordering.

    Args:
        expr (SqlExpr): Expression to evaluate
        offset (SqlExpr | int): Number of rows to look back (default: 1)
        default (SqlExpr | None): Default value if no such row exists (default: NULL)

    Returns:
        SqlExpr
    """
    return func("lag", expr, offset, default)


def last_value(expr: SqlExpr) -> SqlExpr:
    """Returns expr evaluated at the row that is the last row (among rows with a non-null value of expr if IGNORE NULLS is set) of the window frame.

    If an ORDER BY clause is specified, the last row is determined within the frame using the provided ordering instead of the frame ordering.

    Args:
        expr (SqlExpr): Expression to evaluate

    Returns:
        SqlExpr
    """
    return func("last_value", expr)


def lead(
    expr: SqlExpr, offset: SqlExpr | int = 1, default: SqlExpr | None = None
) -> SqlExpr:
    """Returns expr evaluated at the row that is offset rows after the current row (among rows with a non-null value of expr if IGNORE NULLS is set) within the window frame; if there is no such row, instead return default (which must be of the Same type as expr).

    Both offset and default are evaluated with respect to the current row. If omitted, offset defaults to 1 and default to NULL. If an ORDER BY clause is specified, the leading row number is computed within the frame using the provided ordering instead of the frame ordering.

    Args:
        expr (SqlExpr): Expression to evaluate
        offset (SqlExpr | int): Number of rows to look ahead (default: 1)
        default (SqlExpr | None): Default value if no such row exists (default: NULL)

    Returns:
        SqlExpr
    """
    return func("lead", expr, offset, default)


def nth_value(expr: SqlExpr, nth: SqlExpr | int) -> SqlExpr:
    """Returns expr evaluated at the nth row (among rows with a non-null value of expr if IGNORE NULLS is set) of the window frame (counting from 1); NULL if no such row.

    If an ORDER BY clause is specified, the nth row number is computed within the frame using the provided ordering instead of the frame ordering.

    Args:
        expr (SqlExpr): Expression to evaluate
        nth (SqlExpr | int): The row number to retrieve (1-based)

    Returns:
        SqlExpr
    """
    return func("nth_value", expr, nth)


def ntile(num_buckets: SqlExpr | int) -> SqlExpr:
    """An integer ranging from 1 to num_buckets, dividing the partition as equally as possible.

    If an ORDER BY clause is specified, the ntile is computed within the frame using the provided ordering instead of the frame ordering.

    Args:
        num_buckets (SqlExpr | int): Number of buckets to divide into

    Returns:
        SqlExpr
    """
    return func("ntile", num_buckets)


def percent_rank() -> SqlExpr:
    """The relative rank of the current row: (rank() - 1) / (total partition rows - 1).

    If an ORDER BY clause is specified, the relative rank is computed within the frame using the provided ordering instead of the frame ordering.

    Returns:
        SqlExpr
    """
    return func("percent_rank")


def rank() -> SqlExpr:
    """The rank of the current row with gaps; same as row_number of its first peer.

    If an ORDER BY clause is specified, the rank is computed within the frame using the provided ordering instead of the frame ordering.

    Returns:
        SqlExpr
    """
    return func("rank")


def row_number() -> SqlExpr:
    """The number of the current row within the partition, counting from 1.

    If an ORDER BY clause is specified, the row number is computed within the frame using the provided ordering instead of the frame ordering.

    Returns:
        SqlExpr
    """
    return func("row_number")
