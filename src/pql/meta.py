"""metadata table functions."""

from . import sql as _sql
from ._frame import LazyFrame


def approx_database_count(*args: str) -> LazyFrame:
    """SQL approx_database_count table function.

    **SQL name**: *approx_database_count*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.approx_database_count(*args))


def columns(*args: str) -> LazyFrame:
    """SQL columns table function.

    **SQL name**: *columns*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.columns(*args))


def connection_count(*args: str) -> LazyFrame:
    """SQL connection_count table function.

    **SQL name**: *connection_count*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.connection_count(*args))


def constraints(*args: str) -> LazyFrame:
    """SQL constraints table function.

    **SQL name**: *constraints*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.constraints(*args))


def coordinate_systems(*args: str) -> LazyFrame:
    """SQL coordinate_systems table function.

    **SQL name**: *coordinate_systems*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.coordinate_systems(*args))


def databases(*args: str) -> LazyFrame:
    """SQL databases table function.

    **SQL name**: *databases*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.databases(*args))


def dependencies(*args: str) -> LazyFrame:
    """SQL dependencies table function.

    **SQL name**: *dependencies*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.dependencies(*args))


def extensions(*args: str) -> LazyFrame:
    """SQL extensions table function.

    **SQL name**: *extensions*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.extensions(*args))


def external_file_cache(*args: str) -> LazyFrame:
    """SQL external_file_cache table function.

    **SQL name**: *external_file_cache*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.external_file_cache(*args))


def functions(*args: str) -> LazyFrame:
    """SQL functions table function.

    **SQL name**: *functions*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.functions(*args))


def indexes(*args: str) -> LazyFrame:
    """SQL indexes table function.

    **SQL name**: *indexes*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.indexes(*args))


def keywords(*args: str) -> LazyFrame:
    """SQL keywords table function.

    **SQL name**: *keywords*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.keywords(*args))


def log_contexts(*args: str) -> LazyFrame:
    """SQL log_contexts table function.

    **SQL name**: *log_contexts*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.log_contexts(*args))


def logs(denormalized_table: bool | None = None, *args: str) -> LazyFrame:  # noqa: FBT001
    """SQL logs table function.

    **SQL name**: *logs*

    Args:
        denormalized_table (bool | None): Parameter
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.logs(denormalized_table, *args))


def memory(*args: str) -> LazyFrame:
    """SQL memory table function.

    **SQL name**: *memory*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.memory(*args))


def optimizers(*args: str) -> LazyFrame:
    """SQL optimizers table function.

    **SQL name**: *optimizers*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.optimizers(*args))


def prepared_statements(*args: str) -> LazyFrame:
    """SQL prepared_statements table function.

    **SQL name**: *prepared_statements*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.prepared_statements(*args))


def schemas(*args: str) -> LazyFrame:
    """SQL schemas table function.

    **SQL name**: *schemas*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.schemas(*args))


def secret_types(*args: str) -> LazyFrame:
    """SQL secret_types table function.

    **SQL name**: *secret_types*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.secret_types(*args))


def secrets(redact: bool | None = None, *args: str) -> LazyFrame:  # noqa: FBT001
    """SQL secrets table function.

    **SQL name**: *secrets*

    Args:
        redact (bool | None): Parameter
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.secrets(redact, *args))


def sequences(*args: str) -> LazyFrame:
    """SQL sequences table function.

    **SQL name**: *sequences*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.sequences(*args))


def settings(*args: str) -> LazyFrame:
    """SQL settings table function.

    **SQL name**: *settings*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.settings(*args))


def tables(*args: str) -> LazyFrame:
    """SQL tables table function.

    **SQL name**: *tables*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.tables(*args))


def temporary_files(*args: str) -> LazyFrame:
    """SQL temporary_files table function.

    **SQL name**: *temporary_files*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.temporary_files(*args))


def types(*args: str) -> LazyFrame:
    """SQL types table function.

    **SQL name**: *types*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.types(*args))


def variables(*args: str) -> LazyFrame:
    """SQL variables table function.

    **SQL name**: *variables*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.variables(*args))


def views(*args: str) -> LazyFrame:
    """SQL views table function.

    **SQL name**: *views*

    Args:
        *args (str): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.views(*args))
