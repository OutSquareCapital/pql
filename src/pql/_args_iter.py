"""Helpers for iterating over arguments that may or may not be iterables."""

from collections.abc import Iterable, Sequence
from typing import Any

import pyochain as pc

from .sql.typing import NonNestedLiteral


def try_iter[T](val: Iterable[T] | T) -> pc.Iter[T]:
    """Try to iterate over a value that may or may not be iterable.

    Args:
        val (Iterable[T] | T): The value to try to iterate over.

    Returns:
        pc.Iter[T]: An iterator over the value if it is iterable, otherwise an iterator over a single element.
    """
    match val:
        case str() | bytes() | bytearray():
            return pc.Iter[T].once(val)
        case Iterable():
            return pc.Iter(val)  # pyright: ignore[reportUnknownArgumentType]
        case _:
            return pc.Iter[T].once(val)


def check_by_arg[T: NonNestedLiteral](
    compared: pc.Seq[Any], name: str, arg: T | Sequence[T]
) -> pc.Result[pc.Iter[T], ValueError]:
    length = compared.length()
    match arg:
        case Sequence():
            len_arg = len(arg)
            match len_arg == length:
                case True:
                    return pc.Ok(try_iter(arg))
                case False:
                    msg = f"the length of `{name}` ({len_arg}) does not match the length of `by` ({length})"
                    return pc.Err(ValueError(msg))

        case _:
            return pc.Ok(pc.Iter.once(arg).cycle().take(length))


def try_chain[T](vals: T | Iterable[T], other_vals: Iterable[T]) -> pc.Iter[T]:
    """Try to chain a value that may or may not be iterable with another iterable.

    Args:
        vals (T | Iterable[T]): The value to try to chain.
        other_vals (Iterable[T]): The other iterable to chain with.

    Returns:
        pc.Iter[T]: An iterator over the chained values.
    """
    return try_iter(vals).chain(other_vals)


def try_flatten[T](vals: T | Iterable[T]) -> pc.Iter[T]:
    """Try to flatten a value that may be nested iterables.

    A value that is not an iterable will be treated as a single element iterable.

    Args:
        vals (T | Iterable[T]): The value to try to flatten.

    Returns:
        pc.Iter[T]: An iterator over the flattened values.
    """
    return try_iter(vals).flat_map(try_iter)
