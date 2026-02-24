"""Helpers for iterating over arguments that may or may not be iterables."""

from collections.abc import Iterable

import pyochain as pc


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
