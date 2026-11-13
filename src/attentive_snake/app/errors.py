"""Turning library validation errors into HTTP errors at the API boundary."""

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException


@contextmanager
def invalid_input_as_422() -> Iterator[None]:
    """``ValueError``/``TypeError`` raised by config validation become 422 with the message as detail."""
    try:
        yield
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def not_found(message: str) -> HTTPException:
    return HTTPException(status_code=404, detail=message)


def conflict(message: str) -> HTTPException:
    return HTTPException(status_code=409, detail=message)
