"""CLI working-directory isolation for MuJoCo's native warning log."""

from functools import wraps
import os


def preserve_working_directory(function):
    """Restore the caller's directory on success, error or CLI interruption."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = os.getcwd()
        try:
            return function(*args, **kwargs)
        finally:
            os.chdir(previous)

    return wrapped
