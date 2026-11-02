# SPDX-License-Identifier: Apache-2.0

"""
Definition of errors for class APIs.
"""

# Standard
import enum


class L1ReconfigureError(RuntimeError):
    """HTTP-mappable runtime L1 reconfiguration error."""

    def __init__(self, status_code: int, message: str) -> None:
        """Create a runtime error with an ``{"error": message}`` payload.

        Args:
            status_code: HTTP status code the API should return.
            message: Human-readable error message.
        """
        super().__init__(message)
        self.status_code = status_code
        self.payload: dict[str, object] = {"error": message}


class L2ClearError(RuntimeError):
    """Raised when one or more L2 adapters failed to clear (backend error).

    Attributes:
        failures: Maps the adapter id of each adapter that failed to clear to
            the exception it raised. Adapters not listed were cleared.
    """

    def __init__(self, failures: dict[int, Exception]) -> None:
        self.failures = dict(failures)
        details = "; ".join(
            f"adapter {adapter_id}: {type(exc).__name__}: {exc}"
            for adapter_id, exc in sorted(self.failures.items())
        )
        super().__init__(f"Failed to clear L2 adapter(s): {details}")


class L1Error(enum.Enum):
    """Errors for L1Manager class APIs."""

    SUCCESS = enum.auto()
    """ Operation succeeded. """

    KEY_NOT_EXIST = enum.auto()
    """ The specified key does not exist. """

    KEY_NOT_READABLE = enum.auto()
    """ The specified key exists but cannot be read """

    KEY_NOT_WRITABLE = enum.auto()
    """ The specified key exists but cannot be written """

    KEY_IN_WRONG_STATE = enum.auto()
    """ The specified key is in the wrong state for the operation. """

    KEY_IS_LOCKED = enum.auto()
    """ The specified key is locked and cannot perform the operation. """

    OUT_OF_MEMORY = enum.auto()
    """ Not enough memory to complete the operation. """


ErrorType = L1Error


def strerror(error: ErrorType) -> str:
    """Convert error code to human-readable string.

    Args:
        error (ErrorType): The error code.

    Returns:
        str: The human-readable string.
    """
    if isinstance(error, L1Error):
        if error == L1Error.SUCCESS:
            return "Operation succeeded."
        elif error == L1Error.KEY_NOT_EXIST:
            return "The specified key does not exist."
        elif error == L1Error.KEY_NOT_READABLE:
            return "The specified key exists but cannot be read."
        elif error == L1Error.KEY_NOT_WRITABLE:
            return "The specified key exists but cannot be written."
        elif error == L1Error.KEY_IN_WRONG_STATE:
            return "The specified key is in the wrong state for the operation."
        elif error == L1Error.KEY_IS_LOCKED:
            return "The specified key is locked and cannot perform the operation."
        elif error == L1Error.OUT_OF_MEMORY:
            return "Not enough memory to complete the operation."

    return "Unknown error."
