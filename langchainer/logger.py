import os
import logging

# Maps log level names to their corresponding integer values, so that it can be set via env var.
LOG_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR"   : logging.ERROR,
    "WARNING" : logging.WARNING,
    "INFO"    : logging.INFO,
    "DEBUG"   : logging.DEBUG,
    "NOTSET"  : logging.NOTSET,
}


def init_logger(log_level: int | None = None, use_rich: bool | None = None, name: str = "langchainer"):
    """
    Initialize and configure a logger.

    This function sets up a logger with the specified log level and formatting options.
    It can use either the standard logging or the Rich library for enhanced logging output.

    Parameters:
    -----------
    log_level : int | None, optional
        The logging level to set for the logger.
        **Priority:** This parameter has the highest priority. If provided, it overrides
        the `LOG_LEVEL` environment variable and the default level (INFO).
    use_rich : bool | None, optional
        Whether to use the Rich library for logging.
        **Priority:** This parameter has the highest priority. If provided, it overrides
        the `RICH_LOGGING_ENABLED` environment variable and the default behavior (False).
    name : str, optional
        The name of the logger. Defaults to "langchainer".

    Returns:
    --------
    logging.Logger
        A configured logger instance.

    Notes:
    ------
    - **Log Level Precedence:**
        1. `log_level` parameter (if provided)
        2. `LOG_LEVEL` environment variable (if set and valid)
        3. Default: `INFO`

    - **Rich Logging Precedence:**
        1. `use_rich` parameter (if provided)
        2. `RICH_LOGGING_ENABLED` environment variable (if set to "true" - case-insensitive)
        3. Default: `False` (standard logging)

    - If Rich is specified but not installed, the function falls back to standard logging with a warning message.
    """

    logger = logging.getLogger(name)

    if log_level is None:
        log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
        log_level = LOG_LEVEL_MAP.get(log_level_str, LOG_LEVEL_MAP["INFO"])

    logger.setLevel(log_level)

    # Remove any existing handlers to prevent duplicate logging.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if use_rich is None:
        use_rich = os.environ.get("RICH_LOGGING_ENABLED", "false").lower() == "true"

    if use_rich:
        try:
            # noinspection PyUnresolvedReferences
            from .rich_debug import DebugRichHandler
            handler = DebugRichHandler()
            formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        except ImportError:
            if use_rich:
                if os.environ.get("RICH_LOGGING_ENABLED", "").lower() == "true":
                    reason = "the environment variable RICH_LOGGING_ENABLED is set to 'true'"
                else:
                    reason = "`use_rich` parameter was explicitly set to True"
            else:
                reason = "unknown reason"

            print(f"Warning: Rich is not installed, but rich logging was requested because {reason}. Using standard logging instead.")

            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
