import os
import logging

# Default configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
RICH_LOGGING_ENABLED = os.environ.get("RICH_LOGGING_ENABLED", "false").lower() == "true"


def init_logger(log_level: int | str = LOG_LEVEL, use_rich: bool = RICH_LOGGING_ENABLED, name: str = "langchainer"):
    """Initialize and configure a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if use_rich:
        try:
            # noinspection PyUnresolvedReferences
            from .rich_debug import DebugRichHandler
            handler = DebugRichHandler()
            formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        except ImportError as e:
            print("Warning: Rich is not installed. Using standard logging.")
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger