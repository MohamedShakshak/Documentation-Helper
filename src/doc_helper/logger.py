import logging
import sys


class _Colors:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


_logger = logging.getLogger("doc_helper")


def _setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    _logger.addHandler(handler)
    _logger.setLevel(level)


_setup_logging()


def log_info(message: str) -> None:
    _logger.info(message)


def log_success(message: str) -> None:
    _logger.info(f"✅ {message}")


def log_error(message: str) -> None:
    _logger.error(f"❌ {message}")


def log_warning(message: str) -> None:
    _logger.warning(f"⚠️  {message}")


def log_header(message: str) -> None:
    _logger.info(f"{'=' * 60}")
    _logger.info(f"🚀 {message}")
    _logger.info(f"{'=' * 60}")
