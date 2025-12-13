"""
logging_setup.py

Robust, no-basicConfig logging configuration for RAG/ML applications.
Features:
- Root logger with file + real-stderr output (full format)
- Stderr interception with suppression and custom prefix
- Isolation of internal logs from third-party noise
- Safe for notebooks, reloads, and multiprocessing (if called once per process)
"""

import logging
import sys
import os
import re
from typing import List, Pattern, Optional


class LoggingStderrProxy:
    """Proxy sys.stderr to intercept, filter, and log raw stderr writes."""

    def __init__(
        self,
        logger: logging.Logger,
        suppression_patterns: List[Pattern],
        *,
        debug_suppressed: bool = False
    ):
        """
        Parameters
        ----------
        logger : logging.Logger
            Logger to emit non-suppressed messages (WARNING level) and debug-suppressed ones.
        suppression_patterns : List[re.Pattern]
            Regex patterns (case-insensitive by convention) to suppress messages.
        debug_suppressed : bool
            If True, log suppressed messages at DEBUG level with [SUPPRESSED] prefix.
        """
        self.logger = logger
        self.suppression_patterns = suppression_patterns
        self.debug_suppressed = debug_suppressed

    def write(self, msg: str) -> None:
        if not msg.strip():
            return
        # Check suppression
        for pat in self.suppression_patterns:
            if pat.search(msg):
                if self.debug_suppressed:
                    self.logger.debug(f"[SUPPRESSED] {msg.rstrip()}")
                return
        # Emit as warning
        self.logger.warning(msg.rstrip())

    def flush(self) -> None:
        for handler in self.logger.handlers:
            handler.flush()
        # Also flush root handlers (optional but good practice)
        for handler in logging.getLogger().handlers:
            handler.flush()

    @staticmethod
    def fileno() -> int:
        """Required for compatibility with C extensions that call select() on stderr."""
        return sys.__stderr__.fileno()


def setup_logging(
    *,
    log_file_path: str = "./logs/rag_system.log",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    suppress_patterns: Optional[List[str]] = None,
    debug_suppressed: bool = False,
    noisy_libraries: Optional[List[str]] = None,
) -> logging.Logger:
    """
    Configure logging for the application.

    Returns
    -------
    logging.Logger
        A pre-configured application-level logger (e.g., for __name__ or 'RAG').
        You may also use `logging.getLogger(__name__)` in modules.

    Side Effects
    ------------
    - Configures root logger (removes existing handlers)
    - Creates log directory if needed
    - Replaces `sys.stderr` with filtering proxy
    """
    # ----------------------------
    # 1. Prepare log directory
    # ----------------------------
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # ----------------------------
    # 2. Configure ROOT logger
    # ----------------------------
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)  # handlers do the filtering

    # Remove pre-existing handlers (e.g. from Jupyter or libs)
    for h in root.handlers[:]:
        root.removeHandler(h)

    # Formatter (reusable)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    try:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level.upper())
        root.addHandler(file_handler)
    except Exception as e:
        print(f"[FATAL] Failed to create log file '{log_file_path}': {e}", file=sys.__stderr__)
        raise

    # Console handler → REAL stderr
    console_handler = logging.StreamHandler(sys.__stderr__)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level.upper())
    root.addHandler(console_handler)

    # Ensure root level is at least as permissive as the most verbose handler
    root.setLevel(min(file_handler.level, console_handler.level))

    # ----------------------------
    # 3. stderr_proxy logger + proxy
    # ----------------------------
    stderr_logger = logging.getLogger("stderr_proxy")
    stderr_logger.setLevel(logging.WARNING)
    stderr_logger.propagate = False  # 🔑 isolation

    if not stderr_logger.handlers:
        proxy_handler = logging.StreamHandler(sys.__stderr__)
        proxy_handler.setFormatter(logging.Formatter("[STDERR] %(message)s"))
        proxy_handler.setLevel(logging.WARNING)
        stderr_logger.addHandler(proxy_handler)

    # Compile suppression patterns
    default_patterns = [
        r".*incorrect regex pattern.*Mistral.*",
        # Add more defaults here if desired
    ]
    patterns = suppress_patterns or default_patterns
    compiled_patterns: List[Pattern] = [re.compile(p, re.IGNORECASE) for p in patterns]

    # Activate proxy
    proxy = LoggingStderrProxy(
        logger=stderr_logger,
        suppression_patterns=compiled_patterns,
        debug_suppressed=debug_suppressed,
    )
    sys.stderr = proxy

    # ----------------------------
    # 4. Silence noisy libraries
    # ----------------------------
    default_noisy = ["sentence_transformers", "transformers", "urllib3", "faiss", "huggingface_hub"]
    libs_to_quiet = noisy_libraries or default_noisy
    for lib in libs_to_quiet:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # ----------------------------
    # 5. Return convenience app logger
    # ----------------------------
    app_logger = logging.getLogger("RAG")
    return app_logger