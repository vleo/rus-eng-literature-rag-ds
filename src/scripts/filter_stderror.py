import sys
import logging
from logging import getLogger
import re

# --- 1. Configure logger (before any noisy imports) ---
logger = getLogger("stderr_proxy")
logger.setLevel(logging.WARNING)
# Add handler only if not already added (avoid duplication in notebooks)
if not logger.handlers:
    handler = logging.StreamHandler(sys.__stderr__)  # write to REAL stderr
    handler.setFormatter(logging.Formatter("[STDERR] %(message)s"))
    logger.addHandler(handler)

# --- 2. Define suppression rules ---
SUPPRESSION_PATTERNS = [
    re.compile(r".*incorrect regex pattern.*Mistral.*", re.IGNORECASE),
    # Add more as needed:
    # re.compile(r"bla blah tokenizer.*", re.IGNORECASE),
]

class LoggingStderrProxy:
    @staticmethod
    def write(msg: str):
        if not msg.strip():
            return
        # Check if msg should be suppressed
        for pat in SUPPRESSION_PATTERNS:
            if pat.search(msg):
                logger.debug(f"[SUPPRESSED] {msg.rstrip()}")
                return
        # Otherwise — log as warning (or info/error as appropriate)
        logger.warning(msg.rstrip())
    
    @staticmethod
    def flush():
        for h in logger.handlers:
            h.flush()
            
# --- 3. Activate proxy — MUST be before importing tokenizers/transformers ---
sys.stderr = LoggingStderrProxy()
