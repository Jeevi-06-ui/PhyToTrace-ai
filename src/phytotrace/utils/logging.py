"""Structured logging utilities for PhytoTrace AI."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "phytotrace", level: Optional[str] = None) -> logging.Logger:
    """Get or configure a standardized logger.
    
    Args:
        name: Logger name.
        level: Optional log level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    
    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    resolved_level = level or "INFO"
    logger.setLevel(getattr(logging, resolved_level.upper(), logging.INFO))
    return logger

