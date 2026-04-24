import logging
import os
from datetime import datetime

class TimestampedFileHandler(logging.FileHandler):
    """
    Custom handler to ensure every log entry across the project 
    is clearly timestamped with high precision.
    """
    def emit(self, record):
        # Format: [YYYY-MM-DD HH:MM:SS.mmm] Message
        record.msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {record.msg}"
        super().emit(record)

def setup_aether_logger(name: str, log_file: str = "service_activity.log") -> logging.Logger:
    """
    Usage: logger = setup_aether_logger("ModuleName")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if setup is called multiple times
    if not logger.handlers:
        # File Handler (Persistent)
        file_handler = TimestampedFileHandler(log_file)
        file_formatter = logging.Formatter('%(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console Handler (Real-time monitoring)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)

    return logger

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'