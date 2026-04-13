"""
Industry-standard logging module for training.
"""

import logging
import sys
from pathlib import Path


class TrainingLogger:
    """
    Encapsulates training logging with console and file output.
    
    Features:
    - Console output at INFO level
    - File output at DEBUG level
    - Automatic log directory creation
    - Thread-safe logging
    """
    
    _instances = {}  # Cache for logger instances
    
    def __init__(self, name: str, log_dir: str | Path = "logs"):
        """
        Initialize the logger.
        
        Args:
            name: Logger name (typically __name__)
            log_dir: Directory to store log files (default: "logs")
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure and set up the underlying logging.Logger instance.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(self.name)
        
        # Avoid adding handlers multiple times
        if logger.handlers:
            return logger
        
        logger.setLevel(logging.DEBUG)
        
        # Formatter for log messages
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (DEBUG level - logs everything)
        log_file = self.log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self._logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self._logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self._logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log a critical message."""
        self._logger.critical(message)
    
    @classmethod
    def get_logger(cls, name: str, log_dir: str | Path = "logs") -> "TrainingLogger":
        """
        Get or create a logger instance (singleton-like pattern per name).
        
        Args:
            name: Logger name (typically __name__)
            log_dir: Directory to store log files (default: "logs")
        
        Returns:
            TrainingLogger instance
        """
        key = (name, str(log_dir))
        if key not in cls._instances:
            cls._instances[key] = cls(name, log_dir)
        return cls._instances[key]


def get_logger(name: str, log_dir: str | Path = "logs") -> TrainingLogger:
    """
    Convenience function to get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        log_dir: Directory to store log files (default: "logs")
    
    Returns:
        TrainingLogger instance
    """
    return TrainingLogger.get_logger(name, log_dir)
