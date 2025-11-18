"""
Structured logging utility for the Multi-Agent System.

Provides consistent, structured logging with correlation IDs and log levels.
"""
import logging
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter."""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        # Get color for log level
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        
        # Build message
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = f"{color}{record.levelname:8s}{reset}"
        logger_name = f"[{record.name}]" if record.name != "root" else ""
        message = record.getMessage()
        
        # Add correlation ID if present
        correlation = ""
        if hasattr(record, "correlation_id"):
            correlation = f" [ID: {record.correlation_id[:8]}]"
        
        formatted = f"{timestamp} {level} {logger_name}{correlation} {message}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        json_format: Whether to use JSON format (default: human-readable)
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        if json_format:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(ConsoleFormatter())
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        # Always use JSON format for file logs
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, correlation_id: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with optional correlation ID.
    
    Args:
        name: Logger name (typically module name)
        correlation_id: Optional correlation ID for request tracking
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Add correlation ID adapter
    if correlation_id:
        class CorrelationAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                kwargs.setdefault("extra", {})["correlation_id"] = self.extra["correlation_id"]
                return msg, kwargs
        
        return CorrelationAdapter(logger, {"correlation_id": correlation_id})
    
    return logger


# Convenience functions for common log operations
def log_session_start(logger: logging.Logger, session_id: str, query: str) -> None:
    """Log session start."""
    logger.info(
        "Session started",
        extra={"extra_fields": {"session_id": session_id, "query": query}}
    )


def log_memory_search(logger: logging.Logger, query: str, results_count: int) -> None:
    """Log memory search results."""
    logger.info(
        f"Memory search completed: {results_count} results",
        extra={"extra_fields": {"query": query, "results_count": results_count}}
    )


def log_perception_result(logger: logging.Logger, snapshot_type: str, goal_achieved: bool) -> None:
    """Log perception result."""
    logger.info(
        f"Perception completed: {snapshot_type}",
        extra={"extra_fields": {"snapshot_type": snapshot_type, "goal_achieved": goal_achieved}}
    )


def log_decision_plan(logger: logging.Logger, plan_version: int, step_count: int) -> None:
    """Log decision plan generation."""
    logger.info(
        f"Decision plan generated: version {plan_version}",
        extra={"extra_fields": {"plan_version": plan_version, "step_count": step_count}}
    )


def log_step_execution(logger: logging.Logger, step_index: int, step_type: str, status: str) -> None:
    """Log step execution."""
    logger.info(
        f"Step {step_index} executed: {step_type}",
        extra={"extra_fields": {"step_index": step_index, "step_type": step_type, "status": status}}
    )


def log_error(logger: logging.Logger, error_type: str, error_message: str, **kwargs) -> None:
    """Log error with context."""
    logger.error(
        f"{error_type}: {error_message}",
        extra={"extra_fields": {"error_type": error_type, **kwargs}}
    )

