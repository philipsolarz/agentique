"""
Enhanced logging configuration for Agentique using Rich.

Provides beautiful console output with proper formatting and color coding
for better debugging and visibility.
"""

import logging
import sys
from typing import Optional, Union, Literal
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install as install_rich_traceback
from rich.pretty import pprint
from rich.syntax import Syntax
from rich.panel import Panel
from rich.json import JSON

# Create console instance
console = Console()

# Install rich traceback handling
install_rich_traceback(console=console, show_locals=True)

def configure_logging(
    level: Union[int, str] = "INFO",
    show_path: bool = False,
    show_time: bool = True
) -> logging.Logger:
    """
    Configure logging with Rich formatting.
    
    Args:
        level: Logging level
        show_path: Whether to show file path in log messages
        show_time: Whether to show timestamps in log messages
        
    Returns:
        Logger instance
    """
    # Set log format based on configuration
    if show_path:
        log_format = "%(name)s - %(message)s"
    else:
        log_format = "%(message)s"
    
    # Configure handlers
    handlers = [
        RichHandler(
            rich_tracebacks=True,
            console=console,
            show_time=show_time,
            show_path=show_path,
            markup=True
        )
    ]
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt="[%X]",
        handlers=handlers
    )
    
    # Get and return the main library logger
    logger = logging.getLogger("agentique")
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        name: Component name
        
    Returns:
        Logger for the component
    """
    return logging.getLogger(f"agentique.{name}")

# Utility functions for rich logging
def print_json(data, title: Optional[str] = None) -> None:
    """Print formatted JSON data."""
    json_str = JSON.from_data(data)
    if title:
        console.print(Panel(json_str, title=title, expand=False))
    else:
        console.print(json_str)

def print_code(code: str, language: str = "python", title: Optional[str] = None) -> None:
    """Print formatted code block."""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    if title:
        console.print(Panel(syntax, title=title, expand=False))
    else:
        console.print(syntax)

def print_object(obj: any, title: Optional[str] = None) -> None:
    """Pretty print any object."""
    if title:
        console.print(f"[bold]{title}[/bold]")
    pprint(obj, console=console, expand_all=False)

def print_table(data, title: Optional[str] = None) -> None:
    """Print data as a table."""
    from rich.table import Table
    
    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        console.print("[yellow]Cannot create table: Invalid data format[/yellow]")
        pprint(data)
        return
    
    table = Table(title=title)
    
    # Add columns
    columns = list(data[0].keys())
    for column in columns:
        table.add_column(str(column))
    
    # Add rows
    for row in data:
        table.add_row(*[str(row[column]) for column in columns])
    
    console.print(table)