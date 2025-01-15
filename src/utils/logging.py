from rich.console import Console
from typing import Optional
from functools import wraps

class Logger:
    def __init__(self, verbose: bool = False):
        self.console = Console()
        self.verbose = verbose

    def log(self, message: str, level: str = "info") -> None:
        """Log a message if verbosity level allows it"""
        if not self.verbose and level == "debug":
            return
            
        if level == "error":
            self.console.print(f"[red]{message}[/red]")
        elif level == "warning":
            self.console.print(f"[yellow]{message}[/yellow]")
        elif level == "success":
            self.console.print(f"[green]{message}[/green]")
        elif level == "debug" and self.verbose:
            self.console.print(f"[blue]DEBUG: {message}[/blue]")
        elif level == "info":
            self.console.print(message)

# Global logger instance
logger = Logger()

def verbose_only(func):
    """Decorator to run a function only in verbose mode"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if logger.verbose:
            return func(*args, **kwargs)
    return wrapper