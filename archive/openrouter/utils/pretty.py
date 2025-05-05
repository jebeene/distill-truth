from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.pretty import pprint

console = Console()

def log_info(message: str):
    """Print a clean, bold info message."""
    console.print(f"[bold cyan]➤[/bold cyan] {message}")

def log_error(message: str):
    """Print a clean, bold error message."""
    console.print(f"[bold red]{message}[/bold red]")

def print_metrics_summary(accuracy, f1_macro):
    table = Table(title="Model Evaluation Summary")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold magenta")
    table.add_row("Accuracy", f"{accuracy:.4f}")
    table.add_row("F1 Macro", f"{f1_macro:.4f}")
    console.print(table)

def print_confusion_matrix(cm):
    console.print(Panel("Confusion Matrix", style="bold green"))
    pprint(cm)