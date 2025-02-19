import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text
import shutil

from opndb.constants.base import DATA_ROOT


console = Console()

class TerminalBase:
    """Messages & statements printed during stage 1 of the opndb workflow"""
    @classmethod
    def print_welcome(cls):
        """Display a stylized welcome message"""
        console.print("\n" * 2)
        welcome_text = Text()
        welcome_text.append("Welcome to ", style="blue")
        welcome_text.append("OPNDB", style="bold blue")
        welcome_text.append("\nOpen Property Network Database", style="blue")

        message = (
            "\nThis tool provides a standardized workflow for obtaining landlord-linked property datasets."
        )

        panel = Panel(
            Text.assemble(welcome_text, message),
            title="🏘️ 🏘️ 🏘️  Open Property Network Database (opndb)  🏘️ 🏘️ 🏘️",
            border_style="blue"
        )
        console.print(panel)
        console.print()

    @classmethod
    def print_raw_data_message(cls):
        text = Text()
        text.append("\n")
        text.append("The first step is to copy ra thew data files into the project's \"raw\" data directory. Before continuing, ", style="green")
        text.append("be sure all relevant raw data files are stored in one single directory. ", style="bold green")
        text.append("The data should be in CSV format. (CHANGE THIS LATER TO ACCEPT DIFFERENT FORMATS)", style="green")
        text.append("\n")
        text.append("\nThe following raw data sources are REQUIRED in order to generate the landlord-linked property dataset (see readme for a detailed description of the required data columns:", style="green")
        text.append("\n")
        text.append("\n > Taxpayer records")
        text.append("\n > Building class codes")
        text.append("\n > Corporate Records")
        text.append("\n > LLC Records")
        text.append("\n")
        text.append("\nWhen you're ready, paste the absolute path of the raw data directory to get started.", style="green")
        text.append("\n")
        panel = Panel(
            Text.assemble(text),
            title="GATHER RAW DATA SOURCES",
            border_style="green"
        )
        console.print(panel)
        console.print()

    @classmethod
    def display_files_table(cls, files: list[Path]):
        """Display files in a formatted table"""
        table = Table(title="Available Files")

        table.add_column("#", justify="right", style="cyan")
        table.add_column("Filename", style="green")
        table.add_column("Size", justify="right", style="blue")

        for idx, file in enumerate(files, 1):
            # Get file size in KB or MB
            size_bytes = os.path.getsize(file)
            if size_bytes > 1024 * 1024 * 1024:  # Greater than 1 GB
                size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
            elif size_bytes > 1024 * 1024:  # Greater than 1 MB
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            else:  # Less than 1 MB
                size_str = f"{size_bytes / 1024:.1f} KB"

            table.add_row(str(idx), file.name, size_str)

        console.print("\n" * 2)
        console.print(table)

class TerminalInteract(TerminalBase):

    @classmethod
    def get_file_selection(cls, files: list[Path], data_dir, data_type: str, data_type_display: str, required: bool = True) -> Path | None:
        """Prompt user to select a file for a specific data type"""
        while True:
            console.print(f"\n[blue]Please select the file containing your {data_type_display}[/blue]")
            if not required:
                console.print("[yellow]Enter 0 to skip if you don't have this data[/yellow]")

            try:
                file_idx = IntPrompt.ask(
                    "Enter the file number"
                )

                if 1 <= file_idx <= len(files):
                    selected_file = files[file_idx - 1]

                    # Copy file to raw data directory with standardized name
                    new_filename = f"{data_type}.csv"
                    destination = DATA_ROOT / data_dir / new_filename

                    shutil.copy2(selected_file, destination)
                    console.print(f"[green]✓ Copied {selected_file.name} to {destination}[/green]")

                    return selected_file
                else:
                    if file_idx == 0 and not required:
                        return None
                    else:
                        console.print("[red]Invalid file number. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number.[/red]")

