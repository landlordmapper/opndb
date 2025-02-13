import os
from pathlib import Path
from typing import List

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.text import Text
import shutil
import pandas as pd

from opndb.constants.base import DATA_ROOT

console = Console()
REQUIRED_DATA_TYPES = [
    ("taxpayer_records", "Property Taxpayer Data"),
    ("corps", "Corporate Data"),
    ("llcs", "LLC Data"),
    ("class_code_descriptions", "Class Codes Data (descriptions)"),
]
root = Path(DATA_ROOT)
RAW_DATA_DIR = root / "raw"

# This creates the cli function that pyproject.toml is looking for
@click.group()
def cli():
    """OPNDB command line interface"""
    pass





def list_directory_files(directory: Path) -> list[Path]:
    """List all files in the given directory"""
    try:
        files = [f for f in directory.iterdir() if f.is_file()]
        return sorted(files)
    except Exception as e:
        console.print(f"[red]Error reading directory: {e}[/red]")
        return []


def display_files_table(files: list[Path]):
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


def get_file_selection(files: list[Path], data_type: str, data_type_display: str, required: bool = True) -> Path | None:
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
                destination = RAW_DATA_DIR / new_filename

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







# This makes "start" a subcommand of cli (ex: running "opndb start" in command line executes start() command)
@cli.command()
def start():
    """Start the OPNDB workflow"""
    print_welcome()
    print_raw_data_message()

    # Get directory path from user
    while True:
        dir_path = Prompt.ask(
            "\nPlease enter the directory path containing your raw data files"
        )

        directory = Path(dir_path)

        if not directory.exists():
            console.print("[red]Directory does not exist. Please try again.[/red]")
            continue

        if not directory.is_dir():
            console.print("[red]Path is not a directory. Please try again.[/red]")
            continue

        files = list_directory_files(directory)

        if not files:
            console.print("[yellow]No files found in directory. Please try again.[/yellow]")
            continue

        break

    # Display files in directory
    display_files_table(files)

    # Process each required data type
    selected_files = {}
    for data_type, display_name in REQUIRED_DATA_TYPES:
        selected_file = get_file_selection(files, data_type, display_name)
        selected_files[data_type] = selected_file

    while True:
        console.print("\n")
        class_codes_together = Prompt.ask(
            "Does the property taxpayer dataset contain building class codes?",
            choices=["y", "n"],
        )
        if class_codes_together == "Y":
            break
        elif class_codes_together == "n":
            selected_file = get_file_selection(files, "bldg_class_codes", "Class Codes Data (properties)")
            selected_files["bldg_class_codes"] = selected_file
            break
        else:
            console.print("[red]Invalid input. Please try again.[/red]")


    # Summary of selections
    console.print("\n[blue]Summary of selected files:[/blue]")
    table = Table()
    table.add_column("Data Type", style="cyan")
    table.add_column("Selected File", style="green")

    for data_type, display_name in REQUIRED_DATA_TYPES:
        file_name = selected_files[data_type].name
        table.add_row(display_name, file_name)
    if "bldg_class_codes" in selected_files.keys():
        table.add_row("Building Class Codes", selected_files["bldg_class_codes"].name)

    console.print(table)



if __name__ == "__main__":
    cli()
