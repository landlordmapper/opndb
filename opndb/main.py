from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from opndb.constants.base import DATA_ROOT
from opndb.services.terminal_printers import TerminalBase as t, TerminalInteract as ti
from opndb.workflows.base import WorkflowBase as w

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



# This makes "start" a subcommand of cli (ex: running "opndb start" in command line executes start() command)
@cli.command()
def start():
    while True:
        configs = w.load_configs()
        wkfl = w.create_workflow(configs)
        wkfl.execute()


# This makes "start" a subcommand of cli (ex: running "opndb start" in command line executes start() command)
@cli.command()
def start_old():
    """Start the OPNDB workflow"""
    t.print_welcome()
    # ask to generate directory structure in the same path as the raw data if running locally
    # if hitting the s3, the login data will be stored in configs
    t.print_raw_data_message()

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

        files = ti.list_directory_files(directory)

        if not files:
            console.print("[yellow]No files found in directory. Please try again.[/yellow]")
            continue

        break

    # Display files in directory
    t.display_files_table(files)

    # Process each required data type
    selected_files = {}
    # for data_type, display_name in REQUIRED_DATA_TYPES:
        # selected_file = ti.get_file_selection(files, configs.data_dir, data_type, display_name)
        # selected_files[data_type] = selected_file

    while True:
        console.print("\n")
        class_codes_together = Prompt.ask(
            "Does the property taxpayer dataset contain building class codes?",
            choices=["y", "n"],
        )
        if class_codes_together == "Y":
            break
        elif class_codes_together == "n":
            # selected_file = get_file_selection(files, "bldg_class_codes", "Class Codes Data (properties)")
            # selected_files["bldg_class_codes"] = selected_file
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
