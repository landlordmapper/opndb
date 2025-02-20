from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from click.core import Context
from opndb.constants.base import DATA_ROOT
from opndb.constants.files import Dirs
from opndb.services.config import ConfigManager
from opndb.services.terminal_printers import TerminalBase as t, TerminalInteract as ti
from opndb.utils import UtilsBase as utils
from opndb.workflows.base import WorkflowBase as w, WorkflowBase

console = Console()
REQUIRED_DATA_TYPES = [
    ("taxpayer_records", "Property Taxpayer Data"),
    ("corps", "Corporate Data"),
    ("llcs", "LLC Data"),
    ("class_code_descriptions", "Class Codes Data (descriptions)"),
]
root = Path(DATA_ROOT)
RAW_DATA_DIR = root / "raw"

@click.group()  # This creates the main cli group of functions that pyproject.toml is looking for
@click.pass_context  # passes context object into the function - passes data into different commands in this group
def cli(ctx: Context):
    """
    OPNDB command line interface

    Main CLI group. The ctx parameter is of type click.Context.
    ctx.obj can store any object you want to share between commands.
    """
    ctx.obj = ConfigManager()  # ctx.obj stores python object of any type

@cli.command()
@click.argument("data_root", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.pass_obj
def init(config_manager: ConfigManager, data_root: Path):
    """Initialize OPNDB with a data directory"""
    # todo: add check for whether or not the configs file and/or data dirs already exist
    config_manager.generate(data_root)
    utils.generate_data_dirs(data_root)
    console.print("\n")
    console.print(f"Initialized OPNDB with data directory: \n{data_root}")
    console.print("\n")
    console.print(f"Config file location: \n{config_manager.path}")
    console.print("\n")
    console.print("Project data directories created.\n")
    t.print_data_root_tree(data_root)
    console.print("\n")
    raw_data_dir: Path = t.get_raw_data_input()
    console.print("\n")
    utils.copy_raw_data(raw_data_dir, data_root)

@cli.command()
@click.pass_obj
def start(config_manager: ConfigManager):
    t.print_welcome()
    if not t.press_enter_to_continue("continue "):
        console.print("Exiting program...", style="yellow")
        return
    t.print_with_dots("Searching for project settings...")
    if config_manager.exists:
        t.print_with_dots("Configs file located. Loading...")
        config_manager.load()
        t.print_with_dots("Configs successfully loaded.")
    else:
        t.print_with_dots("No configs file was found. Run `opndb init /path/to/your/root/data/dir`")
        return
    t.print_with_dots("Launching workflows...")
    console.print("\n")
    while True:
        wkfl = WorkflowBase.create_workflow(config_manager.configs)
        wkfl.load()
        if not t.press_enter_to_continue("execute string cleaning workflow "):
            t.print_with_dots("Exiting program...", style="yellow")
            return
        wkfl.process()
        wkfl.save()
        # print out summary stats of data found in raw datasets
        # press enter to begin cleaning

if __name__ == "__main__":
    cli()
