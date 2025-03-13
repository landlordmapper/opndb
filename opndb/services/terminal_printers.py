import os
from pathlib import Path
import pandas as pd
import click
from rich.prompt import IntPrompt
import shutil
from opndb.constants.base import DATA_ROOT
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich import box
from rich.rule import Rule
from rich.style import Style
from rich.text import Text
from rich.align import Align
from rich.padding import Padding
from rich.console import Group
from rich.traceback import install
from datetime import datetime
from time import sleep
import random
import questionary


# Install rich traceback handler
install()


console = Console()

class TerminalBase:
    """Messages & statements printed during stage 1 of the opndb workflow"""

    # todo: make color scheme of terminal printers like the pycharm color scheme

    @classmethod
    # Alternative implementation using the questionary library which has better support for arrow key navigation
    def select_workflow(cls) -> str:
        """
        Prompts the user to select a workflow using questionary for better arrow key navigation.
        Returns the identifier string of the selected workflow.
        """
        console = Console()

        # Display selection prompt inside a panel
        panel = Panel("Select a workflow to execute.", title="[bold cyan]Workflow Selection[/bold cyan]",
                      border_style="cyan")
        console.print(panel)

        # Ask if user wants to start from the beginning
        start_from_beginning = questionary.select(
            "Do you want to start from the beginning or execute a specific workflow?",
            choices=["Start from beginning", "Select specific workflow"]
        ).ask()

        if start_from_beginning == "Start from beginning":
            panel = Panel("Starting from the beginning with Data Cleaning workflow.", border_style="green")
            console.print(panel)
            return "data_clean"

        # Define the available workflows
        workflow_choices = [
            {"name": "Data Cleaning", "value": "data_clean"},
            {"name": "Initial Address Cleaning", "value": "address_initial"},
            {"name": "Address Validation (Geocodio)", "value": "address_geocodio"},
            {"name": "Address Merge", "value": "address_merge"},
            {"name": "Name Analysis", "value": "name_analysis"},
            {"name": "Address Analysis", "value": "address_analysis"},
        ]

        # Let user select a workflow
        workflow_id = questionary.select(
            "Select a workflow to execute:",
            choices=[w["name"] for w in workflow_choices]
        ).ask()

        # Find the selected workflow ID
        for w in workflow_choices:
            if w["name"] == workflow_id:
                panel = Panel(f"Selected workflow: [bold green]{w['name']}[/bold green]", border_style="green")
                console.print(panel)
                return w["value"]

        return "data_clean"
        # console = Console()
        # console.print("\n[bold cyan]Workflow Selection[/bold cyan]")
        #
        # # Ask if user wants to start from the beginning
        # start_from_beginning = questionary.select(
        #     "Do you want to start from the beginning or execute a specific workflow?",
        #     choices=["Start from beginning", "Select specific workflow"]
        # ).ask()
        #
        # if start_from_beginning == "Start from beginning":
        #     console.print("[green]Starting from the beginning with Data Cleaning workflow.[/green]")
        #     return "data_clean"
        #
        # # Define the available workflows
        # workflow_choices = [
        #     {"name": "Data Cleaning", "value": "data_clean"},
        #     {"name": "Inital Address Cleaning", "value": "address_initial"},
        #     {"name": "Address Validation (Geocodio)", "value": "address_geocodio"},
        #     {"name": "Address Merge", "value": "address_merge"},
        #     {"name": "Name Analysis", "value": "name_analysis"},
        #     {"name": "Address Analysis", "value": "address_analysis"},
        # ]
        #
        # # Let user select a workflow
        # workflow_id = questionary.select(
        #     "Select a workflow to execute:",
        #     choices=[f"{w['name']}" for w in workflow_choices]
        # ).ask()
        #
        # # Find the selected workflow ID
        # for w in workflow_choices:
        #     if w["name"] == workflow_id:
        #         console.print(f"[green]Selected workflow: {w['name']}[/green]")
        #         console.print("\n")
        #         return w["value"]
        #
        # return "data_clean"

    @classmethod
    def print_dataset_name(cls, dataset: str):
        console.print("\n")
        console.print(Rule(
            title=f"PROCESSING DATAFRAME: {dataset}",
            style="red"
        ))

    @classmethod
    def print_equals(cls, text: str):
        console.print("\n")
        console.print(f"\n==== {text} ====")
        console.print("\n")

    @classmethod
    def print_workflow_name(cls, wkfl_name: str, wkfl_desc: str):
        group = Group(
            Rule(style="green"),
            Rule(style="bold yellow"),
            Rule(title=wkfl_name, style="red"),
            Rule(style="bold yellow"),
            Rule(style="green")
        )
        # Create the panel with padding and print it
        panel = Padding(
            Panel(
                wkfl_desc,
                style="cyan"
            ),
            (2, 4)
        )
        console.print(group)
        console.print(panel)  # Print the panel
        console.print("\n")

    @classmethod
    def print_with_dots(cls, message: str, console: Console = Console(), style: Style = None) -> None:
        """
        Print a message and fill the remaining space to the end of the line with dots.

        Args:
            message: The message to print
            console: Rich Console instance to use for printing
            style: Optional Rich style to apply to the entire line
        """
        # Get the console width
        width = console.width

        # Calculate how many dots we need
        # Account for the message length and leave space for one character at the end
        dots_needed = width - len(message) - 1

        text = Text()
        text.append(message, style="white")

        # Create the full line with dots
        if dots_needed > 0:
            text.append("." * dots_needed, style="cyan")

        console.print(text)

    @classmethod
    def press_enter_to_continue(cls, specify_text: str = "") -> bool:
        """
        Prompts user to press enter to continue or 'q' to quit.

        Returns:
            bool: True if user wants to continue, False if user wants to quit
        """
        response = input(f"\nPress ENTER to {specify_text}(or \"q\" + ENTER to quit): ").lower()
        return response != "q"

    @classmethod
    def get_raw_data_input(cls) -> Path:
        while True:
            response: str = input("\n Enter the path to the raw data directory: ").lower()
            try:
                path: Path = Path(response).resolve().absolute()
                if path.exists():
                    return path
                else:
                    console.print(f"Path does not exist: {path}")
                    continue
            except Exception as e:
                console.print(f"Invalid path: {e}")
                continue

    @classmethod
    # After generating directories:
    def print_data_root_tree(cls, root):
        tree = Tree(f"📁 [bold blue]{root}[/]")
        for dir_path in sorted(root.glob("**/")):  # Recursively get all directories
            if dir_path != root:  # Skip root itself
                # Calculate relative path for proper tree structure
                relative = dir_path.relative_to(root)
                # Add to tree with proper nesting
                current = tree
                for part in relative.parts:
                    current = current.add(f"📁 [green]{part}[/]")
        console.print(tree)

    @classmethod
    def print_test(cls):
        """
        Demonstrate various ways to print styled output using Click and Rich.
        """
        # Original functionality
        click.echo("Basic Click message")
        click.secho("Colored Click message", fg='green')
        click.secho("Bold Click message", bold=True)
        click.secho("Error message", fg='red', bold=True)
        click.secho("Warning message", fg='yellow')
        click.secho("Success message", fg='green', bg='black')

        # Horizontal rule with title
        console.print(Rule(title="Rich Formatting Showcase", style="bold magenta"))

        console.print("\n=== Rich Basic Formatting ===")
        console.print("Basic Rich message")
        console.print("Styled Rich message", style="bold blue")
        console.print("[red]Colored[/red] [green]Rich[/green] [blue]message[/blue]")

        # Interactive prompts
        console.print("\n=== Rich Interactive Prompts ===")
        name = Prompt.ask("Enter your name", default="User")
        console.print(f"Hello, [bold]{name}[/bold]!")
        if Confirm.ask("Would you like to see more examples?"):
            console.print("[green]Continuing with more examples...[/green]")

        # Tables with different box styles
        console.print("\n=== Rich Tables ===")
        # Simple table
        table = Table(title="Sample Data", box=box.DOUBLE_EDGE)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Status", style="green")
        table.add_row("1", "Task One", "Complete")
        table.add_row("2", "Task Two", "Pending")
        console.print(table)

        # Advanced table with multiple styles
        console.print("\n=== Advanced Table Formatting ===")
        advanced_table = Table(
            title="Project Status",
            caption="Updated status as of " + datetime.now().strftime("%Y-%m-%d"),
            caption_style="dim",
            box=box.ROUNDED,
            header_style="bold magenta",
            border_style="blue",
            row_styles=["none", "dim"]
        )
        advanced_table.add_column("Module", justify="left")
        advanced_table.add_column("Progress", justify="center")
        advanced_table.add_column("Status", justify="right")

        advanced_table.add_row("Core", "[green]100%[/green]", "✅")
        advanced_table.add_row("API", "[yellow]60%[/yellow]", "🏗️")
        advanced_table.add_row("Tests", "[red]25%[/red]", "⏳")
        console.print(advanced_table)

        # Tree structure with expanded features
        console.print("\n=== Enhanced Tree Structure ===")
        tree = Tree("📁 [bold]Project Root[/bold]", guide_style="bold bright_blue")
        src = tree.add("📁 [bold]src[/bold]", guide_style="bright_blue")
        src.add("📄 [green]main.py[/green]").add("[dim]# Main application entry[/dim]")
        utils = src.add("📁 [bold]utils[/bold]")
        utils.add("📄 [green]helpers.py[/green]").add("[dim]# Utility functions[/dim]")
        utils.add("📄 [green]config.py[/green]").add("[dim]# Configuration[/dim]")
        tests = tree.add("📁 [bold]tests[/bold]", guide_style="bright_blue")
        tests.add("📄 [yellow]test_main.py[/yellow]")
        tests.add("📄 [yellow]test_utils.py[/yellow]")
        console.print(tree)

        # Live updating display
        console.print("\n=== Live Updating Display ===")
        with Live(auto_refresh=False) as live:
            for i in range(5):
                data = [random.randint(0, 100) for _ in range(3)]
                table = Table()
                table.add_column("Metric")
                table.add_column("Value")
                table.add_row("CPU", f"{data[0]}%")
                table.add_row("Memory", f"{data[1]}%")
                table.add_row("Disk", f"{data[2]}%")
                live.update(table)
                live.refresh()
                sleep(0.5)

        # Enhanced progress bars
        console.print("\n=== Enhanced Progress Indicators ===")
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
        ) as progress:
            task1 = progress.add_task("[red]Downloading...", total=100)
            task2 = progress.add_task("[green]Processing...", total=100)
            task3 = progress.add_task("[blue]Uploading...", total=100)

            while not progress.finished:
                progress.update(task1, advance=0.9)
                progress.update(task2, advance=0.6)
                progress.update(task3, advance=0.3)
                sleep(0.02)

        # Grouped content with padding and alignment
        console.print("\n=== Grouped Content with Styling ===")
        group = Group(
            Rule(style="red"),
            Align.center(Text("Centered Title", style="bold yellow")),
            Padding(
                Panel(
                    "This content has padding and is within a panel",
                    style="blue"
                ),
                (2, 4)
            ),
            Rule(style="red")
        )
        console.print(group)

        # Code syntax highlighting with different themes
        console.print("\n=== Enhanced Syntax Highlighting ===")
        code = '''
    def example_function(name: str) -> str:
        """Docstring with syntax highlighting."""
        if name.strip():
            return f"Hello, {name}!"
        raise ValueError("Name cannot be empty!")
        '''
        console.print(Panel(
            Syntax(code, "python", theme="monokai", line_numbers=True),
            title="[bold]Python Code Example[/bold]",
            border_style="green"
        ))

        # Status summary with enhanced styling
        console.print("\n=== Enhanced Status Summary ===")
        status_panel = Panel(
            Group(
                Text("System Status", style="bold blue underline"),
                Text(""),
                Text("✅ Database Connection", style="green"),
                Text("⚠️ API Response Time", style="yellow"),
                Text("❌ Background Jobs", style="red"),
                Text(""),
                Text("Last Updated: " + datetime.now().strftime("%H:%M:%S"), style="dim")
            ),
            title="Status Dashboard",
            border_style="blue"
        )
        console.print(status_panel)

        # Demonstration of error handling with rich traceback
        console.print("\n=== Rich Error Handling ===")
        try:
            raise Exception("This is a demonstration of Rich's error handling!")
        except Exception as e:
            console.print_exception()

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

    @classmethod
    def display_table(cls, table_data):
        # todo: format large numbers to have commas
        table = Table(title="Dataframes Loaded")
        table.add_column("Dataset Name", justify="right", style="bold yellow")
        table.add_column("File Size", justify="right", style="green")
        table.add_column("Number of Rows", justify="right", style="cyan")
        for td_obj in table_data:
            row_count = int(td_obj["record_count"])
            formatted_count = f"{row_count:,}"
            table.add_row(
                str(td_obj["dataset_name"]),
                str(td_obj["file_size"]),
                formatted_count,
            )
        console.print("\n" * 2)
        console.print(table)

    @classmethod
    def prompt_continue(cls):
        """Prompts user to press Enter to continue."""
        while True:
            response = input("\nPress Enter to continue...")
            if response == "":
                break

    @classmethod
    def print_geocodio_warning(cls, df: pd.DataFrame):
        unique_addrs: str = f"{len(df):,}"
        cost = 0.0005 * (len(df) - 2500) if len(df) > 2500 else 0
        est_cost: str = f"${cost:,.2f}"
        console.print(f"{unique_addrs} unique addresses found.")
        console.print(f"[red] Cost for executing Geocodio calls: {est_cost}.[/red]")


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

