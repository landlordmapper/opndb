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

class TerminalPrinterBase:
    pass

class TerminalPrinter1(TerminalPrinterBase):
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