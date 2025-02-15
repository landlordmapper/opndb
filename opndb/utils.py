from datetime import datetime
from pathlib import Path

from opndb.constants.base import DATA_ROOT
from opndb.workflows import WorkflowStage
from rich.console import Console

console = Console()


class UtilsBase(object):

    @staticmethod
    def generate_filename(filename: str, stage: WorkflowStage, ext: str = "csv") -> str:
        """Returns file name with stage prefix."""
        return f"{stage:02d}_{filename}.{ext}"

    @staticmethod
    def get_timestamp():
        return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    @staticmethod
    def list_directory_files(directory: Path) -> list[Path]:
        """List all files in the given directory"""
        try:
            files = [f for f in directory.iterdir() if f.is_file()]
            return sorted(files)
        except Exception as e:
            console.print(f"[red]Error reading directory: {e}[/red]")
            return []


    @classmethod
    def generate_path(cls, subdir: str, filename: str, stage: WorkflowStage, ext: str = "csv") -> Path:
        """Returns file path for specified file name and subdirectory."""
        filename: str = cls.generate_filename(filename, stage, ext)
        return DATA_ROOT / subdir / filename
