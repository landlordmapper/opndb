import os
from pathlib import Path

from opndb.constants.base import DATA_ROOT
from opndb.workflows import WorkflowStage


class UtilsBase(object):

    @classmethod
    def generate_filename(cls, filename: str, stage: WorkflowStage, ext: str = "csv") -> str:
        """Returns file name with stage prefix."""
        return f"{stage:02d}_{filename}.{ext}"

    @classmethod
    def generate_path(cls, subdir: str, filename: str, stage: WorkflowStage, ext: str = "csv") -> Path:
        """Returns file path for specified file name and subdirectory."""
        filename: str = cls.generate_filename(filename, stage, ext)
        return DATA_ROOT / subdir / filename
