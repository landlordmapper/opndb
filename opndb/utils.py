from datetime import datetime
from pathlib import Path

from opndb.constants.base import DATA_ROOT
from rich.console import Console

from opndb.constants.files import Dirs, Raw, Processed, Geocodio, Analysis
from opndb.types.base import FileExt, WorkflowConfigs
from opndb.workflows.base import WorkflowStage

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
    def generate_path(cls, subdir: str, filename: str, stage: WorkflowStage, ext: FileExt = "csv") -> Path:
        """Returns file path for specified file name and subdirectory."""
        filename: str = cls.generate_filename(filename, stage, ext)
        return DATA_ROOT / subdir / filename

    @classmethod
    def generate_configs(cls):
        """Generates new configs.json file."""
        pass

    @classmethod
    def load_configs(cls):
        """Loads configs.json file associated with the project. Creates it if it doesn't exist"""
        pass

    @classmethod
    def is_encoded_empty(cls, x):
        if isinstance(x, str):
            # Check if string contains mostly non-printable characters
            return any(ord(c) < 32 for c in x)
        return False


class PathGenerators(UtilsBase):
    """
    Helper functions to return the file path for each individual dataset. Methods are named by
    '{dir_name}_{dataset_name}()'
    """
    # -----------
    # ----RAW----
    # -----------
    @classmethod
    def raw_taxpayer_records(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/raw/taxpayer_records[ext]"""
        return cls.generate_path(
            Dirs.RAW,
            Raw.TAXPAYER_RECORDS,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def raw_corps(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/raw/corps[ext]"""
        return cls.generate_path(
            Dirs.RAW,
            Raw.CORPS,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def raw_llcs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/raw/llcs[ext]"""
        return cls.generate_path(
            Dirs.RAW,
            Raw.LLCS,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def raw_class_codes(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/raw/class_codes[ext]"""
        return cls.generate_path(
            Dirs.RAW,
            Raw.CLASS_CODES,
            configs["prev_stage"],
            configs["load_ext"]
        )

    # -----------------
    # ----PROCESSED----
    # -----------------
    @classmethod
    def processed_taxpayer_records(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/taxpayer_records[ext]"""
        return cls.generate_path(
            Dirs.PROCESSED,
            Processed.TAXPAYER_RECORDS,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def processed_properties(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/taxpayer_records[ext]"""
        return cls.generate_path(
            Dirs.PROCESSED,
            Processed.PROPERTIES,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def processed_corps(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/corps[ext]"""
        return cls.generate_path(
            Dirs.PROCESSED,
            Processed.CORPS,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def processed_llcs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/llcs[ext]"""
        return cls.generate_path(
            Dirs.PROCESSED,
            Processed.LLCS,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def processed_class_codes(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/class_codes[ext]"""
        return cls.generate_path(
            Dirs.PROCESSED,
            Processed.CLASS_CODES,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def processed_validated_addrs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/validated_addrs[ext]"""
        return cls.generate_path(
            Dirs.PROCESSED,
            Processed.VALIDATED_ADDRS,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def processed_unvalidated_addrs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/unvalidated_addrs[ext]"""
        return cls.generate_path(
            Dirs.PROCESSED,
            Processed.UNVALIDATED_ADDRS,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def processed_props_subsetted(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/props_subsetted[ext]"""
        return cls.generate_path(
            Dirs.PROCESSED,
            Processed.PROPS_SUBSETTED,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def processed_props_prepped(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/props_prepped[ext]"""
        return cls.generate_path(
            Dirs.PROCESSED,
            Processed.PROPS_PREPPED,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def processed_props_string_matched(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/props_string_matched[ext]"""
        return cls.generate_path(
            Dirs.PROCESSED,
            Processed.PROPS_STRING_MATCHED,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def processed_props_networked(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/props_networked[ext]"""
        return cls.generate_path(
            Dirs.PROCESSED,
            Processed.PROPS_NETWORKED,
            configs["prev_stage"],
            configs["load_ext"]
        )

    # -----------------
    # ----GEOCODIO----
    # -----------------
    @classmethod
    def geocodio_gcd_validated(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/geocodio/gcd_validated[ext]"""
        return cls.generate_path(
            Dirs.GEOCODIO,
            Geocodio.GCD_VALIDATED,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def geocodio_gcd_unvalidated(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/geocodio/gcd_unvalidated[ext]"""
        return cls.generate_path(
            Dirs.GEOCODIO,
            Geocodio.GCD_UNVALIDATED,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def geocodio_gcd_failed(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/geocodio/gcd_failed[ext]"""
        return cls.generate_path(
            Dirs.GEOCODIO,
            Geocodio.GCD_FAILED,
            configs["prev_stage"],
            configs["load_ext"]
        )

    # -----------------
    # ----ANALYSIS----
    # -----------------
    @classmethod
    def analysis_frequent_tax_names(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/analysis/frequent_tax_names[ext]"""
        return cls.generate_path(
            Dirs.ANALYSIS,
            Analysis.FREQUENT_TAX_NAMES,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def analysis_frequent_tax_addrs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/analysis/frequent_tax_addrs[ext]"""
        return cls.generate_path(
            Dirs.ANALYSIS,
            Analysis.FREQUENT_TAX_ADDRS,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def analysis_fixing_tax_names(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/analysis/fixing_tax_names[ext]"""
        return cls.generate_path(
            Dirs.ANALYSIS,
            Analysis.FIXING_TAX_NAMES,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def analysis_fixing_tax_addrs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/analysis/fixing_tax_addrs[ext]"""
        return cls.generate_path(
            Dirs.ANALYSIS,
            Analysis.FIXING_TAX_ADDRS,
            configs["prev_stage"],
            configs["load_ext"]
        )
    @classmethod
    def analysis_address_analysis(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/analysis/address_analysis[ext]"""
        return cls.generate_path(
            Dirs.ANALYSIS,
            Analysis.ADDRESS_ANALYSIS,
            configs["prev_stage"],
            configs["load_ext"]
        )