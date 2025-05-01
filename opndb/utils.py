import shutil
from datetime import datetime
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, FileSizeColumn, TotalFileSizeColumn, TimeRemainingColumn

from opndb.constants.base import DATA_ROOT
from rich.console import Console

from opndb.constants.files import Dirs, Raw, Processed, Geocodio, Analysis, Output, PreProcess
from opndb.types.base import FileExt, WorkflowConfigs, WorkflowStage

console = Console()


class UtilsBase(object):

    @staticmethod
    def generate_filename(filename: str, ext: str = "csv") -> str:
        """Returns file name with stage prefix."""
        return f"{filename}.{ext}"

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
    def generate_path(cls, data_root: str | Path, subdir: str, filename: str, ext: FileExt = "csv") -> Path:
        """Returns file path for specified file name and subdirectory."""
        filename: str = cls.generate_filename(filename, ext)
        return Path(data_root) / subdir / filename

    @classmethod
    def generate_geocodio_partial_path(cls, data_root: str | Path, filename: str, ext: FileExt = "csv") -> Path:
        return Path(data_root) / "geocodio" / "partials" / f"{filename}.{ext}"


    @classmethod
    def is_encoded_empty(cls, x):
        if isinstance(x, str):
            # Check if string contains mostly non-printable characters
            return any(ord(c) < 32 for c in x)
        return False

    @classmethod
    def generate_data_dirs(cls, root: Path):
        """
        Check if required directories exist and create them if they don't.
        Args:
            root (Path): Root directory path where all subdirectories should be created
        Raises:
            ValueError: If root path is not provided or is invalid
        """
        if not root:
            raise ValueError("Root directory path must be provided")
        # define required directories
        directories = [
            Dirs.RAW,
            Dirs.PRE_PROCESS,
            Dirs.PROCESSED,
            Dirs.ANALYSIS,
            Dirs.GEOCODIO,
            Path(Dirs.GEOCODIO) / "partials",
            Dirs.OUTPUT,
            Dirs.SUMMARY_STATS,
            Dirs.VALIDATION_ERRORS
        ]
        # create directories if they don't exist

        for dir_name in directories:
            dir_path: Path = root / dir_name
            if not dir_path.exists():
                try:
                    # print to console # dir_name not found. creating...
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise RuntimeError(f"Failed to create directory {dir_path}: {str(e)}")

    @classmethod
    def copy_raw_data(cls, raw_data_dir: Path, data_root: Path):
        file_names = [
            Raw.PROPS_TAXPAYERS,
            Raw.CORPS,
            Raw.LLCS,
            Raw.CLASS_CODES
        ]
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TotalFileSizeColumn(),
                TimeRemainingColumn(),
        ) as progress:
            console.print("Copying raw data files to project data directory...\n")
            for file_name in file_names:
                source_file = raw_data_dir / f"{file_name}.csv"
                dest_file = data_root / "raw" / f"{file_name}.csv"
                if not source_file.exists():
                    console.print(f"[red]Error: Source file \"{file_name}.csv\" not found.[/red]")
                    return
                try:
                    file_size = source_file.stat().st_size
                    task = progress.add_task(
                        f"Copying {file_name}...",
                        total=file_size,
                        completed=0
                    )
                    with open(source_file, "rb") as src, open(dest_file, "wb") as dst:
                        while chunk := src.read(8192):
                            dst.write(chunk)
                            progress.update(task, advance=len(chunk))
                except PermissionError:
                    console.print(f"[red]Error: Permission denied when copying {file_name}[/red]")
                    return
                except Exception as e:
                    console.print(f"[red]Error copying {file_name}: {str(e)}[/red]")
                    return

        console.print("\nData successfully copied.")

    @classmethod
    def sizeof_fmt(cls, num: float, suffix: str = 'B') -> str:
        """Convert bytes to human readable string."""
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(num) < 1024.0:
                return f"{num:3.1f} {unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f} Y{suffix}"



class PathGenerators(UtilsBase):
    """
    Helper functions to return the file path for each individual dataset. Methods are named by
    '{dir_name}_{dataset_name}()'
    """
    # -------------------
    # ----PRE_PROCESS----
    # -------------------
    @classmethod
    def pre_process_taxpayers_city(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/pre_process/taxpayers_city[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PRE_PROCESS,
            PreProcess.TAXPAYERS_CITY,
            configs["load_ext"]
        )
    @classmethod
    def pre_process_taxpayers_county(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/pre_process/taxpayers_county[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PRE_PROCESS,
            PreProcess.TAXPAYERS_COUNTY,
            configs["load_ext"]
        )
    @classmethod
    def pre_process_business_filings_1(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/pre_process/taxpayers_city[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PRE_PROCESS,
            PreProcess.BUSINESS_FILINGS_1,
            configs["load_ext"]
        )
    @classmethod
    def pre_process_business_filings_3(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/pre_process/taxpayers_city[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PRE_PROCESS,
            PreProcess.BUSINESS_FILINGS_3,
            configs["load_ext"]
        )

    # -----------
    # ----RAW----
    # -----------
    @classmethod
    def raw_props_taxpayers(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/raw/taxpayer_records[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.RAW,
            Raw.PROPS_TAXPAYERS,
            configs["load_ext"]
        )
    @classmethod
    def raw_corps(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/raw/corps[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.RAW,
            Raw.CORPS,
            configs["load_ext"]
        )
    @classmethod
    def raw_llcs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/raw/llcs[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.RAW,
            Raw.LLCS,
            configs["load_ext"]
        )
    @classmethod
    def raw_class_codes(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/raw/class_codes[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.RAW,
            Raw.CLASS_CODES,
            configs["load_ext"]
        )
    @classmethod
    def raw_bus_filings(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/raw/business_filings_1[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.RAW,
            Raw.BUS_FILINGS,
            configs["load_ext"]
        )
    @classmethod
    def raw_bus_names_addrs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/raw/business_filings_1[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.RAW,
            Raw.BUS_NAMES_ADDRS,
            configs["load_ext"]
        )

    # -----------------
    # ----PROCESSED----
    # -----------------
    @classmethod
    def processed_taxpayer_records(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/taxpayer_records[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.TAXPAYER_RECORDS,
            configs["load_ext"]
        )
    @classmethod
    def processed_properties(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/taxpayer_records[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.PROPERTIES,
            configs["load_ext"]
        )
    @classmethod
    def processed_properties_rentals(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/taxpayer_records[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.PROPERTIES_RENTALS,
            configs["load_ext"]
        )
    @classmethod
    def processed_bus_filings(cls, configs: WorkflowConfigs) -> Path:
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.BUS_FILINGS,
            configs["load_ext"]
        )
    @classmethod
    def processed_bus_names_addrs(cls, configs: WorkflowConfigs) -> Path:
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.BUS_NAMES_ADDRS,
            configs["load_ext"]
        )
    @classmethod
    def processed_corps(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/corps[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.CORPS,
            configs["load_ext"]
        )
    @classmethod
    def processed_llcs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/llcs[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.LLCS,
            configs["load_ext"]
        )
    @classmethod
    def processed_class_codes(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/class_codes[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.CLASS_CODES,
            configs["load_ext"]
        )
    @classmethod
    def processed_validated_addrs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/validated_addrs[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.VALIDATED_ADDRS,
            configs["load_ext"]
        )
    @classmethod
    def processed_unvalidated_addrs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/unvalidated_addrs[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.UNVALIDATED_ADDRS,
            configs["load_ext"]
        )
    @classmethod
    def processed_taxpayers_merged(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/taxpayers_merged[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.TAXPAYERS_MERGED,
            configs["load_ext"]
        )
    @classmethod
    def processed_corps_merged(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/corps_merged[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.CORPS_MERGED,
            configs["load_ext"]
        )
    @classmethod
    def processed_llcs_merged(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/llcs_merged[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.LLCS_MERGED,
            configs["load_ext"]
        )
    @classmethod
    def processed_taxpayers_fixed(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/props_subsetted[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.TAXPAYERS_FIXED,
            configs["load_ext"]
        )
    @classmethod
    def processed_taxpayers_subsetted(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/props_subsetted[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.TAXPAYERS_SUBSETTED,
            configs["load_ext"]
        )
    @classmethod
    def processed_taxpayers_prepped(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/props_prepped[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.TAXPAYERS_PREPPED,
            configs["load_ext"]
        )
    @classmethod
    def processed_corps_subsetted(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/corps_subsetted[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.CORPS_SUBSETTED,
            configs["load_ext"]
        )
    @classmethod
    def processed_llcs_subsetted(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/llcs_subsetted[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.LLCS_SUBSETTED,
            configs["load_ext"]
        )
    @classmethod
    def processed_taxpayers_string_matched(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/props_string_matched[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.TAXPAYERS_STRING_MATCHED,
            configs["load_ext"]
        )
    @classmethod
    def processed_taxpayers_networked(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/processed/props_networked[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.PROCESSED,
            Processed.TAXPAYERS_NETWORKED,
            configs["load_ext"]
        )

    # -----------------
    # ----GEOCODIO----
    # -----------------
    @classmethod
    def geocodio_gcd_validated(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/geocodio/gcd_validated[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.GEOCODIO,
            Geocodio.GCD_VALIDATED,
            configs["load_ext"]
        )
    @classmethod
    def geocodio_gcd_unvalidated(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/geocodio/gcd_unvalidated[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.GEOCODIO,
            Geocodio.GCD_UNVALIDATED,
            configs["load_ext"]
        )
    @classmethod
    def geocodio_gcd_failed(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/geocodio/gcd_failed[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.GEOCODIO,
            Geocodio.GCD_FAILED,
            configs["load_ext"]
        )
    @classmethod
    def geocodio_partial(cls, configs: WorkflowConfigs, filename: str) -> Path:
        """:returns: ROOT/geocodio/partials/gcd_partial_{timestamp}[ext]"""
        return cls.generate_geocodio_partial_path(configs["data_root"], filename, configs["load_ext"])

    # ----------------
    # ----ANALYSIS----
    # ----------------
    @classmethod
    def analysis_frequent_tax_names(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/analysis/frequent_tax_names[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.ANALYSIS,
            Analysis.FREQUENT_TAX_NAMES,
            configs["load_ext"]
        )
    @classmethod
    def analysis_frequent_tax_addrs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/analysis/frequent_tax_addrs[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.ANALYSIS,
            Analysis.FREQUENT_TAX_ADDRS,
            configs["load_ext"]
        )
    @classmethod
    def analysis_fixing_tax_names(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/analysis/fixing_tax_names[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.ANALYSIS,
            Analysis.FIXING_TAX_NAMES,
            configs["load_ext"]
        )
    @classmethod
    def analysis_fixing_addrs(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/analysis/fixing_tax_addrs[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.ANALYSIS,
            Analysis.FIXING_ADDRS,
            configs["load_ext"]
        )
    @classmethod
    def analysis_address_analysis(cls, configs: WorkflowConfigs) -> Path:
        """:returns: ROOT/analysis/address_analysis[ext]"""
        return cls.generate_path(
            configs["data_root"],
            Dirs.ANALYSIS,
            Analysis.ADDRESS_ANALYSIS,
            configs["load_ext"]
        )

    # ---------------------
    # ----SUMMARY STATS----
    # ---------------------
    @classmethod
    def summary_stats(cls, configs: WorkflowConfigs, wkfl_name: str) -> Path:
        return cls.generate_path(
            configs["data_root"],
            Dirs.SUMMARY_STATS,
            wkfl_name,
            configs["load_ext"]
        )

    # --------------
    # ----OUTPUT----
    # --------------
    @classmethod
    def output_network_calcs(cls, configs: WorkflowConfigs) -> Path:
        return cls.generate_path(
            configs["data_root"],
            Dirs.OUTPUT,
            Output.NETWORK_CALCS,
            configs["load_ext"]
        )
    @classmethod
    def output_entity_types(cls, configs: WorkflowConfigs) -> Path:
        return cls.generate_path(
            configs["data_root"],
            Dirs.OUTPUT,
            Output.ENTITY_TYPES,
            configs["load_ext"]
        )
    @classmethod
    def output_entities(cls, configs: WorkflowConfigs) -> Path:
        return cls.generate_path(
            configs["data_root"],
            Dirs.OUTPUT,
            Output.ENTITIES,
            configs["load_ext"]
        )
    @classmethod
    def output_validated_addresses(cls, configs: WorkflowConfigs) -> Path:
        return cls.generate_path(
            configs["data_root"],
            Dirs.OUTPUT,
            Output.VALIDATED_ADDRESSES,
            configs["load_ext"]
        )
    @classmethod
    def output_llcs(cls, configs: WorkflowConfigs) -> Path:
        return cls.generate_path(
            configs["data_root"],
            Dirs.OUTPUT,
            Output.LLCS,
            configs["load_ext"]
        )
    @classmethod
    def output_corps(cls, configs: WorkflowConfigs) -> Path:
        return cls.generate_path(
            configs["data_root"],
            Dirs.OUTPUT,
            Output.CORPS,
            configs["load_ext"]
        )
    @classmethod
    def output_networks(cls, configs: WorkflowConfigs) -> Path:
        return cls.generate_path(
            configs["data_root"],
            Dirs.OUTPUT,
            Output.NETWORKS,
            configs["load_ext"]
        )
    @classmethod
    def output_taxpayer_records(cls, configs: WorkflowConfigs) -> Path:
        return cls.generate_path(
            configs["data_root"],
            Dirs.OUTPUT,
            Output.TAXPAYER_RECORDS,
            configs["load_ext"]
        )
    # -------------------------
    # ----VALIDATION ERRORS----
    # -------------------------
    @classmethod
    def validation_errors(cls, configs: WorkflowConfigs, wkfl_name: str, dataset_type: str) -> Path:
        filename: str = f"{wkfl_name.replace(" ", "_")}__{cls.get_timestamp()}__{dataset_type}"
        return cls.generate_path(
            configs["data_root"],
            Dirs.VALIDATION_ERRORS,
            filename,
            configs["load_ext"]
        )