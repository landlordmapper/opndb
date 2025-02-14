from abc import abstractmethod
from pathlib import Path

from opndb.types.base import WorkflowConfigs, FileExt

DATA_ROOT: Path = Path("")
GEOCODIO_URL = "https://api.geocod.io/v1.7/geocode?api_key="

DIRECTIONS: dict[str, str] = {
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
    "NORTHEAST": "NE",
    "SOUTHEAST": "SE",
    "NORTHWEST": "NW",
    "SOUTHWEST": "SW",
}

STREET_SUFFIXES: dict[str, str] = {
    "STREET": "ST",
    "STREE": "ST",
    "AVENUE": "AVE",
    "AVENU": "AVE",
    "AV":"AVE",
    "LANE": "LN",
    "DRIVE": "DR",
    "BOULEVARD": "BLVD",
    "BULEVARD": "BLVD",
    "BOULEVAR": "BLVD",
    "ROAD":"RD",
    "COURT":"CT",
    "PLACE": "PL",
    "WAY": "WAY"
}

SECONDARY_KEYWORDS: set[str] = {
    "#",
    "UNIT",
    "FLOOR",
    "FL",
    "SUITE",
    "STE",
    "APT",
    "ROOM"
}

class DataDirs:
    """Directory names for workflow data file storage, corresponding to classes."""
    RAW: str = "raw"
    GEOCODIO: str = "geocodio"
    GCD_PARTIALS: str = "partials"
    ANALYSIS: str = "analysis"
    PROCESSED: str = "processed"
    SUMMARY_STATS: str = "summary_stats"
    FINAL_OUTPUTS: str = "final_outputs"

class FileNames:
    """Centralized file naming constants organized by directory structure."""
    @classmethod
    def get_raw_filename_ext(cls, filename: str, config: WorkflowConfigs) -> str:
        """Sets file extension on file names. Allows for flexibility in file format of data inputs"""
        return f"{filename}.{config['load_ext']}"

class Raw(FileNames):
    """Raw input files, saved after running initial validation checks."""
    TAXPAYER_RECORDS_RAW: str = "taxpayer_records"
    CORPS_RAW: str = "corps"
    LLCS_RAW: str = "llcs"
    CLASS_CODE_DESCRIPTIONS: str = "class_code_descriptions"


class Geocodio(FileNames):
    GCD_VALIDATED: str = "gcd_validated"
    GCD_UNVALIDATED: str = "gcd_unvalidated"
    GCD_FAILED: str = "gcd_failed"

class Analysis(FileNames):
    """Analysis files generated by the workflow and used for manual research & input."""
    FREQUENT_TAX_NAMES: str = "frequent_tax_names"
    FREQUENT_ADDRS: str = "frequent_tax_addrs"
    FIXING_TAX_NAMES: str = "fixing_tax_names"
    FIXING_ADDRS: str = "fixing_tax_addrs"
    ADDRESS_ANALYSIS: str = "address_analysis"

class Processed(FileNames):
    """Datasets generated by different stages of the workflow."""

    TAXPAYER_RECORDS_CLEAN: str = "taxpayer_records"
    CORPS_CLEAN: str = "corps"
    LLCS_CLEAN: str = "llcs"
    BLDG_CLASS_CODES_CLEAN: str = "bldg_class_codes"

    VALIDATED_ADDRS: str = "validated_addrs"
    UNVALIDATED_ADDRS: str = "unvalidated_addrs"

    PROPS_SUBSETTED: str = "props_subsetted"
    PROPS_PREPPED: str = "props_prepped"
    PROPS_STRING_MATCHED: str = "props_string_matched"
    PROPS_NETWORKED: str = "props_networked"

class SummaryStats(FileNames):
    """Summary statistics for each stage of the workflow."""
    pass

class FinalOutputs(FileNames):
    """Final datasets."""
    pass
