from pathlib import Path

import pandas as pd

from opndb.constants.base import DATA_ROOT, FileNames
from opndb.df_ops import DataFrameOpsBase as df_ops
from opndb.types.base import AddressColObj


class AddressBase:

    dirs = FileNames.DataDirs
    proc = FileNames.Processed

    validated_addrs_path: Path = DATA_ROOT / dirs.PROCESSED / proc.VALIDATED_ADDRS
    unvalidated_addrs_path: Path = DATA_ROOT / dirs.PROCESSED / proc.UNVALIDATED_ADDRS

    @classmethod
    def get_unique_addresses(cls, df: pd.DataFrame, addr_cols: AddressColObj):
        """Returns all unique addresses found in the dataframe."""
        pass

    @classmethod
    def save_unvalidated_addrs_initial(cls, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        pass

    @classmethod
    def save_validated_addrs_initial(cls, df: pd.DataFrame) -> str:
        return df_ops.save_df_csv(df, cls.validated_addrs_path)


class AddressValidatorBase(AddressBase):

    """Handles operations that add to and remove from the master validated & unvalidated address files."""

    def __init__(self):
        super().__init__()
        self.df_validated: pd.DataFrame = df_ops.load_df_csv(self.validated_addrs_path, str)
        self.df_unvalidated: pd.DataFrame = df_ops.load_df_csv(self.unvalidated_addrs_path, str)

    def add_to_df_validated(self):
        pass

    def remove_from_df_unvalidated(self):
        # pass index of row to be removed as argument
        pass

    def save(self) -> dict[str, str]:
        validated_path = df_ops.save_df_csv(self.df_validated, self.validated_addrs_path)
        unvalidated_path = df_ops.save_df_csv(self.df_unvalidated, self.unvalidated_addrs_path)
        return {
            "validated_path": validated_path,
            "unvalidated_path": unvalidated_path,
        }


class GeocodioBase(AddressBase):
    pass