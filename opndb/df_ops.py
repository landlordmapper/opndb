from pathlib import Path
from typing import Any

import pandas as pd


class DataFrameOpsBase(object):

    @classmethod
    def print_cols(cls, df: pd.DataFrame) -> list[str]:
        """Returns list of strings representing pandas dataframe columns."""
        return [col for col in df.columns]

    @classmethod
    def rename_col(cls, df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
        """Renames pandas dataframe column."""
        df.rename(columns={old_name: new_name}, inplace=True)
        return df

    @classmethod
    def load_df_csv(cls, filepath: Path, dtype: str | dict[str, Any]) -> pd.DataFrame:
        return pd.read_csv(str(filepath), dtype=dtype)

    @classmethod
    def save_df_csv(cls, df, path):
        df.to_csv(path, index=False)
        return path

    # @classmethod
    # def get_string_cols(cls, df: pd.DataFrame) -> list[str]:
    #     """Returns names of all columns of a dataframe whose dtype is string."""
    #     string_cols = []
    #     for col in df.columns:
    #         if col.dtype is str:
    #             string_cols.append(col)
    #     return string_cols


class DataFrameOpsCols(DataFrameOpsBase):

    @classmethod
    def trim_whitespace(cls, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            df[col] = df[col].str.strip()
        return df

    def remove_extra_spaces(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        return df