import string
from pathlib import Path
from typing import Any
import word2number as w2n

import pandas as pd

from opndb.string_cleaning_ops import CleanStringBase as clean


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
    def trim_whitespace(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        cols = list(df.columns) if cols is None else cols
        for col in cols:
            df[col] = df[col].apply(lambda x: clean.delete_symbols_spaces())
        return df


    @classmethod
    def remove_symbols_punctuation(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        cols = list(df.columns) if cols is None else cols
        for col in cols:
            df[col] = df[col].apply(lambda x: x.replace("&", "").replace(",", "").replace(".", ""))
            df[col] = df[col].apply(
                lambda x: x.translate(
                    str.maketrans(string.punctuation.replace("/", "").replace("-", "")),
                    " "*len(string.punctuation.replace('/','').replace('-',''))
                )
            )
        return df


    @classmethod
    def remove_extra_spaces(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        cols = list(df.columns) if cols is None else cols
        for col in cols:
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        return df


    # @classmethod
    # def switch_the(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    #     cols = list(df.columns) if cols is None else cols
    #     for col in cols:
    #         df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

    @classmethod
    def convert_ordinals(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        cols = list(df.columns) if cols is None else cols
        for col in cols:
            df[col] = pd.to_numeric
