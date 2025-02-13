import pandas as pd

class UtilsBase(object):

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
    def generate_file_name(cls, name: str, stage: str, ext: str) -> str:
        """Returns file name with stage prefix."""
        return f"{stage:02d}_{name}.{ext}"
