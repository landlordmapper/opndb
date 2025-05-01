from pathlib import Path
from typing import Any, Callable, Type
import pandas as pd
import pandera as pa
from opndb.services.string_clean import (
    CleanStringBase as clean_base,
    CleanStringName as clean_name,
    CleanStringAddress as clean_addr,
    CleanStringAccuracy as clean_acc
)
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TotalFileSizeColumn,
)
from rich.console import Console

from opndb.validator.df_model import OPNDFModel

console = Console()

class DataFrameOpsBase(object):

    """
    Base dataframe operations class. Low-level dataframe functions that can be imported/used in other services classes
    without circular import issues. High-level dataframe operations that import from other services classes belong in
    services/dataframe/ops.py
    """

    @classmethod
    def load_df(
        cls,
        path: Path,
        schema: Type[OPNDFModel] = None,
        recursive_bools: bool = False,
    ) -> pd.DataFrame | None:
        """
        Loads dataframes based on file format. Reads extension and loads dataframe using corresponding pd.read method.
        Returns None if the path doesn't exist.'

        :param filepath: Complete path to data file to be loaded into dataframe (UtilsBase.generate_file_path())
        :param dtype: Specify data types for columns or entire dataset
        :return: Dataframe containing data from specified file
        """

        # Set boolean columns
        def coerce_booleans(df: pd.DataFrame, bool_cols: list[str]) -> pd.DataFrame:
            for col in bool_cols:
                df[col] = df[col].map(
                    lambda x: str(x).strip().lower() in {"true", "1", "t", "yes"}
                    if pd.notnull(x) else False
                )
            return df

        # no file found, return None
        if not path.exists():
            console.print(f"[yellow]File not found: {path}[/yellow]")
            return None

        # File is empty, return None
        if path.stat().st_size == 0:
            console.print(f"[yellow]File is empty: {path}[/yellow]")
            return None

        # Check file content to see if it's a valid CSV
        if path.suffix.lower() == ".csv":
            try:
                with open(path, "r") as f:
                    # Read first few lines to see if there's anything parseable
                    sample = "".join([f.readline() for _ in range(5)])
                if not sample.strip() or "," not in sample:
                    console.print(f"[yellow]File exists but doesn't contain valid CSV data: {path}[/yellow]")
                    return None
            except Exception as e:
                console.print(f"[yellow]Error inspecting file content: {str(e)}[/yellow]")
                # Continue trying to load anyway, pandas will handle the error

        # Get file size, initiate progress bar & load dataframe
        file_size = path.stat().st_size
        format = path.suffix[1:].lower()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TotalFileSizeColumn(),
        ) as progress:
            task = progress.add_task(
                f"Loading {path.name} into dataframe...",
                total=file_size,
                completed=0
            )
            try:
                if format == "csv":
                    df: pd.DataFrame = pd.read_csv(str(path), dtype=str)
                elif format == "parquet":
                    df: pd.DataFrame = pd.read_parquet(str(path), dtype=str)
                elif format == "xlsx" or format == "xls":
                    df: pd.DataFrame = pd.read_excel(str(path), dtype=str)
                elif format == "json":
                    df: pd.DataFrame = pd.read_json(str(path), dtype=str)
                else:
                    raise ValueError(f"Unsupported file format: {format}")
                progress.update(task, completed=file_size)
                # Convert boolean columns to pandas bool type
                if schema:
                    df = coerce_booleans(df, schema.boolean_fields(recursive=recursive_bools))
                return df
            except Exception as e:
                progress.stop()
                console.print(f"[red]Error loading {path.name}: {str(e)}[/red]")
                raise

    @classmethod
    def save_df(cls, df: pd.DataFrame, path: Path) -> str:
        """
        Saves dataframe to file based on file format. Reads extension and saves dataframe using corresponding pd.save
        method.

        :param df: Dataframe to be saved
        :param path: Path to save dataframe, including file extension (UtilsBase.generate_file_path())
        :return: Path to saved dataframe
        """
        df.to_csv(str(path), index=False)
        # format = path.suffix[1:].lower()
        # if format == "csv":
        #     df.to_csv(str(path), index=False)
        # elif format == "parquet":
        #     df.to_parquet(str(path), index=False)
        # elif format == "xlsx":
        #     df.to_excel(str(path), index=False)
        # elif format == "json":
        #     df.to_json(str(path), orient='records', indent=4)
        # else:
        #     raise ValueError(f"Unsupported file format: {format}")
        return str(path)

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
    def combine_columns(cls, c1, c2):
        """Combines two dataframe columns."""
        # todo: check where and how this is used to provide a more detailed description
        if pd.isnull(c1) == True:
            return c2
        else:
            return c1

    @classmethod
    def combine_columns_parallel(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicated, redundant columns (based on _x and _y auto-added suffixes)"""
        # todo: check where and how this is used to provide a more detailed description
        for column in df.columns:
            if column[-2:] == '_x':
                df[column[:-2]] = df.apply(lambda x: cls.combine_columns(x[column], x[column[:-2] + '_y']), axis =1)
                df.drop(columns = [column, column[:-2] + '_y'], inplace = True)
        return df

    @classmethod
    def get_frequency_df(cls, df: pd.DataFrame, unique_col: str) -> pd.DataFrame:
        """
        Returns single dataframe with a column of unique values from the original dataframe and their frequency counts
        """
        df_freq = df[unique_col].value_counts().reset_index()
        df_freq.columns = [unique_col, "count"]
        return df_freq

    @classmethod
    def rental_class_check(cls, class_codes: list[str], rental_classes: list[str]) -> bool:
        """
        Checks class codes and returns True if at least one of them is a rental class code. Returns false if none are.
        """
        return any(code in rental_classes for code in class_codes)

    @classmethod
    def exclude_raw_cols(cls, df: pd.DataFrame) -> list[str]:
        return list(filter(lambda x: not x.startswith("raw"), df.columns))


class DataFrameCleaners(DataFrameOpsBase):
    """Base dataframe cleaning methods."""
    @classmethod
    def apply_string_cleaner(
        cls,
        df: pd.DataFrame,
        cleaner_func: Callable[[str], str],
        cols: list[str] | None = None
    ) -> pd.DataFrame:
        """Generic method to apply any string cleaner function to specified columns"""
        cols = list(df.columns) if cols is None else cols
        for col in cols:
            df[col] = df[col].apply(cleaner_func)
        return df


class DataFrameBaseCleaners(DataFrameCleaners):

    """
    Methods that clean dataframe columns by applying string cleaning functions to specific columns. Default behavior
    is to execute string cleaning on all columns. Each method corresponds to a single string cleaning method from the
    CleanStringBase class.
    """

    @classmethod
    def make_upper(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.make_upper, cols)

    @classmethod
    def remove_symbols_punctuation(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.remove_symbols_punctuation, cols)

    @classmethod
    def trim_whitespace(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.trim_whitespace, cols)

    @classmethod
    def replace_with_nan(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.replace_with_nan, cols)

    @classmethod
    def remove_extra_spaces(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.remove_extra_spaces, cols)

    @classmethod
    def fix_llcs(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.fix_llcs, cols)

    @classmethod
    def deduplicate(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.deduplicate, cols)

    @classmethod
    def convert_ordinals(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.convert_ordinals, cols)

    @classmethod
    def take_first(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.take_first, cols)

    @classmethod
    def combine_numbers(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.combine_numbers, cols)


class DataFrameNameCleaners(DataFrameCleaners):

    """
    Methods that clean dataframe columns by applying string cleaning functions to specific columns. Default behavior
    is to execute string cleaning on all columns. Each method corresponds to a single string cleaning method from the
    CleanStringName class.
    """

    @classmethod
    def switch_the(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_name.switch_the, cols)


class DataFrameAddressCleaners(DataFrameCleaners):

    """
    Methods that clean dataframe columns by applying string cleaning functions to specific columns. Default behavior
    is to execute string cleaning on all columns. Each method corresponds to a single string cleaning method from the
    CleanStringAddress class.
    """

    @classmethod
    def convert_nsew(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_addr.convert_nsew, cols)

    @classmethod
    def remove_secondary_designators(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_addr.remove_secondary_designators, cols)

    @classmethod
    def convert_street_suffixes(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_addr.convert_street_suffixes, cols)

    @classmethod
    def fix_zip(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_addr.fix_zip, cols)


class DataFrameCleanersAccuracy(DataFrameCleaners):

    """
    Methods that clean dataframe columns by applying string cleaning functions to specific columns. Default behavior
    is to execute string cleaning on all columns. Each method corresponds to a single string cleaning method from the
    CleanStringAccuracy class.

    NOTE - Applying these string cleaners could meaningfully impact accuracy during matching processes.
    """

    @classmethod
    def drop_letters(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_acc.drop_letters, cols)

    @classmethod
    def convert_mixed(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_acc.convert_mixed, cols)

    @classmethod
    def remove_secondary_component(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_acc.remove_secondary_component, cols)
