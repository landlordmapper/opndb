from pathlib import Path
from typing import Any, Callable, Type

import pandas as pd

from opndb.constants.columns import TaxpayerRecords as tr, ValidatedAddrs as va, ClassCodes as cc, Properties as p, \
    UnvalidatedAddrs as ua
from opndb.services.string_clean import CleanStringBase as clean_base
from opndb.services.string_clean import CleanStringName as clean_name
from opndb.services.string_clean import CleanStringAddress as clean_addr
from opndb.services.string_clean import CleanStringAccuracy as clean_acc


class DataFrameOpsBase(object):

    @classmethod
    def load_df(cls, path: Path, dtype: Type | dict[str, Any]) -> pd.DataFrame:
        """
        Loads dataframes based on file format. Reads extension and loads dataframe using corresponding pd.read method.

        :param filepath: Complete path to data file to be loaded into dataframe (UtilsBase.generate_file_path())
        :param dtype: Specify data types for columns or entire dataset
        :return: Dataframe containing data from specified file
        """
        format = path.suffix[1:].lower()
        if format == "csv":
            return pd.read_csv(str(path), dtype=dtype)
        elif format == "parquet":
            return pd.read_parquet(str(path), dtype=dtype)
        elif format == "xlsx" or format == "xls":
            return pd.read_excel(str(path), dtype=dtype)
        elif format == "json":
            return pd.read_json(str(path), dtype=dtype)
        else:
            raise ValueError(f"Unsupported file format: {format}")

    @classmethod
    def save_df(cls, df: pd.DataFrame, path: Path) -> str:
        """
        Saves dataframe to file based on file format. Reads extension and saves dataframe using corresponding pd.save
        method.

        :param df: Dataframe to be saved
        :param path: Path to save dataframe, including file extension (UtilsBase.generate_file_path())
        :return: Path to saved dataframe
        """
        format = path.suffix[1:].lower()
        if format == "csv":
            df.to_csv(str(path), index=False)
        elif format == "parquet":
            df.to_parquet(str(path), index=False)
        elif format == "xlsx":
            df.to_excel(str(path), index=False)
        elif format == "json":
            df.to_json(str(path), orient='records', indent=4)
        else:
            raise ValueError(f"Unsupported file format: {format}")
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


class DataFrameMergers(DataFrameOpsBase):
    """
    Dataframe operations that merge multiple dataframes and handle post-merge cleanup (dropping duplicates, combining
    columns, etc.)
    """
    @classmethod
    def merge_orgs(cls, df_taxpayers: pd.DataFrame, df_orgs: pd.DataFrame) -> pd.DataFrame:
        pass

    @classmethod
    def merge_validated_addrs(cls, df: pd.DataFrame, df_addrs: pd.DataFrame, clean_addr_cols: list[str]) -> pd.DataFrame:
        """Merges validated address data into property taxpayer dataset."""
        # todo: remove repeated cols (combine columns function from old workflow)
        # todo: remove unnecessary columns before returning
        # todo: test to confirm whether we should drop duplicates here
        return pd.merge(
            df,
            df_addrs[[va.CLEAN_ADDRESS, va.FORMATTED_ADDRESS]],
            how="left",
            left_on=clean_addr_cols[0],
            right_on=va.CLEAN_ADDRESS
        )


class DataFrameColumnGenerators(DataFrameOpsBase):
    """Dataframe operations that generate new columns."""
    @classmethod
    def set_is_rental(cls, df_props: pd.DataFrame, df_class_codes: pd.DataFrame) -> pd.DataFrame:
        """
        Subsets property dataframe to only include properties associated with rental class codes. Handles situations in
        which multiple class codes are associated with a single property and are separated by commas.
        :param df_props: Dataframe containing entire property taxpayer dataset
        :param df_class_codes: Dataframe containing building class code descriptions
        :return: D
        """
        df_rental_codes: pd.DataFrame = df_class_codes[df_class_codes[cc.IS_RENTAL] == True]
        rental_codes: list[str] = list(df_rental_codes[cc.IS_RENTAL])
        df_props[tr.IS_RENTAL] = df_props[p.CLASS_CODE].apply(
            lambda codes: cls.rental_class_check(
                [code.strip() for code in codes.split(",")],
                rental_codes
            )
        )
        return df_props

    @classmethod
    def set_core_name(cls, df: pd.DataFrame, name_col: str) -> pd.DataFrame:
        """
        Adds core_name column to dataframe.

        :param df: Dataframe to add column to
        :param name_col: Column in the dataframe to be used to derive core name
        """
        df["core_name"] = df[name_col].apply(lambda name: clean_base.core_name(name))
        return df

    @classmethod
    def set_is_bank(cls, df: pd.DataFrame, col: str, suffix: str | None = None) -> pd.DataFrame:
        """
        Adds is_bank boolean column to dataframe.

        :param df: Dataframe to add column to
        :param name_col: Column in the dataframe to be used to set is_bank
        :param suffix: Suffix to add to column name (ex: is_bank_president, is_bank_agent, etc.)
        """
        is_bank_col: str = f"is_bank_{suffix}" if suffix else "is_bank"
        df[is_bank_col] = df[col].apply(lambda name: clean_base.get_is_bank(name))
        return df

    @classmethod
    def set_is_trust(cls, df: pd.DataFrame, col: str, suffix: str | None = None) -> pd.DataFrame:
        """
        Adds is_trust boolean column to dataframe.

        :param df: Dataframe to add column to
        :param name_col: Column in the dataframe to be used to set is_trust
        :param suffix: Suffix to add to column name (ex: is_trust_president, is_trust_agent, etc.)
        """
        is_trust_col: str = f"is_trust_{suffix}" if suffix else "is_trust"
        df[is_trust_col] = df[col].apply(lambda name: clean_base.get_is_trust(name))
        return df

    @classmethod
    def set_is_person(cls, df: pd.DataFrame, col: str, suffix: str | None = None) -> pd.DataFrame:
        """
        Adds is_person boolean column to dataframe.

        :param df: Dataframe to add column to
        :param name_col: Column in the dataframe to be used to set is_person
        :param suffix: Suffix to add to column name (ex: is_person_president, is_person_agent, etc.)
        """
        is_person_col: str = f"is_person_{suffix}" if suffix else "is_person"
        df[is_person_col] = df[col].apply(lambda name: clean_base.get_is_person(name))
        return df

    @classmethod
    def set_is_common_name(cls, df: pd.DataFrame, col: str, suffix: str | None = None) -> pd.DataFrame:
        """
        Adds is_common_name boolean column to dataframe.

        :param df: Dataframe to add column to
        :param name_col: Column in the dataframe to be used to set is_common_name
        :param suffix: Suffix to add to column name (ex: is_common_name_president, is_common_name_agent, etc.)
        """
        is_common_name_col: str = f"is_common_name_{suffix}" if suffix else "is_common_name"
        df[is_common_name_col] = df[col].apply(lambda name: clean_base.get_is_common_name(name))
        return df

    @classmethod
    def set_is_org(cls, df: pd.DataFrame, col: str, suffix: str | None = None) -> pd.DataFrame:
        """
        Adds is_org boolean column to dataframe.

        :param df: Dataframe to add column to
        :param name_col: Column in the dataframe to be used to set is_org
        :param suffix: Suffix to add to column name (ex: is_org_president, is_org_agent, etc.)
        """
        is_org_col: str = f"is_org_{suffix}" if suffix else "is_org"
        df[is_org_col] = df[col].apply(lambda name: clean_base.get_is_org(name))
        return df

    @classmethod
    def set_is_llc(cls, df: pd.DataFrame, col: str, suffix: str | None = None) -> pd.DataFrame:
        """
        Adds is_llc boolean column to dataframe.

        :param df: Dataframe to add column to
        :param name_col: Column in the dataframe to be used to set is_llc
        :param suffix: Suffix to add to column name (ex: is_llc_president, is_llc_agent, etc.)
        """
        is_llc_col: str = f"is_llc_{suffix}" if suffix else "is_llc"
        df[is_llc_col] = df[col].apply(lambda name: clean_base.get_is_llc(name))
        return df

    @classmethod
    def set_is_pobox(cls, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Adds is_pobox boolean column to dataframe.

        :param df: Dataframe to add column to
        :param col: Column in the dataframe to be used to set is_pobox
        """
        df["is_pobox"] = df[col].apply(lambda addr: clean_base.get_is_pobox(addr))
        return df


class DataFrameSubsetters(DataFrameOpsBase):
    """Dataframe operations that return subsets."""
    @classmethod
    def get_nonrentals_from_addrs(cls, df_all: pd.DataFrame, df_rentals_initial: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts properties whose class codes are NOT associated with rental class codes, but whose taxpayer mailing
        address matches with an address from the rental property subset.

        :param df_all: Dataframe containing entire property taxpayer dataset
        :param df_rentals_initial: Dataframe containing initial rental subset
        :return: Dataframe containing properties excluded from initial rental subset with matching validated taxpayer addresses
        """
        rental_addrs: list[str] = list(df_rentals_initial[tr.FORMATTED_ADDRESS].dropna().unique())
        df_nonrentals: pd.DataFrame = df_all[df_all[tr.IS_RENTAL] == False]
        return df_nonrentals[df_nonrentals[tr.FORMATTED_ADDRESS].isin(rental_addrs)]

    @classmethod
    def get_active(cls, df: pd.DataFrame, col: str) -> pd.DataFrame:
        pass

    @classmethod
    def get_is_pobox(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Subsets dataframe to include only addresses identified as matching a PO Box pattern."""
        return df[df["is_pobox"] == True]

    @classmethod
    def update_unvalidated_addrs(cls, df: pd.DataFrame, addrs: list[str]) -> pd.DataFrame:
        """
        Removes addresses from the unvalidated_addrs master dataset. This should be called after validating new
        addresses.

        :param df: Dataframe containing the previous unvalidated addresses dataset
        :param addrs: Addresses to remove from the unvalidated_addrs master dataset
        """
        # todo: figure out how to handle saving dfs, whether or not it should be done in the workflows or in these
        return df[~df[ua.TAX_FULL_ADDRESS].isin(addrs)]


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
    def remove_extra_spaces(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.remove_extra_spaces, cols)

    @classmethod
    def words_to_num(cls, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return cls.apply_string_cleaner(df, clean_base.words_to_num, cols)

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
