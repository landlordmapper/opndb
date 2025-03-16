from pathlib import Path
from typing import Any, Callable, Type, Tuple
from pprint import pprint

import numpy as np
import pandas as pd

from opndb.constants.columns import (
    PropsTaxpayers as pt,
    TaxpayerRecords as tr,
    ValidatedAddrs as va,
    ClassCodes as cc,
    Properties as p,
    UnvalidatedAddrs as ua
)
from opndb.services.address import AddressBase as addr
from opndb.services.dataframe.base import DataFrameOpsBase
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
    FileSizeColumn,
    TotalFileSizeColumn,
    TimeRemainingColumn, TimeElapsedColumn
)
from rich.console import Console

from opndb.services.terminal_printers import TerminalBase as t

console = Console()

class DataFrameMergers(DataFrameOpsBase):
    """
    Dataframe operations that merge multiple dataframes and handle post-merge cleanup (dropping duplicates, combining
    columns, etc.)
    """
    @classmethod
    def merge_orgs(
        cls,
        df_taxpayers: pd.DataFrame,
        df_orgs: pd.DataFrame,
        clean_core: str,
        string_match: bool = False
    ) -> pd.DataFrame:
        right_on: str = f"entity_{string_match}_name" if not string_match else "entity_string_match"
        df_merge = pd.merge(
            df_taxpayers,
            df_orgs,
            how="left",
            left_on=f"{clean_core}_name",
            right_on=right_on,
        )
        df_merge_clean = cls.combine_columns_parallel(df_merge)
        return df_merge_clean

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

    @classmethod
    def merge_validated_address(cls, df: pd.DataFrame, df_addrs: pd.DataFrame, addr_col: str) -> pd.DataFrame:
        """Merges validated addresses into specified address column in df."""
        # print("DF COLS:")
        # for col in df.columns:
        #     print(col)
        # print("DF ADDR COLS:")
        # for col in df_addrs.columns:
        #     print(col)
        # print("ADDR COL:", addr_col)
        df_merged: pd.DataFrame = pd.merge(
            df,
            df_addrs[["clean_address", "formatted_address_v"]],
            how="left",
            left_on=addr_col,
            right_on="clean_address"
        )
        # print("DF MERGED COLS:")
        # for col in df_merged.columns:
        #     print(col)
        df_merged.rename(columns={"formatted_address": f"{addr_col}_v"}, inplace=True)
        return df_merged


class DataFrameColumnGenerators(DataFrameOpsBase):
    """Dataframe operations that generate new columns."""
    @classmethod
    def set_is_rental_initial(cls, df_props: pd.DataFrame, df_class_codes: pd.DataFrame) -> pd.DataFrame:
        """
        Subsets property dataframe to only include properties associated with rental class codes. Handles situations in
        which multiple class codes are associated with a single property and are separated by commas.
        :param df_props: Dataframe containing entire property taxpayer dataset
        :param df_class_codes: Dataframe containing building class code descriptions
        :return: D
        """
        df_rental_codes: pd.DataFrame = df_class_codes[df_class_codes["is_rental"] == "t"]
        rental_codes: list[str] = list(df_rental_codes["code"])
        df_props["is_rental"] = df_props["class_code"].apply(lambda code: code in rental_codes)
        return df_props

    @classmethod
    def set_is_rental_final(cls, df_props: pd.DataFrame, nonrental_records: list[str]) -> pd.DataFrame:
        # Find rows where "raw_name_address" is in nonrental_records
        mask = df_props["raw_name_address"].isin(nonrental_records)
        # Set "is_rental" column to True for matching rows
        df_props.loc[mask, "is_rental"] = True
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
    def set_is_landlord_org(cls, df_taxpayers: pd.DataFrame, df_addr_analysis: pd.DataFrame) -> pd.DataFrame:
        """
        Adds is_landlord_org boolean column to dataframe. Landlord orgs are identified manually by user and stored in
        address_analysis dataset.

        :param df_taxpayers: Dataframe containing taxpayer record data
        :param df_addr_analysis: Dataframe containing address analysis spreadsheet
        """
        df_orgs: pd.DataFrame = df_addr_analysis[df_addr_analysis["is_landlord_org"] == "t"]
        org_addrs: list[str] = list(df_orgs["value"])
        df_taxpayers["is_landlord_org"] = df_taxpayers["clean_address"].apply(
            lambda addr: clean_base.get_is_landlord_org(addr, org_addrs)
        )
        return df_taxpayers

    @classmethod
    def set_is_common_name(cls, df_taxpayers: pd.DataFrame, df_freq_names: pd.DataFrame) -> pd.DataFrame:
        """
        Adds is_common_name boolean column to dataframe. Common names are currently obtained from frequent_tax_names
        dataset manually inputted.

        :param df_taxpayers: Dataframe containing taxpayer record data
        :param df_freq_names: Dataframe containing name analysis spreadsheet
        """
        df_common: pd.DataFrame = df_freq_names[df_freq_names["is_common_name"] == "t"]
        common_names: list[str] = list(df_common["value"])
        df_taxpayers["is_common_name"] = df_taxpayers["clean_name"].apply(
            lambda name: clean_base.get_is_common_name(name, common_names)
        )
        return df_taxpayers

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
        df = df.fillna({col: ""})
        df["is_pobox"] = df[col].apply(lambda addr: clean_base.get_is_pobox(addr))
        return df

    @classmethod
    def set_name_address_concat(cls, df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
        """
        Returns concatenation of taxpayer name and address.

        :param df: Dataframe to add column to
        :param col_map: Map of name to address columns
        """
        df[col_map["name_addr"]] = df[col_map["name"]] + " -- " + df[col_map["addr"]]
        return df

    @classmethod
    def set_full_address_fields(cls, df: pd.DataFrame, raw_address_map, id: str) -> pd.DataFrame:
        if raw_address_map is None:
            return df
        llc: bool = False
        if id == "llcs":
            llc: bool = True
        for map in raw_address_map:
            df[map["full_address"]] = df.apply(
                lambda row: addr.get_full_address(
                    row,
                    map["address_cols"],
                    llc
                ), axis=1
            )
        return df

    @classmethod
    def set_raw_columns(cls, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        # todo: this will have to be handled more gracefully to account for optional columns
        for col in cols:
            if col[:3] == "tax":  # exception for props_taxpayer records fields
                df[f"raw_{col[4:]}"] = df[col].copy()
            else:
                df[f"raw_{col}"] = df[col].copy()
        return df

    @classmethod
    def set_check_sec_num(cls, df: pd.DataFrame, addr_col: str) -> pd.DataFrame:
        """
        Adds "check_sec_num" column to dataframe, containing any numbers found at the end of the street address
        component of clean addresses. Returns dataframe subset that excludes PO box addresses.
        """
        df["check_sec_num"] = df[addr_col].apply(lambda address: clean_addr.check_sec_num(address))
        return df[df["is_pobox"] == "False"]

    @classmethod
    def set_formatted_address_v(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["formatted_address_v"] = df.apply(lambda row: addr.get_formatted_address_v(row), axis=1)
        return df


class DataFrameColumnManipulators(DataFrameOpsBase):
    """Dataframe operations that manipulate or transform an existing dataframe column."""

    @classmethod
    def fix_pobox(cls, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Cleans up PO box addresses for the list of specified columns in the dataframe.

        :param df: Dataframe to manipulate columns
        :param addr_cols: List of columns containing address strings to be cleaned
        """
        df[col] = df[col].apply(lambda clean_addr: addr.fix_pobox(clean_addr))
        return df

    @classmethod
    def fix_tax_names(cls, df: pd.DataFrame, banks: dict[str, str]) -> pd.DataFrame:
        """
        Checks cleaned taxpayer names against raw/standardized name pairs in banks dictionary. Returns taxpayer name
        with raw strings replaced with standardized.
        """
        df = df.apply(lambda row: clean_name.fix_banks(row, banks), axis=1)
        return df


class DataFrameSubsetters(DataFrameOpsBase):
    """Dataframe operations that return subsets."""
    @classmethod
    def get_duplicates(cls, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Finds duplicate values in the specified column and returns two dataframes:
        1. A dataframe containing only rows with duplicates in the specified column
        2. A dataframe with the duplicated rows removed and all other original rows
        Returns:
            tuple: (duplicate_rows, non_duplicate_rows)
        """
        # Find which values are duplicated
        duplicated_names = df[col][df[col].duplicated(keep=False)]
        # Create a mask for rows with duplicated values
        duplicate_mask = df[col].isin(duplicated_names)
        # Return both dataframes
        df_dups = df[duplicate_mask]
        df_non_dups = df[~duplicate_mask]
        return df_dups, df_non_dups


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
        return df[df[col].isin(["0", "1"])]

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

    @classmethod
    def split_props_taxpayers(cls, df: pd.DataFrame, props_cols: list[str], taxpayer_cols: list[str]):
        """
        Splits up the cleaned 'props_taxpayers' dataset into two separate datasets: 'taxpayer_records' and 'properties'
        """
        try:
            # pull out required columns for each final dataset
            # separate out properties dataset from props_taxpayers, handling for cases in which NUM_UNITS is missing
            df_props: pd.DataFrame = df[props_cols].copy()
            df_taxpayers: pd.DataFrame = df[taxpayer_cols].copy()
            # rename columns with clean_ prefix
            df_taxpayers.rename(columns={
                pt.TAX_NAME: tr.CLEAN_NAME,
                pt.TAX_ADDRESS: tr.CLEAN_ADDRESS,
                pt.TAX_STREET: tr.CLEAN_STREET,
                pt.TAX_CITY: tr.CLEAN_CITY,
                pt.TAX_STATE: tr.CLEAN_STATE,
                pt.TAX_ZIP: tr.CLEAN_ZIP,
            })
            df_taxpayers.drop_duplicates(subset=[tr.RAW_NAME_ADDRESS], inplace=True)
            return df_props, df_taxpayers
        except KeyError as e:
            print("KEY ERROR")
            for col in df.columns:
                print(col)
            raise

    @classmethod
    def generate_unvalidated_df(cls, dfs: dict[str, pd.DataFrame], col_map: dict[str, Any]) -> pd.DataFrame:
        """
        Initial generator for unvalidated addresses. Extracts all address columns from datasets, concatenates into a
        single dataframe, drops duplicates. Called at the end of the initial data cleaning workflow.

        :param dfs: Dictionary mapping dataset IDs to dataframes. Required datasets: taxpayer_records, corps, llcs
        :params col_map: Object mapping dataset IDs to required columns for outputted dataframes
        """
        dfs_to_concat: list[pd.DataFrame] = []
        for id, df in dfs.items():
            if id == "class_codes":
                continue
            elif id == "properties":
                continue
            elif id == "taxpayer_records":
                df = df[col_map[id]]
                df["type"] = id
                df["status"] = None
                df["active"] = None
                dfs_to_concat.append(df)
            else:
                for addr_cols in col_map[id]:  # col_map[id]: list[dict[str, str]]
                    df_addr = df[addr_cols.keys()].copy()
                    df_addr["type"] = id
                    df_addr["active"] = df_addr["status"].isin(["0", "1"])
                    df_addr.rename(columns=addr_cols, inplace=True)
                    dfs_to_concat.append(df_addr)
        df_out: pd.DataFrame = pd.concat(dfs_to_concat, ignore_index=True)
        df_out.dropna(subset=[ua.RAW_ADDRESS], inplace=True)
        df_out.drop_duplicates(subset=[tr.RAW_ADDRESS], inplace=True)
        return df_out

    @classmethod
    def remove_unvalidated_addrs(
        cls,
        df_unvalidated: pd.DataFrame,
        addrs_to_remove: list[str]
    ) -> pd.DataFrame:
        """
        Removes validated addresses from the unvalidated master address list by filtering existing unvalidated address
        dataset by clean_address.
        """
        return df_unvalidated[~df_unvalidated["clean_address"].isin(addrs_to_remove)]

    @classmethod
    def generate_frequency_df(cls, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Returns frequency dataframe with two columns: unique values contained in the column "col" (function parameter)
        and their frequency count.

        :param df: Input DataFrame to analyze
        :param col: Column name to calculate frequencies for
        :return: DataFrame with values and their frequencies, sorted by frequency in descending order
        """
        counts = df[col].value_counts().to_dict()
        freq_df = pd.DataFrame({
            "value": list(counts.keys()),
            "frequency": list(counts.values())
        })
        freq_df = freq_df.sort_values("frequency", ascending=False).reset_index(drop=True)
        return freq_df

class DataFrameDeduplicators(DataFrameOpsBase):

    @classmethod
    def drop_dups_corps_llcs(cls, id: str, df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
        """
        Drop duplicates with priority given to:
        1. Rows with non-null values for most columns
        2. Rows with status="0" or status="1" if status values differ

        Args:
            df: Input dataframe, containing ONLY rows with duplicate values for clean_name
            subset: Columns to consider for identifying duplicates

        Returns:
            Dataframe with duplicates removed according to priority rules
        """
        result_df = pd.DataFrame()

        # Group by the subset columns that identify duplicates
        groups = df.groupby(subset)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            "•",
            TextColumn("[bold cyan]{task.fields[processed]}/{task.total} records"),
        ) as progress:
            task = progress.add_task(
                f"[yellow]Processing {id} records...",
                total=len(groups),
                processed=0,
            )
            processed_count = 0
            for _, group in groups:
                # Start with an empty Series to build our merged row
                merged_row = pd.Series(index=df.columns, dtype="object")
                # Fill in the subset values (these should be identical within the group)
                for col in subset:
                    merged_row[col] = group[col].iloc[0]
                # Handle status column specially
                if "status" in df.columns:
                    # Priority for status is "0" or "1" if present
                    if group["status"].isin(["0", "1"]).any():
                        # Take "0" or "1", preferring "0" if both exist
                        if '0' in group["status"].values:
                            merged_row["status"] = "0"
                        else:
                            merged_row["status"] = "1"
                    else:
                        # If no "0" or "1", take the first non-null value
                        merged_row["status"] = group["status"].dropna().iloc[0] if not group[
                            "status"].isna().all() else None
                # For all other columns, take the first non-null value
                for col in [c for c in df.columns if c != "status" and c not in subset]:
                    non_null_values = group[col].dropna()
                    if len(non_null_values) > 0:
                        merged_row[col] = non_null_values.iloc[0]
                    else:
                        merged_row[col] = None
                # Add the merged row to the result
                result_df = pd.concat([result_df, pd.DataFrame([merged_row])], ignore_index=True)
                processed_count += 1
                progress.update(
                    task,
                    advance=1,
                    processed=processed_count,
                    description=f"[yellow]Processing record {processed_count}/{len(groups)}"
                )
        return result_df


class DataFrameConcatenators(DataFrameOpsBase):

    @classmethod
    def get_merge_address(cls, row: pd.Series, addr_col: str) -> pd.Series:
        if pd.notnull(row[f"{addr_col}_v"]):
            return row[f"{addr_col}_v"]
        else:
            return row[addr_col]

    @classmethod
    def combine_corps_llcs(cls, df_corps: pd.DataFrame, df_llcs: pd.DataFrame) -> pd.DataFrame:
        # rename address cols to address_1, address_2, address_3
        df_corps["merge_address_1"] = df_corps.apply(
            lambda row: cls.get_merge_address(row, "clean_president_address"), axis=1
        )
        df_corps["merge_address_2"] = df_corps.apply(
            lambda row: cls.get_merge_address(row, "clean_secretary_address"), axis=1
        )
        df_llcs["merge_address_1"] = df_llcs.apply(
            lambda row: cls.get_merge_address(row, "clean_office_address"), axis=1
        )
        df_llcs["merge_address_2"] = df_llcs.apply(
            lambda row: cls.get_merge_address(row, "clean_manager_member_address"), axis=1
        )
        df_llcs["merge_address_3"] = df_llcs.apply(
            lambda row: cls.get_merge_address(row, "clean_agent_address"), axis=1
        )
        # concatenate, take slice of only necessary columns
        df_corps = df_corps[["clean_name", "core_name", "merge_address_1", "merge_address_2"]]
        df_llcs = df_llcs[["clean_name", "core_name", "merge_address_1", "merge_address_2", "merge_address_3"]]
        df_out: pd.DataFrame = pd.concat([df_corps, df_llcs], ignore_index=True)
        df_out.rename(columns={"clean_name": "entity_clean_name"})
        df_out.rename(columns={"core_name": "entity_core_name"})
        return df_out
