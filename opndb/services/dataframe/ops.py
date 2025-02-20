from pathlib import Path
from typing import Any, Callable, Type

import pandas as pd

from opndb.constants.columns import (
    PropsTaxpayers as pt,
    TaxpayerRecords as tr,
    ValidatedAddrs as va,
    ClassCodes as cc,
    Properties as p,
    UnvalidatedAddrs as ua
)
from opndb.services.address import AddressBase
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
    TimeRemainingColumn
)
from rich.console import Console


console = Console()

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

    @classmethod
    def set_name_address_concat(cls, df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
        df[col_map["name_addr"]]: pd.DataFrame = df.apply(
            lambda row: f"{row[col_map['name']]} -- {row[col_map['addr']]}", axis=1
        )
        return df

    @classmethod
    def set_full_address_fields(
        cls,
        df: pd.DataFrame,
        full_addr_fields: dict[str, str],
        raw_clean_prefix: str | None = None,
    ) -> pd.DataFrame:
        """
        Concatenates address component fields into a single string and adds as new column to the dataframe. Dynamically
        sets field names based on which full address is being calculated (taxpayer vs corp president vs LLC agent etc.)
        and based on whether the raw or clean address is being concatenated.

        EXAMPLE:
            - TAX_STREET: 123 Oak St
            - TAX_CITY: Chicago
            - TAX_STATE: IL
            - TAX_ZIP: 12345
            - TAX_ADDRESS (output of AddressBase.get_full_address()): "123 Oak St, Chicago, IL, 12345"

        :param df: Dataframe to add column to
        :param full_addr_fields: Dictionary whose keys are the names of the full address fields for the specific dataset (ex: "tax_address", "president_address", etc.), and whose values are used to prefix the column names (ex: "tax", "president", etc.).
        :param raw_clean_prefix: Optional address column prefix, either "raw" or "clean" (ex: "raw" > "raw_tax_address")
        """
        for field in full_addr_fields.keys():
            field_prefixed: str = f"{raw_clean_prefix}_{field}" if raw_clean_prefix else field
            if field_prefixed not in df.columns:
                df = df.apply(
                    lambda row: AddressBase.get_full_address(
                        row,
                        list(df.columns),
                        full_addr_fields[field],
                        raw_clean_prefix
                    ), axis=1
                )
        return df

    @classmethod
    def set_raw_columns(cls, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            if col[:3] == "tax":  # exception for props_taxpayer records fields
                df[f"raw_{col[4:]}"] = df[col].copy()
            else:
                df[f"raw_{col}"] = df[col].copy()
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

    @classmethod
    def split_props_taxpayers(cls, df: pd.DataFrame, props_cols: list[str], taxpayer_cols: list[str]):
        """
        Splits up the cleaned 'props_taxpayers' dataset into two separate datasets: 'taxpayer_records' and 'properties'
        """
        df: pd.DataFrame = DataFrameColumnGenerators.set_name_address_concat(
            df,
            {
                "name_addr": p.CLEAN_NAME_ADDRESS,
                "name": pt.TAX_NAME,
                "addr": pt.TAX_ADDRESS,
            }
        )
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
                df.drop_duplicates(subset=[tr.RAW_ADDRESS], inplace=True)
                dfs_to_concat.append(df)
            else:
                for key in col_map[id].keys():
                    df = df[col_map[id][key].keys()]
                    df.rename(columns=col_map[id][key], inplace=True)
                    df.drop_duplicates(subset=[tr.RAW_ADDRESS], inplace=True)
                    dfs_to_concat.append(df)
        df_out: pd.DataFrame = pd.concat(dfs_to_concat, ignore_index=True)
        df_out.drop_duplicates(subset=[tr.RAW_ADDRESS], inplace=True)
        return df_out

