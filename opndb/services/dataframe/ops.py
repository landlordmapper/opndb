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
from opndb.services.address import AddressBase as addr, AddressBase
from opndb.services.dataframe.base import DataFrameOpsBase
from opndb.services.match import MatchBase, StringMatch
from opndb.services.string_clean import (
    CleanStringBase as clean_base,
    CleanStringName as clean_name,
    CleanStringAddress as clean_addr,
    CleanStringAccuracy as clean_acc, safe_string_cleaner
)
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn, TimeElapsedColumn
)
from rich.console import Console

from opndb.services.terminal_printers import TerminalBase as t
from opndb.utils import UtilsBase

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
        right_on: str
    ) -> pd.DataFrame:
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
        df_merged: pd.DataFrame = pd.merge(
            df,
            df_addrs[[
                "clean_address",
                "formatted_address_v0",
                "formatted_address_v1",
                "formatted_address_v2",
                "formatted_address_v3",
                "formatted_address_v4",
            ]],
            how="left",
            left_on=addr_col,
            right_on="clean_address"
        )
        df_merged.rename(columns={
            "formatted_address_v0": f"{addr_col}_v0",
            "formatted_address_v1": f"{addr_col}_v1",
            "formatted_address_v2": f"{addr_col}_v2",
            "formatted_address_v3": f"{addr_col}_v3",
            "formatted_address_v4": f"{addr_col}_v4",
        }, inplace=True)
        return df_merged

    @classmethod
    def merge_proptax_match(
        cls,
        df_taxpayers: pd.DataFrame,
        df_props: pd.DataFrame,
        df_prop_addrs: pd.DataFrame,
        df_valid_addrs: pd.DataFrame,
    ) -> pd.DataFrame:
        df_props_addrs = pd.merge(df_props, df_prop_addrs, how="left", on="pin")
        df_props_addrs_taxpayers = pd.merge(df_props_addrs, df_taxpayers, how="left", on="raw_name_address")
        df_merge_final = pd.merge(
            df_props_addrs_taxpayers,
            df_valid_addrs,
            how="left",
            left_on="clean_address",
            right_on="clean_address_validated"
        )
        df_merge_final.drop_duplicates(subset="pin", inplace=True)
        return df_merge_final


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
        df_rental_codes: pd.DataFrame = df_class_codes[df_class_codes["is_rental"] == True]
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
        df_orgs: pd.DataFrame = df_addr_analysis[df_addr_analysis["is_landlord_org"] == True]
        org_addrs: list[str] = list(df_orgs["value"])
        df_taxpayers["is_landlord_org"] = df_taxpayers["raw_address_v"].apply(
            lambda addr: clean_base.get_is_landlord_org(addr, org_addrs)
        )
        return df_taxpayers

    @classmethod
    def set_exclude_name(cls, df_taxpayers: pd.DataFrame, df_freq_names: pd.DataFrame) -> pd.DataFrame:
        """
        Adds exclude_name boolean column to dataframe. Common names are currently obtained from frequent_tax_names
        dataset manually inputted.

        :param df_taxpayers: Dataframe containing taxpayer record data
        :param df_freq_names: Dataframe containing name analysis spreadsheet
        """
        df_common: pd.DataFrame = df_freq_names[df_freq_names["exclude_name"] == True]
        names_to_exclude: list[str] = list(df_common["value"])
        df_taxpayers["exclude_name"] = df_taxpayers["clean_name"].apply(
            lambda name: clean_base.get_exclude_name(name, names_to_exclude)
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
        def get_name_address(row: pd.Series) -> str:
            if col_map["name_2"] in row.keys() and pd.notnull(row[col_map["name_2"]]):
                return row[col_map["name"]] + " -- " + row[col_map["name_2"]] + " -- " + row[col_map["addr"]]
            else:
                return row[col_map["name"]] + " -- " + row[col_map["addr"]]
        df[col_map["name_addr"]] = df.apply(lambda row: get_name_address(row), axis=1)
        return df

    @classmethod
    def set_name_address_concat_fix(cls, df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
        """
        Returns concatenation of taxpayer name and address.

        :param df: Dataframe to add column to
        :param col_map: Map of name to address columns
        """
        df[col_map["name_addr"]] = df.apply(lambda row: row[col_map["name"]] + " -- " + row[col_map["addr"]], axis=1)
        return df

    @classmethod
    def set_full_address_fields(cls, df: pd.DataFrame, raw_address_map, id: str | None = None) -> pd.DataFrame:
        if raw_address_map is None:
            return df
        llc: bool = False
        if id and id == "llcs":
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
        return df[df["is_pobox"] == False]

    @classmethod
    def set_formatted_address_v0(cls, df: pd.DataFrame) -> pd.DataFrame:
        # All fields included (full address)
        df["formatted_address_v0"] = df.apply(lambda row: addr.format_address(row), axis=1)
        return df

    @classmethod
    def set_formatted_address_v1(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Omit suite prefixes only (e.g. keep "403" but drop "APT")
        df["formatted_address_v1"] = df.apply(
            lambda row: addr.format_address(
                row,
                include_suite_prefix=False
            ),
            axis=1
        )
        return df

    @classmethod
    def set_formatted_address_v2(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Omit all suite info
        df["formatted_address_v2"] = df.apply(
            lambda row: addr.format_address(
                row,
                include_suite_prefix=False,
                include_suite_number=False
            ),
            axis=1)
        return df

    @classmethod
    def set_formatted_address_v3(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Omit directionals, keep suite info
        df["formatted_address_v3"] = df.apply(
            lambda row: addr.format_address(row, include_directionals=False), axis=1)
        return df

    @classmethod
    def set_formatted_address_v4(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Omit directionals and suite info
        df["formatted_address_v4"] = df.apply(lambda row: addr.format_address(row, include_directionals=False, include_suite_prefix=False, include_suite_number=False), axis=1)
        return df


    @classmethod
    def concatenate_name_addr(cls, df: pd.DataFrame, name_col: str, addr_col: str, suffix: str = "") -> pd.DataFrame:
        """Generates column with name and address concatenated. Used for string matching workflow."""
        df[f"{name_col}_address{suffix}"] = df[name_col] + " - " + df[addr_col]
        return df

    @classmethod
    def set_match_address_t(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["match_address_t"] = df.apply(
            lambda row: MatchBase.set_matching_address(row), axis=1
        )
        return df

    @classmethod
    def set_match_address(cls, df: pd.DataFrame, address_col: str, suffix: str = "") -> pd.DataFrame:
        """
        Sets match_address column for taxpayer and business filing records to be used in string matching and network
        graph generation. Returns validated address if one exists, and cleaned unvalidated address if otherwise. Set
        suffix to specify which address to set (e.g. clean_address_v1, clean_address_v2, etc.)
        """
        def get_match_address(row: pd.Series) -> str:
            if pd.notnull(row[address_col]):
                return row[address_col]
            else:
                return row["clean_address"]
        df[f"match_address{suffix}"] = df.apply(lambda row: get_match_address(row), axis=1)
        return df

    @classmethod
    def set_exclude_address(
        cls,
        exclude_addrs: list[str],
        df: pd.DataFrame,
        match_col: str,
        suffix: str = ""
    ) -> pd.DataFrame:
        df[f"exclude_address{suffix}"] = df[match_col].apply(lambda addr: addr in exclude_addrs)
        return df

    @classmethod
    def set_is_researched(
        cls,
        researched_addrs: list[str],
        df: pd.DataFrame,
        match_col: str,
        suffix: str = ""
    ) -> pd.DataFrame:
        df[f"is_researched{suffix}"] = df[match_col].apply(lambda addr: addr in researched_addrs)
        return df

    @classmethod
    def set_is_org_address(
        cls,
        org_addrs: list[str],
        df: pd.DataFrame,
        match_col: str,
        suffix: str = ""
    ) -> pd.DataFrame:
        df[f"is_org_address{suffix}"] = df[match_col].apply(lambda addr: addr in org_addrs)
        return df

    @classmethod
    def set_is_validated(
        cls,
        df: pd.DataFrame,
        valid_col: str,
        suffix: str = ""
    ) -> pd.DataFrame:
        df[f"is_validated{suffix}"] = df[valid_col].apply(
            lambda addr: pd.notnull(addr)
        )
        return df

    @classmethod
    def set_is_missing_suite(
        cls,
        missing_suite_addrs: list[str],
        df: pd.DataFrame,
        valid_col: str,
        suffix: str = ""
    ) -> pd.DataFrame:
        df[f"is_missing_suite{suffix}"] = df[valid_col].apply(lambda addr: addr in missing_suite_addrs)
        return df

    @classmethod
    def set_is_problem_suite(
        cls,
        problem_suite_addrs: list[str],
        df: pd.DataFrame,
        valid_col: str,
        suffix: str = ""
    ) -> pd.DataFrame:
        df[f"is_problem_suite{suffix}"] = df[valid_col].apply(lambda addr: addr in problem_suite_addrs)
        return df

    @classmethod
    def set_is_realtor(
        cls,
        realtor_addrs: list[str],
        df: pd.DataFrame,
        valid_col: str,
        suffix: str = ""
    ) -> pd.DataFrame:
        df[f"is_realtor{suffix}"] = df[valid_col].apply(lambda addr: addr in realtor_addrs)
        return df

    @classmethod
    def set_is_incomplete_address_mnsos(cls, row: pd.Series) -> bool:
        """MNSOS-specific boolean generator indicating whether an address is complete or not."""
        # todo: standardize this
        # todo: have this follow the pattern of returning dataframes
        if pd.isnull(row["street_1"]):
            return True
        elif pd.isnull(row["zip_code"]):
            return True
        else:
            return False

    @classmethod
    def concatenate_addr_mpls(cls, row):
        if pd.notnull(row["clean_city"]) and pd.notnull(row["clean_state"]) and pd.notnull(row["clean_zip_code"]):
            return f"{row["clean_street"]}, {row["clean_city"]}, {row["clean_state"]} {row[f"clean_zip_code"]}"
        else:
            return f"{row['clean_street']}, {row['clean_city_state_zip']}"

    @classmethod
    def concatenate_addr_mnsos(cls, row, prefix: str = "") -> str | float:
        """MNSOS-specific address full address concatenator."""
        # todo: standardize this
        if row["is_incomplete_address"] == True:
            return np.nan

        street_1: str = row[f"{prefix}street_1"]
        street_2: str | float = row[f"{prefix}street_2"]
        city: str | float = row[f"{prefix}city"]
        state: str | float = row[f"{prefix}state"]
        zip_code: str = row[f"{prefix}zip_code"]

        address: str = ""
        if pd.notnull(street_2):
            address += f"{street_1} {street_2}"
        else:
            address += street_1
        address += ", "
        if pd.notnull(city):
            address += f"{city}, "
        if pd.notnull(state):
            address += f"{state} "
        address += zip_code

        return address

    @classmethod
    def set_city_state_zip(cls, df: pd.DataFrame) -> pd.DataFrame:

        def parse_city_state_zip(row: pd.Series) -> pd.Series:
            """
            Parses single field for city_state_zip. If the state and zip code values are valid, return row with city, state
            and zip fields populated. If not, returns row as-is.
            """
            city_state_zip = row["clean_city_state_zip"].split()
            if len(city_state_zip) > 1:
                zip_code = city_state_zip[-1]
                state = city_state_zip[-2]
                city = " ".join(city_state_zip[:-2])
                if not AddressBase.is_irregular_zip(zip_code) and not AddressBase.is_irregular_state(state):
                    row["clean_city"] = city
                    row["clean_state"] = state
                    row["clean_zip_code"] = zip_code
            return row

        df["clean_city"] = np.nan
        df["clean_state"] = np.nan
        df["clean_zip_code"] = np.nan
        df = df.apply(lambda row: parse_city_state_zip(row), axis=1)

        return df

    @classmethod
    def set_clean_street_mnsos(cls, row: pd.Series) -> str:
        street_1: str | float = row[f"clean_street_1"]
        street_2: str | float = row[f"clean_street_2"]
        if pd.notnull(street_1) and pd.notnull(street_2):
            return f"{street_1} {street_2}"
        else:
            return str(street_1)

    # GEOCODIO STRING MATCH PROCESSING COLUMNS
    @classmethod
    def is_match_street(cls, row: pd.Series) -> bool:
        street_num_original = row["original_doc"].split()[0]
        street_num_match = row["matched_doc"].split()[0]
        return street_num_original.strip() == street_num_match.strip()

    @classmethod
    def is_match_zip(cls, row: pd.Series) -> bool:
        zip_original = row["original_doc"].split()[-1]
        zip_match = row["matched_doc"].split()[-1]
        return zip_original.strip() == zip_match.strip()

    @classmethod
    def is_match_secondary(cls, row: pd.Series) -> bool | float:
        if pd.isnull(row["secondary_number"]):
            return np.nan
        sec_num = row["secondary_number"].strip()
        clean_addr_split = row["original_doc"].split(",")[0]
        street_split = clean_addr_split.split()
        return sec_num in street_split

    @classmethod
    def set_is_not_secondarynumber(cls, df: pd.DataFrame) -> pd.DataFrame:
        def get_is_not_secondarynumber(clean_addr: str) -> bool:
            clean_split: list[str] = clean_addr.split(",")
            addr_split: list[str] = clean_split[0].split()
            if UtilsBase.is_int(addr_split[-1]):
                if len(addr_split) > 1:
                    if addr_split[-2] in ["HWY", "HIGHWAY"]:
                        if " ".join(addr_split[-3:-1]) != "DUPONT HWY":
                            return True
                    if " ".join(addr_split[-3:-1]) in [
                        "CO RD",
                        "COUNTY RD",
                        "STATE RD",
                        "ST RD"
                    ]:
                        return True
            return False
        df["is_not_secondarynumber"] = df["clean_address"].apply(get_is_not_secondarynumber)
        return df

    @classmethod
    def set_is_invalid_street(cls, df: pd.DataFrame) -> pd.DataFrame:
        def get_is_invalid_street(clean_addr: str) -> bool:
            clean_split: list[str] = clean_addr.split(",")
            if UtilsBase.is_int(clean_split[0]):
                return True
            return False
        df["is_invalid_street"] = df["clean_address"].apply(get_is_invalid_street)
        return df

    @classmethod
    def set_is_unit_gte_1(cls, df: pd.DataFrame) -> pd.DataFrame:
        def is_gte_1(val):
            if UtilsBase.is_int(val):
                val_int = int(val)
                return val_int > 1
            return False
        df["is_unit_gte_1"] = df["num_units"].apply(lambda unit: is_gte_1(unit))
        return df

    @classmethod
    def set_is_match_number_street_zip(cls, df: pd.DataFrame) -> pd.DataFrame:
        def get_is_match_number_street_zip(row: pd.Series) -> bool:
            prop_number = row["number"]
            prop_street = row["street"]
            prop_zip = row["zip_code"]
            if pd.notnull(row["clean_address_v1"]):
                tax_number = row["number_validated"]
                tax_street = row["street_validated"]
                tax_zip = row["zip_code_validated"]
            else:
                tax_addr = row["clean_address"].split()
                tax_number = tax_addr[0]
                tax_street = tax_addr[1]
                tax_zip = tax_addr[-1]
            if prop_number != tax_number:
                return False
            elif prop_street != tax_street:
                return False
            elif tax_zip != prop_zip:
                return False
            else:
                return True
        df["is_match_number_street_zip"] = df.apply(lambda row: get_is_match_number_street_zip(row), axis=1)
        return df

    @classmethod
    def set_is_match_number_zip(cls, df: pd.DataFrame) -> pd.DataFrame:
        def get_is_match_number_zip(row: pd.Series) -> bool | float:
            if row["is_match_number_street_zip"] == True:
                return np.nan
            prop_number = row["number"]
            prop_zip = row["zip_code"]
            if pd.notnull(row["clean_address_v1"]):
                tax_number = row["number_validated"]
                tax_zip = row["zip_validated"]
            else:
                tax_addr = row["clean_address"].split()
                tax_number = tax_addr[0]
                tax_zip = tax_addr[-1]
            if prop_number != tax_number:
                return False
            elif tax_zip != prop_zip:
                return False
            else:
                return True
        df["is_match_number_zip"] = df.apply(lambda row: get_is_match_number_zip(row), axis=1)
        return df

    @classmethod
    def set_match_conf(cls, df: pd.DataFrame) -> pd.DataFrame:
        def get_match_conf(row: pd.Series) -> str | float:
            if row["is_match_number_street_zip"] == True:
                return np.nan
            # address missing for that property - can't check similarity with tax address
            if pd.isnull(row["formatted_address_prop"]):
                return np.nan
            # if it's not a number or zip match, it's definitively not a match
            if row["is_match_number_zip"] == False:
                return np.nan
            if pd.notnull(row["clean_address_v1"]):
                return StringMatch.test_string_similarity(row["clean_address_v1"], row["formatted_address_prop"])
            else:
                return StringMatch.test_string_similarity(row["clean_address"], row["formatted_address_prop"])
        df["match_conf"] = df.apply(lambda row: get_match_conf(row), axis=1)
        return df

    @classmethod
    def set_is_match(cls, df: pd.DataFrame) -> pd.DataFrame:
        def get_is_match(row: pd.Series) -> bool:
            if row["is_match_number_street_zip"] == True:
                return True
            if row["is_match_number_zip"] == True and row["match_conf"] > .9:
                return True
            return False
        df["is_match"] = df.apply(lambda row: get_is_match(row), axis=1)
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

    @classmethod
    def set_business_names_taxpayers(cls, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["entity_clean_name"].notna()
        df.loc[mask, "clean_name"] = df.loc[mask, "entity_clean_name"]
        df.loc[mask, "core_name"] = df.loc[mask, "entity_core_name"]
        return df

    @classmethod
    def fix_corp_llc_exclude_names(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fixes rows whose taxpayer name should be excluded from matching in case they mistakenly got matched by corp/LLC
        match workflows.
        """
        cols_to_set = [
            "entity_clean_name",
            "entity_core_name",
            "entity_address_1",
            "entity_address_2",
            "entity_address_3"
        ]
        mask = df["exclude_name"] == True
        for col in cols_to_set:
            df.loc[mask, col] = np.nan
        return df

    @classmethod
    def fix_states(cls, state_raw: str, state_fixer: dict[str, str]) -> str:
        """
        Fixes state values in raw addresses.

        :param state_raw: raw value for the state component of the address being processed
        :param state_fixer: state fixer object mapping raw messy state values to their cleaned and corrected values.
        """
        if state_raw in state_fixer.keys():
            return state_fixer[state_raw]
        if state_raw in AddressBase.STATES_ABBREVS.keys():
            return AddressBase.STATES_ABBREVS[state_raw]
        return state_raw


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
            # df_taxpayers.rename(columns={
            #     pt.TAX_NAME: tr.CLEAN_NAME,
            #     pt.TAX_ADDRESS: tr.CLEAN_ADDRESS,
            #     pt.TAX_STREET: tr.CLEAN_STREET,
            #     pt.TAX_CITY: tr.CLEAN_CITY,
            #     pt.TAX_STATE: tr.CLEAN_STATE,
            #     pt.TAX_ZIP: tr.CLEAN_ZIP,
            # })
            df_taxpayers.drop_duplicates(subset=["raw_name_address"], inplace=True)
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
                df["origin"] = id
                df["status"] = None
                df["active"] = None
                dfs_to_concat.append(df)
            else:
                for addr_cols in col_map[id]:  # col_map[id]: list[dict[str, str]]
                    df_addr = df[addr_cols.keys()].copy()
                    df_addr["addr_type"] = id
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
    def get_entity_match_address(cls, row: pd.Series, addr_col: str) -> pd.Series:
        """
        Checks whether a validated address exists for the specified address column passed as a parameter. If it DOES
        exist, it returns it. If it does NOT exist, it returns the raw address.
        """
        if pd.notnull(row[f"{addr_col}_v"]):
            return row[f"{addr_col}_v"]
        else:
            return row[addr_col]

    @classmethod
    def combine_corps_llcs(cls, df_corps: pd.DataFrame, df_llcs: pd.DataFrame) -> pd.DataFrame:
        # rename address cols to address_1, address_2, address_3
        df_corps["match_address_e1"] = df_corps.apply(
            lambda row: cls.get_entity_match_address(row, "raw_president_address"), axis=1
        )
        df_corps["match_address_e2"] = df_corps.apply(
            lambda row: cls.get_entity_match_address(row, "raw_secretary_address"), axis=1
        )
        df_llcs["match_address_e1"] = df_llcs.apply(
            lambda row: cls.get_entity_match_address(row, "raw_office_address"), axis=1
        )
        df_llcs["match_address_e2"] = df_llcs.apply(
            lambda row: cls.get_entity_match_address(row, "raw_manager_member_address"), axis=1
        )
        df_llcs["match_address_e3"] = df_llcs.apply(
            lambda row: cls.get_entity_match_address(row, "raw_agent_address"), axis=1
        )

        df_corps.rename(columns={
            "raw_president_address": "entity_address_1",
            "raw_president_address_v": "entity_address_1_v",
            "raw_secretary_address": "entity_address_2",
            "raw_secretary_address_v": "entity_address_2_v",
        }, inplace=True)
        df_llcs.rename(columns={
            "raw_office_address": "entity_address_1",
            "raw_office_address_v": "entity_address_1_v",
            "raw_manager_member_address": "entity_address_2",
            "raw_manager_member_address_v": "entity_address_2_v",
            "raw_agent_address": "entity_address_3",
            "raw_agent_address_v": "entity_address_3_v",
        }, inplace=True)

        # add column to ID origin
        df_llcs["origin"] = "llc"
        df_corps["origin"] = "corp"
        # concatenate, take slice of only necessary columns
        df_corps = df_corps[[
            "origin",
            "clean_name",
            "core_name",
            "entity_address_1",
            "entity_address_1_v",
            "match_address_e1",
            "entity_address_2",
            "entity_address_2_v",
            "match_address_e2",
        ]]
        df_llcs = df_llcs[[
            "origin",
            "clean_name",
            "core_name",
            "entity_address_1",
            "entity_address_1_v",
            "match_address_e1",
            "entity_address_2",
            "entity_address_2_v",
            "match_address_e2",
            "entity_address_3",
            "entity_address_3_v",
            "match_address_e3",
        ]]
        df_out: pd.DataFrame = pd.concat([df_corps, df_llcs], ignore_index=True)
        df_out.rename(columns={"clean_name": "entity_clean_name"}, inplace=True)
        df_out.rename(columns={"core_name": "entity_core_name"}, inplace=True)
        return df_out


class DataFrameCellShifters(DataFrameOpsBase):

    @classmethod
    def shift_taxpayer_data_cells(cls, row: pd.Series) -> pd.Series:
        """Hennepin County-specific function to shift messy taxpayer name data."""
        if pd.isnull(row["taxpayer_3"]):
            row["taxpayer_3"] = row["taxpayer_2"]
            row["taxpayer_2"] = row["taxpayer_1"]
            row["taxpayer_1"] = np.nan
        return row

    @classmethod
    def shift_street_addrs(cls, row: pd.Series, prefix: str = "") -> pd.Series:
        """MNSOS-specific function shifting missing address data cells."""
        street_1: str | float = row[f"{prefix}street_1"]
        street_2: str | float = row[f"{prefix}street_2"]
        if pd.isnull(street_1) and pd.notnull(street_2):
            row[f"{prefix}street_1"] = street_2
            row[f"{prefix}street_2"] = np.nan
        return row
