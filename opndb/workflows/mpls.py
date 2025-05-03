# 1. Standard library imports
import csv
import gc
import shutil
import sys
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from pprint import pprint
from typing import Any, ClassVar, Optional, Tuple, List, Type
import networkx as nx
import nmslib
import numpy as np
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TaskID
import pandera as pa
from itertools import product
import warnings

# 2. Third-party imports
import pandas as pd

# 3. Constants (these should have no dependencies on other local modules)
from opndb.constants.columns import (
    ValidatedAddrs as va,
    UnvalidatedAddrs as ua,
    TaxpayerRecords as tr,
    Corps as c,
    LLCs as l,
    Properties as p,
    PropsTaxpayers as pt, ValidatedAddrs
)
from opndb.constants.files import Raw as r, Dirs as d, Geocodio as g
from opndb.schema.v0_1.process import TaxpayerRecords, Properties, UnvalidatedAddrs, Geocodio, UnvalidatedAddrsClean, \
    Corps, LLCs, FixingAddrs, FixingTaxNames, AddressAnalysis, FrequentTaxNames, TaxpayersFixed, \
    TaxpayersStringMatched, TaxpayersMerged, TaxpayersSubsetted, CorpsMerged, LLCsMerged, TaxpayersPrepped, \
    TaxpayersNetworked, TaxpayerRecordsMN, PropertiesMN
from opndb.schema.v0_1.raw import (
    PropsTaxpayers,
    Corps as CorpsRaw,
    LLCs as LLCsRaw,
    ClassCodes, BusinessRecordsBase, PropsTaxpayersMN, BusinessFilings, BusinessNamesAddrs
)
from opndb.services.summary_stats import SummaryStatsBase as ss, SSDataClean, SSAddressClean, SSAddressGeocodio, \
    SSFixUnitsInitial, SSFixUnitsFinal, SSAddressMerge, SSNameAnalysisInitial, SSAddressAnalysisInitial, \
    SSAnalysisFinal, SSRentalSubset, SSCleanMerge, SSStringMatch, SSNetworkGraph, SSFinalOutput
from opndb.services.config import ConfigManager

# 4. Types (these should only depend on constants)
from opndb.types.base import (
    WorkflowConfigs,
    NmslibOptions,
    StringMatchParams,
    NetworkMatchParams,
    CleaningColumnMap,
    BooleanColumnMap, WorkflowStage, GeocodioReturnObject,
)

# 5. Utils (these should only depend on constants and types)
from opndb.utils import UtilsBase as utils, PathGenerators as path_gen

# 6. Services (these can depend on everything else)
from opndb.services.match import StringMatch, NetworkMatchBase, MatchBase
from opndb.services.address import AddressBase as addr
from opndb.services.terminal_printers import TerminalBase as t
from opndb.services.terminal_printers import TerminalInteract as ti
from opndb.services.dataframe.base import (
    DataFrameOpsBase as ops_df,
    DataFrameBaseCleaners as clean_df_base,
    DataFrameNameCleaners as clean_df_name,
    DataFrameAddressCleaners as clean_df_addr,
    DataFrameCleanersAccuracy as clean_df_acc, DataFrameOpsBase,
)
from opndb.services.dataframe.ops import (
    DataFrameMergers as merge_df,
    DataFrameSubsetters as subset_df,
    DataFrameColumnGenerators as cols_df,
    DataFrameColumnManipulators as colm_df,
    DataFrameDeduplicators as dedup_df,
    DataFrameConcatenators as concat_df,
    DataFrameCellShifters as shift_df
)
from rich.console import Console
from rich.prompt import Prompt

from opndb.validator.df_model import OPNDFModel

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
console = Console()

class WorkflowBase(ABC):
    """
    Base workflow class the controls execution of data processing tasks required for each stage of the opndb workflow.
    Each child class that inherits from WorkflowBase corresponds to the broader workflow stage.

    CHILD WORKFLOW REQUIREMENTS:
        - Required dataframes object: instance variable containing all required dataframes and their file paths
        - Execute method: executes data load, required logic and transformations, and saves outputs
    """
    def __init__(self, config_manager: ConfigManager):
        self.config_manager: ConfigManager = config_manager
        self.dfs_in: dict[str, pd.DataFrame | None] = {}
        self.dfs_out: dict[str, pd.DataFrame | None] = {}

    def load_dfs(self, load_map: dict[str, dict[str, Any]]) -> None:
        """
        Sets the self.dfs_in object. Sets keys as dataframe ID values. Sets values to dataframes, or None if the file
        path specified is not found. Sets schema_map instance variable to associated ID values with pandera schemas.
        """
        for id, params in load_map.items():
            # unpack params object
            path: Path = params["path"]
            schema: Type[OPNDFModel] = params["schema"]
            recursive_bools: bool = params["recursive_bools"] if "recursive_bools" in params.keys() else False
            if id.startswith("mnsos"):
                self.dfs_in[id] = pd.read_csv(
                    str(path),
                    dtype=str,
                    encoding="windows-1252",
                    encoding_errors="replace",
                    delimiter=",",
                    quoting=csv.QUOTE_NONE,  # disables quote logic entirely
                    engine="python",  # fallback parser that handles irregular rows better
                    on_bad_lines="skip",  # logs but keeps going
                )
            else:
                # load df and add to self.dfs_in
                self.dfs_in[id] = ops_df.load_df(path, schema, recursive_bools)
            # print success message
            if self.dfs_in[id] is not None:
            # self.dfs_out[id] = ops_df.load_df(path, str)
            # if self.dfs_out[id] is not None:
                console.print(f"\"{id}\" successfully loaded from: \n{path}")
        # print summary stats for loaded dfs
        console.print("\n")
        ss.display_load_table(self.dfs_in)
        ss.display_load_stats_table(self.dfs_in)

    def run_validator(self, id: str, df: pd.DataFrame, configs: WorkflowConfigs, wkfl_name: str, schema) -> None:
        """
        Executes pandera validator
        """
        console.print("\n")
        t.print_with_dots(f"Executing validator for {id} dataset")
        try:
            schema.validate(df, lazy=True)
            console.print("\n")
            console.print("✅ Validation successful ✅")
            console.print("\n")
        except pa.errors.SchemaErrors as err:
            console.print("\n")
            console.print("❌ Validation failed ❌")
            console.print(f"Number of validation errors: {len(err.failure_cases)}")
            console.print("\n")
            if hasattr(err, "failure_cases"):
                error_df: pd.DataFrame = err.failure_cases
                console.print(f"{error_df.head()}")
                error_indices: np.ndarray = error_df["index"].dropna().unique()
                error_rows_df: pd.DataFrame = df.loc[error_indices]
                # proceed, save_error_df = t.validator_failed()
                # if save_error_df:
                ops_df.save_df(
                    error_df,
                    path_gen.validation_errors(configs, wkfl_name, "summary")
                )
                ops_df.save_df(
                    error_rows_df,
                    path_gen.validation_errors(configs, wkfl_name, "error_rows")
                )
                # if not proceed:
                #     sys.exit()
                console.print("\n")

    def save_dfs(self, save_map: dict[str, Path]) -> None:
        """Saves dataframes to their specified paths."""
        console.print("\n")
        for id, path in save_map.items():
            ops_df.save_df(self.dfs_out[id], path)
            console.print(f"\"{id}\" successfully saved to: \n{path}")

    @classmethod
    def create_workflow(cls, config_manager: ConfigManager, wkfl_id: str) -> Optional['WorkflowBase']:
        """Instantiates workflow object based on last saved progress (config['wkfl_stage'])."""
        if wkfl_id == "raw_prep":
            return WkflRawDataPrep(config_manager)
        elif wkfl_id == "data_clean":
            return WkflDataClean(config_manager)
        elif wkfl_id == "bus_merge":
            return WkflBusinessMerge(config_manager)
        elif wkfl_id == "unvalidated_addrs":
            return WkflUnvalidatedAddrs(config_manager)
        elif wkfl_id == "address_geocodio":
            return WkflAddressGeocodio(config_manager)
        return None

    @abstractmethod
    def execute(self) -> None:
        pass


class WorkflowStandardBase(WorkflowBase):
    """Base class for workflows that follow the standard load->process->save pattern"""
    def execute(self) -> None:
        """Template method implementation"""
        try:
            self.load()
            self.process()
            self.summary_stats()
            self.save()
            # self.update_configs()
        except Exception as e:
            raise

    @abstractmethod
    def load(self) -> None:
        """Loads data files into dataframes. Returns dictionary mapping dataframes to IDs."""
        pass

    @abstractmethod
    def process(self) -> None:
        """
        Executes business & transformation logic for the workflow. Saves and stores processed dataframes in self.dfs_out.
        """
        pass

    @abstractmethod
    def summary_stats(self):
        """Executes summary stats builder for the workflow."""
        # todo: determine summary stats data type (return data hint)
        pass

    @abstractmethod
    def save(self) -> None:
        """
        Saves processed dataframes. Keys of save_map object must match EXACTLY those of the dataframe dictionary
        returned by process().
        """
        # todo: figure out a way to enforce structure of load map/process output/save map across workflow
        pass

    @abstractmethod
    def update_configs(self) -> None:
        """Updates configurations based on current workflow stage."""
        pass


class WkflRawDataPrep(WorkflowStandardBase):
    """Prepares raw data for further processing."""
    WKFL_NAME: str = "RAW DATA PREPARATION WORKFLOW"
    WKFL_DESC: str = "Prepares raw Minnesota/Minneapolis data for processing."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "taxpayers_city": {
                "path": path_gen.pre_process_taxpayers_city(configs),
                "schema": None  # TaxpayersCity
            },
            "taxpayers_county": {
                "path": path_gen.pre_process_taxpayers_county(configs),
                "schema": None  # TaxpayersCounty
            },
            "mnsos_type1": {
                "path": path_gen.pre_process_business_filings_1(configs),
                "schema": None  # BusinessFilings
            },
            "mnsos_type3": {
                "path": path_gen.pre_process_business_filings_3(configs),
                "schema": None  # BusinessNamesAddrs
            }
        }
        self.load_dfs(load_map)

    def execute_taxpayer_pre_processing(self, df_city: pd.DataFrame, df_county: pd.DataFrame) -> pd.DataFrame:
        # extract relevant columns from both datasets
        df_city = df_city[[
            "PIN",
            "PRIMARY_PROP_TYPE",
            "IS_EXEMPT",
            "IS_HOMESTEAD",
        ]]
        df_city.rename(columns={
            "PIN": "pin",
            "PRIMARY_PROP_TYPE": "prop_type",
            "IS_EXEMPT": "is_exempt",
            "IS_HOMESTEAD": "is_homestead",
        }, inplace=True)
        df_county = df_county[[
            "PID",
            "TAXPAYER_NM",
            "TAXPAYER_NM_1",
            "TAXPAYER_NM_2",
            "TAXPAYER_NM_3",
            "MUNIC_NM",
        ]]
        df_county.rename(columns={
            "PID": "pin",
            "TAXPAYER_NM": "taxpayer",
            "TAXPAYER_NM_1": "taxpayer_1",
            "TAXPAYER_NM_2": "taxpayer_2",
            "TAXPAYER_NM_3": "taxpayer_3",
            "MUNIC_NM": "municipality",
        }, inplace=True)

        # fix pins on df_city
        df_city["pin"] = df_city["pin"].apply(lambda pin: pin[1:])

        # basic cleaning
        df_county = clean_df_base.trim_whitespace(df_county, [
            "taxpayer",
            "taxpayer_1",
            "taxpayer_2",
            "taxpayer_3",
            "municipality",
        ])  # run cleaners only where necessary

        t.print_equals("Processing raw taxpayer records")
        # basic cleaning operations - bare minimum to execute subset, shift cells etc.
        t.print_with_dots("Convering text to upper case")
        df_county = clean_df_base.make_upper(df_county)
        t.print_with_dots("Trimming whitespace")
        df_county = clean_df_base.trim_whitespace(df_county)
        t.print_with_dots("Replacing empty strings with np.nan")
        df_county = clean_df_base.replace_with_nan(df_county)
        t.print_with_dots("Removing extra spaces")
        df_county = clean_df_base.remove_extra_spaces(df_county)

        df_mpls = df_county[df_county["municipality"] == "MINNEAPOLIS"]

        # shift taxpayer data
        t.print_with_dots("Shifting taxpayer data cells")
        df_mpls = df_mpls.apply(lambda row: shift_df.shift_taxpayer_data_cells(row), axis=1)

        # drop Hennepin county forfeited land properties
        df_mpls.drop(df_mpls[df_mpls["taxpayer"] == "HENNEPIN FORFEITED LAND"].index, inplace=True)

        # drop nulls
        t.print_with_dots("Dropping rows with missing taxpayer data")
        df_mpls.dropna(subset=["taxpayer"], inplace=True)
        df_mpls.dropna(subset=["taxpayer_2"], inplace=True)
        df_mpls.dropna(subset=["taxpayer_3"], inplace=True)

        # fix mpls/mnpls
        df_mpls = clean_df_addr.fix_mpls(df_mpls, ["taxpayer_3"])

        # set address columns
        t.print_with_dots("Setting address column")
        df_mpls["tax_address"] = df_mpls.apply(lambda row: f"{row['taxpayer_2']}, {row['taxpayer_3']}", axis=1)

        # merge relevant data from df_city
        t.print_with_dots("Merging city dataset into county dataset")
        df_taxpayers: pd.DataFrame = pd.merge(df_mpls, df_city[[
            "pin",
            "prop_type",
            "is_exempt",
            "is_homestead",
        ]], how="left", on="pin")

        # drop & rename columns
        df_taxpayers.rename(columns={
            "PID": "pin",
            "taxpayer": "tax_name",
            "taxpayer_1": "tax_name_2",
            "taxpayer_2": "tax_street",
            "taxpayer_3": "tax_city_state_zip",
        }, inplace=True)

        return df_taxpayers

    def execute_business_filings_preprocessing(
        self,
        df_bus1: pd.DataFrame,
        df_bus3: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        t.print_equals("Processing raw business filings")
        t.print_with_dots("Extracting relevant columns")
        df_bus1 = df_bus1[[
            "master_id",
            "minnesota_business_name",
            "business_filing_status",
        ]]
        df_bus1.rename(columns={
            "master_id": "uid",
            "minnesota_business_name": "name",
            "business_filing_status": "status",
        }, inplace=True)
        df_bus3 = df_bus3[[
            "master_id",
            "name_type_number",
            "address_type_number",
            "party_name",
            "street_address_line_1",
            "street_address_line_2",
            "city_name",
            "region_code",
            "postal_code",
            "postal_code_extension",
            "country_name",
        ]]
        df_bus3.rename(columns={
            "master_id": "uid",
            "name_type_number": "name_type",
            "address_type_number": "address_type",
            "street_address_line_1": "street_1",
            "street_address_line_2": "street_2",
            "city_name": "city",
            "region_code": "state",
            "postal_code": "zip_code",
            "postal_code_extension": "zip_code_ext",
            "country_name": "country",
        }, inplace=True)

        t.print_with_dots("Trimming whitespace")
        df_bus3 = clean_df_base.trim_whitespace(df_bus3)
        t.print_with_dots("Replacing empty strings with np.nan")
        df_bus3 = clean_df_base.replace_with_nan(df_bus3)
        t.print_with_dots("Shifting street address cells")
        df_bus3 = df_bus3.apply(lambda row: shift_df.shift_street_addrs(row), axis=1)
        t.print_with_dots("Identifying incomplete addresses")
        df_bus3["is_incomplete_address"] = df_bus3.apply(
            lambda row: cols_df.set_is_incomplete_address_mnsos(row),
            axis=1
        )
        t.print_with_dots("Setting full address column")
        df_bus3["address"] = df_bus3.apply(lambda row: cols_df.concatenate_addr_mnsos(row), axis=1)

        return df_bus1, df_bus3

    def process(self) -> None:

        df_city: pd.DataFrame = self.dfs_in["taxpayers_city"].copy()
        df_county: pd.DataFrame = self.dfs_in["taxpayers_county"].copy()
        df_tax_out = self.execute_taxpayer_pre_processing(df_city, df_county)
        self.dfs_out["props_taxpayers"] = df_tax_out

        df_bus1: pd.DataFrame = self.dfs_in["mnsos_type1"].copy()
        df_bus3: pd.DataFrame = self.dfs_in["mnsos_type3"].copy()
        df_bus_1_out, df_bus_3_out = self.execute_business_filings_preprocessing(df_bus1, df_bus3)
        self.dfs_out["bus_filings"] = df_bus_1_out
        self.dfs_out["bus_names_addrs"] = df_bus_3_out

    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "props_taxpayers": path_gen.raw_props_taxpayers(configs),
            "bus_filings": path_gen.raw_bus_filings(configs),
            "bus_names_addrs": path_gen.raw_bus_names_addrs(configs),
        }
        self.save_dfs(save_map)

    def update_configs(self) -> None:
        pass


class WkflDataClean(WorkflowStandardBase):
    """
    Initial data cleaning. Runs cleaners and validators on raw datasets. if they pass the validation checks, raw
    datasets are broken up and stored in their appropriate locations.
    """
    WKFL_NAME: str = "INITIAL DATA CLEANING WORKFLOW"
    WKFL_DESC: str = "Runs basic string cleaners on raw inputted datasets."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_taxpayer_cleaning(self, df: pd.DataFrame, schema_map) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # pre-cleaning
        t.print_equals("Executing pre-cleaning operations")
        # generate raw_prefixed columns
        t.print_with_dots(f"Setting raw columns for \"{id}\"")
        df: pd.DataFrame = cols_df.set_raw_columns(df, schema_map["props_taxpayers"].raw())
        console.print(f"Raw columns generated ✅")
        # rename columns to prepare for cleaning
        df.rename(columns=schema_map["props_taxpayers"].clean_rename_map(), inplace=True)
        t.print_with_dots(f"Concatenating raw name and address fields")
        df: pd.DataFrame = cols_df.set_name_address_concat(
            df, schema_map["props_taxpayers"].name_address_concat_map()["raw"]
        )
        console.print(f"\"name_address\" field generated ✅")

        # basic cleaning
        t.print_equals(f"Executing basic operations (all data columns)")
        cols: list[str] = schema_map["props_taxpayers"].basic_clean()
        t.print_with_dots("Removing symbols and punctuation")
        df = clean_df_base.remove_symbols_punctuation(df, cols)
        t.print_with_dots("Handling LLCs")
        df = clean_df_base.fix_llcs(df, cols)
        t.print_with_dots("Deduplicating repeated words")
        df = clean_df_base.deduplicate(df, cols)
        # t.print_with_dots("Converting ordinal numbers to digits...")  # todo: this one breaks the df, returns all null values
        # df = clean_df_base.convert_ordinals(df, cols)
        t.print_with_dots("Combining numbers separated by spaces")
        df = clean_df_base.combine_numbers(df, cols)
        console.print("Preliminary cleaning complete ✅")

        # name cleaning only
        t.print_equals(f"Executing cleaning operations (name columns only)")
        name_cols: list[str] = schema_map["props_taxpayers"].name_clean()
        t.print_with_dots("Replacing number ranges with first number in range")
        df = clean_df_base.take_first(df, name_cols)
        df = clean_df_name.switch_the(df, name_cols)
        console.print(f"Name field cleaning complete ✅")

        # address cleaning only
        t.print_equals(f"Executing cleaning operations (address columns only)")
        for col in schema_map["props_taxpayers"].address_clean()["street"]:
            t.print_with_dots("Converting street suffixes")
            df = clean_df_addr.convert_street_suffixes(df, [col])
            t.print_with_dots("Converting directionals to their abbreviations")
            df = clean_df_addr.convert_nsew(df, [col])
            t.print_with_dots("Removing secondary designators")
            df = clean_df_addr.remove_secondary_designators(df, [col])
        console.print("Address field cleaning complete ✅")

        # post-cleaning
        t.print_equals(f"Executing post-cleaning operations")
        t.print_with_dots("Parsing city_state_zip")
        df = cols_df.set_city_state_zip(df)
        # add clean full address fields
        t.print_with_dots("Setting clean full address fields")
        df["clean_address"] = df.apply(lambda row: cols_df.concatenate_addr_mpls(row), axis=1)
        df = cols_df.set_name_address_concat(
            df, schema_map["props_taxpayers"].name_address_concat_map()["clean"]
        )
        t.print_with_dots("Full clean address fields generated ✅")

        # split dataframe into properties and taxpayer_records
        t.print_with_dots(f"Splitting \"{id}\" into \"taxpayer_records\" and \"properties\"...")
        df_props, df_taxpayers = subset_df.split_props_taxpayers(
            df,
            schema_map["properties"].out(),
            schema_map["taxpayer_records"].out()
        )
        df_taxpayers.dropna(subset=["raw_name"], inplace=True)
        console.print(f"props_taxpayers successfully split into \"taxpayer_records\" and \"properties\" ✅")

        return df_props, df_taxpayers

    def execute_business_filings_cleaning(
        self,
        df_bus1: pd.DataFrame,
        df_bus3: pd.DataFrame,
        schema_map
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        for id, df in {"bus_filings": df_bus1, "bus_names_addrs": df_bus3}.items():

            t.print_equals(f"Executing pre-cleaning operations ({id})")
            # generate raw_prefixed columns
            t.print_with_dots(f"Setting raw columns for \"{id}\"")
            df: pd.DataFrame = cols_df.set_raw_columns(df, schema_map[id].raw())
            console.print(f"Raw columns generated ✅")
            # rename columns to prepare for cleaning
            df.rename(columns=schema_map[id].clean_rename_map(), inplace=True)

            # basic cleaning
            t.print_equals(f"Executing basic operations (all data columns) ({id})")
            cols = schema_map[id].basic_clean()
            # basic cleaning in preparation for address concatenation
            t.print_with_dots("Converting letters to uppercase")
            df = clean_df_base.make_upper(df, cols)
            t.print_with_dots("Removing symbols and punctuation")
            df = clean_df_base.remove_symbols_punctuation(df, cols)
            t.print_with_dots("Stripping dashes")
            df = clean_df_base.strip_dashes(df, cols)
            t.print_with_dots("Trimming whitespace")
            df = clean_df_base.trim_whitespace(df, cols)
            t.print_with_dots("Removing extra spaces")
            df = clean_df_base.remove_extra_spaces(df, cols)
            t.print_with_dots("Handling LLCs")
            df = clean_df_base.fix_llcs(df, cols)
            t.print_with_dots("Deduplicating repeated words")
            df = clean_df_base.deduplicate(df, cols)
            t.print_with_dots("Combining numbers separated by spaces")
            df = clean_df_base.combine_numbers(df, cols)

            # name-only cleaning
            t.print_equals(f"Executing cleaning operations (name columns only)")
            name_cols: list[str] = schema_map[id].name_clean()
            df = clean_df_name.switch_the(df, name_cols)
            console.print(f"Name field cleaning complete ✅")

            # address-only cleaning
            t.print_equals(f"Executing cleaning operations (address columns only)")
            if id == "bus_names_addrs":
                for col in schema_map[id].address_clean()["street"]:
                    t.print_with_dots("Converting street suffixes")
                    df = clean_df_addr.convert_street_suffixes(df, [col])
                    t.print_with_dots("Converting directionals to their abbreviations")
                    df = clean_df_addr.convert_nsew(df, [col])
                    t.print_with_dots("Removing secondary designators")
                    df = clean_df_addr.remove_secondary_designators(df, [col])
                    # address-specific operations
                    state_fixer: dict[str, str] = BusinessRecordsBase.mn_state_fixer()
                    t.print_with_dots("Cleaning up states")
                    df["clean_state"] = df["clean_state"].apply(lambda state: colm_df.fix_states(state, state_fixer))
                df["clean_address"] = df.apply(lambda row: cols_df.concatenate_addr_mnsos(row, prefix="clean_"), axis=1)
                df = df.apply(lambda row: shift_df.shift_street_addrs(row, prefix="clean_"), axis=1)

        return df_bus1, df_bus3

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "props_taxpayers": {
                "path": path_gen.raw_props_taxpayers(configs),
                "schema": PropsTaxpayersMN
            },
            "bus_filings": {
                "path": path_gen.raw_bus_filings(configs),
                "schema": BusinessFilings
            },
            "bus_names_addrs": {
                "path": path_gen.raw_bus_names_addrs(configs),
                "schema": BusinessNamesAddrs,
                "recursive_bools": True
            }
        }
        self.load_dfs(load_map)

    def process(self) -> None:

        schema_map = {
            "props_taxpayers": PropsTaxpayersMN,
            "bus_filings": BusinessFilings,
            "bus_names_addrs": BusinessNamesAddrs,
            "properties": PropertiesMN,
            "taxpayer_records": TaxpayerRecordsMN,
        }

        # taxpayer records
        df_taxpayers: pd.DataFrame = self.dfs_in["props_taxpayers"].copy()
        df_properties, df_taxpayer_records = self.execute_taxpayer_cleaning(df_taxpayers, schema_map)
        self.dfs_out["taxpayer_records"] = df_taxpayer_records
        self.dfs_out["properties"] = df_properties

        # business filings
        df_bus1: pd.DataFrame = self.dfs_in["bus_filings"].copy()
        df_bus3: pd.DataFrame = self.dfs_in["bus_names_addrs"].copy()
        df_filings, df_names_addrs = self.execute_business_filings_cleaning(df_bus1, df_bus3, schema_map)
        self.dfs_out["bus_filings"] = df_filings
        self.dfs_out["bus_names_addrs"] = df_names_addrs

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "properties": path_gen.processed_properties(configs),
            "taxpayer_records": path_gen.processed_taxpayer_records(configs),
            "bus_filings": path_gen.processed_bus_filings(configs),
            "bus_names_addrs": path_gen.processed_bus_names_addrs(configs),
        }
        self.save_dfs(save_map)

    def summary_stats(self) -> None:
        pass

    def update_configs(self) -> None:
        pass


class WkflBusinessMerge(WorkflowStandardBase):

    WKFL_NAME: str = "PRE-MATCH CLEANING & MERGING WORKFLOW"
    WKFL_DESC: str = "Adds boolean columns identifying patterns in taxpayer names, merges corporate/LLC records into taxpayer data."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_column_generators(
        self,
        df_taxpayers: pd.DataFrame,
        df_bus: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        t.print_with_dots("Adding core name columns to taxpayer, corporate and LLC records")
        df_taxpayers = cols_df.set_core_name(df_taxpayers, "clean_name")
        df_bus = cols_df.set_core_name(df_bus, "clean_name")

        # t.print_with_dots("Adding is_bank boolean column to taxpayer records")
        # df_taxpayers = cols_df.set_is_bank(df_taxpayers, "clean_name")
        t.print_with_dots("Adding is_trust boolean column to taxpayer records")
        df_taxpayers = cols_df.set_is_trust(df_taxpayers, "clean_name")
        t.print_with_dots("Adding is_person boolean column to taxpayer records")
        df_taxpayers = cols_df.set_is_person(df_taxpayers, "clean_name")
        t.print_with_dots("Adding is_org boolean column to taxpayer records")
        df_taxpayers = cols_df.set_is_org(df_taxpayers, "clean_name")
        t.print_with_dots("Adding is_llc boolean column to taxpayer records")
        df_taxpayers = cols_df.set_is_llc(df_taxpayers, "clean_name")

        return df_taxpayers, df_bus

    def execute_clean_merge(
        self,
        df_taxpayers: pd.DataFrame,
        df_bus: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        t.print_with_dots("Merging corporations & LLCs to taxpayers on clean_name")
        df_bus.rename(columns={"clean_name": "entity_clean_name"}, inplace=True)
        df_clean_merge: pd.DataFrame = pd.merge(
            df_taxpayers,
            df_bus,
            how="left",
            left_on="clean_name",
            right_on="entity_clean_name"
        )
        df_clean_merge = ops_df.combine_columns_parallel(df_clean_merge)
        df_clean_merge.drop_duplicates(subset=["raw_name_address"], inplace=True)
        df_clean_merge["is_clean_match"] = df_clean_merge["entity_clean_name"].apply(lambda name: pd.notnull(name))
        df_taxpayers_remaining: pd.DataFrame = df_clean_merge[df_clean_merge["is_clean_match"] == False]
        return df_clean_merge, df_taxpayers_remaining

    def execute_core_merge(
        self,
        df_taxpayers_remaining: pd.DataFrame,
        df_bus: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_bus.rename(columns={"core_name": "entity_core_name"}, inplace=True)
        df_core_merge: pd.DataFrame = pd.merge(
            df_taxpayers_remaining,
            df_bus,
            how="left",
            left_on="core_name",
            right_on="entity_core_name"
        )
        df_core_merge = ops_df.combine_columns_parallel(df_core_merge)
        df_core_merge.drop_duplicates(subset=["raw_name_address"], inplace=True)
        df_core_merge["is_core_match"] = df_core_merge["entity_core_name"].apply(lambda name: pd.notnull(name))
        df_taxpayers_remaining: pd.DataFrame = df_core_merge[df_core_merge["is_core_match"] == False]
        return df_core_merge, df_taxpayers_remaining

    def execute_string_match_merge(
        self,
        df_taxpayers_remaining: pd.DataFrame,
        df_bus: pd.DataFrame
    ) -> pd.DataFrame:
        ref_docs = list(df_bus["entity_clean_name"].dropna().unique())
        query_docs = list(df_taxpayers_remaining["clean_name"].dropna().unique())
        df_string_matches = StringMatch.match_strings(
            ref_docs,
            query_docs,
            params={
                "name_col": None,
                "match_threshold": .89,
                "include_unvalidated": True,
                "include_unresearched": False,
                "include_orgs": False,
                "nmslib_opts": {
                    "method": "hnsw",
                    "space": "cosinesimil_sparse_fast",
                    "data_type": nmslib.DataType.SPARSE_VECTOR
                },
                "query_batch_opts": {
                    "num_threads": 8,
                    "K": 1
                }
            }
        )
        # merge string match results and drop duplicates to get org names
        df_taxpayers_remaining.drop(columns=["entity_clean_name"], inplace=True)
        df_string_merge: pd.DataFrame = pd.merge(
            df_taxpayers_remaining,
            df_string_matches,
            how="left",
            left_on="clean_name",
            right_on="original_doc"
        )
        df_string_merge = ops_df.combine_columns_parallel(df_string_merge)
        df_string_merge.rename(columns={"matched_doc": "entity_clean_name"}, inplace=True)
        df_string_merge.drop_duplicates(subset=["raw_name_address"], inplace=True)
        df_string_merge.drop(columns=["original_doc", "conf", "ldist", "conf1"], inplace=True)

        # take slice of business filing dataframe to get address data & merge again
        df_matched_orgs: pd.DataFrame = df_bus[df_bus["entity_clean_name"].isin(
            list(df_string_matches["matched_doc"].unique())
        )]
        df_string_merge_final: pd.DataFrame = pd.merge(
            df_string_merge,
            df_matched_orgs,
            how="left",
            on="entity_clean_name"
        )
        df_string_merge_final = ops_df.combine_columns_parallel(df_string_merge_final)
        df_string_merge_final.drop_duplicates(subset=["raw_name_address"], inplace=True)
        df_string_merge_final["is_string_match"] = df_string_merge_final["entity_clean_name"].apply(
            lambda name: pd.notnull(name)
        )
        return df_string_merge_final

    def execute_post_merge_column_ops(self, df_taxpayers: pd.DataFrame) -> pd.DataFrame:
        # set taxpayer clean and core name equal to corporate/llc records clean and core names
        df_taxpayers = colm_df.set_business_names_taxpayers(df_taxpayers)
        return df_taxpayers

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "taxpayer_records": {
                "path": path_gen.processed_taxpayer_records(configs),
                "schema": TaxpayerRecordsMN,
                "recursive_bools": True
            },
            "bus_filings": {
                "path": path_gen.processed_bus_filings(configs),
                "schema": BusinessFilings
            }
        }
        self.load_dfs(load_map)

    def process(self) -> None:

        schema_map = {
            "taxpayer_records": TaxpayerRecordsMN,
            "bus_filings": BusinessFilings
        }
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayer_records"].copy()
        df_bus: pd.DataFrame = self.dfs_in["bus_filings"].copy()

        df_taxpayers, df_bus = self.execute_column_generators(df_taxpayers, df_bus)
        df_clean_merge, df_taxpayers_remaining = self.execute_clean_merge(df_taxpayers, df_bus)
        df_core_merge, df_taxpayers_remaining = self.execute_core_merge(df_taxpayers_remaining, df_bus)
        df_string_merge: pd.DataFrame = self.execute_string_match_merge(df_taxpayers_remaining, df_bus)
        df_taxpayers = pd.concat([
            df_clean_merge[df_clean_merge["entity_clean_name"].notnull()],
            df_core_merge[df_core_merge["entity_core_name"].notnull()],
            df_string_merge
        ], ignore_index=True)
        df_taxpayers = self.execute_post_merge_column_ops(df_taxpayers)

        self.dfs_out["taxpayers_bus_merged"] = df_taxpayers


    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "taxpayers_bus_merged": path_gen.processed_taxpayers_bus_merged(configs),
        }
        self.save_dfs(save_map)

    def update_configs(self) -> None:
        pass


class WkflUnvalidatedAddrs(WorkflowStandardBase):

    WKFL_NAME: str = "UNVALIDATED ADDRESS GENERATION WORKFLOW"
    WKFL_DESC: str = "Fetches addresses from taxpayer and business records to run through validation service."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "taxpayers_bus_merged": {
                "path": path_gen.processed_taxpayers_bus_merged(configs),
                "schema": None  # TaxpayersBusMergedMN,
            },
            "bus_names_addrs": {
                "path": path_gen.processed_bus_names_addrs(configs),
                "schema": BusinessNamesAddrs
            }
        }
        self.load_dfs(load_map)

    def process(self) -> None:

        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_bus_merged"].copy()
        df_bus_names_addrs: pd.DataFrame = self.dfs_in["bus_names_addrs"].copy()

        unique_entities: list[str] = list(df_taxpayers["uid"].dropna().unique())

        t.print_with_dots("Fetching unique addresses from taxpayer records")
        df_tax_u: pd.DataFrame = df_taxpayers[["clean_street", "clean_city", "clean_state", "clean_zip_code", "clean_address"]]
        df_tax_u.drop_duplicates(subset=["clean_address"], inplace=True)
        df_tax_u.rename(columns={"clean_street": "clean_street_1"}, inplace=True)

        # fetch all unique clean addresses from business records for matching orgs
        t.print_with_dots("Fetching unique addresses from business records")
        df_bus_u: pd.DataFrame = df_bus_names_addrs[df_bus_names_addrs["uid"].isin(unique_entities)]
        df_bus_u = df_bus_u[["clean_street_1", "clean_street_2", "clean_city", "clean_state", "clean_zip_code", "clean_country", "clean_address"]]
        df_bus_u.dropna(subset=["clean_address"], inplace=True)
        df_bus_u.drop_duplicates(subset=["clean_address"], inplace=True)

        # generate final unvalidated address file
        t.print_with_dots("Generating master unvalidated dataset")
        df_unvalidated: pd.DataFrame = pd.concat([df_tax_u, df_bus_u], ignore_index=True)
        df_unvalidated.drop_duplicates(subset=["clean_address"], inplace=True)
        self.dfs_out["unvalidated_addrs"] = df_unvalidated


    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(configs),
        }
        self.save_dfs(save_map)

    def update_configs(self) -> None:
        pass


class WkflAddressGeocodio(WorkflowStandardBase):

    WKFL_NAME: str = "ADDRESS VALIDATION (GEOCODIO) WORKFLOW"
    WKFL_DESC: str = ("Executes Geocodio API calls for unvalidated addresses. Processes and stores results in data "
                      "directories.")

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_gcd_address_subset(self, df_unvalidated: pd.DataFrame) -> pd.DataFrame:
        """
        Subsets unvalidated master address file by filtering out addresses that have already been run through the
        validator. Returns dataframe slice containing only columns required for processing Geocodio API calls.
        """
        # fetch addresses already processed
        addrs: list[str] = []
        if self.dfs_in["gcd_validated"] is not None:
            addrs.extend(list(self.dfs_in["gcd_validated"]["clean_address"].unique()))
        if self.dfs_in["gcd_unvalidated"] is not None:
            addrs.extend(list(self.dfs_in["gcd_unvalidated"]["clean_address"].unique()))
        # filter out addresses already processed
        if len(addrs) > 0:
            df_addrs: pd.DataFrame = df_unvalidated[~df_unvalidated["clean_address"].isin(set(addrs))]
        else:
            df_addrs: pd.DataFrame = df_unvalidated
        # get number of addresses to run through geocodio
        num_addrs_to_geocodio: int = ti.prompt_geocode_count(len(df_addrs))
        return df_addrs[[
            "clean_street_1",
            "clean_street_2",
            "clean_city",
            "clean_state",
            "clean_zip_code",
            "clean_country",
            "clean_address",
        ]].head(num_addrs_to_geocodio)

    def execute_api_key_handler_warning(self, df_addrs: pd.DataFrame) -> bool:
        """
        Prompts user to enter geocodio API key if not already found in configs.json. Prints out warning telling user
        how many addresses are set to be geocoded and how much id could cost them, and aborts workflow if user quits.
        """
        if "geocodio_api_key" not in self.config_manager.configs.keys():
            while True:
                console.print("\n")
                api_key: str = Prompt.ask(
                    "[bold cyan]Copy & paste your geocodio API key for address validation[/bold cyan]"
                )
                if len(api_key) > 0:
                    self.config_manager.set("geocodio_api_key", api_key)
                else:
                    console.print("You must enter an API key to continue.")
        t.print_geocodio_warning(df_addrs)
        cont: bool = t.press_enter_to_continue("execute geocodio API calls ")
        if not cont:
            console.print("Aborted!")
        return cont

    def execute_geocodio_postprocessor(self, gcd_results_obj: GeocodioReturnObject) -> None:
        """
        Processes results of run_geocodio. Updates gcd_validated, gcd_unvalidated and gcd_failed based on results of
        run_geocodio call. Stores final dataframes to be saved in self.dfs_out.
        """
        t.print_with_dots("Merging validated address data")
        # create new dataframes from the resulting geocodio call
        df_gcd_validated_new: pd.DataFrame = pd.DataFrame(gcd_results_obj["validated"])
        df_gcd_unvalidated_new: pd.DataFrame = pd.DataFrame(gcd_results_obj["unvalidated"])

        # if there were already gcd_validated, gcd_unvalidated and gcd_failed in the directories, concatenate the new one and set to dfs_out
        if self.dfs_in["gcd_validated"] is not None:
            df_gcd_validated_out = pd.concat([self.dfs_in["gcd_validated"], df_gcd_validated_new], ignore_index=True)
        else:
            df_gcd_validated_out = df_gcd_validated_new
        if self.dfs_in["gcd_unvalidated"] is not None:
            df_gcd_unvalidated_out = pd.concat([self.dfs_in["gcd_unvalidated"], df_gcd_unvalidated_new], ignore_index=True)
        else:
            df_gcd_unvalidated_out = df_gcd_unvalidated_new
        # calculate stats to print
        if not df_gcd_validated_out.empty:
            validated_before: int = len(self.dfs_in["gcd_validated"]) if self.dfs_in["gcd_validated"] is not None else 0
            validated_after: int = len(df_gcd_validated_out)
            validated_diff: int = validated_after - validated_before
            console.print(f"Total validated addresses: {validated_after} (+{validated_diff})")

        if not df_gcd_unvalidated_out.empty:
            unvalidated_before: int = len(self.dfs_in["gcd_unvalidated"]["clean_address"].unique()) if self.dfs_in["gcd_unvalidated"] is not None else 0
            unvalidated_after: int = len(df_gcd_unvalidated_out["clean_address"].unique())
            unvalidated_diff: int = unvalidated_after - unvalidated_before
            console.print(f"Total unvalidated addresses: {unvalidated_after} (+{unvalidated_diff})")

        # set dfs_in with updated validated address datasets to run again
        self.dfs_in["gcd_validated"] = df_gcd_validated_out
        self.dfs_in["gcd_unvalidated"] = df_gcd_unvalidated_out

        # set dfs_out and save progress
        self.dfs_out["gcd_validated"] = df_gcd_validated_out
        self.dfs_out["gcd_unvalidated"] = df_gcd_unvalidated_out
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "gcd_validated": path_gen.geocodio_gcd_validated(configs),
            "gcd_unvalidated": path_gen.geocodio_gcd_unvalidated(configs),
        }
        self.save_dfs(save_map)

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "unvalidated_addrs": {
                "path": path_gen.processed_unvalidated_addrs(configs),
                "schema": UnvalidatedAddrs,
            },
            "gcd_unvalidated": {
                "path": path_gen.geocodio_gcd_unvalidated(configs),
                "schema": Geocodio,
            },
            "gcd_validated": {
                "path": path_gen.geocodio_gcd_validated(configs),
                "schema": Geocodio,
            },
        }
        self.load_dfs(load_map)

    def process(self) -> None:

        df_unvalidated_master: pd.DataFrame = self.dfs_in["unvalidated_addrs"].copy()

        while True:
            # fetch addresses to be geocoded
            df_addrs: pd.DataFrame = self.execute_gcd_address_subset(df_unvalidated_master)
            # get geocodio api key from user if not already exists in configs.json
            cont: bool = self.execute_api_key_handler_warning(df_addrs)
            if not cont: return
            # call geocodio or exit
            gcd_results_obj: GeocodioReturnObject = addr.run_geocodio_mpls(
                self.config_manager.configs,
                df_addrs,
            )
            # execute post-processor
            self.execute_geocodio_postprocessor(gcd_results_obj)

    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        pass

    def update_configs(self) -> None:
        pass
