# 1. Standard library imports
import csv
import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, Type
import nmslib
import numpy as np
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
import pandera as pa
from itertools import product
import warnings

# 2. Third-party imports
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

# 3. Constants (these should have no dependencies on other local modules)
from opndb.schema.base.process import UnvalidatedAddrs, Geocodio, UnvalidatedAddrsClean, FixingAddrs, FixingTaxNames, \
    AddressAnalysis, FrequentTaxNames, GeocodioFormatted
from opndb.schema.mpls.process import TaxpayerRecords, Properties, \
    TaxpayersStringMatched, TaxpayersSubsetted, TaxpayersPrepped, TaxpayersNetworked, TaxpayersBusMerged, \
    TaxpayersAddrMerged, BusinessNamesAddrsMerged, TaxpayersFixed, BusinessNamesAddrsSubsetted
from opndb.schema.mpls.raw import PropsTaxpayers, BusinessFilings, BusinessNamesAddrs, BusinessRecordsBase
from opndb.services.summary_stats import SummaryStatsBase as ss, \
    SSFixUnitsInitial, SSFixUnitsFinal, SSAddressMerge, SSNameAnalysisInitial, SSAddressAnalysisInitial, \
    SSAnalysisFinal, SSStringMatch, SSNetworkGraph
from opndb.services.config import ConfigManager

# 4. Types (these should only depend on constants)
from opndb.types.base import (
    WorkflowConfigs,
    NmslibOptions,
    NetworkMatchParams,
    GeocodioReturnObject, GeocodioResultProcessed, GeocodioResultFlat, CleanAddress,
    StringMatchParamsMN,
)

# 5. Utils (these should only depend on constants and types)
from opndb.utils import PathGenerators as path_gen

# 6. Services (these can depend on everything else)
from opndb.services.match import StringMatch, NetworkMatchBase, MatchBase
from opndb.services.address import AddressBase as addr, AddressBase
from opndb.services.terminal_printers import TerminalBase as t
from opndb.services.terminal_printers import TerminalInteract as ti
from opndb.services.dataframe.base import (
    DataFrameOpsBase as ops_df,
    DataFrameBaseCleaners as clean_df_base,
    DataFrameNameCleaners as clean_df_name,
    DataFrameAddressCleaners as clean_df_addr,
)
from opndb.services.dataframe.ops import (
    DataFrameMergers as merge_df,
    DataFrameSubsetters as subset_df,
    DataFrameColumnGenerators as cols_df,
    DataFrameColumnManipulators as colm_df,
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
        elif wkfl_id == "geocodio_fix":
            return WkflGeocodioFix(config_manager)
        elif wkfl_id == "gcd_string_match":
            return WkflGeocodioStringMatch(config_manager)
        elif wkfl_id == "fix_units_initial":
            return WkflFixUnitsInitial(config_manager)
        elif wkfl_id == "fix_units_final":
            return WkflFixUnitsFinal(config_manager)
        elif wkfl_id == "fix_addrs_initial":
            return WkflFixAddrsInitial(config_manager)
        elif wkfl_id == "fix_addrs_final":
            return WkflFixAddrsFinal(config_manager)
        elif wkfl_id == "set_address_columns":
            return WkflSetAddressColumns(config_manager)
        elif wkfl_id == "address_merge":
            return WkflAddressMerge(config_manager)
        elif wkfl_id == "name_analysis_initial":
            return WkflNameAnalysisInitial(config_manager)
        elif wkfl_id == "address_analysis_initial":
            return WkflAddressAnalysisInitial(config_manager)
        elif wkfl_id == "analysis_final":
            return WkflAnalysisFinal(config_manager)
        elif wkfl_id == "rental_subset":
            return WkflRentalSubset(config_manager)
        elif wkfl_id == "match_addr_cols":
            return WkflMatchAddressCols(config_manager)
        elif wkfl_id == "string_match":
            return WkflStringMatch(config_manager)
        elif wkfl_id == "network_graph":
            return WkflNetworkGraph(config_manager)
        elif wkfl_id == "final_output":
            return WkflFinalOutput(config_manager)
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
            self.validate()
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
    def validate(self):
        """Runs validators on each loaded dataframe based on schema map set in the function itself."""
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

    def execute_taxpayer_pre_processing(self, df_city: pd.DataFrame, df_county: pd.DataFrame) -> pd.DataFrame:
        # extract relevant columns from both datasets
        df_city = df_city[[
            "PIN",
            "LANDUSE",
            "BUILDINGUSE",
            "YEARBUILT",
            "PRIMARY_PROP_TYPE",
            "IS_EXEMPT",
            "IS_HOMESTEAD",
            "TOTAL_UNITS",
        ]]
        df_city.rename(columns={
            "PIN": "pin",
            "LANDUSE": "land_use",
            "BUILDINGUSE": "building_use",
            "PRIMARY_PROP_TYPE": "prop_type",
            "IS_EXEMPT": "is_exempt",
            "IS_HOMESTEAD": "is_homestead",
            "TOTAL_UNITS": "num_units",
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
            "land_use",
            "building_use",
            "prop_type",
            "is_exempt",
            "is_homestead",
            "num_units",
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
            "filing_date",
            "expiration_date",
            "home_jurisdiction",
            "home_business_name",
            "is_llc_non_profit",
            "is_lllp",
            "is_professional",
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

        t.print_with_dots("Setting zeros and ones to booleans")
        for col in ["is_llc_non_profit", "is_lllp", "is_professional"]:
            df_bus1[col] = df_bus1[col].apply(lambda val: val.strip() == "1")
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

    # --------------
    # ----LOADER----
    # --------------
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

    # -----------------
    # ----PROCESSOR----
    # -----------------
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

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "props_taxpayers": path_gen.raw_props_taxpayers(configs),
            "bus_filings": path_gen.raw_bus_filings(configs),
            "bus_names_addrs": path_gen.raw_bus_names_addrs(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
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

    def execute_taxpayer_cleaning(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # pre-cleaning
        t.print_equals("Executing pre-cleaning operations")
        # generate raw_prefixed columns
        t.print_with_dots(f"Setting raw columns for \"{id}\"")
        df: pd.DataFrame = cols_df.set_raw_columns(df, PropsTaxpayers.raw())
        console.print(f"Raw columns generated ✅")
        # rename columns to prepare for cleaning
        df.rename(columns=PropsTaxpayers.clean_rename_map(), inplace=True)
        t.print_with_dots(f"Concatenating raw name and address fields")
        df: pd.DataFrame = cols_df.set_name_address_concat(df, PropsTaxpayers.name_address_concat_map()["raw"])
        console.print(f"\"name_address\" field generated ✅")

        # basic cleaning
        t.print_equals(f"Executing basic operations (all data columns)")
        cols: list[str] = PropsTaxpayers.basic_clean()
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
        name_cols: list[str] = PropsTaxpayers.name_clean()
        t.print_with_dots("Replacing number ranges with first number in range")
        df = clean_df_base.take_first(df, name_cols)
        df = clean_df_name.switch_the(df, name_cols)
        console.print(f"Name field cleaning complete ✅")

        # address cleaning only
        t.print_equals(f"Executing cleaning operations (address columns only)")
        for col in PropsTaxpayers.address_clean()["street"]:
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
        df = cols_df.set_name_address_concat(df, PropsTaxpayers.name_address_concat_map()["clean"])
        t.print_with_dots("Full clean address fields generated ✅")

        # split dataframe into properties and taxpayer_records
        t.print_with_dots(f"Splitting \"{id}\" into \"taxpayer_records\" and \"properties\"...")
        df_props, df_taxpayers = subset_df.split_props_taxpayers(df, Properties.out(), TaxpayerRecords.out())
        df_taxpayers.dropna(subset=["raw_name"], inplace=True)
        console.print(f"props_taxpayers successfully split into \"taxpayer_records\" and \"properties\" ✅")

        return df_props, df_taxpayers

    def execute_business_filings_cleaning(
        self, df_bus1: pd.DataFrame, df_bus3: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        schema_map = {
            "bus_filings": BusinessFilings,
            "bus_names_addrs": BusinessNamesAddrs
        }

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

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "props_taxpayers": {
                "path": path_gen.raw_props_taxpayers(configs),
                "schema": PropsTaxpayers
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

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "props_taxpayers": PropsTaxpayers,
            "bus_filings": BusinessFilings,
            "bus_names_addrs": BusinessNamesAddrs,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # taxpayer records
        df_taxpayers: pd.DataFrame = self.dfs_in["props_taxpayers"].copy()
        df_properties, df_taxpayer_records = self.execute_taxpayer_cleaning(df_taxpayers)
        self.dfs_out["taxpayer_records"] = df_taxpayer_records
        self.dfs_out["properties"] = df_properties

        # business filings
        df_bus1: pd.DataFrame = self.dfs_in["bus_filings"].copy()
        df_bus3: pd.DataFrame = self.dfs_in["bus_names_addrs"].copy()
        df_filings, df_names_addrs = self.execute_business_filings_cleaning(df_bus1, df_bus3)
        self.dfs_out["bus_filings"] = df_filings
        self.dfs_out["bus_names_addrs"] = df_names_addrs

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "properties": path_gen.processed_properties(configs),
            "taxpayer_records": path_gen.processed_taxpayer_records(configs),
            "bus_filings": path_gen.processed_bus_filings(configs),
            "bus_names_addrs": path_gen.processed_bus_names_addrs(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
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

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "taxpayer_records": {
                "path": path_gen.processed_taxpayer_records(configs),
                "schema": TaxpayerRecords,
                "recursive_bools": True
            },
            "bus_filings": {
                "path": path_gen.processed_bus_filings(configs),
                "schema": BusinessFilings
            }
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "taxpayer_records": TaxpayerRecords,
            "bus_filings": BusinessFilings
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # get copies
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayer_records"].copy()
        df_bus: pd.DataFrame = self.dfs_in["bus_filings"].copy()
        # execute logic
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
        # set out dfs
        self.dfs_out["taxpayers_bus_merged"] = df_taxpayers

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "taxpayers_bus_merged": path_gen.processed_taxpayers_bus_merged(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflUnvalidatedAddrs(WorkflowStandardBase):

    WKFL_NAME: str = "UNVALIDATED ADDRESS GENERATION WORKFLOW"
    WKFL_DESC: str = "Fetches addresses from taxpayer and business records to run through validation service."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "taxpayers_bus_merged": {
                "path": path_gen.processed_taxpayers_bus_merged(configs),
                "schema": TaxpayersBusMerged,
            },
            "bus_names_addrs": {
                "path": path_gen.processed_bus_names_addrs(configs),
                "schema": BusinessNamesAddrs
            }
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "taxpayers_bus_merged": TaxpayersBusMerged,
            "bus_names_addrs": BusinessNamesAddrs
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # get copies
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_bus_merged"].copy()
        df_bus_names_addrs: pd.DataFrame = self.dfs_in["bus_names_addrs"].copy()
        # execute logic
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

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
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

    # --------------
    # ----LOADER----
    # --------------
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

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "unvalidated_addrs": UnvalidatedAddrs
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # get copy
        df_unvalidated_master: pd.DataFrame = self.dfs_in["unvalidated_addrs"].copy()
        # execute logic
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

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        pass

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflGeocodioFix(WorkflowStandardBase):

    WKFL_NAME: str = "WORKFLOW GEOCODIO FIX"
    WKFL_DESC: str = "Runs filters on all Geocodio partials and creates validated/unvalidated address datasets resulting from the filters."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_geocodio_partial_concatenator(self) -> pd.DataFrame:
        partials_path: Path = Path(self.config_manager.configs["data_root"]) / "geocodio" / "partials"
        dfs: list[pd.DataFrame] = []
        for file in partials_path.iterdir():
            if file.is_file() and file.suffix == ".csv":
                df = pd.read_csv(file, dtype=str)
                dfs.append(df)
        df_results = pd.concat(dfs, ignore_index=True)
        return df_results

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "unvalidated_addrs": {
                "path": path_gen.processed_unvalidated_addrs(configs),
                "schema": UnvalidatedAddrsClean,
            },
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "unvalidated_addrs": UnvalidatedAddrsClean
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:

        df_results: pd.DataFrame = self.execute_geocodio_partial_concatenator()
        clean_addrs: list[str] = list(df_results["clean_address"].unique())
        grouped: DataFrameGroupBy = df_results.groupby("clean_address")
        gcd_results_obj: GeocodioReturnObject = {  # object to be used to create/update dataframes in workflow process
            "validated": [],
            "unvalidated": [],
        }

        # Set up Rich progress display
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
            TextColumn("[bold cyan]{task.fields[processed]}/{task.total} addresses"),
        ) as progress:
            geocodio_task = progress.add_task(
                "[yellow]Processing Geocodio API call results...",
                total=len(clean_addrs),
                processed=0,
            )
            processed_count = 0
            for addr in clean_addrs:
                group = grouped.get_group(addr).reset_index(drop=True)
                flattened_results: list[GeocodioResultFlat] = []
                clean_addr_row: pd.Series = self.dfs_in["unvalidated_addrs"].loc[
                    self.dfs_in["unvalidated_addrs"]["clean_address"] == addr
                ].iloc[0]
                clean_address: CleanAddress = AddressBase.build_clean_address_object(clean_addr_row)
                for i, row in group.iterrows():
                    flattened_results.append(row.to_dict())
                results_processed: GeocodioResultProcessed = AddressBase.process_geocodio_results(
                    clean_address,
                    flattened_results
                )
                if len(results_processed["results_parsed"]) == 1:
                    new_validated = results_processed["results_parsed"][0]
                    if pd.isnull(new_validated["number"]) or new_validated["number"] == "":
                        new_unvalidated = new_validated
                        new_unvalidated["clean_address"] = clean_address["clean_address"]
                        gcd_results_obj["unvalidated"].append(new_unvalidated)
                    else:
                        new_validated["clean_address"] = row["clean_address"]
                        # new_validated["is_pobox"] = clean_address["is_pobox"]
                        gcd_results_obj["validated"].append(new_validated)
                else:
                    for result in results_processed["results_parsed"]:
                        new_unvalidated = result
                        new_unvalidated["clean_address"] = clean_address["clean_address"]
                        gcd_results_obj["unvalidated"].append(new_unvalidated)
                processed_count += 1
                progress.update(
                    geocodio_task,
                    advance=1,
                    processed=processed_count,
                    description=f"[yellow]Processing address {processed_count}/{len(clean_addrs)}"
                )

        df_gcd_validated: pd.DataFrame = pd.DataFrame(gcd_results_obj["validated"])
        df_gcd_unvalidated: pd.DataFrame = pd.DataFrame(gcd_results_obj["unvalidated"])

        self.dfs_out["gcd_validated"] = df_gcd_validated
        self.dfs_out["gcd_unvalidated"] = df_gcd_unvalidated

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "gcd_validated": path_gen.geocodio_gcd_validated(configs, suffix="_test"),
            "gcd_unvalidated": path_gen.geocodio_gcd_unvalidated(configs, suffix="_test"),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflGeocodioStringMatch(WorkflowStandardBase):

    WKFL_NAME: str = "GEOCODIO STRING MATCH WORKFLOW"
    WKFL_DESC: str = "Matches geocodio results based on similarity to original address, filtering for street number and zip code equality."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "gcd_validated": {
                "path": path_gen.geocodio_gcd_validated(configs),
                "schema": Geocodio
            },
            "gcd_unvalidated": {
                "path": path_gen.geocodio_gcd_unvalidated(configs),
                "schema": Geocodio
            },
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "gcd_validated": Geocodio,
            "gcd_unvalidated": Geocodio,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # copy df for processing
        df_gcd_valid: pd.DataFrame = self.dfs_in["gcd_validated"].copy()
        df_gcd_un: pd.DataFrame = self.dfs_in["gcd_unvalidated"].copy()
        # set ref_docs
        t.print_with_dots("Executing string matching for geocodio results")
        df_gcd_un = df_gcd_un.dropna(subset=["number"])
        ref_docs: list[str] = list(df_gcd_un["formatted_address"].dropna().unique())
        console.print("REF DOC COUNT:", len(ref_docs))
        query_docs: list[str] = list(df_gcd_un["clean_address"].dropna().unique())
        console.print("QUERY DOC COUNT:", len(query_docs))
        df_string_matches: pd.DataFrame = StringMatch.match_strings(
            ref_docs,
            query_docs,
            params={
                "name_col": None,
                "match_threshold": .7,
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
        df_merged: pd.DataFrame = pd.merge(
            df_string_matches,
            df_gcd_un,
            how="left",
            left_on="matched_doc",
            right_on="formatted_address",
        )
        df_merged.drop_duplicates(subset=["original_doc"], inplace=True)
        df_merged["is_good_match_street"] = df_merged.apply(lambda row: cols_df.is_match_street(row), axis=1)
        df_merged["is_good_match_zip"] = df_merged.apply(lambda row: cols_df.is_match_zip(row), axis=1)
        df_merged["is_good_match_sec_num"] = df_merged.apply(lambda row: cols_df.is_match_secondary(row), axis=1)
        df_valid_new = df_merged[
            (df_merged["is_good_match_street"] == True) &
            (df_merged["is_good_match_zip"] == True) &
            (df_merged["is_good_match_sec_num"] != False)
        ]
        df_valid_new.drop(columns=[
            "formatted_address",
            "clean_address",
            "conf",
            "conf1",
            "ldist",
            "is_good_match_street",
            "is_good_match_zip",
            "is_good_match_sec_num",
        ], inplace=True)
        df_valid_new.rename(columns={
            "original_doc": "clean_address",
            "matched_doc": "formatted_address"
        }, inplace=True)

        df_valid_new = pd.concat([self.dfs_in["gcd_validated"],df_valid_new], ignore_index=True)
        df_unvalid_new: pd.DataFrame = self.dfs_in["gcd_unvalidated"][
            ~self.dfs_in["gcd_unvalidated"]["clean_address"].isin(
                list(df_valid_new["clean_address"].unique())
            )
        ]

        self.dfs_out["gcd_validated"] = df_valid_new
        self.dfs_out["gcd_unvalidated"] = df_unvalid_new

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "gcd_validated": path_gen.geocodio_gcd_validated(configs),
            "gcd_unvalidated": path_gen.geocodio_gcd_unvalidated(configs)
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflFixUnitsInitial(WorkflowStandardBase):
    """
    Outputs validated address subset dataset for rows whose raw (or clean) addresses contain secondary unit numbers
    but whose validated addresses do not. To be used for manual investigation and address fixing
    """
    WKFL_NAME: str = "INITIAL FIX UNITS WORKFLOW"
    WKFL_DESC: str = ("Outputs validated addresses whose raw addresses contain secondary unit numbers but whose "
                      "validated addresses do not.")

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "gcd_validated": {
                "path": path_gen.geocodio_gcd_validated(configs),
                "schema": Geocodio,
            },
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "gcd_validated": Geocodio,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # set copy
        df_unit: pd.DataFrame = self.dfs_in["gcd_validated"].copy()
        df_unit = cols_df.set_is_pobox(df_unit, "clean_address")
        # subset validated addresses for only ones which do not have a secondary number
        df_unit = df_unit[df_unit["secondarynumber"].isnull()]
        df_unit = cols_df.set_is_not_secondarynumber(df_unit)
        # subset addresses with invalid streets
        df_unit = cols_df.set_is_invalid_street(df_unit)
        df_unit = df_unit[df_unit["is_invalid_street"] == False]
        df_unit = df_unit[df_unit["is_not_secondarynumber"] == False]
        # check street addresses for digits at the end
        df_unit = cols_df.set_check_sec_num(df_unit, "clean_address")
        # subset check_sec_num results to only include rows where a number WAS detected
        df_unit = df_unit[df_unit["check_sec_num"].notnull()]
        self.dfs_out["fixing_addrs"] = df_unit

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        ss_obj = SSFixUnitsInitial(
            self.config_manager.configs,
            self.WKFL_NAME,
            self.dfs_out
        )
        ss_obj.calculate()
        ss_obj.print()
        ss_obj.save()

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "fixing_addrs": path_gen.analysis_fixing_addrs(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflFixUnitsFinal(WorkflowStandardBase):
    """
    Changes validated addresses to include unit numbers not initially detected by the workflow.
    """
    WKFL_NAME: str = "FINAL FIX UNITS WORKFLOW"
    WKFL_DESC: str = "Changes validated addresses to include unit numbers not initially detected by the workflow."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "gcd_validated": {
                "path": path_gen.geocodio_gcd_validated(configs),
                "schema": Geocodio,
            },
            "fixing_addrs": {
                "path": path_gen.analysis_fixing_addrs(configs),
                "schema": FixingAddrs,
            },
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "gcd_validated": Geocodio,
            "fixing_addrs": FixingAddrs,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # get copies
        df_valid: pd.DataFrame = self.dfs_in["gcd_validated"].copy()
        df_fix: pd.DataFrame = self.dfs_in["fixing_addrs"].copy()
        # execute logic
        t.print_equals("Adding missing unit numbers to validated addresses")
        total_addresses = len(df_fix["clean_address"])
        # Set up Rich progress display
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
            TextColumn("[bold cyan]{task.fields[processed]}/{task.total} addresses"),
        ) as progress:
            task = progress.add_task(
                "[yellow]Fixing validated addresses...",
                total=total_addresses,
                processed=0,
            )
            processed_count = 0
            for _, row in df_fix.iterrows():  # todo: add progress bar
                address: str = row["clean_address"]
                mask = df_valid["clean_address"] == address
                matching_indices = df_valid.index[mask]
                df_valid.loc[matching_indices, "secondarynumber"] = row["secondarynumber"]
                for idx in matching_indices:
                    row_to_fix = df_valid.loc[idx]
                    df_valid.loc[idx, "formatted_address"] = addr.fix_formatted_address_unit(row_to_fix)
                processed_count += 1
                progress.update(
                    task,
                    advance=1,
                    processed=processed_count,
                    description=f"[yellow]Processing address {processed_count}/{total_addresses}"
                )
        # generate formatted_address_v
        df_valid = cols_df.set_formatted_address_v1(df_valid)
        self.dfs_out["gcd_validated"] = df_valid

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        ss_obj = SSFixUnitsFinal(
            self.config_manager.configs,
            self.WKFL_NAME,
            self.dfs_out
        )
        ss_obj.calculate()
        ss_obj.print()
        ss_obj.save()

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "gcd_validated": path_gen.geocodio_gcd_validated(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflFixAddrsInitial(WorkflowStandardBase):


    WKFL_NAME: str = "FIX ADDRESSES (INITIAL)"
    WKFL_DESC: str = "Outputs subset of gcd_validated for addresses identified from manual research as requiring manual fixing."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "gcd_validated": {
                "path": path_gen.geocodio_gcd_validated(configs),
                "schema": Geocodio,
            },
            "address_analysis": {
                "path": path_gen.analysis_address_analysis(configs),
                "schema": AddressAnalysis,
                "recursive_bools": True
            }
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "gcd_validated": Geocodio,
            "address_analysis": AddressAnalysis,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # set copies
        df_analysis: pd.DataFrame = self.dfs_in["address_analysis"].copy()
        df_valid: pd.DataFrame = self.dfs_in["gcd_validated"].copy()
        # subset validated addresses
        df_to_fix = df_analysis[df_analysis["fix_address"] == True]
        addrs_to_fix: list[str] = list(df_to_fix["value"].unique())
        df_out = df_valid[df_valid["formatted_address_v"].isin(addrs_to_fix)]
        # set out df
        self.dfs_out["fixing_addrs_analysis"] = df_out

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "fixing_addrs_analysis": path_gen.analysis_fixing_addrs_analysis(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflFixAddrsFinal(WorkflowStandardBase):

    WKFL_NAME: str = "FIX ADDRESSES (FINAL)"
    WKFL_DESC: str = "Updates validated address master list with manual changes & adjustments made in fix_addrs dataset."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "fixing_addrs_analysis": {
                "path": path_gen.analysis_fixing_addrs_analysis(configs),
                "schema": Geocodio,
            },
            "gcd_validated": {
                "path": path_gen.geocodio_gcd_validated(configs),
                "schema": Geocodio,
            }
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "fixing_addrs_analysis": Geocodio,
            "gcd_validated": Geocodio,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # set copies
        df_fixed: pd.DataFrame = self.dfs_in["fixing_addrs_analysis"].copy()
        df_valid: pd.DataFrame = self.dfs_in["gcd_validated"].copy()
        # remove manually fixed address rows from validated master list
        addrs_to_drop: list[str] = list(df_fixed["clean_address"].unique())
        df_valid_dropped = df_valid[~df_valid["clean_address"].isin(addrs_to_drop)]
        # concatenate fixed addresses to master list
        df_valid_out = pd.concat([df_valid_dropped, df_fixed], ignore_index=True)
        df_valid_out = cols_df.set_formatted_address_v1(df_valid_out)
        # set out dfs
        self.dfs_out["gcd_validated_fixed"] = df_valid_out

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "gcd_validated_fixed": path_gen.geocodio_gcd_validated(configs, "_fixed"),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflSetAddressColumns(WorkflowStandardBase):

    WKFL_NAME: str = "SET VALIDATED ADDRESS COLUMNS WORKFLOW"
    WKFL_DESC: str = "Sets columns for validated address matching."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "gcd_validated": {
                "path": path_gen.geocodio_gcd_validated(configs, "_fixed"),
                "schema": Geocodio,
            }
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "gcd_validated": Geocodio,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        df_valid: pd.DataFrame = self.dfs_in["gcd_validated"].copy()
        t.print_with_dots("Setting formatted_address_v1")
        df_valid = cols_df.set_formatted_address_v1(df_valid)
        t.print_with_dots("Setting formatted_address_v2")
        df_valid = cols_df.set_formatted_address_v2(df_valid)
        t.print_with_dots("Setting formatted_address_v3")
        df_valid = cols_df.set_formatted_address_v3(df_valid)
        t.print_with_dots("Setting formatted_address_v4")
        df_valid = cols_df.set_formatted_address_v4(df_valid)
        self.dfs_out["gcd_validated_formatted"] = df_valid

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "gcd_validated_formatted": path_gen.geocodio_gcd_validated(configs, "_formatted"),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflAddressMerge(WorkflowStandardBase):

    WKFL_NAME: str = "ADDRESS MERGE WORKFLOW"
    WKFL_DESC: str = "Merges validated addresses to address fields in taxpayer and business filing records."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "gcd_validated_formatted": {
                "path": path_gen.geocodio_gcd_validated(configs, "_formatted"),
                "schema": GeocodioFormatted,
            },
            "taxpayers_bus_merged": {
                "path": path_gen.processed_taxpayers_bus_merged(configs),
                "schema": TaxpayersBusMerged,
            },
            "bus_names_addrs": {
                "path": path_gen.processed_bus_names_addrs(configs),
                "schema": BusinessNamesAddrs
            }
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "gcd_validated_formatted": GeocodioFormatted,
            "taxpayers_bus_merged": TaxpayersBusMerged,
            "bus_names_addrs": BusinessNamesAddrs,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # set df copies
        df_valid = self.dfs_in["gcd_validated_formatted"].copy()
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_bus_merged"].copy()
        df_bus_names_addrs: pd.DataFrame = self.dfs_in["bus_names_addrs"].copy()
        # run validator on validated address dataset
        # self.run_validator("gcd_validated", df_valid, self.config_manager.configs, self.WKFL_NAME, Geocodio)
        t.print_dataset_name("taxpayers_bus_merged")
        t.print_with_dots("Merging validated addresses into taxpayers_bus_merged")
        df_tax_merge = merge_df.merge_validated_address(df_taxpayers, df_valid, "clean_address")
        df_tax_merge = clean_df_base.combine_columns_parallel(df_tax_merge)
        df_tax_merge.drop_duplicates(subset=["raw_name_address"], inplace=True)
        t.print_dataset_name("bus_names_addrs")
        t.print_with_dots("Merging validated addresses into bus_names_addrs")
        df_bus_merge = merge_df.merge_validated_address(df_bus_names_addrs, df_valid, "clean_address")
        df_bus_merge = clean_df_base.combine_columns_parallel(df_bus_merge)

        console.print("Validated addresses merged ✅ 🗺️ 📍")

        self.dfs_out["taxpayers_addr_merged"] = df_tax_merge
        self.dfs_out["bus_names_addrs_merged"] = df_bus_merge


    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        ss_obj = SSAddressMerge(
            self.config_manager.configs,
            self.WKFL_NAME,
            self.dfs_out
        )
        ss_obj.calculate()
        ss_obj.print()
        ss_obj.save()

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "taxpayers_addr_merged": path_gen.processed_taxpayers_addr_merged(configs),
            "bus_names_addrs_merged": path_gen.processed_bus_names_addrs_merged(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflNameAnalysisInitial(WorkflowStandardBase):
    """
    Initial taxpayer name analysis workflow. Generates frequency and name analysis dataframes.

    INPUTS:
    OUTPUTS:
    """

    WKFL_NAME: str = "NAME ANALYSIS INITIAL WORKFLOW"
    WKFL_DESC: str = "Generates & saves dataframe with most commonly appearing names in taxpayer records."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "taxpayers_addrs_merged": {
                "path": path_gen.processed_taxpayers_addr_merged(configs),
                "schema": TaxpayersAddrMerged,
                "recursive_bools": True
            },
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "taxpayers_addrs_merged": TaxpayersAddrMerged,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        df = self.dfs_in["taxpayers_addrs_merged"].copy()
        # run validator
        # self.run_validator("taxpayer_records", df, self.config_manager.configs, self.WKFL_NAME, TaxpayerRecords)
        df_freq: pd.DataFrame = subset_df.generate_frequency_df(df, "clean_name")
        # frequent_tax_names
        self.dfs_out["frequent_tax_names"] = df_freq
        self.dfs_out["frequent_tax_names"]["exclude_name"] = ""
        # fixing_tax_names
        self.dfs_out["fixing_tax_names"] = pd.DataFrame(
            columns=["raw_value", "standardized_value"],
            data=[
                [
                    "Paste the EXACT string from the messy data to be standardized in this column",
                    "Paste the fixed version in this column"
                ],
                ["EXAMPLES","EXAMPLES"],
                ["COMMUNITY SAV BK LT", "COMMUNITY SAVINGS BANK"],
                ["COMMUNITY SAV BK TR", "COMMUNITY SAVINGS BANK"],
                ["COMMUNITY SAV BK", "COMMUNITY SAVINGS BANK"],
                ["COMMUNITY SAV BANK", "COMMUNITY SAVINGS BANK"],
                ["COMMUNITY BK TR LT", "COMMUNITY SAVINGS BANK"],
                ["COMM SAVINGS BK LT", "COMMUNITY SAVINGS BANK"],
                ["COMM SAVGS BK LT", "COMMUNITY SAVINGS BANK"],
            ]
        )

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        ss_obj = SSNameAnalysisInitial(
            self.config_manager.configs,
            self.WKFL_NAME,
            self.dfs_out
        )
        ss_obj.calculate()
        ss_obj.print()
        ss_obj.save()

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "frequent_tax_names": path_gen.analysis_frequent_tax_names(configs),
            "fixing_tax_names": path_gen.analysis_fixing_tax_names(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflAddressAnalysisInitial(WorkflowStandardBase):
    """
    Address analysis

    INPUTS:
        - Master validated address file
            - 'ROOT/processed/validated_addrs[FileExt]'
        - User inputs manual address research data
    OUTPUTS:
        - Address analysis dataset
            - 'ROOT/analysis/address_freq[FileExt]'
    """
    WKFL_NAME: str = "ADDRESS ANALYSIS INITIAL WORKFLOW"
    WKFL_DESC: str = "Generates & saves dataframe with most commonly appearing names in taxpayer records."

    ANALYSIS_FIELDS: list[str] = [
        "name",
        "urls",
        "notes",
        "is_landlord_org",
        "is_govt_agency",
        "is_lawfirm",
        "is_missing_suite",
        "is_problematic_suite",
        "is_religious",
        "is_realtor",
        "is_financial_services",
        "is_assoc_bus",
        "fix_address",
        "is_virtual_office_agent",
        "yelp_urls",
        "is_nonprofit",
        "google_urls",
        "is_ignore_misc",
        "google_place_id",
        "is_researched"
    ]

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "gcd_validated_formatted": {
                "path": path_gen.geocodio_gcd_validated(configs, suffix="_formatted"),
                "schema": GeocodioFormatted,
            },
            "taxpayers_addrs_merged": {
                "path": path_gen.processed_taxpayers_addr_merged(configs),
                "schema": TaxpayersAddrMerged,
                "recursive_bools": True
            },
            "bus_names_addrs_merged": {
                "path": path_gen.processed_bus_names_addrs_merged(configs),
                "schema": BusinessNamesAddrsMerged,
            }
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "gcd_validated_formatted": GeocodioFormatted,
            "taxpayers_addrs_merged": TaxpayersAddrMerged,
            "bus_names_addrs_merged": BusinessNamesAddrsMerged,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        addrs = []
        for id, df in self.dfs_in.items():
            # self.run_validator(id, df_in, self.config_manager.configs, self.WKFL_NAME, schema_map[id])
            if id == "gcd_validated":
                continue
            addrs.extend(
                [
                    addr
                    for addr in df["clean_address_v"]
                    if pd.notnull(addr) and addr != ""
                ]
            )
        df_addrs: pd.DataFrame = pd.DataFrame(columns=["address"], data=addrs)
        df_freq: pd.DataFrame = subset_df.generate_frequency_df(df_addrs, "address")
        for field in self.ANALYSIS_FIELDS:
            df_freq[field] = ""
        df_freq["is_researched"] = "f"
        self.dfs_out["address_analysis"] = df_freq

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        ss_obj = SSAddressAnalysisInitial(
            self.config_manager.configs,
            self.WKFL_NAME,
            self.dfs_out
        )
        ss_obj.calculate()
        ss_obj.print()
        ss_obj.save()

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "address_analysis": path_gen.analysis_address_analysis(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflAnalysisFinal(WorkflowStandardBase):
    """
    Fixes taxpayer names based on standardized spellings manually specified in the fixing_tax_names dataset. Adds
    boolean columns to taxpayer records for exclude_name, based on manual input in
    fixing_tax_names dataset.
    """
    WKFL_NAME: str = "FIX NAMES ADDRESSES WORKFLOW"
    WKFL_DESC: str = "Changes taxpayer names and validated addresses based on manual input."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def get_banks_dict(self, df: pd.DataFrame) -> dict:
        banks = {}
        for standard_name in list(df["standardized_value"].unique()):
            df_name: pd.DataFrame = df[df["standardized_value"] == standard_name]
            for raw_name in list(df_name["raw_value"].unique()):
                banks[raw_name] = standard_name
        return banks

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "fixing_tax_names": {
                "path": path_gen.analysis_fixing_tax_names(configs),
                "schema": FixingTaxNames,
            },
            "frequent_tax_names": {
                "path": path_gen.analysis_frequent_tax_names(configs),
                "schema": FrequentTaxNames,
            },
            "taxpayers_addr_merged": {
                "path": path_gen.processed_taxpayers_addr_merged(configs),
                "schema": TaxpayersAddrMerged,
                "recursive_bools": True
            },
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "fixing_tax_names": FixingTaxNames,
            "frequent_tax_names": FrequentTaxNames,
            "taxpayers_addr_merged": TaxpayersAddrMerged,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # copy dfs
        df_fix_names: pd.DataFrame = self.dfs_in["fixing_tax_names"].copy()
        df_freq_names: pd.DataFrame = self.dfs_in["frequent_tax_names"].copy()
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_addr_merged"].copy()

        df_taxpayers.dropna(subset=["raw_name"], inplace=True)
        # create banks dict
        t.print_with_dots("Setting standardized name dictionary")
        banks: dict[str, str] = self.get_banks_dict(df_fix_names)
        t.print_with_dots("Fixing taxpayer names")
        df_taxpayers = colm_df.fix_tax_names(df_taxpayers, banks)
        t.print_with_dots("Setting exclude_name boolean column")
        df_taxpayers = cols_df.set_exclude_name(df_taxpayers, df_freq_names)
        # t.print_with_dots("Setting is_landlord_org boolean column")
        # df_taxpayers = cols_df.set_is_landlord_org(df_taxpayers, df_analysis)
        t.print_with_dots("Setting clean_name_address field with fixed names")
        df_taxpayers = cols_df.set_name_address_concat_fix(
            df_taxpayers,
            {
                "name_addr": "clean_name_address",
                "name": "clean_name",
                "name_2": "clean_name_2",
                "addr": "clean_address"
            }
        )
        self.dfs_out["taxpayers_fixed"] = df_taxpayers

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        ss_obj = SSAnalysisFinal(
            self.config_manager.configs,
            self.WKFL_NAME,
            self.dfs_out
        )
        ss_obj.calculate()
        ss_obj.print()
        ss_obj.save()

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "taxpayers_fixed": path_gen.processed_taxpayers_fixed(configs)
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflRentalSubset(WorkflowStandardBase):

    WKFL_NAME: str = "RENTAL SUBSET WORKFLOW"
    WKFL_DESC: str = "Subsets property and taxpayer record datasets for rental properties only."

    PROP_TYPES: list[str] = [  # todo: move to schema class
        "2 UNIT RESIDENTIAL",
        "3 UNIT RESIDENTIAL",
        "APARTMENT",
        "COMMERCIAL",
        "VACANT LAND - RESIDENTIAL"
    ]
    LAND_USES: list[str] = [
        "2 UNIT RESIDENTIAL - DUPLEX",
        "2 UNIT RESIDENTIAL - SF HOUSE AND ADU",
        "2 UNIT RESIDENTIAL - SF HOUSE AND CARRIAGE HOUSE",
        "2 UNIT RESIDENTIAL - TWO HOUSES",
        "3 UNIT RESIDENTIAL - DUPLEX AND ADU",
        "3 UNIT RESIDENTIAL - DUPLEX AND SF HOUSE",
        "3 UNIT RESIDENTIAL - TRIPLEX",
        "MIXED OFFICE, RETAIL, RESIDENTIAL, ETC",
        "MULTI - FAMILY APARTMENT",
        "MULTI - FAMILY RESIDENTIAL",
        "OFFICE STRUCTURE",
        "VACANT"
    ]
    BUILDING_USES: list[str] = [
        "APARTMENT 4 OR 5 UNIT",
        "APARTMENT CONVERTED",
        "BAR / FOOD / REST.W RES",
        "BOARDING OR LODGING",
        "COMMERCIAL",
        "DUPLEX",
        "DUPLEX W / ADU",
        "GROUP HOME",
        "TRIPLEX"
    ]

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "taxpayers_fixed": {
                "path": path_gen.processed_taxpayers_fixed(configs),
                "schema": TaxpayersFixed,
                "recursive_bools": True
            },
            "properties": {
                "path": path_gen.processed_properties(configs),
                "schema": Properties,
            },
            "rental_licenses": {
                "path": path_gen.pre_process_rental_licenses(configs),
                "schema": None,
            },
            "address_analysis": {
                "path": path_gen.analysis_address_analysis(configs),
                "schema": AddressAnalysis,
            }
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "taxpayers_fixed": TaxpayersFixed,
            "properties": Properties,
            "address_analysis": AddressAnalysis,
        }
        for id, df in self.dfs_in.items():
            if id == "rental_licenses":
                continue
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # copy dfs
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_fixed"].copy()
        df_props: pd.DataFrame = self.dfs_in["properties"].copy()
        df_lic: pd.DataFrame = self.dfs_in["rental_licenses"].copy()
        df_analysis: pd.DataFrame = self.dfs_in["address_analysis"].copy()
        t.print_with_dots("Fetching property pins for rental property subset of taxpayer data")
        # 1. subset by rental licenses
        license_pins: list[str] = list(df_lic["apn"].dropna().unique())
        # 2. subset by non-homesteaded
        is_homestead_pins: list[str] = list(df_props[df_props["is_homestead"] == "NON-HOMESTEADED"]["pin"])
        # 3. subset by prop_type
        type_pins: list[str] = list(df_props[df_props["prop_type"].isin(self.PROP_TYPES)]["pin"])
        # 4. subset by land_use
        land_use_pins: list[str] = list(df_props[df_props["land_use"].isin(self.LAND_USES)]["pin"])
        # 5. subset by building_use
        bldg_use_pins: list[str] = list(df_props[df_props["building_use"].isin(self.BUILDING_USES)]["pin"])
        # 6. subset by num_units
        df_props = cols_df.set_is_unit_gte_1(df_props)
        num_units_pins: list[str] = list(df_props[df_props["is_unit_gte_1"] == True]["pin"])
        # use set of pins to subset original props df
        rental_pins: set[str] = set(license_pins + is_homestead_pins + type_pins + land_use_pins + bldg_use_pins + num_units_pins)
        df_rentals: pd.DataFrame = df_props[df_props["pin"].isin(rental_pins)]
        # 7. subset by rental addresses - pull in remaining properties based on matching addresses from rental subset
        # 7a. get taxpayer addresses
        rental_taxpayers: list[str] = list(df_rentals["raw_name_address"].unique())
        t.print_with_dots("Executing sub-subset on properties excluded from initial subset")
        # get addrs to exclude
        registered_agent_addrs: list[str] = list(df_analysis[df_analysis["is_virtual_office_agent"] == True]["value"])
        financial_services_addrs: list[str] = list(df_analysis[df_analysis["is_financial_services"] == True]["value"])
        law_firm_addrs: list[str] = list(df_analysis[df_analysis["is_lawfirm"] == True]["value"])
        exclude_addrs: set[str] = set(registered_agent_addrs + financial_services_addrs + law_firm_addrs)
        # subset taxpayers for rentals only
        df_taxpayers_r: pd.DataFrame = df_taxpayers[df_taxpayers["raw_name_address"].isin(rental_taxpayers)]
        df_taxpayers_nr: pd.DataFrame = df_taxpayers[~df_taxpayers["raw_name_address"].isin(rental_taxpayers)]
        # fetch taxpayer addresses to execute final subset
        addrs_to_subset: list[str] = list(df_taxpayers_r[~df_taxpayers_r["clean_address_v1"].isin(exclude_addrs)])
        df_nonrentals_r: pd.DataFrame = df_taxpayers_nr[df_taxpayers_nr["clean_address_v1"].isin(addrs_to_subset)]
        df_subset_final: pd.DataFrame = pd.concat([df_taxpayers_r, df_nonrentals_r], ignore_index=True)

        self.dfs_out["properties_rentals"] = df_rentals
        self.dfs_out["taxpayers_subsetted"] = df_subset_final

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "properties_rentals": path_gen.processed_properties_rentals(configs),
            "taxpayers_subsetted": path_gen.processed_taxpayers_subsetted(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflMatchAddressCols(WorkflowStandardBase):

    WKFL_NAME: str = "MATCH ADDRESS & BOOLEAN IDENTIFIER GENERATORS WORKFLOW"
    WKFL_DESC: str = "Assigns boolean identifiers for various address analysis categories. Subsets business filings data by presence of validated addresses from initial subset. Generates name + address concatenated columns for string matching."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "taxpayers_subsetted": {
                "path": path_gen.processed_taxpayers_subsetted(configs),
                "schema": TaxpayersSubsetted,
                "recursive_bools": True
            },
            "bus_names_addrs_merged": {
                "path": path_gen.processed_bus_names_addrs_merged(configs),
                "schema": BusinessNamesAddrsMerged,
            },
            "address_analysis": {
                "path": path_gen.analysis_address_analysis(configs),
                "schema": AddressAnalysis,
                "recursive_bools": True
            },
            "gcd_validated_formatted": {
                "path": path_gen.geocodio_gcd_validated(configs, "_formatted"),
                "schema": GeocodioFormatted,
            }
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "taxpayers_subsetted": TaxpayersSubsetted,
            "bus_names_addrs_merged": BusinessNamesAddrsMerged,
            "address_analysis": AddressAnalysis,
            "gcd_validated_formatted": GeocodioFormatted
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # copy dfs
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_subsetted"].copy()
        df_bus: pd.DataFrame = self.dfs_in["bus_names_addrs_merged"].copy()
        df_analysis: pd.DataFrame = self.dfs_in["address_analysis"].copy()
        df_valid: pd.DataFrame = self.dfs_in["gcd_validated_formatted"].copy()

        # subset business filings
        t.print_with_dots("Subsetting business filings data")
        # fetch all uids merged into taxpayer records via name matching
        uids: list[str] = list(df_taxpayers["uid"].dropna().unique())
        # subset business name and address records for ONLY records associated with matched uids
        df_uids: pd.DataFrame = df_bus[df_bus["uid"].isin(uids)]
        # fetch unique validated addresses for matched entities
        addrs_for_uids: list[str] = list(df_uids["clean_address_v1"].dropna().unique())

        # subset entire dataset for ALL records associated with validated addresses
        uids_sub: set[str] = set(list(df_bus[df_bus["clean_address_v1"].isin(addrs_for_uids)]["uid"].unique()) + uids)
        df_bus_subset: pd.DataFrame = df_bus[df_bus["uid"].isin(uids_sub)]

        # set address lists for bool col generators
        t.print_with_dots("Fetching researched address lists to be used in boolean column generators")
        df_exclude: pd.DataFrame = df_analysis[
            (df_analysis["is_lawfirm"] == True) |
            (df_analysis["is_financial_services"] == True) |
            (df_analysis["is_virtual_office_agent"] == True)
        ]
        exclude_addrs: list[str] = list(df_exclude["value"])
        pobox_addrs: list[str] = list(df_valid[df_valid["street"] == "PO BOX"]["formatted_address_v1"].unique())
        researched_addrs: list[str] = list(df_analysis[df_analysis["is_researched"] == True]["value"])
        org_addrs: list[str] = list(df_analysis[df_analysis["is_landlord_org"] == True]["value"])
        missing_suite_addrs: list[str] = list(df_analysis[df_analysis["is_missing_suite"] == True]["value"])
        problem_suite_addrs: list[str] = list(df_analysis[df_analysis["is_problematic_suite"] == True]["value"])
        realtor_addrs: list[str] = list(df_analysis[df_analysis["is_realtor"] == True]["value"])

        for id, df in {"taxpayers": df_taxpayers, "business_filings": df_bus_subset}.items():
            t.print_dataset_name(id)
            t.print_with_dots(f"Setting match_address_v1 for {id}")
            df = cols_df.set_match_address(df, "clean_address_v1", "_v1")
            t.print_with_dots(f"Setting match_address_v2 for {id}")
            df = cols_df.set_match_address(df, "clean_address_v2", "_v2")
            t.print_with_dots(f"Setting match_address_v3 for {id}")
            df = cols_df.set_match_address(df, "clean_address_v3", "_v3")
            t.print_with_dots(f"Setting match_address_v4 for {id}")
            df = cols_df.set_match_address(df, "clean_address_v4", "_v4")
            t.print_with_dots(f"Setting is_validated for {id}")
            df = cols_df.set_is_validated(df, "clean_address_v1")
            t.print_with_dots(f"Setting exclude_address for {id}")
            df = cols_df.set_exclude_address(exclude_addrs, df, "clean_address_v1")
            t.print_with_dots(f"Setting is_researched for {id}")
            df = cols_df.set_is_researched(researched_addrs + pobox_addrs, df, "clean_address_v1")
            t.print_with_dots(f"Setting is_org_address for {id}")
            df = cols_df.set_is_org_address(org_addrs, df, "clean_address_v1")
            t.print_with_dots(f"Setting is_missing_suite for {id}")
            df = cols_df.set_is_missing_suite(missing_suite_addrs, df, "clean_address_v1")
            t.print_with_dots(f"Setting is_problem_suite for {id}")
            df = cols_df.set_is_problem_suite(problem_suite_addrs, df, "clean_address_v1")
            t.print_with_dots(f"Setting is_realtor for {id}")
            df = cols_df.set_is_realtor(realtor_addrs, df, "clean_address_v1")

        # add name+address concatenated columns to use for matching
        t.print_with_dots("Adding name + address concatenation columns")
        for suffix in ["_v1", "_v2", "_v3", "_v4"]:
            df_taxpayers = cols_df.concatenate_name_addr(
                df_taxpayers, "clean_name", f"match_address{suffix}", suffix
            )
            df_taxpayers = cols_df.concatenate_name_addr(
                df_taxpayers, "core_name", f"match_address{suffix}", suffix
            )

        self.dfs_out["taxpayers_prepped"] = df_taxpayers
        self.dfs_out["bus_names_addrs_subsetted"] = df_bus_subset

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "taxpayers_prepped": path_gen.processed_taxpayers_prepped(configs),
            "bus_names_addrs_subsetted": path_gen.processed_bus_names_addrs_subsetted(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflStringMatch(WorkflowStandardBase):

    WKFL_NAME: str = "TAXPAYER RECORD STRING-MATCHING WORKFLOW"
    WKFL_DESC: str = "Executes string matching based on concatenation of taxpayer name and address."

    DEFAULT_NMSLIB: NmslibOptions = {
        "method": "hnsw",
        "space": "cosinesimil_sparse_fast",
        "data_type": nmslib.DataType.SPARSE_VECTOR
    }
    DEFAULT_QUERY_BATCH = {
        "num_threads": 8,
        "K": 3
    }

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.params_matrix: list[StringMatchParamsMN] = []
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_param_builder(self) -> None:
        t.print_with_dots("Building string matching params object")
        # set options for params matrix
        taxpayer_name_col: list[str] = ["clean_name"]
        match_threshold_options: list[float] = [0.85]
        unvalidated_options: list[bool] = [False, True]
        unresearched_options: list[bool] = [False, True]
        org_options: list[bool] = [False, True]
        missing_suite_options: list[bool] = [False, True]
        problem_suite_options: list[bool] = [False, True]
        address_suffix: list[str] = ["v1", "v2", "v3", "v4"]
        # loop through unique combinations of param matrix options
        for i, params in enumerate(product(
            taxpayer_name_col,
            match_threshold_options,
            unvalidated_options,
            unresearched_options,
            org_options,
            missing_suite_options,
            problem_suite_options,
            address_suffix,
        )):
            self.params_matrix.append({
                "name_col": params[0],
                "match_threshold": params[1],
                "include_unvalidated": params[2],
                "include_unresearched": params[3],
                "include_orgs": params[4],
                "include_missing_suites": params[5],
                "include_problem_suites": params[6],
                "address": params[7],
                "nmslib_opts": self.DEFAULT_NMSLIB,
                "query_batch_opts": self.DEFAULT_QUERY_BATCH,
            })

    def execute_string_matching(self,df_taxpayers: pd.DataFrame) -> pd.DataFrame:
        """Returns final dataset to be outputted"""
        t.print_with_dots("Executing string matching")
        for i, params in enumerate(self.params_matrix):
            t.print_equals(f"Matching strings for STRING_MATCHED_NAME_{i+1}")
            console.print("NAME COLUMN:", params["name_col"])
            console.print("MATCH THRESHOLD:", params["match_threshold"])
            console.print("INCLUDE ORGS:", params["include_orgs"])
            console.print("INCLUDE UNRESEARCHED ADDRESSES:", params["include_unresearched"])
            console.print("INCLUDE MISSING SUITES:", params["include_missing_suites"])
            console.print("INCLUDE PROBLEMATIC SUITES:", params["include_problem_suites"])
            console.print("ADDRESS COLUMN:", params["address_suffix"])
            t.print_with_dots("Setting include_address")
            df_taxpayers["include_address"] = df_taxpayers.apply(
                lambda row: MatchBase.check_address_mpls(
                    row["match_address_v1"],  # this is just used to test nan values in the address field
                    row["is_validated"],
                    row["is_researched"],
                    row["exclude_address"],
                    row["is_org_address"],
                    row["is_missing_suite"],
                    row["is_problem_suite"],
                    params["include_unvalidated"],
                    params["include_unresearched"],
                    params["include_orgs"],
                    params["include_missing_suites"],
                    params["include_problem_suites"],
                    params["address_suffix"]
                ), axis=1
            )
            # filter out addresses
            t.print_with_dots("Filtering out taxpayer records where include_address is False")
            df_filtered: pd.DataFrame = df_taxpayers[df_taxpayers["include_address"] == True][[
                "clean_name",
                "core_name",
                f"match_address_{params['address']}",
                f"clean_name_address_{params['address']}",
                f"core_name_address_{params['address']}"
            ]]
            # set ref & query docs
            t.print_with_dots("Setting document objects for HNSW index")
            ref_docs: list[str] = list(
                df_filtered[f"{params['name_col']}_address_{params['address']}"].dropna().unique()
            )
            query_docs: list[str] = list(
                df_filtered[f"{params['name_col']}_address_{params['address']}"].dropna().unique()
            )
            # get string matches
            df_matches: pd.DataFrame = StringMatch.match_strings(
                ref_docs=ref_docs,
                query_docs=query_docs,
                params=params
            )
            # generate network graph to associated matches
            df_matches_networked: pd.DataFrame = NetworkMatchBase.string_match_network_graph(df_matches)
            # t.print_with_dots("Merging string match results into taxpayer records dataset")
            df_taxpayers = pd.merge(
                df_taxpayers,
                df_matches_networked[["original_doc", "fuzzy_match_combo"]],
                how="left",
                left_on=f"{params['name_col']}_address_{params['address']}",
                right_on="original_doc"
            )
            gc.collect()
            df_taxpayers.drop(columns="original_doc", inplace=True)
            df_taxpayers.drop_duplicates(subset="raw_name_address", inplace=True)
            df_taxpayers.rename(columns={"fuzzy_match_combo": f"string_matched_name_{i+1}"}, inplace=True)

        return df_taxpayers

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "taxpayers_prepped": {
                "path": path_gen.processed_taxpayers_prepped(configs),
                "schema": TaxpayersPrepped,
                "recursive_bools": True
            },
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "taxpayers_prepped": TaxpayersPrepped,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # copy df
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_prepped"].copy()
        # generate matrix parameters
        self.execute_param_builder()
        # run matching
        df_taxpayers_matched = self.execute_string_matching(df_taxpayers)
        # set out dfs
        self.dfs_out["taxpayers_string_matched"] = df_taxpayers_matched

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        ss_obj = SSStringMatch(
            self.config_manager.configs,
            self.WKFL_NAME,
            self.dfs_out,
            self.params_matrix
        )
        ss_obj.calculate()
        ss_obj.print()
        ss_obj.save()

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "taxpayers_string_matched": path_gen.processed_taxpayers_string_matched(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflNetworkGraph(WorkflowStandardBase):

    WKFL_NAME: str = "NETWORK GRAPH WORKFLOW"
    WKFL_DESC: str = "Executes network graph generation linking taxpayer, corporate and LLC records."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.params_matrix: list[NetworkMatchParams] = []
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_param_builder(self) -> None:
        t.print_with_dots("Building network graph params object")
        # set options for params matrix
        taxpayer_name_col: list[str] = ["clean_name"]
        unvalidated_options: list[bool] = [False, True]
        unresearched_options: list[bool] = [False, True]
        org_options: list[bool] = [False, True]
        missing_suite_options: list[bool] = [False, True]
        problem_suite_options: list[bool] = [False, True]
        address_suffix: list[str] = ["v1", "v2", "v3", "v4"]
        string_match_names: list[str] = [
            "string_matched_name_4",
            "string_matched_name_23",
            "string_matched_name_40",
            "string_matched_name_106"
        ]
        # loop through unique combinations of param matrix options
        for i, params in enumerate(product(
            taxpayer_name_col,
            unvalidated_options,
            unresearched_options,
            org_options,
            missing_suite_options,
            problem_suite_options,
            address_suffix,
            string_match_names
        )):
            if params[6] == "v2" and params[4] == False:
                continue
            if params[6] == "v4" and params[4] == False:
                continue
            self.params_matrix.append({
                "taxpayer_name_col": params[0],
                "include_unvalidated": params[1],
                "include_unresearched": params[2],
                "include_orgs": params[3],
                "include_missing_suites": params[4],
                "include_problem_suites": params[5],
                "address_suffix": params[6],
                "string_match_name": params[7],
            })
        # self.params_matrix = [
        #     {
        #         "taxpayer_name_col": "clean_name",
        #         "include_unvalidated": False,
        #         "include_unresearched": False,
        #         "include_orgs": False,
        #         "include_missing_suites": True,
        #         "include_problem_suites": True,
        #         "address_suffix": "v1",
        #         "string_match_name": "string_matched_name_4",
        #     },
        #     {
        #         "taxpayer_name_col": "clean_name",
        #         "include_unvalidated": False,
        #         "include_unresearched": False,
        #         "include_orgs": False,
        #         "include_missing_suites": True,
        #         "include_problem_suites": True,
        #         "address_suffix": "v2",
        #         "string_match_name": "string_matched_name_4",
        #     },
        #     {
        #         "taxpayer_name_col": "clean_name",
        #         "include_unvalidated": False,
        #         "include_unresearched": False,
        #         "include_orgs": False,
        #         "include_missing_suites": True,
        #         "include_problem_suites": True,
        #         "address_suffix": "v4",
        #         "string_match_name": "string_matched_name_4",
        #     },
        #     {
        #         "taxpayer_name_col": "clean_name",
        #         "include_unvalidated": False,
        #         "include_unresearched": True,
        #         "include_orgs": False,
        #         "include_missing_suites": True,
        #         "include_problem_suites": True,
        #         "address_suffix": "v1",
        #         "string_match_name": "string_matched_name_4",
        #     },
        #     {
        #         "taxpayer_name_col": "clean_name",
        #         "include_unvalidated": False,
        #         "include_unresearched": True,
        #         "include_orgs": False,
        #         "include_missing_suites": True,
        #         "include_problem_suites": True,
        #         "address_suffix": "v2",
        #         "string_match_name": "string_matched_name_4",
        #     },
        #     {
        #         "taxpayer_name_col": "clean_name",
        #         "include_unvalidated": False,
        #         "include_unresearched": True,
        #         "include_orgs": False,
        #         "include_missing_suites": True,
        #         "include_problem_suites": True,
        #         "address_suffix": "v4",
        #         "string_match_name": "string_matched_name_4",
        #     },
        # ]

    def execute_network_graph_generator(self, df_taxpayers: pd.DataFrame, df_bus: pd.DataFrame) -> pd.DataFrame:
        for i, params in enumerate(self.params_matrix):
            console.print("TAXPAYER NAME COLUMN:", params["taxpayer_name_col"])
            console.print("INCLUDE UNVALIDATED ADDRESSES:", params["include_unvalidated"])
            console.print("INCLUDE UNRESEARCHED ADDRESSES:", params["include_unresearched"])
            console.print("INCLUDE ORGS:", params["include_orgs"])
            console.print("INCLUDE MISSING SUITES:", params["include_missing_suites"])
            console.print("INCLUDE PROBLEMATIC SUITES:", params["include_problem_suites"])
            console.print("ADDRESS COLUMN:", params["address_suffix"])
            console.print("STRING MATCH NAME:", params["string_match_name"])
            # build network graph object
            gMatches = NetworkMatchBase.taxpayers_network_mpls(df_taxpayers, df_bus, params)
            # assign IDs to taxpayer records based on name/address presence in graph object
            df_taxpayers = NetworkMatchBase.set_taxpayer_component_mpls(i+1, df_taxpayers, df_bus, gMatches, params)
            # set network names for each taxpayer record (long AND short)
            df_taxpayers = NetworkMatchBase.set_network_name(i+1, df_taxpayers)
            # set text for node/edge data
            # df_taxpayers = NetworkMatchBase.set_network_text(i+1, gMatches, df_taxpayers)
        return df_taxpayers

    # --------------
    # ----LOADER----
    # --------------
    def load(self):
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "taxpayers_string_matched": {
                "path": path_gen.processed_taxpayers_string_matched(configs),
                "schema": TaxpayersPrepped,
                "recursive_bools": True
            },
            "bus_names_addrs_subsetted": {
                "path": path_gen.processed_bus_names_addrs_subsetted(configs),
                "schema": BusinessNamesAddrsSubsetted,
                "recursive_bools": True
            }
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "taxpayers_string_matched": TaxpayersStringMatched,
            "bus_names_addrs_subsetted": BusinessNamesAddrsSubsetted,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # copy dfs
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_string_matched"].copy()
        df_bus: pd.DataFrame = self.dfs_in["bus_names_addrs_subsetted"].copy()
        # generate matrix parameters
        self.execute_param_builder()
        # run network graph
        df_networked: pd.DataFrame = self.execute_network_graph_generator(df_taxpayers, df_bus)
        self.dfs_out["taxpayers_networked"] = df_networked

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        ss_obj = SSNetworkGraph(
            self.config_manager.configs,
            self.WKFL_NAME,
            self.dfs_out,
            self.params_matrix
        )
        ss_obj.calculate()
        ss_obj.print()
        ss_obj.save()

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "taxpayers_networked": path_gen.processed_taxpayers_networked(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflFinalOutput(WorkflowStandardBase):

    WKFL_NAME: str = "FINAL OUTPUT WORKFLOW"
    WKFL_DESC: str = "Generates final output datasets to be used for landlord database creation."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_network_calcs(self) -> pd.DataFrame:
        # todo: tbd, whatever Tony decides
        rows_network_calcs: list[dict[str, Any]] = [
            {
                "network_id": "",
                "taxpayer_name": "",
                "entity_name": "",
                "string_match": "",
                "include_orgs": False,
                "include_orgs_string": False,
                "include_unresearched": False,
                "include_missing_suite": False,
                "include_problem_suite": False,
                "address_suffix": "v1",
                "include_unresearched_string": False,
                "include_missing_suite_string": False,
                "include_problem_suite_string": False,
                "address_suffix_string": "v1",
            },
        ]
        return pd.DataFrame(rows_network_calcs)

    def execute_entity_types(self) -> pd.DataFrame:
        rows_entity_types: list[dict[str, Any]] = [
            {
                "name": "Landlord Organization",
                "description": "Property management company, real estate developer, real estate investment firm, or any other organization type that deals with property ownership, management, investment or development. Note that given the nature of these organizations, their direct ownership of properties cannot be definitively established, however their responsibility as the taxpayer does mean they can be held accountable for living conditions and treatment of tenants."
            },
            {
                "name": "Realtor",
                "description": "Realty company or individual realtor associated with taxpayer address.",
            },
            {
                "name": "Healthcare / Senior Care Organization",
                "description": "Healthcare company or senior care company associated with taxpayer address."
            },
            {
                "name": "Religious Institution",
                "description": "Church, mosque or other religious institution associated with taxpayer address.",
            },
            {
                "name": "Government Agency",
                "description": "Government agency at any level (local, state or federal)."
            },
            {
                "name": "Law Firm",
                "description": "Legal services firm whose office address has been confirmed and submitted to the state as the legal property taxpayer."
            },
            {
                "name": "Financial Services Company",
                "description": "Financial firm whose office address has been confirmed and submitted to the state as the legal property taxpayer. These firms offer a wide range of services, including mortgage services, tax services, accounting services, or any other financial service."
            },
            {
                "name": "Associated Business",
                "description": "Any business unrelated to property ownership, management or development but that shares the same mailing address as the property taxpayer."
            },
            {
                "name": "Virtual Office / Registered Agent",
                "description": "Virtual office or registered agent service. These services allow landlords to submit the virtual office or registered agent's mailing address as the property taxpayer instead of themselves."
            },
            {
                "name": "Nonprofit Organization",
                "description": "501(c)(3) tax-exempt not-for-profit organization."
            },
            {
                "name": "Other / Unknown",
                "description": ""
            }
        ]
        return pd.DataFrame(rows_entity_types)

    def execute_entities(self, df_researched: pd.DataFrame) -> pd.DataFrame:
        rows_entities: list[dict[str, Any]] = []
        for _, row in df_researched.iterrows():
            out_row = {
                "name": row["name"],
                "urls": row["urls"],
                "yelp_urls": row["yelp_urls"],
                "google_urls": row["google_urls"],
                "google_place_id": row["google_place_id"],
            }
            if row["is_landlord_org"] == True:
                out_row["entity_type"] = "Landlord Organization"
            elif row["is_govt_agency"] == True:
                out_row["entity_type"] = "Government Agency"
            elif row["is_lawfirm"] == True:
                out_row["entity_type"] = "Law Firm"
            elif row["is_financial_services"] == True:
                out_row["entity_type"] = "Financial Services Company"
            elif row["is_assoc_bus"] == True:
                out_row["entity_type"] = "Associated Business"
            elif row["is_virtual_office_agent"] == True:
                out_row["entity_type"] = "Virtual Office / Registered Agent"
            elif row["is_nonprofit"] == True:
                out_row["entity_type"] = "Nonprofit Organization"
            elif row["is_religious"] == True:
                out_row["entity_type"] = "Religious Institution"
            elif row["is_healthcare_senior"] == True:
                out_row["entity_type"] = "Healthcare / Senior Care Organization"
            elif row["is_realtor"] == True:
                out_row["entity_type"] = "Realtor"
            else:
                out_row["entity_type"] = "Other / Unknown"
            rows_entities.append(out_row)
        return pd.DataFrame(rows_entities)

    def execute_validated_addresses(self, df_researched: pd.DataFrame) -> pd.DataFrame:
        df_validated_addresses: pd.DataFrame = self.dfs_in["gcd_validated"][[
            "number",
            "predirectional",
            "prefix",
            "street",
            "suffix",
            "postdirectional",
            "secondaryunit",
            "secondarynumber",
            "city",
            "county",
            "state",
            "zip",
            "country",
            "lng",
            "lat",
            "accuracy",
            "formatted_address",
            "formatted_address_v1"
        ]]
        df_validated_addresses.drop_duplicates(subset=["formatted_address"], inplace=True)
        df_validated_addresses["landlord_entity"] = None
        with t.create_progress_bar(
            "[yellow]Setting landlord_entity names for validated addresses...", len(df_researched)
        )[0] as progress:
            task = progress.tasks[0]
            processed_count = 0
            for _, row in df_researched.iterrows():
                mask = df_validated_addresses["formatted_address_v1"] == row["value"]
                df_validated_addresses.loc[mask, "landlord_entity"] = row["name"]
                processed_count += 1
                progress.update(
                    task.id,
                    advance=1,
                    processed=processed_count,
                    description=f"[yellow]Processing entity {processed_count}/{len(df_researched)}",
                )
        return df_validated_addresses

    def execute_business_filings(self) -> pd.DataFrame:
        df_bus_filings: pd.DataFrame = self.dfs_in["business_filings"]
        return df_bus_filings

    def execute_business_names_addrs(self) -> pd.DataFrame:
        df_bus_names_addrs: pd.DataFrame = self.dfs_in["business_names_addrs"][[
            "uid",
            "name_type",
            "address_type",
            "raw_party_name",
            "clean_party_name",
            "clean_address",
            "clean_address_v1",
        ]]
        df_bus_names_addrs.rename(columns={
            "raw_party_name": "raw_name",
            "clean_party_name": "clean_name",
            "clean_address": "address",
            "clean_address_v1": "address_v",
        })
        return df_bus_names_addrs

    def execute_networks(self) -> pd.DataFrame:
        df_networks_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_networked"][[
            "network_1",
            "network_1_short",
            "network_1_text",
            "network_2",
            "network_2_short",
            "network_2_text",
            "network_3",
            "network_3_short",
            "network_3_text",
            "network_4",
            "network_4_short",
            "network_4_text",
            "network_5",
            "network_5_short",
            "network_5_text",
            "network_6",
            "network_6_short",
            "network_6_text"
        ]]
        rows_networks: list[dict[str, Any]] = []
        network_ids: list[str] = ["network_1", "network_2", "network_3", "network_4", "network_5", "network_6"]
        for ntwk in network_ids:
            df_network: pd.DataFrame = df_networks_taxpayers[[
                ntwk,
                f"{ntwk}_short",
                f"{ntwk}_text"
            ]]
            df_network.dropna(subset=[ntwk], inplace=True)
            df_network.drop_duplicates(subset=[ntwk], inplace=True)
            for _, row in df_network.iterrows():
                row = {
                    "name": row[ntwk],
                    "short_name": row[f"{ntwk}_short"],
                    "network_calc": ntwk,
                    "nodes_edges": row[f"{ntwk}_text"]
                }
                rows_networks.append(row)
        return pd.DataFrame(rows_networks)

    def execute_taxpayer_records(self) -> pd.DataFrame:
        # taxpayer_records
        df_taxpayer_records: pd.DataFrame = self.dfs_in["taxpayers_networked"][[
            "raw_name",
            "raw_name_2",
            "clean_name",
            "clean_name_2",
            "clean_address",
            "clean_address_v1",
            "uid",
            "network_1",
            "network_2",
            "network_3",
            "network_4",
            "network_5",
            "network_6"
        ]]
        df_taxpayer_records.rename(columns={
            "uid": "entity_uid",
            "clean_address": "address",
            "clean_address_v1": "address_v",
        })
        return df_taxpayer_records

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, dict[str, Any]] = {
            "taxpayers_networked": {
                "path": path_gen.processed_taxpayers_networked(configs),
                "schema": None,
            },
            "business_filings": {
                "path": path_gen.processed_bus_filings(configs),
                "schema": None,
            },
            "business_names_addrs": {
                "path": path_gen.processed_bus_names_addrs_subsetted(configs),
                "schema": None,
            },
            "address_analysis": {
                "path": path_gen.analysis_address_analysis(configs),
                "schema": AddressAnalysis,
                "recursive_bools": True
            },
            "gcd_validated_formatted": {
                "path": path_gen.geocodio_gcd_validated(configs, suffix="_formatted"),
                "schema": GeocodioFormatted,
            },
        }
        self.load_dfs(load_map)

    # -----------------
    # ----VALIDATOR----
    # -----------------
    def validate(self) -> None:
        schema_map = {
            "taxpayers_networked": TaxpayersNetworked,
            "business_filings": BusinessFilings,
            "business_names_addrs": BusinessNamesAddrs,
            "address_analysis": AddressAnalysis,
            "gcd_validated_formatted": GeocodioFormatted,
        }
        for id, df in self.dfs_in.items():
            self.run_validator(id, df, self.config_manager.configs, self.WKFL_NAME, schema_map[id])

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        df_researched: pd.DataFrame = self.dfs_in["address_analysis"][
            self.dfs_in["address_analysis"]["is_researched"] == True
        ]
        df_researched.dropna(subset=["name"], inplace=True)
        self.dfs_out["network_calcs"] = self.execute_network_calcs()
        self.dfs_out["entity_types"] = self.execute_entity_types()
        self.dfs_out["entities"] = self.execute_entities(df_researched)
        self.dfs_out["validated_addresses"] = self.execute_validated_addresses(df_researched)
        self.dfs_out["business_filings"] = self.execute_business_filings()
        self.dfs_out["business_names_addrs"] = self.execute_business_names_addrs()
        self.dfs_out["networks"] = self.execute_networks()
        self.dfs_out["taxpayer_records"] = self.execute_taxpayer_records()

    # -------------------------------
    # ----SUMMARY STATS GENERATOR----
    # -------------------------------
    def summary_stats(self) -> None:
        pass

    # -------------
    # ----SAVER----
    # -------------
    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "network_calcs": path_gen.output_network_calcs(configs),
            "entity_types": path_gen.output_entity_types(configs),
            "entities": path_gen.output_entities(configs),
            "validated_addresses": path_gen.output_validated_addresses(configs),
            "business_filings": path_gen.output_business_filings(configs),
            "business_names_addrs": path_gen.output_business_names_addrs(configs),
            "networks": path_gen.output_networks(configs),
            "taxpayer_records": path_gen.output_taxpayer_records(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass