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

# 2. Third-party imports
import pandas as pd

from opndb.constants.base import STATES_ABBREVS
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
    TaxpayersNetworked
from opndb.schema.v0_1.raw import (
    PropsTaxpayers,
    Corps as CorpsRaw,
    LLCs as LLCsRaw,
    ClassCodes
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
)
from rich.console import Console
from rich.prompt import Prompt

from opndb.validator.df_model import OPNDFModel

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
        elif wkfl_id == "address_clean":
            return WkflAddressClean(config_manager)
        elif wkfl_id == "address_geocodio":
            return WkflAddressGeocodio(config_manager)
        elif wkfl_id == "fix_units_initial":
            return WkflFixUnitsInitial(config_manager)
        elif wkfl_id == "fix_units_final":
            return WkflFixUnitsFinal(config_manager)
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
        elif wkfl_id == "clean_merge":
            return WkflCleanMerge(config_manager)
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

        # define minneapolis-specific functions
        def shift_taxpayer_data_cells(row: pd.Series) -> pd.Series:
            if pd.isnull(row["TAXPAYER_NM_3"]):
                row["TAXPAYER_NM_3"] = row["TAXPAYER_NM_2"]
                row["TAXPAYER_NM_2"] = row["TAXPAYER_NM_1"]
                row["TAXPAYER_NM_1"] = np.nan
            return row

        def fix_mpls(text: str) -> str:
            return text.replace("MPLS", "MINNEAPOLIS").replace("MNPLS", "MINNEAPOLIS")

        # extract relevant columns from both datasets
        df_city = df_city[[
            "PIN",
            "HOUSE_NO",
            "STREET_NAME",
            "UNIT_NO",
            "ZIP_POSTAL",
            "OWNERNM",
            "TAXPAYER1",
            "TAXPAYER2",
            "TAXPAYER3",
            "TAXPAYER4",
            "X",
            "Y",
            "PRIMARY_PROP_TYPE",
            "LANDVALUE",
            "BLDGVALUE",
            "TOTALVALUE",
            "IS_EXEMPT",
            "IS_HOMESTEAD",
            "TOTAL_UNITS",
            "FID"
        ]]
        df_county = df_county[[
            "PID",
            "OWNER_NM",
            "TAXPAYER_NM",
            "TAXPAYER_NM_1",
            "TAXPAYER_NM_2",
            "TAXPAYER_NM_3",
            "MUNIC_NM",
            "BUILD_YR",
            "SALE_DATE",
            "SALE_PRICE",
            "MKT_VAL_TOT",
            "TAXABLE_VAL_TOT",
            "TOT_NET_TAX",
            "TOT_SPEC_TAX",
            "TAX_TOT",
            "NET_TAX_PD",
            "TOT_PENALTY_PD",
            "PR_TYP_NM1",
            "LAT",
            "LON"
        ]]

        # misc cleaning
        t.print_with_dots("Trimming whitespace")
        df_county = clean_df_base.trim_whitespace(df_county)
        t.print_with_dots("Replacing empty strings with np.nan")
        df_county = clean_df_base.replace_with_nan(df_county)
        t.print_with_dots("Removing extra spaces")
        df_county = clean_df_base.remove_extra_spaces(df_county)
        t.print_with_dots("Removing symbols & punctuation")
        df_county = clean_df_base.remove_symbols_punctuation(df_county)

        # fix pins on df_city
        df_city["PIN"] = df_city["PIN"].apply(lambda pin: pin[1:])

        # subset county data for Minneapolis only
        df_mpls = df_county[df_county["MUNIC_NM"] == "MINNEAPOLIS"]

        # further subset relevant columns
        df_mpls = df_mpls[[
            "PID",
            "TAXPAYER_NM",
            "TAXPAYER_NM_1",
            "TAXPAYER_NM_2",
            "TAXPAYER_NM_3",
        ]]

        # shift taxpayer data
        t.print_with_dots("Shifting taxpayer data cells")
        df_mpls = df_mpls.apply(lambda row: shift_taxpayer_data_cells(row), axis=1)

        # drop Hennepin county forfeited land properties
        df_mpls.drop(df_mpls[df_mpls["TAXPAYER_NM"] == "HENNEPIN FORFEITED LAND"].index, inplace=True)

        # drop nulls
        t.print_with_dots("Dropping rows with missing taxpayer data")
        df_mpls.dropna(subset=["TAXPAYER_NM"], inplace=True)
        df_mpls.dropna(subset=["TAXPAYER_NM_2"], inplace=True)
        df_mpls.dropna(subset=["TAXPAYER_NM_3"], inplace=True)

        # fix mpls/mnpls
        df_mpls["TAXPAYER_NM_3"] = df_mpls["TAXPAYER_NM_3"].apply(lambda x: fix_mpls(x))

        # set address columns
        t.print_with_dots("Setting address column")
        df_mpls["tax_address"] = df_mpls.apply(lambda row: f"{row['TAXPAYER_NM_2']}, {row['TAXPAYER_NM_3']}", axis=1)

        # merge relevant data from df_city
        t.print_with_dots("Merging city dataset into county dataset")
        df_taxpayers: pd.DataFrame = pd.merge(df_mpls, df_city[[
            "PIN",
            "PRIMARY_PROP_TYPE",
            "IS_EXEMPT",
            "IS_HOMESTEAD",
            "TOTAL_UNITS",
        ]], how="left", left_on="PID", right_on="PIN")

        # drop & rename columns
        df_taxpayers.rename(columns={
            "PID": "pin",
            "TAXPAYER_NM": "tax_name",
            "TAXPAYER_NM_1": "tax_name_2",
            "TAXPAYER_NM_2": "tax_street",
            "TAXPAYER_NM_3": "tax_city_state_zip",
            "IS_EXEMPT": "is_exempt",
            "IS_HOMESTEAD": "is_homestead",
            "PRIMARY_PROP_TYPE": "prop_type",
            "TOTAL_UNITS": "num_units",
        }, inplace=True)
        df_taxpayers.drop(columns=["PIN"], inplace=True)

        return df_taxpayers

    def execute_business_filings_preprocessing(
        self,
        df_bus1: pd.DataFrame,
        df_bus3: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        df_bus1 = df_bus1[[
            "master_id",
            "business_type_code",
            "original_filing_number",
            "minnesota_business_name",
            "business_filing_status",
            "filing_date",
            "expiration_date",
            "next_renewal_due_date",
            "home_jurisdiction",
            "is_llc_non_profit",
            "home_business_name"
        ]]
        df_bus1.rename(columns={
            "master_id": "uid",
            "original_filing_number": "file_number",
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

        return df_bus1, df_bus3

    def process(self) -> None:

        # load df copies
        df_city: pd.DataFrame = self.dfs_in["taxpayers_city"].copy()
        df_county: pd.DataFrame = self.dfs_in["taxpayers_county"].copy()
        df_bus1: pd.DataFrame = self.dfs_in["mnsos_type1"].copy()
        df_bus3: pd.DataFrame = self.dfs_in["mnsos_type3"].copy()

        df_tax_out = self.execute_taxpayer_pre_processing(df_city, df_county)
        df_bus_1_out, df_bus_3_out = self.execute_business_filings_preprocessing(df_bus1, df_bus3)

        # set out dfs
        self.dfs_out["props_taxpayers"] = df_tax_out
        self.dfs_out["business_filings_1"] = df_bus_1_out
        self.dfs_out["business_filings_3"] = df_bus_3_out

    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "props_taxpayers": path_gen.raw_props_taxpayers(configs),
            "business_filings_1": path_gen.raw_business_filings_1(configs),
            "business_filings_3": path_gen.raw_business_filings_3(configs),
        }
        self.save_dfs(save_map)

    def update_configs(self) -> None:
        pass
