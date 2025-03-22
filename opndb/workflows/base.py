# 1. Standard library imports
import gc
import shutil
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from pprint import pprint
from typing import Any, ClassVar, Optional, Tuple, List
import networkx as nx
import nmslib
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TaskID

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
    PropsTaxpayers as pt
)
from opndb.constants.files import Raw as r, Dirs as d, Geocodio as g
from opndb.services.column import ColumnPropsTaxpayers, ColumnCorps, ColumnLLCs, ColumnProperties, \
    ColumnTaxpayerRecords, ColumnClassCodes, ColumnUnvalidatedAddrs, ColumnValidatedAddrs
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


console = Console()

class WorkflowBase(ABC):
    """
    Base workflow class the controls execution of data processing tasks required for each stage of the opndb workflow.
    Each child class that inherits from WorkflowBase corresponds to the broader workflow stage.

    CHILD WORKFLOW REQUIREMENTS:
        - Required dataframes object: instance variable containing all required dataframes and their file paths
        - Execute method: executes data load, required logic and transformations, and saves outputs
        - Load
    """
    def __init__(self, config_manager: ConfigManager):
        self.config_manager: ConfigManager = config_manager
        self.dfs_in: dict[str, pd.DataFrame | None] = {}
        self.dfs_out: dict[str, pd.DataFrame | None] = {}

    def load_dfs(self, load_map: dict[str, Path]) -> None:
        """
        Sets the self.dfs_in object. Sets keys as dataframe ID values. Sets values to dataframes, or None if the file
        path specified is not found.
        """
        for id, path in load_map.items():
            self.dfs_in[id] = ops_df.load_df(path, str)
            if self.dfs_in[id] is not None:
                console.print(f"\"{id}\" successfully loaded from: \n{path}")
        # prep data for summary table printing
        table_data = []
        for id, df in self.dfs_in.items():  # todo: standardize this, enforce types
            if df is None:
                continue
            memory_usage = df.memory_usage(deep=True).sum()
            table_data.append({
                "dataset_name": id,
                "file_size": utils.sizeof_fmt(memory_usage),
                "record_count": len(df)
            })
        t.display_table(table_data)
        # todo: add summary tables listing columns found in each table

    def save_dfs(self, save_map: dict[str, Path]) -> None:
        """Saves dataframes to their specified paths."""
        console.print("\n")
        for id, path in save_map.items():
            ops_df.save_df(self.dfs_out[id], path)
            console.print(f"\"{id}\" successfully saved to: \n{path}")

    @classmethod
    def create_workflow(cls, config_manager: ConfigManager, wkfl_id: str) -> Optional['WorkflowBase']:
        """Instantiates workflow object based on last saved progress (config['wkfl_stage'])."""
        if wkfl_id == "data_clean":
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
        # elif configs["wkfl_type"] == "final_output":
        #     return WkflFinalOutput(config_manager)
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
            # self.summary_stats()
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


class WkflDataClean(WorkflowStandardBase):
    """
    Initial data cleaning. Runs cleaners and validators on raw datasets. if they pass the validation checks, raw
    datasets are broken up and stored in their appropriate locations.

    INPUTS:
        - Raw taxpayer, corporate and LLC data
            - 'ROOT/raw/props_taxpayers[FileExt]'
            - 'ROOT/raw/corps[FileExt]'
            - 'ROOT/raw/llcs[FileExt]'
            - 'ROOT/raw/class_codes[FileExt]'
    OUTPUTS:
        - Cleaned taxpayer, corporate and LLC data
            = 'ROOT/processed/properties[FileExt]'
            - 'ROOT/processed/taxpayer_records[FileExt]'
            - 'ROOT/processed/corps[FileExt]'
            - 'ROOT/processed/llcs[FileExt]'
            - 'ROOT/processed/class_codes[FileExt]'
            - 'ROOT/processed/unvalidated_addrs[FileExt]'
    """

    WKFL_NAME: str = "INITIAL DATA CLEANING WORKFLOW"
    WKFL_DESC: str = "Runs basic string cleaners on raw inputted datasets."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_pre_cleaning(self, id: str, df: pd.DataFrame, column_manager) -> pd.DataFrame:

        t.print_equals("Executing pre-cleaning operations")

        # generate raw_prefixed columns
        t.print_with_dots(f"Setting raw columns for \"{id}\"")
        df: pd.DataFrame = cols_df.set_raw_columns(df, column_manager.raw)
        console.print(f"Raw columns generated ✅")

        # rename columns to prepare for cleaning
        df.rename(columns=column_manager.clean_rename_map, inplace=True)

        # create raw_address field
        t.print_with_dots(f"Generating full raw address columns (if not already present)")
        df: pd.DataFrame = cols_df.set_full_address_fields(df, column_manager.raw_address_map, id)
        console.print(f"Full raw address columns generated ✅")

        # props_taxpayers-specific logic - concatenate raw name + raw address
        if id == "props_taxpayers":
            t.print_with_dots(f"Concatenating raw name and address fields")
            df: pd.DataFrame = cols_df.set_name_address_concat(
                df, column_manager.name_address_concat_map["raw"]
            )
            console.print(f"\"name_address\" field generated ✅")

        return df

    def execute_basic_cleaning(self, df: pd.DataFrame, column_manager) -> pd.DataFrame:

        t.print_equals(f"Executing basic operations (all data columns)")

        cols: list[str] = column_manager.basic_clean

        t.print_with_dots("Converting letters to uppercase")
        df = clean_df_base.make_upper(df, cols)

        t.print_with_dots("Removing symbols and punctuation")
        df = clean_df_base.remove_symbols_punctuation(df, cols)

        t.print_with_dots("Trimming whitespace")
        df = clean_df_base.trim_whitespace(df, cols)

        t.print_with_dots("Removing extra spaces")
        df = clean_df_base.remove_extra_spaces(df, cols)

        t.print_with_dots("Deduplicating repeated words")
        df = clean_df_base.deduplicate(df, cols)

        # t.print_with_dots("Converting ordinal numbers to digits...")  # todo: this one breaks the df, returns all null values
        # df = clean_df_base.convert_ordinals(df, cols)

        t.print_with_dots("Replacing number ranges with first number in range")
        df = clean_df_base.take_first(df, cols)

        t.print_with_dots("Combining numbers separated by spaces")
        df = clean_df_base.combine_numbers(df, cols)

        console.print("Preliminary cleaning complete ✅")

        return df

    def execute_name_column_cleaning(self, df: pd.DataFrame, column_manager) -> pd.DataFrame:

        t.print_equals(f"Executing cleaning operations (name columns only)")
        cols: list[str] = column_manager.name_clean
        df = clean_df_name.switch_the(df, cols)
        console.print(f"Name field cleaning complete ✅")

        return df

    def execute_address_column_cleaning(self, df: pd.DataFrame, column_manager) -> pd.DataFrame:

        t.print_equals(f"Executing cleaning operations (address columns only)")

        for col in column_manager.address_clean["street"]:

            t.print_with_dots("Converting directionals to their abbreviations")
            df = clean_df_addr.convert_nsew(df, [col])

            t.print_with_dots("Removing secondary designators")
            df = clean_df_addr.remove_secondary_designators(df, [col])

            t.print_with_dots("Converting street suffixes")
            df = clean_df_addr.convert_street_suffixes(df, [col])

        for col in column_manager.address_clean["zip"]:

            t.print_with_dots("Fixing zip codes...")
            df = clean_df_addr.fix_zip(df, [col])

        console.print("Address field cleaning complete ✅")

        return df

    # def execute_accuracy_implications(self, id: str, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    #     if self.configs["accuracy"] == "low":
    #         console.print(f"Accuracy set to low. Executing additional string cleaners with accuracy implications on {id}")
    #         df = clean_df_acc.remove_secondary_component(df, cols)
    #         df = clean_df_acc.convert_mixed(df, cols)
    #         df = clean_df_acc.drop_letters(df, cols)
    #         console.print(f"{id} additional string cleaning complete.")
    #     else:
    #         console.print(f"Skipping additional string cleaning.")
    #     return df

    def execute_post_cleaning(self, id: str, df: pd.DataFrame, column_manager, out_column_managers) -> None:

        t.print_equals(f"Executing post-cleaning operations")

        # add clean full address fields
        t.print_with_dots("Setting clean full address fields")
        df: pd.DataFrame = cols_df.set_full_address_fields(df, column_manager.clean_address_map, id)
        t.print_with_dots("Full clean address fields generated ✅")

        # props_taxpayers-specific logic
        if id == "props_taxpayers":
            # generate clean name + address concat field
            # t.print_with_dots(f"Concatenating clean name and address fields")
            # df: pd.DataFrame = cols_df.set_name_address_concat(
            #     df, column_manager.name_address_concat_map["clean"]
            # )
            # console.print(f"\"name_address\" field generated ✅")
            # split dataframe into properties and taxpayer_records
            t.print_with_dots(f"Splitting \"{id}\" into \"taxpayer_records\" and \"properties\"...")
            col_manager_p = out_column_managers["properties"]
            col_manager_t = out_column_managers["taxpayer_records"]
            df_props, df_taxpayers = subset_df.split_props_taxpayers(df, col_manager_p.out, col_manager_t.out)
            df_taxpayers.dropna(subset=["raw_name"], inplace=True)
            self.dfs_out["properties"] = df_props
            self.dfs_out["taxpayer_records"] = df_taxpayers
            console.print(f"\"{id}\" successfully split into \"taxpayer_records\" and \"properties\" ✅")
        else:
            t.print_with_dots("Setting final dataframe")
            self.dfs_out[id] = df[column_manager.out]
            console.print(f"Final dataframe set.")

    def execute_unvalidated_generator(self, column_managers, out_column_managers) -> None:
        t.print_with_dots(f"Generating \"unvalidated_addrs\"...")
        col_map = {
            "taxpayer_records": out_column_managers["taxpayer_records"].unvalidated_addr_cols,
            "corps": column_managers["corps"].unvalidated_col_objs,
            "llcs": column_managers["llcs"].unvalidated_col_objs,
        }
        df_unvalidated: pd.DataFrame = subset_df.generate_unvalidated_df(self.dfs_out, col_map)
        self.dfs_out["unvalidated_addrs"] = df_unvalidated.drop_duplicates(subset=["clean_address"])
        console.print("\"unvalidated_addrs\" successfully generated ✅")

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {  # todo: change these back
            "props_taxpayers": path_gen.raw_props_taxpayers(configs),
            "corps": path_gen.raw_corps(configs),
            "llcs": path_gen.raw_llcs(configs),
            "class_codes": path_gen.raw_class_codes(configs)
        }
        # load_map: dict[str, Path] = {  # todo: change these back
        #     "taxpayer_records": path_gen.processed_taxpayer_records(configs),
        #     "corps": path_gen.processed_corps(configs),
        #     "llcs": path_gen.processed_llcs(configs),
        #     "class_codes": path_gen.raw_class_codes(configs)
        # }
        self.load_dfs(load_map)

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:

        # todo: add validator that checks for required columns, throw error/failure immediately if not
        # todo: add to columns validator: if _ADDRESS is empty, _STREET and _ZIP must be present; otherwise, if _ADDRESS exists, all others can be empty/missing
        # todo: add to properties/taxpayer records validator: unique column constraint on PIN
        # todo: add checks to confirm that dfs_in were loaded correctly that stop this workflow from executing if not

        column_manager = {
            "props_taxpayers": ColumnPropsTaxpayers(),
            "corps": ColumnCorps(),
            "llcs": ColumnLLCs(),
            "class_codes": ColumnClassCodes()
        }
        out_column_manager = {
            "properties": ColumnProperties(),
            "taxpayer_records": ColumnTaxpayerRecords(),
        }

        for id, df_in in self.dfs_in.items():
            t.print_dataset_name(id)
            df: pd.DataFrame = df_in.copy()  # make copy to preserve the loaded dataframes

            # specific logic for class_codes
            if id == "class_codes":
                df: pd.DataFrame = self.execute_basic_cleaning(df, column_manager[id])
                self.dfs_out[id] = df
                continue

            # PRE-CLEANING OPERATIONS
            df: pd.DataFrame = self.execute_pre_cleaning(id, df, column_manager[id])

            # # MAIN CLEANING OPERATIONS
            df: pd.DataFrame = self.execute_basic_cleaning(df, column_manager[id])
            df: pd.DataFrame = self.execute_name_column_cleaning(df, column_manager[id])
            df: pd.DataFrame = self.execute_address_column_cleaning(df, column_manager[id])
            # df: pd.DataFrame = self.execute_accuracy_implications("props_taxpayers", df)

            # POST-CLEANING OPERATIONS
            self.execute_post_cleaning(id, df, column_manager[id], out_column_manager)

        # GENERATE UNVALIDATED ADDRESS DATASET
        self.execute_unvalidated_generator(column_manager, out_column_manager)

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
            "corps": path_gen.processed_corps(configs),
            "llcs": path_gen.processed_llcs(configs),
            "class_codes": path_gen.processed_class_codes(configs),
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(configs),
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflAddressClean(WorkflowStandardBase):
    """
    Initial address validation workflow. Cleans up & validates PO box addresses, updates validated and unvalidated
    master address files.

    INPUTS:
        - Unvalidated address file
            - 'ROOT/processed/unvalidated_addrs[FileExt]'
    OUTPUTS:
        - Creates validated_addrs file and stores validated PO boxes
            - 'ROOT/processed/validated_addrs[FileExt]'
        - Updates unvalidated_addrs file so that the PO box rows are removed
            -'ROOT/processed/unvalidated_addrs[FileExt]'
    """
    # todo: specify to use clean_address to associate with validated addr object

    WKFL_NAME: str = "INITIAL ADDRESS WORKFLOW"
    WKFL_DESC: str = "Cleans up PO box addresses in preparation for Geocodio validation."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def load(self):
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(configs),
        }
        self.load_dfs(load_map)

    def process(self) -> None:
        t.print_dataset_name("unvalidated_addrs")
        t.print_equals("Executing initial address operations")
        column_manager = {"unvalidated_addrs": ColumnUnvalidatedAddrs()}
        df: pd.DataFrame = self.dfs_in["unvalidated_addrs"].copy()
        t.print_with_dots("Setting is_pobox column")
        df = cols_df.set_is_pobox(df, column_manager["unvalidated_addrs"].CLEAN_ADDRESS)
        t.print_with_dots("Cleaning up PO box addresses")
        df = colm_df.fix_pobox(df, column_manager["unvalidated_addrs"].CLEAN_ADDRESS)
        self.dfs_out["unvalidated_addrs"] = df

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
    """
    Address validation

    INPUTS:
        - Unvalidated address file
            - 'ROOT/processed/unvalidated_addrs[FileExt]'
    OUTPUTS:
        - Updated master dataset for validated and unvalidated addresses
            - 'ROOT/processed/validated_addrs[FileExt]'
            - 'ROOT/processed/unvalidated_addrs[FileExt]'
        - Geocodio master files for validated and unvalidated addresses
            - 'ROOT/geocodio/gcd_validated[FileExt]'
            - 'ROOT/geocodio/gcd_unvalidated[FileExt]'
            - 'ROOT/geocodio/gcd_failed[FileExt]'
        - Geocodio partial files for all API call results, in 'ROOT/geocodio/partials'
    """

    WKFL_NAME: str = "ADDRESS VALIDATION (GEOCODIO) WORKFLOW"
    WKFL_DESC: str = ("Executes Geocodio API calls for unvalidated addresses. Processes and stores results in data "
                      "directories.")

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_gcd_address_subset(self, df_unvalidated: pd.DataFrame, column_manager) -> pd.DataFrame:
        """
        Subsets unvalidated master address file by filtering out addresses that have already been validated. Returns
        dataframe slice containing only columns required for processing Geocodio API calls.
        """
        if self.dfs_in["gcd_validated"] is not None:
            validated_addrs: list[str] = list(self.dfs_in["gcd_validated"]["clean_address"].unique())
            df_addrs: pd.DataFrame = df_unvalidated[~df_unvalidated["clean_address"].isin(validated_addrs)]
        else:
            df_addrs: pd.DataFrame = df_unvalidated
        return df_addrs[column_manager["unvalidated_addrs"].geocodio_columns]

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

    def execute_geocodio_postprocessor(self, gcd_results_obj: GeocodioReturnObject):
        """
        Processes results of run_geocodio. Updates gcd_validated, gcd_unvalidated and gcd_failed based on results of
        run_geocodio call. Stores final dataframes to be saved in self.dfs_out.
        """
        # create new dataframes from the resulting geocodio call
        df_gcd_validated: pd.DataFrame = pd.DataFrame(gcd_results_obj["validated"])
        df_gcd_unvalidated: pd.DataFrame = pd.DataFrame(gcd_results_obj["unvalidated"])
        df_gcd_failed: pd.DataFrame = pd.DataFrame(gcd_results_obj["failed"])

        # if there were already gcd_validated, gcd_unvalidated and gcd_failed in the directories, concatenate the new one and set to dfs_out
        if self.dfs_in["gcd_validated"] is not None:
            df_gcd_validated = pd.concat([self.dfs_in["gcd_validated"], df_gcd_validated], ignore_index=True)
        if self.dfs_in["gcd_unvalidated"] is not None:
            df_gcd_unvalidated = pd.concat([self.dfs_in["gcd_unvalidated"], df_gcd_unvalidated], ignore_index=True)
        if self.dfs_in["gcd_failed"] is not None:
            df_gcd_failed = pd.concat([self.dfs_in["gcd_failed"], df_gcd_failed], ignore_index=True)

        # calculate stats to print
        if not df_gcd_validated.empty:
            validated_before: int = len(self.dfs_in["gcd_validated"]) if self.dfs_in["gcd_validated"] is not None else 0
            validated_after: int = len(df_gcd_validated)
            validated_diff: int = validated_after - validated_before
            console.print(f"Total validated addresses: {validated_after} (+{validated_diff})")

        if not df_gcd_unvalidated.empty:
            unvalidated_before: int = len(self.dfs_in["gcd_unvalidated"]["clean_address"].unique()) if self.dfs_in["gcd_unvalidated"] is not None else 0
            unvalidated_after: int = len(df_gcd_unvalidated["clean_address"].unique())
            unvalidated_diff: int = unvalidated_after - unvalidated_before
            console.print(f"Total unvalidated addresses: {unvalidated_after} (+{unvalidated_diff})")

        if not df_gcd_failed.empty:
            failed_before: int = len(self.dfs_in["gcd_failed"]) if self.dfs_in["gcd_failed"] is not None else 0
            failed_after: int = len(df_gcd_failed)
            failed_diff: int = failed_after - failed_before
            console.print(f"Total failed addresses: {failed_after} (+{failed_diff})")


        self.dfs_out: dict[str, pd.DataFrame] = {
            "gcd_validated": df_gcd_validated,
            "gcd_unvalidated": df_gcd_unvalidated,
            "gcd_failed": df_gcd_failed,
        }

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(configs),
            "gcd_unvalidated": path_gen.geocodio_gcd_unvalidated(configs),
            "gcd_validated": path_gen.geocodio_gcd_validated(configs),
            "gcd_failed": path_gen.geocodio_gcd_failed(configs),
        }
        self.load_dfs(load_map)

    def process(self):

        configs = self.config_manager.configs
        column_manager = {
            "unvalidated_addrs": ColumnUnvalidatedAddrs(),
            "validated_addrs": ColumnValidatedAddrs(),
        }
        dfs = {id: df.copy() for id, df in self.dfs_in.items() if df is not None}

        # fetch addresses to be geocoded
        df_addrs: pd.DataFrame = self.execute_gcd_address_subset(dfs["unvalidated_addrs"], column_manager)

        # get geocodio api key from user if not already exists in configs.json
        cont: bool = self.execute_api_key_handler_warning(df_addrs)
        if not cont: return

        # call geocodio or exit
        gcd_results_obj: GeocodioReturnObject = addr.run_geocodio(
            configs["geocodio_api_key"],
            df_addrs,
            column_manager["unvalidated_addrs"].CLEAN_ADDRESS,
            self.config_manager.configs
        )

        # execute post-processor
        self.execute_geocodio_postprocessor(gcd_results_obj)

    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "gcd_validated": path_gen.geocodio_gcd_validated(configs),
            "gcd_unvalidated": path_gen.geocodio_gcd_unvalidated(configs),
            "gcd_failed": path_gen.geocodio_gcd_failed(configs),
        }
        self.save_dfs(save_map)

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
        load_map: dict[str, Path] = {  # todo: change these back
            "gcd_validated": path_gen.geocodio_gcd_validated(configs),
        }
        self.load_dfs(load_map)

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        # detect missing unit numbers in validated addresses via regex analysis
        df_unit: pd.DataFrame = self.dfs_in["gcd_validated"].copy()
        # subset to exclude pobox addresses
        df_unit = df_unit[df_unit["is_pobox"] == "False"]
        # subset validated addresses for only ones which do not have a secondary number
        df_unit = df_unit[df_unit["secondarynumber"].isnull()]
        # check street addresses for digits at the end
        df_unit = cols_df.set_check_sec_num(df_unit, "clean_address")
        # subset check_sec_num results to only include rows where a number WAS detected
        df_unit = df_unit[df_unit["check_sec_num"].notnull()]
        self.dfs_out["fixing_addrs"] = df_unit

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
        load_map: dict[str, Path] = {  # todo: change these back
            "gcd_validated": path_gen.geocodio_gcd_validated(configs),
            "fixing_addrs": path_gen.analysis_fixing_addrs(configs),
        }
        self.load_dfs(load_map)

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:
        t.print_equals("Adding missing unit numbers to validated addresses")
        df_valid: pd.DataFrame = self.dfs_in["gcd_validated"].copy()
        df_fix: pd.DataFrame = self.dfs_in["fixing_addrs"].copy()
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
        df_valid = cols_df.set_formatted_address_v(df_valid)
        self.dfs_out["gcd_validated"] = df_valid

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
        }
        self.save_dfs(save_map)

    # -----------------------
    # ----CONFIGS UPDATER----
    # -----------------------
    def update_configs(self) -> None:
        pass


class WkflAddressMerge(WorkflowStandardBase):
    """
    Merges validated addresses to taxpayer, corporate and LLC records. Creates new columns for validated addresses
    suffixed with "_v"

    INPUTS:
        - Processed taxpayer, corporate and LLC data
            - 'ROOT/processed/taxpayer_records[FileExt]'
            - 'ROOT/processed/corps[FileExt]'
            - 'ROOT/processed/llcs[FileExt]'
    OUTPUTS:
        - Taxpayer, corporate and LLC records with validated addresses
            - 'ROOT/processed/taxpayer_records_merged[FileExt]'
            - 'ROOT/processed/corps_merged[FileExt]'
            - 'ROOT/processed/llcs_merged[FileExt]'
    """

    WKFL_NAME: str = "ADDRESS MERGE WORKFLOW"
    WKFL_DESC: str = "Merges validated addresses to address fields in taxpayer, corporate and LLC records."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    # --------------
    # ----LOADER----
    # --------------
    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {  # todo: change these back
            "gcd_validated": path_gen.geocodio_gcd_validated(configs),
            "taxpayer_records": path_gen.processed_taxpayer_records(configs),
            "corps": path_gen.processed_corps(configs),
            "llcs": path_gen.processed_llcs(configs),
        }
        self.load_dfs(load_map)

    # -----------------
    # ----PROCESSOR----
    # -----------------
    def process(self) -> None:

        column_manager = {
            "gcd_validated": ColumnValidatedAddrs(),
            "taxpayer_records": ColumnTaxpayerRecords(),
            "corps": ColumnCorps(),
            "llcs": ColumnLLCs(),
        }
        df_valid = self.dfs_in["gcd_validated"].copy()

        for id, df_in in self.dfs_in.items():
            if id == "gcd_validated":
                continue
            t.print_dataset_name(id)
            df: pd.DataFrame = df_in.copy()
            for addr_col in column_manager[id].validated_address_merge:
                t.print_with_dots(f"Merging validated address into {addr_col} for \"{id}\"")
                df = merge_df.merge_validated_address(df, df_valid, addr_col)
            df = clean_df_base.combine_columns_parallel(df)
            self.dfs_out[id] = df
            console.print("Validated addresses merged ✅ 🗺️ 📍")

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
            "taxpayer_records": path_gen.processed_taxpayer_records_merged(configs),
            "corps": path_gen.processed_corps_merged(configs),
            "llcs": path_gen.processed_llcs_merged(configs),
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

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {
            "taxpayer_records": path_gen.processed_taxpayer_records(configs)
        }
        self.load_dfs(load_map)

    def process(self) -> None:
        df = self.dfs_in["taxpayer_records"].copy()
        df_freq: pd.DataFrame = subset_df.generate_frequency_df(df, "clean_name")
        # frequent_tax_names
        self.dfs_out["frequent_tax_names"] = df_freq
        self.dfs_out["frequent_tax_names"]["is_common_name"] = ""
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

    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "frequent_tax_names": path_gen.analysis_frequent_tax_names(configs),
            "fixing_tax_names": path_gen.analysis_fixing_tax_names(configs),
        }
        self.save_dfs(save_map)

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
        "researched",
        "name",
        "urls",
        "notes",
        "is_landlord_org",
        "is_govt_agency",
        "is_lawfirm",
        "is_missing_suite",
        "is_financial_services",
        "is_assoc_bus",
        "fix_address",
        "is_virtual_office_agent",
        "yelp_urls",
        "is_nonprofit",
        "google_urls",
        "is_ignore_misc",
        "google_place_id"
    ]

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {
            "gcd_validated": path_gen.geocodio_gcd_validated(configs),
            "taxpayer_records": path_gen.processed_taxpayer_records_merged(configs),
            "corps": path_gen.processed_corps_merged(configs),
            "llcs": path_gen.processed_llcs_merged(configs),
        }
        self.load_dfs(load_map)

    def process(self) -> None:
        column_manager = {
            "gcd_validated": ColumnValidatedAddrs(),
            "taxpayer_records": ColumnTaxpayerRecords(),
            "corps": ColumnCorps(),
            "llcs": ColumnLLCs(),
        }
        addrs = []
        for id, df_in in self.dfs_in.items():
            if id == "gcd_validated":
                continue
            for addr_col in column_manager[id].validated_address_merge:
                addrs.extend(
                    [
                        addr
                        for addr in df_in[f"{addr_col}_v"]
                        if pd.notnull(addr) and addr != ""
                    ]
                )
        df_addrs: pd.DataFrame = pd.DataFrame(columns=["address"], data=addrs)
        df_freq: pd.DataFrame = subset_df.generate_frequency_df(df_addrs, "address")
        for field in self.ANALYSIS_FIELDS:
            df_freq[field] = ""
        df_freq["researched"] = "f"
        self.dfs_out["address_analysis"] = df_freq

    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "address_analysis": path_gen.analysis_address_analysis(configs),
        }
        self.save_dfs(save_map)

    def update_configs(self) -> None:
        pass


class WkflAnalysisFinal(WorkflowStandardBase):
    """
    Fixes taxpayer names based on standardized spellings manually specified in the fixing_tax_names dataset. Adds
    boolean columns to taxpayer records for is_common_name and is_landlord_org, based on manual input in
    fixing_tax_names and address_analysis datasets.

    INPUTS:
    OUTPUTS:
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

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {
            "fixing_tax_names": path_gen.analysis_fixing_tax_names(configs),
            "address_analysis": path_gen.analysis_address_analysis(configs),
            "frequent_tax_names": path_gen.analysis_frequent_tax_names(configs),
            "taxpayer_records_merged": path_gen.processed_taxpayer_records_merged(configs),
        }
        self.load_dfs(load_map)

    def process(self) -> None:
        # copy dfs
        df_fix_names: pd.DataFrame = self.dfs_in["fixing_tax_names"].copy()
        df_analysis: pd.DataFrame = self.dfs_in["address_analysis"].copy()
        df_freq_names: pd.DataFrame = self.dfs_in["frequent_tax_names"].copy()
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayer_records_merged"].copy()

        df_taxpayers.dropna(subset=["raw_name"], inplace=True)
        # create banks dict
        t.print_with_dots("Setting standardized name dictionary")
        banks: dict[str, str] = self.get_banks_dict(df_fix_names)
        t.print_with_dots("Fixing taxpayer names")
        df_taxpayers = colm_df.fix_tax_names(df_taxpayers, banks)
        t.print_with_dots("Setting is_common_name boolean column")
        df_taxpayers = cols_df.set_is_common_name(df_taxpayers, df_freq_names)
        t.print_with_dots("Setting is_landlord_org boolean column")
        df_taxpayers = cols_df.set_is_landlord_org(df_taxpayers, df_analysis)
        self.dfs_out["taxpayers_fixed"] = df_taxpayers

    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "taxpayers_fixed": path_gen.processed_taxpayers_fixed(configs)
        }
        self.save_dfs(save_map)

    def update_configs(self) -> None:
        pass


class WkflRentalSubset(WorkflowStandardBase):
    """
    Subset rental properties based on class code descriptions.

    INPUT:
        - Building class codes dataset
            - 'ROOT/processed/bldg_class_codes[FileExt]'
        - Taxpayer record dataset
            - 'ROOT/processed/taxpayer_records[FileExt]'
        - Property dataset
    OUTPUT:
        - Rental-subsetted taxpayer dataset
            - 'ROOT/processed/tax_records_subsetted[FileExt]'
    """
    WKFL_NAME: str = "RENTAL SUBSET WORKFLOW"
    WKFL_DESC: str = "Subsets property and taxpayer record datasets for rental properties only."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {
            "properties": path_gen.processed_properties(configs),
            "taxpayers_fixed": path_gen.processed_taxpayers_fixed(configs),
            "class_codes": path_gen.processed_class_codes(configs)
        }
        self.load_dfs(load_map)

    def process(self) -> None:

        # copy dfs
        df_props: pd.DataFrame = self.dfs_in["properties"].copy()
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_fixed"].copy()
        df_class_codes: pd.DataFrame = self.dfs_in["class_codes"].copy()

        # add is_rental column to properties
        t.print_with_dots("Subsetting rental properties by class code")
        df_props = cols_df.set_is_rental_initial(df_props, df_class_codes)
        # execute initial subset based on is_rental
        df_rentals_initial: pd.DataFrame = df_props[df_props["is_rental"] == True]
        console.print("Properties subsetted by class code ✅")

        # get unique raw name+addr values for rental subset
        t.print_with_dots("Subsetting rental properties by validated address")
        rental_records: list[str] = list(df_rentals_initial["raw_name_address"].unique())
        # subset taxpayer records for rental and non-rental properties
        df_taxpayer_rentals: pd.DataFrame = df_taxpayers[df_taxpayers["raw_name_address"].isin(rental_records)]
        df_taxpayer_nonrentals: pd.DataFrame = df_taxpayers[~df_taxpayers["raw_name_address"].isin(rental_records)]

        # fetch validated addresses associated with rental taxpayer records
        rental_addrs: list[str] = list(df_taxpayer_rentals["raw_address_v"].unique())
        # fetch taxpayer records NOT pulled in by initial subset but that have matching addresses
        df_taxpayers_missed: pd.DataFrame = df_taxpayer_nonrentals[df_taxpayer_nonrentals["raw_address_v"].isin(rental_addrs)]
        nonrental_records: list[str] = list(df_taxpayers_missed["raw_name_address"].unique())

        # set is_rental in df_props again
        t.print_with_dots("Adding final is_rental columns to property and taxpayer datasets")
        df_props = cols_df.set_is_rental_final(df_props, nonrental_records)
        rental_records_final: list[str] = rental_records + nonrental_records
        df_taxpayers["is_rental"] = df_taxpayers["raw_name_address"].isin(rental_records_final)
        console.print("is_rental columns generated ✅")

        # output dfs
        self.dfs_out["properties_rentals"] = df_props
        self.dfs_out["taxpayers_subsetted"] = df_taxpayers[df_taxpayers["is_rental"] == True]

    def summary_stats(self):
        # how many initially subsetted with class codes only
        # how many additional rentals obtained from validated addresses from initial subset
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "properties_rentals": path_gen.processed_properties_rentals(configs),
            "taxpayers_subsetted": path_gen.processed_taxpayers_subsetted(configs),
        }
        self.save_dfs(save_map)

    def update_configs(self) -> None:
        pass


class WkflCleanMerge(WorkflowStandardBase):
    """
    Additional data cleaning & merging.

    INPUTS:
        - Cleaned taxpayer, corporate and LLC data
            - 'ROOT/processed/props_subsetted[FileExt]'
            - 'ROOT/processed/corps[FileExt]'
            - 'ROOT/processed/llcs[FileExt]'
    OUTPUTS:
        - Single property dataset containing all columns required for string matching & network graph generation
            - 'ROOT/processed/props_prepped[FileExt]'
    """
    WKFL_NAME: str = "PRE-MATCHING DATA CLEANING WORKFLOW"
    WKFL_DESC: str = "Runs basic string cleaners on raw inputted datasets."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_column_generators(
        self,
        df_taxpayers: pd.DataFrame,
        df_corps: pd.DataFrame,
        df_llcs: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        t.print_with_dots("Adding core name columns to taxpayer, corporate and LLC records")
        df_taxpayers = cols_df.set_core_name(df_taxpayers, tr.CLEAN_NAME)
        df_corps = cols_df.set_core_name(df_corps, c.CLEAN_NAME)
        df_llcs = cols_df.set_core_name(df_llcs, l.CLEAN_NAME)

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

        return df_taxpayers, df_corps, df_llcs

    def execute_corp_llc_combine_subset(self, df_corps: pd.DataFrame, df_llcs: pd.DataFrame) -> pd.DataFrame:
        # todo: ask for user input to indicate whether only active llcs/corps should be merged into properties, or if all of them should
        t.print_with_dots("Subsetting corporate and LLC record dataframes for active only")
        df_corps_active = subset_df.get_active(df_corps, c.STATUS)
        df_llcs_active = subset_df.get_active(df_llcs, l.STATUS)
        t.print_with_dots("Combining corporate and LLC record datasets for string matching and merging")
        df_combined = concat_df.combine_corps_llcs(df_corps_active, df_llcs_active)
        return df_combined

    def execute_clean_merge(
        self,
        df_taxpayers: pd.DataFrame,
        df_combined: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        t.print_with_dots("Merging corporations & LLCs to taxpayers on clean_name")
        df_clean_merge: pd.DataFrame = pd.merge(
            df_taxpayers,
            df_combined,
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
        df_combined: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_core_merge: pd.DataFrame = pd.merge(
            df_taxpayers_remaining,
            df_combined,
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
        df_combined: pd.DataFrame
    ) -> pd.DataFrame:
        ref_docs = list(df_combined["entity_clean_name"].dropna().unique())
        query_docs = list(df_taxpayers_remaining["clean_name"].dropna().unique())
        df_string_matches = StringMatch.match_strings(
            ref_docs,
            query_docs,
            params={
                "name_col": None,
                "match_threshold": .89,
                "include_orgs": False,
                "include_unresearched": False,
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

        # take slice of combined org dataframe to get address data & merge again
        df_matched_orgs: pd.DataFrame = df_combined[df_combined["entity_clean_name"].isin(
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

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {
            "taxpayers_subsetted": path_gen.processed_taxpayers_subsetted(configs),
            "corps_merged": path_gen.processed_corps_merged(configs),
            "llcs_merged": path_gen.processed_llcs_merged(configs),
        }
        self.load_dfs(load_map)

    def process(self) -> None:
        # todo: add workflow for fixing taxpayer addresses based on manual research - services/address.py
        # todo: add workflow for fixing taxpayer names based on manual research - services/base.py
        # copy dfs
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_subsetted"].copy()
        df_corps: pd.DataFrame = self.dfs_in["corps_merged"].copy()
        df_llcs: pd.DataFrame = self.dfs_in["llcs_merged"].copy()

        # generate columns for data processing
        df_taxpayers, df_corps, df_llcs = self.execute_column_generators(df_taxpayers, df_corps, df_llcs)

        # subset & combine corps and LLCs for merging
        df_combined: pd.DataFrame= self.execute_corp_llc_combine_subset(df_corps, df_llcs)

        # merge corps/llcs to taxpayer records on clean_name
        df_clean_merge, df_taxpayers_remaining = self.execute_clean_merge(df_taxpayers, df_combined)

        # merge corps/llcs to taxpayer records on core_name
        df_core_merge, df_taxpayers_remaining = self.execute_core_merge(df_taxpayers_remaining, df_combined)

        # string matching
        df_string_merge: pd.DataFrame = self.execute_string_match_merge(df_taxpayers_remaining, df_combined)

        # output dfs
        df_taxpayers = pd.concat([
            df_clean_merge[df_clean_merge["entity_clean_name"].notnull()],
            df_core_merge[df_core_merge["entity_core_name"].notnull()],
            df_string_merge
        ], ignore_index=True)
        self.dfs_out["taxpayers_prepped"] = df_taxpayers

    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "taxpayers_prepped": path_gen.processed_taxpayers_prepped(configs),
        }
        self.save_dfs(save_map)

    def update_configs(self) -> None:
        pass


class WkflStringMatch(WorkflowStandardBase):
    """
    String matching taxpayer records

    INPUTS:
        - Cleaned and merged property dataset with taxpayer records, corporations and LLCs
            - 'ROOT/processed/props_prepped[FileExt]'
        - User inputs matrix params
    OUTPUTS:
        - Inputted dataset with string matching result columns
    """
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
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_param_builder(self) -> List[StringMatchParams]:
        t.print_with_dots("Building string matching params object")
        return [
            {
                "name_col": "clean_name",
                "match_threshold": 0.85,
                "include_orgs": False,
                "include_unresearched": False,
                "nmslib_opts": self.DEFAULT_NMSLIB,
                "query_batch_opts": self.DEFAULT_QUERY_BATCH,
            },
            {
                "name_col": "clean_name",
                "include_orgs": False,
                "match_threshold": 0.85,
                "include_unresearched": True,
                "nmslib_opts": self.DEFAULT_NMSLIB,
                "query_batch_opts": self.DEFAULT_QUERY_BATCH,
            },
            {
                "name_col": "clean_name",
                "match_threshold": 0.8,
                "include_orgs": False,
                "include_unresearched": True,
                "nmslib_opts": self.DEFAULT_NMSLIB,
                "query_batch_opts": self.DEFAULT_QUERY_BATCH,
            },
        ]

    def execute_column_generators(self, df_taxpayers) -> pd.DataFrame:
        t.print_with_dots("Generating columns for matching")
        # set address column so that if there is no validated address for a property, the raw address is used instead
        df_taxpayers = cols_df.set_match_address(df_taxpayers)
        # add name+address concatenated columns to use for matching
        df_taxpayers = cols_df.concatenate_name_addr(df_taxpayers, "clean_name", "match_address")
        df_taxpayers = cols_df.concatenate_name_addr(df_taxpayers, "core_name", "match_address")
        return df_taxpayers

    def execute_string_matching(
        self,
        params_matrix: List[StringMatchParams],
        df_taxpayers: pd.DataFrame,
        df_analysis: pd.DataFrame
    ):
        """Returns final dataset to be outputted"""
        t.print_with_dots("Executing string matching")
        df_researched: pd.DataFrame = df_analysis[df_analysis["researched"] == "t"]
        for i, params in enumerate(params_matrix):
            t.print_equals(f"Matching strings for STRING_MATCHED_NAME_{i+1}")
            console.print("NAME COLUMN:", params["name_col"])
            console.print("MATCH THRESHOLD:", params["match_threshold"])
            console.print("INCLUDE ORGS:", params["include_orgs"])
            console.print("INCLUDE UNRESEARCHED ADDRESSES:", params["include_unresearched"])
            console.print("NMSLIB OPTIONS:", params["nmslib_opts"])
            console.print("QUERY BATCH OPTIONS:", params["query_batch_opts"])
            t.print_with_dots("Setting include_address")
            df_taxpayers["include_address"] = df_taxpayers["match_address"].apply(
                lambda addr: MatchBase.check_address(
                    addr,
                    df_researched,
                    params["include_orgs"],
                    params["include_unresearched"],
                )
            )
            # filter out addresses
            t.print_with_dots("Filtering out taxpayer records where include_address is False")
            df_filtered: pd.DataFrame = df_taxpayers[df_taxpayers["include_address"] == True][[
                "clean_name", "core_name", "match_address", "clean_name_address", "core_name_address"
            ]]
            # set ref & query docs
            t.print_with_dots("Setting document objects for HNSW index")
            ref_docs: list[str] = list(df_filtered[f"{params['name_col']}_address"].dropna().unique())
            query_docs: list[str] = list(df_filtered[f"{params['name_col']}_address"].dropna().unique())
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
                left_on=f"{params['name_col']}_address",
                right_on="original_doc"
            )
            gc.collect()
            df_taxpayers.drop(columns="original_doc", inplace=True)
            df_taxpayers.drop_duplicates(subset="raw_name_address", inplace=True)
            df_taxpayers.rename(columns={"fuzzy_match_combo": f"string_matched_name_{i+1}"}, inplace=True)

        return df_taxpayers

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {
            "taxpayers_prepped": path_gen.processed_taxpayers_prepped(configs),
            "address_analysis": path_gen.analysis_address_analysis(configs),
        }
        self.load_dfs(load_map)

    def process(self) -> None:
        # todo: add log/printout of the parameter matrix combinations and how they correspond to the string match columns generated
        # copy dfs
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_prepped"].copy()
        df_analysis: pd.DataFrame = self.dfs_in["address_analysis"].copy()
        # generate matrix parameters
        params_matrix: List[StringMatchParams] = self.execute_param_builder()
        # generate columns
        df_taxpayers = self.execute_column_generators(df_taxpayers)
        # run matching
        df_taxpayers_matched = self.execute_string_matching(params_matrix, df_taxpayers, df_analysis)
        # set out dfs
        self.dfs_out["taxpayers_string_matched"] = df_taxpayers_matched

    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "taxpayers_string_matched": path_gen.processed_taxpayers_string_matched(configs),
        }
        self.save_dfs(save_map)

    def update_configs(self) -> None:
        pass


class WkflNetworkGraph(WorkflowStandardBase):
    """
    Network graph generation

    INPUTS:
        - Cleaned and merged property dataset with string matching results
            - 'ROOT/processed/props_string_matched[FileExt]'
        - User inputx matrix params
    OUTPUTS:
        - Networked property dataset with columns containing node/edge data for connected components
            - 'ROOT/processed/props_networked[FileExt]'
    """
    WKFL_NAME: str = "NETWORK GRAPH WORKFLOW"
    WKFL_DESC: str = "Executes network graph generation linking taxpayer, corporate and LLC records."

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        t.print_workflow_name(self.WKFL_NAME, self.WKFL_DESC)

    def execute_param_builder(self) -> List[NetworkMatchParams]:
        t.print_with_dots("Building network graph params object")
        return [
            {
                "taxpayer_name_col": "clean_name",
                "entity_name_col": "entity_clean_name",
                "include_orgs": False,
                "include_unresearched": False,
                "string_match_name": "string_matched_name_1",
            },
            {
                "taxpayer_name_col": "clean_name",
                "entity_name_col": "entity_core_name",
                "include_orgs": False,
                "include_unresearched": False,
                "string_match_name": "string_matched_name_3",
            },
            {
                "taxpayer_name_col": "core_name",
                "entity_name_col": "entity_clean_name",
                "include_orgs": False,
                "include_unresearched": True,
                "string_match_name": "string_matched_name_3",
            },
            {
                "taxpayer_name_col": "clean_name",
                "entity_name_col": "entity_clean_name",
                "include_orgs": True,
                "include_unresearched": False,
                "string_match_name": "string_matched_name_2",
            },
            {
                "taxpayer_name_col": "clean_name",
                "entity_name_col": "entity_clean_name",
                "include_orgs": True,
                "include_unresearched": True,
                "string_match_name": "string_matched_name_3",
            },
            {
                "taxpayer_name_col": "core_name",
                "entity_name_col": "entity_clean_name",
                "include_orgs": True,
                "include_unresearched": True,
                "string_match_name": "string_matched_name_3",
            },
        ]

    def execute_network_graph_generator(
        self,
        params_matrix: List[NetworkMatchParams],
        df_taxpayers: pd.DataFrame,
        df_analysis: pd.DataFrame
    ) -> pd.DataFrame:
        df_researched: pd.DataFrame = df_analysis[df_analysis["researched"] == "t"]
        # for i, params in enumerate(params_matrix):
        params = params_matrix[0]
        i = 0
        console.print("TAXPAYER NAME COLUMN:", params["taxpayer_name_col"])
        console.print("ENTITY NAME COLUMN:", params["entity_name_col"])
        console.print("INCLUDE ORGS:", params["include_orgs"])
        console.print("INCLUDE UNRESEARCHED ADDRESSES:", params["include_unresearched"])
        console.print("STRING MATCH NAME:", params["string_match_name"])
        # generate network graph and dataframe with component ID column
        gMatches = NetworkMatchBase.taxpayers_network(
            i+1,
            df_taxpayers,
            df_researched,
            params
        )
        console.print("CONNECTED COMPONENT COUNT:", nx.number_connected_components(gMatches))
        console.print("NODES COUNT:", nx.number_of_nodes(gMatches))
        console.print("EDGES COUNT:", nx.number_of_edges(gMatches))

        # df_taxpayers_networked = NetworkMatchBase.set_taxpayer_component(
        #     i+1,
        #     df_taxpayers,
        #     gMatches,
        #     params
        # )
        # # merge to original dataset
        # df_taxpayers = pd.merge(
        #     df_taxpayers,
        #     df_taxpayers_networked[["name_address_clean", f"final_component_{i+1}"]],
        #     how="left",
        #     on="name_address_clean",
        # )
        # # set network canonical names
        # df_process: pd.DataFrame = NetworkMatchBase.set_network_name(
        #     i+1,
        #     df_process,
        #     f"final_component_{i+1}",
        #     f"network_{i+1}"
        # )
        # # set text for node/edge data - should this be a separate dataset? probably
        # df_process: pd.DataFrame = NetworkMatchBase.set_network_text(
        #     gMatches,
        #     df_process,
        #     f"final_component_{i+1}",
        #     f"network_{i+1}"
        # )
        return gMatches

    def load(self):
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {
            "taxpayers_string_matched": path_gen.processed_taxpayers_string_matched(configs),
            "address_analysis": path_gen.analysis_address_analysis(configs),
        }
        self.load_dfs(load_map)

    def process(self) -> None:
        # copy dfs
        df_taxpayers: pd.DataFrame = self.dfs_in["taxpayers_string_matched"].copy()
        df_analysis: pd.DataFrame = self.dfs_in["address_analysis"].copy()
        # generate matrix parameters
        params_matrix: List[NetworkMatchParams] = self.execute_param_builder()
        # run network graph
        df_networked: pd.DataFrame = self.execute_network_graph_generator(
            params_matrix, df_taxpayers, df_analysis
        )
        self.dfs_out["taxpayers_networked"] = df_networked

    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "taxpayers_networked": path_gen.processed_taxpayers_networked(configs),
        }
        self.save_dfs(save_map)

    def update_configs(self) -> None:
        pass


class WkflFinalOutput(WorkflowBase):
    """
    Produces final datasets to be converted into standardized format

    INPUTS:
        - Networked property dataset
            - 'ROOT/processed/props_networked[FileExt]'
    OUTPUTS:
        - Parquet files & database schema (?)
    """
    stage = WorkflowStage.FINAL_OUTPUT
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)

    def execute(self):
        pass
