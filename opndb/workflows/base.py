# 1. Standard library imports
import shutil
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Optional

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
    DataFrameCleanersAccuracy as clean_df_acc,
)
from opndb.services.dataframe.ops import (
    DataFrameMergers as merge_df,
    DataFrameSubsetters as subset_df,
    DataFrameColumnGenerators as cols_df,
    DataFrameColumnManipulators as colm_df
)
from rich.console import Console


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
            console.print(f"\"{id}\" successfully loaded from: \n{path}")
        # prep data for summary table printing
        table_data = []
        for id, df in self.dfs_in.items():  # todo: standardize this, enforce types
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
        for id, path in save_map.items():
            ops_df.save_df(self.dfs_out[id], path)
            console.print(f"\"{id}\" successfully saved to: \n{path}")

    @classmethod
    def create_workflow(cls, config_manager: ConfigManager) -> Optional['WorkflowBase']:
        """Instantiates workflow object based on last saved progress (config['wkfl_stage'])."""
        configs = config_manager.configs
        if configs["wkfl_type"] == "data_clean":
            return WkflDataClean(config_manager)
        elif configs["wkfl_type"] == "address_initial":
            return WkflAddressInitial(config_manager)
        elif configs["wkfl_type"] == "address_geocodio":
            return WkflAddressGeocodio(config_manager)
        # elif configs["wkfl_type"] == "name_analysis":
        #     return WkflNameAnalysis(config_manager)
        # elif configs["wkfl_type"] == "address_analysis":
        #     return WkflAddressAnalysis(config_manager)
        # elif configs["wkfl_type"] == "rental_subset":
        #     return WkflRentalSubset(config_manager)
        # elif configs["wkfl_type"] == "clean_merge":
        #     return WkflCleanMerge(config_manager)
        # elif configs["wkfl_type"] == "string_match":
        #     return WkflStringMatch(config_manager)
        # elif configs["wkfl_type"] == "network_graph":
        #     return WkflNetworkGraph(config_manager)
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
            self.summary_stats()
            self.save()
            self.update_configs()
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
        t.print_with_dots(f"Setting raw columns for \"{id}\"...")
        df: pd.DataFrame = cols_df.set_raw_columns(df, column_manager.raw)
        console.print(f"Raw columns generated ✅")

        # rename columns to prepare for cleaning
        df.rename(columns=column_manager.clean_rename_map, inplace=True)

        # create raw_address field
        t.print_with_dots(f"Generating full raw address columns (if not already present)")
        df: pd.DataFrame = cols_df.set_full_address_fields(df, column_manager.raw_address_map)
        console.print(f"Full raw address columns generated ✅")

        # props_taxpayers-specific logic - concatenate raw name + raw address
        if id == "props_taxpayers":
            t.print_with_dots(f"Concatenating raw name and address fields")
            df: pd.DataFrame = cols_df.set_name_address_concat(
                df, column_manager.name_address_concat_map["raw"]
            )
            console.print(f"\"name_address\" field generated ✅")

        return df

    def execute_basic_cleaning(self, id: str, df: pd.DataFrame, column_manager) -> pd.DataFrame:

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

    def execute_name_column_cleaning(self, id: str, df: pd.DataFrame, column_manager) -> pd.DataFrame:

        t.print_equals(f"Executing cleaning operations (name columns only)")
        cols: list[str] = column_manager.name_clean
        df = clean_df_name.switch_the(df, cols)
        console.print(f"Name field cleaning complete ✅")

        return df

    def execute_address_column_cleaning(self, id: str, df: pd.DataFrame, column_manager) -> pd.DataFrame:

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
        df: pd.DataFrame = cols_df.set_full_address_fields(df, column_manager.clean_address_map)
        t.print_with_dots("Full clean address fields generated ✅")

        # props_taxpayers-specific logic
        if id == "props_taxpayers":

            # generate clean name + address concat field
            t.print_with_dots(f"Concatenating clean name and address fields")
            df: pd.DataFrame = cols_df.set_name_address_concat(
                df, column_manager.name_address_concat_map["clean"]
            )
            console.print(f"\"name_address\" field generated ✅")

            # split dataframe into properties and taxpayer_records
            t.print_with_dots(f"Splitting \"{id}\" into \"taxpayer_records\" and \"properties\"...")
            col_manager_p = out_column_managers["properties"]
            col_manager_t = out_column_managers["taxpayer_records"]
            df_props, df_taxpayers = subset_df.split_props_taxpayers(df, col_manager_p.out, col_manager_t.out)
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
        self.dfs_out["unvalidated_addrs"] = subset_df.generate_unvalidated_df(self.dfs_out, col_map)
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
                df: pd.DataFrame = self.execute_basic_cleaning(id, df, column_manager[id])
                self.dfs_out[id] = df
                continue

            # PRE-CLEANING OPERATIONS
            df: pd.DataFrame = self.execute_pre_cleaning(id, df, column_manager[id])

            # # MAIN CLEANING OPERATIONS
            df: pd.DataFrame = self.execute_basic_cleaning(id, df, column_manager[id])
            df: pd.DataFrame = self.execute_name_column_cleaning(id, df, column_manager[id])
            df: pd.DataFrame = self.execute_address_column_cleaning(id, df, column_manager[id])
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


class WkflAddressInitial(WorkflowStandardBase):
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
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)

    def load(self):
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(configs),
        }
        self.load_dfs(load_map)

    def process(self) -> None:
        column_manager = {"unvalidated_addrs": ColumnUnvalidatedAddrs()}
        df: pd.DataFrame = self.dfs_in["unvalidated_addrs"].copy()
        df["is_pobox"] = subset_df.get_is_pobox(df["unvalidated_addrs"])
        df: pd.DataFrame = colm_df.fix_pobox(df, column_manager["unvalidated_addrs"].CLEAN_ADDRESS)
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
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)

    def load(self) -> None:
        configs = self.config_manager.configs
        load_map: dict[str, Path] = {
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(configs),
            "validated_addrs": path_gen.processed_validated_addrs(configs),
            "gcd_unvalidated": path_gen.geocodio_gcd_unvalidated(configs),
            "gcd_validated": path_gen.geocodio_gcd_validated(configs),
            "gcd_failed": path_gen.geocodio_gcd_failed(configs),
        }
        self.load_dfs(load_map)

    def process(self):

        dfs = {id: df.copy() for id, df in self.dfs_in.items()}

        configs = self.config_manager.configs
        column_manager = {
            "unvalidated_addrs": ColumnUnvalidatedAddrs(),
            "validated_addrs": ColumnValidatedAddrs(),
        }
        df_addrs: pd.DataFrame = self.dfs_in["unvalidated_addrs"][
            column_manager["unvalidated_addrs"].geocodio_columns
        ].copy()


        # get geocodio api key from user if not already exists in configs.json
        if "geocodio_api_key" not in self.config_manager.configs.keys():
            while True:
                api_key: str = input("Copy & paste your geocodio API key for address validation: ")
                if len(api_key) > 0:
                    self.config_manager.set("geocodio_api_key", api_key)
                    break
                else:
                    console.print("You must enter an API key to continue.")

        # print out final warning and estimated cost
        # print out address count to be validated using geocodio
        # number of unique clean_address values (i.e., number of addresses to be geocoded)
        t.print_geocodio_warning(dfs["unvalidated_addrs"])

        cont: bool = t.press_enter_to_continue("execute geocodio API calls ")
        if not cont:
            console.print("Aborted!")
            return

        # call geocodio or exit
        gcd_results_obj: GeocodioReturnObject = addr.run_geocodio(
            configs["geocodio_api_key"],
            df_addrs,
            column_manager["unvalidated_addrs"].CLEAN_ADDRESS
        )

        # create new dataframes from the resulting geocodio call
        df_validated: pd.DataFrame = pd.DataFrame(gcd_results_obj["validated"])
        df_unvalidated: pd.DataFrame = subset_df.remove_unvalidated_addrs(  # remove all successfully validated addresses from unvalidated master address dataset loaded at beginning of workflow
            dfs["unvalidated_addrs"],
            [d["clean_address"] for d in gcd_results_obj["validated"]]
        )
        df_failed: pd.DataFrame = pd.DataFrame(gcd_results_obj["failed"])

        # if there were already gcd_validated, gcd_unvalidated and gcd_failed in the directories, concatenate the new one and set to dfs_out
        if self.dfs_in["gcd_validated"] is not None:
            df_validated = pd.concat([self.dfs_in["gcd_validated"], df_validated], ignore_index=True)
        if self.dfs_in["gcd_unvalidated"] is not None:
            df_unvalidated = pd.concat([self.dfs_in["gcd_unvalidated"], df_unvalidated], ignore_index=True)
        if self.dfs_in["gcd_failed"] is not None:
            df_failed = pd.concat([self.dfs_in["gcd_failed"], df_failed], ignore_index=True)

        validated_before: int = len(self.dfs_in["validated_addrs"]) if self.dfs_in["validated_addrs"] is not None else 0
        validated_after: int = len(df_validated)
        validated_diff: int = validated_after - validated_before

        unvalidated_before: int = len(self.dfs_in["unvalidated_addrs"])
        unvalidated_after: int = unvalidated_before - (validated_diff)
        unvalidated_diff: int = unvalidated_before - unvalidated_after

        failed_before: int = len(self.dfs_in["gcd_failed"]) if self.dfs_in["gcd_failed"] is not None else 0
        failed_after: int = len(df_failed)
        failed_diff: int = failed_after - failed_before

        console.print(f"Total validated addresses: {validated_after} (+{validated_diff})")
        console.print(f"Total unvalidated addresses: {unvalidated_after} (-{unvalidated_diff})")
        console.print(f"Total failed addresses: {failed_after} (+{failed_diff})")

        self.dfs_out: dict[str, pd.DataFrame] = {
            "gcd_validated": df_validated,
            "gcd_unvalidated": df_unvalidated,
            "gcd_failed": df_failed,
            "validated_addrs": df_validated,
            "unvalidated_addrs": df_unvalidated,
        }


    def summary_stats(self) -> None:
        pass

    def save(self) -> None:
        configs = self.config_manager.configs
        save_map: dict[str, Path] = {
            "gcd_validated": path_gen.geocodio_gcd_validated(configs),
            "gcd_unvalidated": path_gen.geocodio_gcd_unvalidated(configs),
            "gcd_failed": path_gen.geocodio_gcd_failed(configs),
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(configs),
            "validated_addrs": path_gen.processed_validated_addrs(configs),
        }
        self.save_dfs(save_map)

    def update_configs(self) -> None:
        pass


class WkflNameAnalysis(WorkflowStandardBase):
    """
    Taxpayer name analysis workflow.

    INPUTS:
    OUTPUTS:
    """
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def load(self) -> dict[str, pd.DataFrame]:
        load_map: dict[str, Path] = {}
        return self.set_working_dfs(load_map)

    def process(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        pass

    def summary_stats(self, dfs_load: dict[str, pd.DataFrame], dfs_process: dict[str, pd.DataFrame]):
        pass

    def save(self, dfs: dict[str, pd.DataFrame], summary_stats) -> None:
        save_map: dict[str, Path] = {}
        self.save_dfs(dfs, save_map)

    def update_configs(self, configs: WorkflowConfigs) -> None:
        pass


class WkflAddressAnalysis(WorkflowStandardBase):
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
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def load(self) -> dict[str, pd.DataFrame]:
        load_map: dict[str, Path] = {
            "validated_addrs": path_gen.processed_validated_addrs(self.configs),
        }
        return self.set_working_dfs(load_map)

    def process(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        # create & save dataframe with unique validated addresses & their count
        df_addr_counts: pd.DataFrame = ops_df.get_frequency_df(
            dfs["validated_addrs"],
            va.FORMATTED_ADDRESS
        )
        return dfs

    def summary_stats(self, dfs_load: dict[str, pd.DataFrame], dfs_process: dict[str, pd.DataFrame]):
        pass

    def save(self, dfs: dict[str, pd.DataFrame], summary_stats) -> None:
        save_map: dict[str, Path] = {}
        self.save_dfs(dfs, save_map)

    def update_configs(self, configs: WorkflowConfigs) -> None:
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
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def load(self) -> dict[str, pd.DataFrame]:
        load_map: dict[str, Path] = {
            "properties": path_gen.processed_properties(self.configs),
            "taxpayer_records": path_gen.processed_taxpayer_records(self.configs),
            "validated_addrs": path_gen.processed_validated_addrs(self.configs),
            "class_codes": path_gen.processed_class_codes(self.configs)
        }
        return self.set_working_dfs(load_map)

    def process(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        # merge validated_addrs
        df_taxpayers_addrs: pd.DataFrame = merge_df.merge_validated_addrs(
            dfs["taxpayer_records"],
            dfs["validated_addrs"],
        )
        # add is_rental column to properties
        dfs["properties"]: pd.DataFrame = cols_df.set_is_rental(
            df_taxpayers_addrs,
            dfs["class_codes"],
        )
        # todo: fix logic to handle property & taxpayer record datasets distinctly
        # execute initial subset based on is_rental
        df_rentals_initial: pd.DataFrame = dfs["properties"][dfs["properties"][tr.IS_RENTAL] == True]
        # fetch properties left out of initial subset with matching validated taxpayer addresses
        df_rentals_addrs: pd.DataFrame = subset_df.get_nonrentals_from_addrs(dfs["properties"], df_rentals_initial)
        # pull non-rentals with matching rental taxpayer addresses
        df_rentals_final: pd.DataFrame = pd.concat([df_rentals_initial, df_rentals_addrs], axis=1)
        return {
            "taxpayers_subsetted": df_rentals_final,
            "properties_subsetted": dfs["properties"],
        }

    def summary_stats(self, dfs_load: dict[str, pd.DataFrame], dfs_process: dict[str, pd.DataFrame]):
        pass

    def save(self, dfs: dict[str, pd.DataFrame], summary_stats) -> None:
        save_map: dict[str, Path] = {}
        self.save_dfs(dfs, save_map)

    def update_configs(self, configs: WorkflowConfigs) -> None:
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
    BOOL_COL_MAP: BooleanColumnMap = {
        "taxpayer_records": [tr.CLEAN_NAME],
        "corps": [c.PRESIDENT_NAME, c.SECRETARY_NAME],
        "llcs": [l.MANAGER_MEMBER_NAME, l.AGENT_NAME],
    }
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def load(self) -> dict[str, pd.DataFrame]:
        load_map: dict[str, Path] = {
            "taxpayer_records": path_gen.processed_props_subsetted(self.configs),
            "corps": path_gen.raw_corps(self.configs),
            "llcs": path_gen.raw_llcs(self.configs),
            "validated_addrs": path_gen.processed_validated_addrs(self.configs),
        }
        return self.set_working_dfs(load_map)

    def process(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        # todo: add workflow for fixing taxpayer addresses based on manual research - services/address.py
        # todo: add workflow for fixing taxpayer names based on manual research - services/base.py
        # add "core_name" columns
        dfs["taxpayer_records"]["core_name"] = cols_df.set_core_name(dfs["taxpayer_records"], tr.CLEAN_NAME)
        dfs["corps"]["core_name"] = cols_df.set_core_name(dfs["corps"], c.CLEAN_NAME)
        dfs["llcs"]["core_name"] = cols_df.set_core_name(dfs["llcs"], l.CLEAN_NAME)
        # add bool columns for is_common_name, is_org, is_llc, is_person, is_bank
        for id, cols in self.BOOL_COL_MAP.items():
            for col in cols:
                dfs[id]["is_bank"] = cols_df.set_is_bank(dfs[id], col)
                dfs[id]["is_person"] = cols_df.set_is_person(dfs[id], col)
                dfs[id]["is_common_name"] = cols_df.set_is_common_name(dfs[id], col)
                dfs[id]["is_org"] = cols_df.set_is_org(dfs[id], col)
                dfs[id]["is_llc"] = cols_df.set_is_llc(dfs[id], col)
        # subset active corps/llcs
        # todo: ask for user input to indicate whether only active llcs/corps should be merged into properties, or if all of them should
        dfs["corps"] = subset_df.get_active(dfs["corps"], c.STATUS)
        dfs["llcs"] = subset_df.get_active(dfs["llcs"], l.STATUS)
        # drop duplicates corps/llcs
        dfs["corps"].drop_duplicates(subset=[c.CLEAN_NAME], inplace=True)
        dfs["llcs"].drop_duplicates(subset=[l.CLEAN_NAME], inplace=True)
        # add validated addrs to corps/llcs
        for id, df in dfs.items():
            if id == "taxpayer_records":  # it already has it
                continue
            # todo: fix merge_valid_addrs to handle different column names
            df = merge_df.merge_validated_addrs(df, dfs["validated_addrs"], [])
        # merge corps/llcs to taxpayer records
        dfs["taxpayer_records"] = merge_df.merge_orgs(dfs["taxpayer_records"], dfs["corps"])
        dfs["taxpayer_records"] = merge_df.merge_orgs(dfs["taxpayer_records"], dfs["corps"])
        return dfs

    def summary_stats(self, dfs_load: dict[str, pd.DataFrame], dfs_process: dict[str, pd.DataFrame]):
        pass

    def save(self, dfs: dict[str, pd.DataFrame], summary_stats) -> None:
        save_map: dict[str, Path] = {}
        self.save_dfs(dfs, save_map)

    def update_configs(self, configs: WorkflowConfigs) -> None:
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
    DEFAULT_NMSLIB: NmslibOptions = {
        "method": "hnsw",
        "space": "cosinesimil_sparse_fast",
        # "data_type": nmslib.DataType.SPARSE_VECTOR
    }
    DEFAULT_QUERY_BATCH = {
        "num_threads": 8,
        "K": 1
    }
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def load(self) -> dict[str, pd.DataFrame]:
        load_map: dict[str, Path] = {
            "props_prepped": path_gen.processed_props_prepped(self.configs),
            "address_analysis": path_gen.analysis_address_analysis(self.configs),
        }
        return self.set_working_dfs(load_map)

    def process(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:

        # 1. SET PARAMS
        # define default params matrix
        default_matrix: list[StringMatchParams] = [
            {
                "name_col": "",  # todo: handle the name columns
                "match_threshold": 0.85,
                "include_orgs": False,
                "include_unresearched": False,
                "nmslib_opts": self.DEFAULT_NMSLIB,
                "query_batch_opts": self.DEFAULT_QUERY_BATCH,
            },
            {
                "name_col": "",
                "include_orgs": False,
                "match_threshold": 0.85,
                "include_unresearched": True,
                "nmslib_opts": self.DEFAULT_NMSLIB,
                "query_batch_opts": self.DEFAULT_QUERY_BATCH,
            },
            {
                "name_col": "",
                "match_threshold": 0.8,
                "include_orgs": False,
                "include_unresearched": True,
                "nmslib_opts": self.DEFAULT_NMSLIB,
                "query_batch_opts": self.DEFAULT_QUERY_BATCH,
            },
        ]
        # ask user if they want to change the params matrix, make adjustments accordingly
        # if not, set matrix instance var to default
        self.params_matrix = default_matrix

        # 2. PREP DATA
        # set address column so that if there is no validated address for a property, the raw address is used instead
        dfs["props_prepped"]["matching_address"] = dfs["props_prepped"].apply(
            lambda row: MatchBase.set_matching_address(row), axis=1
        )
        # add name+address concatenated columns to use for matching
        dfs["props_prepped"]["name_address_clean"] = StringMatch.concatenate_name_addr(
            dfs["props_prepped"],
            "name_address_clean",
            "clean_name",
            "matching_address"
        )
        dfs["props_prepped"]["name_address_core"] = StringMatch.concatenate_name_addr(
            dfs["props_prepped"],
            "name_address_core",
            "core_name",
            "matching_address"
        )

        # 3. RUN STRING MATCHING
        # raise exceptions is parameter matrix or input dataframe are empty
        if self.params_matrix is None:
            raise Exception("Parameter matrix is empty.")

        # execute param matrix loop for string matching workflow
        for i, params in enumerate(self.params_matrix):
            # set include_address column
            dfs["props_prepped"]["include_address"] = dfs["props_prepped"]["matching_address"].apply(
                lambda addr: MatchBase.check_address(
                    addr,
                    dfs["address_analysis"],
                    params["include_orgs"],
                    params["include_unresearched"],
                )
            )
            # filter out addresses
            df_filtered_addrs: pd.DataFrame = dfs["props_prepped"][dfs["props_prepped"]["include_address"] == True]
            df_filtered: pd.DataFrame = df_filtered_addrs[[
                "clean_name", "core_name", "matching_address", "name_address_clean", "name_address_core"
            ]]
            # set ref & query docs
            if params["name_col"] == "clean_name":
                ref_docs: list[str] = list(df_filtered["name_address_clean"].dropna().unique())
                query_docs: list[str] = list(df_filtered["name_address_clean"].dropna().unique())
            else:
                ref_docs: list[str] = list(df_filtered["name_address_core"].dropna().unique())
                query_docs: list[str] = list(df_filtered["name_address_core"].dropna().unique())

            # get string matches
            df_matches: pd.DataFrame = StringMatch.match_strings(
                ref_docs=ref_docs,
                query_docs=query_docs,
                params=params
            )
            # generate network graph to associated matches
            self.df_process_results: pd.DataFrame = NetworkMatchBase.string_match_network_graph(
                df_process_input=dfs["props_prepped"],
                df_matches=df_matches,
                match_count=i,
                name_address_column=""  # todo: fix this
            )

        # 4. MERGE TO ORIGINAL DF
        # merge output with original properties dataset
        df_process_output: pd.DataFrame = pd.merge(
            dfs["props_prepped"],
            dfs["props_prepped"],
            how="left",
            on=""  # todo: fix this
        )
        # clean up
        df_process_output: pd.DataFrame = ops_df.combine_columns_parallel(df_process_output)
        df_process_output.drop_duplicates(tr.PIN, inplace=True)
        df_process_output.drop(columns=["original_doc"], inplace=True)  # todo: which other columns should be dropped? - NOT name_address_clean

        return {"props_string_matched": df_process_output}

    def summary_stats(self, dfs_load: dict[str, pd.DataFrame], dfs_process: dict[str, pd.DataFrame]):
        pass

    def save(self, dfs: dict[str, pd.DataFrame], summary_stats) -> None:
        save_map: dict[str, Path] = {}
        self.save_dfs(dfs, save_map)

    def update_configs(self, configs: WorkflowConfigs) -> None:
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
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)
        self.stage: ClassVar[WorkflowStage] = WorkflowStage.NETWORK_GRAPH

    def load(self) -> dict[str, pd.DataFrame]:
        load_map: dict[str, Path] = {
            "props_string_matched": path_gen.processed_props_string_matched(self.configs),
            "address_analysis": path_gen.analysis_address_analysis(self.configs),
        }
        return self.set_working_dfs(load_map)

    def process(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        # 1. SET PARAMS
        # define default params matrix
        default_matrix: list[NetworkMatchParams] = [
            {
                "taxpayer_name_col": "",
                "entity_name_col": "",
                "include_orgs": False,
                "include_unresearched": False,
                "string_match_name": "",
            },
            {
                "taxpayer_name_col": "",
                "entity_name_col": "",
                "include_orgs": False,
                "include_unresearched": False,
                "string_match_name": "",
            },
            {
                "taxpayer_name_col": "",
                "entity_name_col": "",
                "include_orgs": False,
                "include_unresearched": True,
                "string_match_name": "",
            },
            {
                "taxpayer_name_col": "",
                "entity_name_col": "",
                "include_orgs": True,
                "include_unresearched": False,
                "string_match_name": "",
            },
            {
                "taxpayer_name_col": "",
                "entity_name_col": "",
                "include_orgs": True,
                "include_unresearched": True,
                "string_match_name": "",
            },
            {
                "taxpayer_name_col": "",
                "entity_name_col": "",
                "include_orgs": True,
                "include_unresearched": True,
                "string_match_name": "",
            },
        ]
        # ask user if they want to change the params matrix, make adjustments accordingly
        # if not, set matrix instance var to default
        self.params_matrix = default_matrix

        # 2. DATA PREP
        dfs["props_string_matched"].drop_duplicates(subset=["name_address_clean"], inplace=True)

        # 3. EXECUTE NETWORK CODE
        df_process: pd.DataFrame = dfs["props_string_matched"].copy()  # todo: determine if this is necessary
        for i, params in enumerate(self.params_matrix):
            # generate network graph and dataframe with component ID column
            df_rentals_components, gMatches = NetworkMatchBase.rentals_network(
                dfs["props_string_matched"],
                dfs["address_analysis"],
                params
            )
            # merge to original dataset
            df_process: pd.DataFrame = pd.merge(
                df_process,
                df_rentals_components[["name_address_clean", f"final_component_{i+1}"]],
                how="left",
                on="name_address_clean",
            )
            # set network canonical names
            df_process: pd.DataFrame = NetworkMatchBase.set_network_name(
                i+1,
                df_process,
                f"final_component_{i+1}",
                f"network_{i+1}"
            )
            # set text for node/edge data - should this be a separate dataset? probably
            df_process: pd.DataFrame = NetworkMatchBase.set_network_text(
                gMatches,
                df_process,
                f"final_component_{i+1}",
                f"network_{i+1}"
            )

        return {"props_networked": df_process}

    def summary_stats(self, dfs_load: dict[str, pd.DataFrame], dfs_process: dict[str, pd.DataFrame]):
        pass

    def save(self, dfs: dict[str, pd.DataFrame], summary_stats) -> None:
        save_map: dict[str, Path] = {}
        self.save_dfs(dfs, save_map)

    def update_configs(self, configs: WorkflowConfigs) -> None:
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
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        # saves final outputs to "final_outputs" directory
        # set summary stats
        # update configuration file
        pass
