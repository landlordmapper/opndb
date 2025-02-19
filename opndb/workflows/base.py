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
)
from opndb.constants.files import Raw as r, Dirs as d, Geocodio as g

# 4. Types (these should only depend on constants)
from opndb.types.base import (
    WorkflowConfigs,
    NmslibOptions,
    StringMatchParams,
    NetworkMatchParams,
    CleaningColumnMap,
    BooleanColumnMap, WorkflowStage,
)

# 5. Utils (these should only depend on constants and types)
from opndb.utils import UtilsBase as utils, PathGenerators as path_gen

# 6. Services (these can depend on everything else)
from opndb.services.match import StringMatch, NetworkMatchBase, MatchBase
from opndb.services.address import AddressBase as addr
from opndb.services.terminal_printers import TerminalBase as terminal
from opndb.services.dataframe import (
    DataFrameOpsBase as ops_df,
    DataFrameBaseCleaners as clean_df_base,
    DataFrameNameCleaners as clean_df_name,
    DataFrameAddressCleaners as clean_df_addr,
    DataFrameCleanersAccuracy as clean_df_acc,
    DataFrameMergers as merge_df,
    DataFrameSubsetters as subset_df,
    DataFrameColumnGenerators as cols_df,
)

class WorkflowBase(ABC):
    """
    Base workflow class the controls execution of data processing tasks required for each stage of the opndb workflow.
    Each child class that inherits from WorkflowBase corresponds to the broader workflow stage.

    CHILD WORKFLOW REQUIREMENTS:
        - Required dataframes object: instance variable containing all required dataframes and their file paths
        - Execute method: executes data load, required logic and transformations, and saves outputs
        - Load
    """
    def __init__(self, configs: WorkflowConfigs):
        self.configs: WorkflowConfigs = configs
        self._dfs: dict[str, pd.DataFrame] = {}

    def load_dfs(self, key: str, path: Path, dtype: Any = str) -> pd.DataFrame:
        """Lazy load dataframes only when needed and cache them."""
        if key not in self._dfs:
            self._dfs[key] = ops_df.load_df(path, dtype)
        return self._dfs[key].copy()

    def set_working_dfs(self, required_dfs: dict[str, Path]) -> dict[str, pd.DataFrame]:
        """Loads required dataframes for a specific workflow."""
        dfs = {}
        for key, path in required_dfs.items():
            print(f"Loading {key} dataset into pandas dataframe...")
            dfs[key] = self.load_dfs(key, path)
        return dfs

    @classmethod
    def save_dfs(cls, dfs: dict[str, pd.DataFrame], save_map: dict[str, Path]) -> None:
        """Saves dataframes to their specified paths."""
        for id, path in save_map.items():
            ops_df.save_df(dfs[id], path)

    @classmethod
    def create_workflow(cls, configs: WorkflowConfigs) -> Optional['WorkflowBase']:
        """Instantiates workflow object based on last saved progress (config['wkfl_stage'])."""
        if configs["wkfl_type"] == "data_clean":
            return WkflDataClean(configs)
        elif configs["wkfl_type"] == "address_initial":
            return WkflAddressInitial(configs)
        elif configs["wkfl_type"] == "address_geocodio":
            return WkflAddressGeocodio(configs)
        elif configs["wkfl_type"] == "name_analysis":
            return WkflNameAnalysis(configs)
        elif configs["wkfl_type"] == "address_analysis":
            return WkflAddressAnalysis(configs)
        elif configs["wkfl_type"] == "rental_subset":
            return WkflRentalSubset(configs)
        elif configs["wkfl_type"] == "clean_merge":
            return WkflCleanMerge(configs)
        elif configs["wkfl_type"] == "string_match":
            return WkflStringMatch(configs)
        elif configs["wkfl_type"] == "network_graph":
            return WkflNetworkGraph(configs)
        elif configs["wkfl_type"] == "final_output":
            return WkflFinalOutput(configs)
        return None

    @abstractmethod
    def execute(self) -> None:
        pass


class WorkflowStandardBase(WorkflowBase):
    """Base class for workflows that follow the standard load->process->save pattern"""
    def execute(self) -> None:
        """Template method implementation"""
        try:
            dfs_load = self.load()
            dfs_process = self.process(dfs_load)
            summary_stats = self.summary_stats(dfs_load, dfs_process)
            self.save(dfs_process, summary_stats)
            self.update_configs(self.configs)
        except Exception as e:
            raise

    @abstractmethod
    def load(self) -> dict[str, pd.DataFrame]:
        """Loads data files into dataframes. Returns dictionary mapping dataframes to IDs."""
        pass

    @abstractmethod
    def process(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Executes business & transformation logic for the workflow. Returns dictionary mapping processed dataframes to
        IDs, ready to be saved. Returned dictionary keys must EXACTLY match the keys of the save_map object in save().
        """
        pass

    @abstractmethod
    def summary_stats(self, dfs_load: dict[str, pd.DataFrame], dfs_process: dict[str, pd.DataFrame]):
        """Executes summary stats builder for the workflow."""
        # todo: determine summary stats data type (return data hint)
        pass

    @abstractmethod
    def save(self, dfs: dict[str, pd.DataFrame], summary_stats) -> None:
        """
        Saves processed dataframes. Keys of save_map object must match EXACTLY those of the dataframe dictionary
        returned by process().
        """
        # todo: figure out a way to enforce structure of load map/process output/save map across workflow
        pass

    @abstractmethod
    def update_configs(self, configs: WorkflowConfigs) -> None:
        """Updates configurations based on current workflow stage."""
        pass


class WkflDataClean(WorkflowStandardBase):
    """
    Initial data cleaning. Runs cleaners and validators on raw datasets. if they pass the validation checks, raw
    datasets are broken up and stored in their appropriate locations.

    INPUTS:
        - Raw taxpayer, corporate and LLC data
            - 'ROOT/raw/taxpayer_records[FileExt]'
            - 'ROOT/raw/corps[FileExt]'
            - 'ROOT/raw/llcs[FileExt]'
            - 'ROOT/raw/class_code_descriptions[FileExt]'
    OUTPUTS:
        - Cleaned taxpayer, corporate and LLC data
            = 'ROOT/processed/properties[FileExt]'
            - 'ROOT/processed/taxpayer_records[FileExt]'
            - 'ROOT/processed/corps[FileExt]'
            - 'ROOT/processed/llcs[FileExt]'
            - 'ROOT/processed/class_code_descriptions[FileExt]'
            - 'ROOT/processed/unvalidated_addrs[FileExt]'
    """
    CLEANING_COL_MAP: CleaningColumnMap = {
        "name": {
            "taxpayer_records": [
                # string column names from raw data to have name cleaners run on them
                # USE CONSTANTS
            ],
            "corps": [],
            "llcs": [],
        },
        "address": {
            "taxpayer_records": [
                # string column names from raw data to have name cleaners run on them
                # USE CONSTANTS
            ],
            "corps": [],
            "llcs": [],
        },
        "accuracy": {
            "name": {
                "taxpayer_records": [
                    # string column names from raw data to have name cleaners run on them
                    # USE CONSTANTS
                ],
                "corps": [],
                "llcs": [],
            },
            "address": {
                "taxpayer_records": [
                    # string column names from raw data to have name cleaners run on them
                    # USE CONSTANTS
                ],
                "corps": [],
                "llcs": [],
            }
        }
    }
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def load(self):
        load_map: dict[str, Path] = {
            "taxpayer_records": path_gen.raw_taxpayer_records(self.configs),
            "corps": path_gen.raw_corps(self.configs),
            "llcs": path_gen.raw_llcs(self.configs),
            "class_codes": path_gen.raw_class_codes(self.configs)
        }
        return self.set_working_dfs(load_map)

    def process(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:

        # todo: add validator that checks for required columns, throw error/failure immediately if not
        # execute on all dataframes and columns
        for df in dfs.values():
            df = clean_df_base.make_upper(df)
            df = clean_df_base.remove_symbols_punctuation(df)
            df = clean_df_base.trim_whitespace(df)
            df = clean_df_base.remove_extra_spaces(df)
            df = clean_df_base.words_to_num(df)
            df = clean_df_base.deduplicate(df)
            df = clean_df_base.convert_ordinals(df)
            df = clean_df_base.take_first(df)
            df = clean_df_base.combine_numbers(df)

        # execute on name columns only
        for id, df in dfs.items():
            # leave out additional class code cleaning for now
            # will have to be updated as class code format varies by municipality
            if id == "class_code_descriptions":
                continue
            dfs[id] = clean_df_name.switch_the(
                dfs[id],
                self.CLEANING_COL_MAP["name"][id]
            )

        # execute on the address columns only
        for id, df in dfs.items():
            # leave out additional class code cleaning for now
            # will have to be updated as class code format varies by municipality
            if id == "class_code_descriptions":
                continue
            dfs[id] = clean_df_addr.convert_nsew(
                dfs[id],
                self.CLEANING_COL_MAP["address"][id]
            )
            dfs[id] = clean_df_addr.remove_secondary_designators(
                dfs[id],
                self.CLEANING_COL_MAP["address"][id]
            )
            dfs[id] = clean_df_addr.convert_street_suffixes(
                dfs[id],
                self.CLEANING_COL_MAP["address"][id]
            )
            dfs[id] = clean_df_addr.fix_zip(
                dfs[id],
                self.CLEANING_COL_MAP["address"][id]
            )

        # execute optional cleaners that reduce accuracy
        # can customize level of accuracy by including/excluding which string cleaning functions get called
        if self.configs["accuracy"] == "low":
            # execute on address columns only
            for id, df in dfs.items():
                dfs[id] = clean_df_acc.remove_secondary_component(
                    dfs[id],
                    self.cleaning_column_map["accuracy"]["address"]
                )
                dfs[id] = clean_df_acc.convert_mixed(
                    dfs[id],
                    self.cleaning_column_map["accuracy"]["address"]
                )
                dfs[id] = clean_df_acc.drop_letters(
                    dfs[id],
                    self.cleaning_column_map["accuracy"]["address"]
                )

        # run validators
        # when they pass, continue
        # todo: add separation of properties and taxpayer datasets to include only unique combinations of raw name+address concatenations
        # todo: change column name to clean_name and clean_address
        # todo: merge raw taxpayer data into cleaned data, rename raw_name and raw_address
        # todo: create clean_address column with comma-separated addresses - will be single source of truth
        # todo: fetch unique raw addresses and store in PROCESSED/unvalidated_addrs
        # set summary stats
        # update configuration file
        return dfs

    def summary_stats(self, dfs_load: dict[str, pd.DataFrame], dfs_process: dict[str, pd.DataFrame]):
        pass

    def save(self, dfs: dict[str, pd.DataFrame], summary_stats) -> None:
        save_map: dict[str, Path] = {
            "properties": path_gen.processed_properties(self.configs),
            "taxpayer_records": path_gen.processed_taxpayer_records(self.configs),
            "corps": path_gen.processed_corps(self.configs),
            "llcs": path_gen.processed_llcs(self.configs),
            "class_codes": path_gen.processed_class_codes(self.configs),
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(self.configs),
        }
        self.save_dfs(dfs, save_map)

    def update_configs(self, configs: WorkflowConfigs) -> None:
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
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def load(self) -> dict[str, pd.DataFrame]:
        load_map: dict[str, Path] = {
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(self.configs),
        }
        return self.set_working_dfs(load_map)

    def process(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        # todo: add conditional handling for whether or not street column is separate from full address column
        # todo: use pydantic/panderas model to created validated object address rows
        # add is_pobox
        dfs["unvalidated_addrs"]["is_pobox"] = cols_df.set_is_pobox(dfs["unvalidated_addrs"], ua.TAX_STREET)
        # subset is_pobox
        df_pobox: pd.DataFrame = subset_df.get_is_pobox(dfs["unvalidated_addrs"])
        # cleans up pobox addresses
        poboxes_cleaned = addr.clean_poboxes(df_pobox)  # these should be pydantic/pandera model objects
        # checks city names and zipcodes
        poboxes_validated = addr.validate_poboxes(df_pobox)  # these should be pydantic/pandera model objects
        # adds pobox rows with validated city/state/zip to df_validated_addrs
        df_validated_addrs: pd.DataFrame = pd.DataFrame(poboxes_validated)
        # removes validated pobox addrs from unvalidated_addrs, saves
        df_unvalidated_addrs: pd.DataFrame = subset_df.update_unvalidated_addrs(
            dfs["unvalidated_addrs"],
            list(df_validated_addrs[va.CLEAN_ADDRESS].unique())
        )
        return dfs

    def summary_stats(self, dfs_load: dict[str, pd.DataFrame], dfs_process: dict[str, pd.DataFrame]):
        pass

    def save(self, dfs: dict[str, pd.DataFrame], summary_stats) -> None:
        save_map: dict[str, Path] = {
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(self.configs),
            "validated_addrs": path_gen.processed_validated_addrs(self.configs),
        }
        self.save_dfs(dfs, save_map)

    def update_configs(self, configs: WorkflowConfigs) -> None:
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
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def load(self) -> dict[str, pd.DataFrame]:
        load_map: dict[str, Path] = {
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(self.configs),
        }
        return self.set_working_dfs(load_map)

    def process(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        # print out address count to be validated using geocodio
        terminal.print_geocodio_warning(dfs["unvalidated_addrs"])
        # prompt user for api key and display warning with number of calls and estimated cost
        self.api_key = terminal.enter_geocodio_api_key()
        # call geocodio or exit
        addr.run_geocodio(self.api_key, dfs["unvalidated_addrs"], ua.TAX_FULL_ADDRESS)  # have it return the dfs?
        # add validated addrs to the master files and save to data dirs
        df_validated_gcd: pd.DataFrame = ops_df.load_df(
            utils.generate_path(
                d.GEOCODIO,
                g.get_raw_filename_ext(g.GCD_VALIDATED, self.configs),
                self.configs["prev_stage"],
                self.configs["load_ext"]
            )
        )
        # remove validated raw address from unvalidated master file
        dfs["unvalidated_gcd"] = subset_df.update_unvalidated_addrs(
            dfs["unvalidated_gcd"],
            df_validated_gcd[va.CLEAN_ADDRESS]
        )
        return dfs

    def summary_stats(self, dfs_load: dict[str, pd.DataFrame], dfs_process: dict[str, pd.DataFrame]):
        pass

    def save(self, dfs: dict[str, pd.DataFrame], summary_stats) -> None:
        save_map: dict[str, Path] = {
            "gcd_validated": path_gen.geocodio_gcd_validated(self.configs),
            "gcd_unvalidated": path_gen.geocodio_gcd_unvalidated(self.configs),
            "gcd_failed": path_gen.geocodio_gcd_failed(self.configs),
            "unvalidated_addrs": path_gen.processed_unvalidated_addrs(self.configs),
            "validated_addrs": path_gen.processed_validated_addrs(self.configs),
        }
        self.save_dfs(dfs, save_map)

    def update_configs(self, configs: WorkflowConfigs) -> None:
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
