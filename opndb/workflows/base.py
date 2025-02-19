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
from opndb.services.terminal_printers import TerminalBase as terminal, TerminalBase
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
            dfs[key] = self.load_dfs(key, path)
        # prep data for summary table printing
        table_data = []
        for id, df in self._dfs.items():  # todo: standardize this, enforce types
            memory_usage = df.memory_usage(deep=True).sum()
            table_data.append({
                "dataset_name": id,
                "file_size": utils.sizeof_fmt(memory_usage),
                "record_count": len(df)
            })
        TerminalBase.display_table(table_data)
        return dfs

    @classmethod
    def save_dfs(cls, dfs: dict[str, pd.DataFrame], save_map: dict[str, Path]) -> None:
        """Saves dataframes to their specified paths."""
        for id, path in save_map.items():
            ops_df.save_df(dfs[id], path)

    @classmethod
    def exclude_raw_cols(cls, df: pd.DataFrame) -> list[str]:
        return list(filter(lambda x: not x.startswith("raw"), df.columns))

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
    CLEANING_COL_MAP = {
        "name": {
            "props_taxpayers": [
                pt.TAX_NAME
            ],
            "corps": [
                c.NAME,
                c.PRESIDENT_NAME,
                c.SECRETARY_NAME
            ],
            "llcs": [
                l.NAME,
                l.MANAGER_MEMBER_NAME,
                l.AGENT_NAME
            ],
        },
        "address": {
            "props_taxpayers": {
                "street": [
                    pt.TAX_ADDRESS,
                    pt.TAX_STREET
                ],
                "zip": [
                    pt.TAX_ZIP
                ],
            },
            "corps": {
                "street": [
                    c.PRESIDENT_ADDRESS,
                    c.PRESIDENT_STREET,
                    c.SECRETARY_ADDRESS,
                    c.SECRETARY_STREET
                ],
                "zip": [
                    c.PRESIDENT_ZIP,
                    c.SECRETARY_ZIP
                ],
            },
            "llcs": {
                "street": [
                    l.OFFICE_ADDRESS,
                    l.OFFICE_STREET,
                    l.MANAGER_MEMBER_ADDRESS,
                    l.MANAGER_MEMBER_STREET,
                    l.AGENT_ADDRESS,
                    l.AGENT_STREET
                ],
                "zip": [
                    l.OFFICE_ZIP,
                    l.MANAGER_MEMBER_ZIP,
                    l.AGENT_ZIP
                ],
            },
        },
        # "accuracy": {
        #     "name": {
        #         "taxpayer_records": [],
        #         "corps": [],
        #         "llcs": [],
        #     },
        #     "address": {
        #         "taxpayer_records": {
        #             "street": [p.TAX_STREET],
        #             "zip": [p.TAX_ZIP],
        #         },
        #         "corps": {
        #             "street": [p.TAX_STREET],
        #             "zip": [p.TAX_ZIP],
        #         },
        #         "llcs": {
        #             "street": [p.TAX_STREET],
        #             "zip": [p.TAX_ZIP],
        #         },
        #     }
        # }
    }
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def load(self):
        load_map: dict[str, Path] = {
            "props_taxpayers": path_gen.raw_props_taxpayers(self.configs),
            "corps": path_gen.raw_corps(self.configs),
            "llcs": path_gen.raw_llcs(self.configs),
            "class_codes": path_gen.raw_class_codes(self.configs)
        }
        return self.set_working_dfs(load_map)

    def process(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:

        # todo: add validator that checks for required columns, throw error/failure immediately if not
        # todo: add to columns validator: if _ADDRESS is empty, _STREET and _ZIP must be present; otherwise, if _ADDRESS exists, all others can be empty/missing
        # todo: add to properties/taxpayer records validator: unique column constraint on PIN

        # -------------------------------
        # ----PRE-CLEANING OPERATIONS----
        # -------------------------------
        # creating formatted complete raw address if doesn't already exist
        if pt.TAX_ADDRESS not in dfs["props_taxpayers"].columns:
            dfs["props_taxpayers"][pt.TAX_ADDRESS] = dfs["props_taxpayers"].apply(
                lambda row: f"{row[pt.TAX_STREET]}, {row[pt.TAX_CITY]}, {row[pt.TAX_STATE]} {row[pt.TAX_ZIP]}"
            )
        # concatenate raw taxpayer name+address
        dfs["props_taxpayers"][p.RAW_NAME_ADDRESS] = dfs["props_taxpayers"].apply(
            lambda row: row[pt.TAX_NAME] + " -- " + row[pt.TAX_ADDRESS],
        )
        # generate raw-prefixed columns - store raw data
        dfs["props_taxpayers"][tr.RAW_NAME] = dfs["props_taxpayers"][pt.TAX_NAME].copy()
        dfs["props_taxpayers"][tr.RAW_ADDRESS] = dfs["props_taxpayers"][pt.TAX_ADDRESS].copy()
        dfs["props_taxpayers"][tr.RAW_STREET] = dfs["props_taxpayers"][pt.TAX_STREET].copy()
        dfs["props_taxpayers"][tr.RAW_CITY] = dfs["props_taxpayers"][pt.TAX_CITY].copy()
        dfs["props_taxpayers"][tr.RAW_STATE] = dfs["props_taxpayers"][pt.TAX_STATE].copy()
        dfs["props_taxpayers"][tr.RAW_ZIP] = dfs["props_taxpayers"][pt.TAX_ZIP].copy()
        dfs["corps"][c.RAW_NAME] = dfs["corps"][c.NAME].copy()
        dfs["corps"][c.RAW_PRESIDENT_NAME] = dfs["corps"][c.PRESIDENT_NAME].copy()
        dfs["corps"][c.RAW_PRESIDENT_ADDRESS] = dfs["corps"][c.PRESIDENT_ADDRESS].copy()
        dfs["corps"][c.RAW_PRESIDENT_STREET] = dfs["corps"][c.PRESIDENT_STREET].copy()
        dfs["corps"][c.RAW_PRESIDENT_CITY] = dfs["corps"][c.PRESIDENT_CITY].copy()
        dfs["corps"][c.RAW_PRESIDENT_STATE] = dfs["corps"][c.PRESIDENT_STATE].copy()
        dfs["corps"][c.RAW_PRESIDENT_ZIP] = dfs["corps"][c.PRESIDENT_ZIP].copy()
        dfs["corps"][c.RAW_SECRETARY_NAME] = dfs["corps"][c.SECRETARY_NAME].copy()
        dfs["corps"][c.RAW_SECRETARY_ADDRESS] = dfs["corps"][c.SECRETARY_ADDRESS].copy()
        dfs["corps"][c.RAW_SECRETARY_STREET] = dfs["corps"][c.SECRETARY_STREET].copy()
        dfs["corps"][c.RAW_SECRETARY_CITY] = dfs["corps"][c.SECRETARY_CITY].copy()
        dfs["corps"][c.RAW_SECRETARY_STATE] = dfs["corps"][c.SECRETARY_STATE].copy()
        dfs["corps"][c.RAW_SECRETARY_ZIP] = dfs["corps"][c.SECRETARY_ZIP].copy()
        dfs["llcs"][l.RAW_NAME] = dfs["llcs"][l.NAME].copy()
        dfs["llcs"][l.RAW_MANAGER_MEMBER_NAME] = dfs["llcs"][l.MANAGER_MEMBER_NAME].copy()
        dfs["llcs"][l.RAW_MANAGER_MEMBER_ADDRESS] = dfs["llcs"][l.MANAGER_MEMBER_ADDRESS].copy()
        dfs["llcs"][l.RAW_MANAGER_MEMBER_STREET] = dfs["llcs"][l.MANAGER_MEMBER_STREET].copy()
        dfs["llcs"][l.RAW_MANAGER_MEMBER_CITY] = dfs["llcs"][l.MANAGER_MEMBER_CITY].copy()
        dfs["llcs"][l.RAW_MANAGER_MEMBER_STATE] = dfs["llcs"][l.MANAGER_MEMBER_STATE].copy()
        dfs["llcs"][l.RAW_MANAGER_MEMBER_ZIP] = dfs["llcs"][l.MANAGER_MEMBER_ZIP].copy()
        dfs["llcs"][l.RAW_AGENT_NAME] = dfs["llcs"][l.AGENT_NAME].copy()
        dfs["llcs"][l.RAW_AGENT_ADDRESS] = dfs["llcs"][l.AGENT_ADDRESS].copy()
        dfs["llcs"][l.RAW_AGENT_STREET] = dfs["llcs"][l.AGENT_STREET].copy()
        dfs["llcs"][l.RAW_AGENT_CITY] = dfs["llcs"][l.AGENT_CITY].copy()
        dfs["llcs"][l.RAW_AGENT_STATE] = dfs["llcs"][l.AGENT_STATE].copy()
        dfs["llcs"][l.RAW_AGENT_ZIP] = dfs["llcs"][l.AGENT_ZIP].copy()
        dfs["llcs"][l.RAW_OFFICE_ADDRESS] = dfs["llcs"][l.OFFICE_ADDRESS].copy()
        dfs["llcs"][l.RAW_OFFICE_STREET] = dfs["llcs"][l.OFFICE_STREET].copy()
        dfs["llcs"][l.RAW_OFFICE_CITY] = dfs["llcs"][l.OFFICE_CITY].copy()
        dfs["llcs"][l.RAW_OFFICE_STATE] = dfs["llcs"][l.OFFICE_STATE].copy()
        dfs["llcs"][l.RAW_OFFICE_ZIP] = dfs["llcs"][l.OFFICE_ZIP].copy()

        # ----------------------------------
        # ----BASIC CLEANING: ALL COLUMNS---
        # ----------------------------------
        console.print("Executing preliminary string cleaning on all columns...")
        for id, df in dfs.items():
            # todo: change clean_df_base functions to not return anything?
            console.print(f"Cleaning {id}...")
            df = clean_df_base.make_upper(df, self.exclude_raw_cols(df))
            df = clean_df_base.remove_symbols_punctuation(df, self.exclude_raw_cols(df))
            df = clean_df_base.trim_whitespace(df, self.exclude_raw_cols(df))
            df = clean_df_base.remove_extra_spaces(df, self.exclude_raw_cols(df))
            df = clean_df_base.words_to_num(df, self.exclude_raw_cols(df))
            df = clean_df_base.deduplicate(df, self.exclude_raw_cols(df))
            df = clean_df_base.convert_ordinals(df, self.exclude_raw_cols(df))
            df = clean_df_base.take_first(df, self.exclude_raw_cols(df))
            df = clean_df_base.combine_numbers(df, self.exclude_raw_cols(df))
            console.print(f"{id} preliminary cleaning complete.")

        # -----------------------------------------
        # ----BASIC CLEANING: NAME COLUMNS ONLY----
        # -----------------------------------------
        # NOTE - unlike the address columns, all of these must be present
        console.print("Executing string cleaning on all datasets (name columns only)...")
        for id, df in dfs.items():
            if id == "class_codes":  # no more cleaning necessary for class code descriptions
                continue
            console.print(f"Cleaning {id}...")
            dfs[id] = clean_df_name.switch_the(
                dfs[id],
                self.CLEANING_COL_MAP["name"][id],
            )
            console.print(f"{id} name field cleaning complete.")

        # --------------------------------------------
        # ----BASIC CLEANING: ADDRESS COLUMNS ONLY----
        # --------------------------------------------
        # NOTE - must take into account variation in available address columns
        # Ex: datasets may have single, unparsed address field, as opposed to having the fields broken out into street, city, state and zip (TAX_ADDRESS, PRESIDENT_ADDRESS, etc.)
        # Ex: datasets may have separate fields for zip codes, but not city or state
        console.print("Executing string cleaning on all datasets (address columns only)...")
        for id, df in dfs.items():
            if id == "class_codes":  # no more cleaning necessary for class code descriptions
                continue

            # clean street columns if they exist
            for col in self.CLEANING_COL_MAP["address"][id]["street"]:
                if col not in df.columns:
                    console.print(f"WARNING: column \"{col}\" not found in \"{id}\" dataframe.")
                    continue
                dfs[id] = clean_df_addr.convert_nsew(dfs[id], [col])
                dfs[id] = clean_df_addr.remove_secondary_designators(dfs[id], [col])
                dfs[id] = clean_df_addr.convert_street_suffixes(dfs[id], [col])

            # clean zip code columns if they exist
            for col in self.CLEANING_COL_MAP["address"][id]["zip"]:
                if col not in df.columns:
                    console.print(f"WARNING: column \"{col}\" not found in \"{id}\" dataframe.")
                    continue
                dfs[id] = clean_df_addr.fix_zip(dfs[id], [col])

        # ------------------------------------------------
        # ----OPTIONAL CLEANING: ACCURACY IMPLICATIONS----
        # ------------------------------------------------
        # can customize level of accuracy by including/excluding which string cleaning functions get called
        # todo: add user input here to determine whether these should be executed, add explanation to documentation
        if self.configs["accuracy"] == "low":
            # execute on address columns only
            for id, df in dfs.items():
                dfs[id] = clean_df_acc.remove_secondary_component(
                    dfs[id],
                    self.CLEANING_COL_MAP["accuracy"]["address"]
                )
                dfs[id] = clean_df_acc.convert_mixed(
                    dfs[id],
                    self.CLEANING_COL_MAP["accuracy"]["address"]
                )
                dfs[id] = clean_df_acc.drop_letters(
                    dfs[id],
                    self.CLEANING_COL_MAP["accuracy"]["address"]
                )

        # ----------------------------
        # ----DATAFRAME OPERATIONS----
        # ----------------------------
        # get clean name+address concatenation
        dfs["props_taxpayers"][p.CLEAN_NAME_ADDRESS] = dfs["props_taxpayers"].apply(
            lambda row: row[pt.TAX_NAME] + " -- " + row[pt.TAX_ADDRESS],
        )
        # separate out properties dataset from props_taxpayers, handling for cases in which NUM_UNITS is missing
        properties_cols: list[str] = [
            p.PIN,
            p.RAW_NAME_ADDRESS,
            p.CLEAN_NAME_ADDRESS,
            p.CLASS_CODE,
        ]
        if p.NUM_UNITS in dfs["properties"].columns:
            properties_cols.append(p.NUM_UNITS)
        # properties dataframe to be saved to ROOT/processed/properties[FileExt]
        df_props: pd.DataFrame = dfs["props_taxpayers"][properties_cols].copy()

        # separate out taxpayer records
        taxpayers_cols : list[str] = [
            # raw columns
            tr.RAW_NAME_ADDRESS,
            tr.RAW_NAME,
            tr.RAW_ADDRESS,
            tr.RAW_STREET,
            tr.RAW_CITY,
            tr.RAW_STATE,
            tr.RAW_ZIP,
            # cleaned columns
            tr.CLEAN_NAME_ADDRESS,
            pt.TAX_NAME,
            pt.TAX_ADDRESS,
            pt.TAX_STREET,
            pt.TAX_CITY,
            pt.TAX_STATE,
            pt.TAX_ZIP,
        ]
        df_taxpayers: pd.DataFrame = dfs["props_taxpayers"][taxpayers_cols].copy()
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

        unvalidated_addrs_cols: list[str] = [
            tr.RAW_ADDRESS,
            tr.RAW_STREET,
            tr.RAW_CITY,
            tr.RAW_STATE,
            tr.RAW_ZIP,
            tr.CLEAN_ADDRESS,
            tr.CLEAN_STREET,
            tr.CLEAN_CITY,
            tr.CLEAN_STATE,
            tr.CLEAN_ZIP,
        ]
        df_unvalidated: pd.DataFrame = df_taxpayers[unvalidated_addrs_cols].copy()
        df_unvalidated.drop_duplicates(subset=[tr.RAW_NAME_ADDRESS], inplace=True)

        return {
            "properties": df_props,
            "taxpayer_records": df_taxpayers,
            "corps": dfs["corps"],
            "llcs": dfs["llcs"],
            "class_codes": dfs["class_codes"],
            "unvalidated_addrs": df_unvalidated,
        }

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
