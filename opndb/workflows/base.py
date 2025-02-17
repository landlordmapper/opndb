import shutil
from asyncio.subprocess import Process
from enum import IntEnum
from pathlib import Path
from typing import ClassVar, Optional
from abc import abstractmethod, ABC

import pandas as pd

from opndb.constants.columns import ValidatedAddrs as va, TaxpayerRecords as tr, Corps as c, LLCs as l
from opndb.constants.files import Raw as r, Dirs as d, Processed as p, Analysis as a
from opndb.services.dataframe import DataFrameOpsBase as ops_df, DataFrameBaseCleaners as clean_df_base, \
    DataFrameNameCleaners as clean_df_name, DataFrameAddressCleaners as clean_df_addr, DataFrameCleanersAccuracy as clean_df_acc, \
    DataFrameMergers as merge_df, DataFrameSubsetters as subset_df, DataFrameColumnGenerators as cols_df
from opndb.types.base import WorkflowConfigs, CleaningColumnMap, BooleanColumnMap
from opndb.utils import UtilsBase as utils
from opndb.workflows.address import WkflAddressBase


class WorkflowStage(IntEnum):
    """Keeps track of workflow stages."""
    PRE = 0
    DATA_LOAD = 1
    DATA_CLEANING = 2
    ADDRESS_VALIDATION = 3
    NAME_ANALYSIS = 4
    ADDRESS_ANALYSIS = 5
    RENTAL_SUBSET = 6
    CLEAN_MERGE = 7
    STRING_MATCH = 8
    NETWORK_GRAPH = 9
    FINAL_OUTPUT = 10


class WorkflowBase(ABC):
    """
    Base workflow class the controls execution of data processing tasks required for each stage of the opndb workflow.
    Each child class that inherits from WorkflowBase corresponds to the broader workflow stage.
    """
    def __init__(self, configs: WorkflowConfigs):
        self.configs: WorkflowConfigs = configs

    @classmethod
    def create_workflow(cls, configs: WorkflowConfigs) -> Optional['WorkflowBase']:
        """Instantiates workflow object based on last saved progress (config['wkfl_stage'])."""
        if configs["wkfl_type"] == "preliminary":
            return WkflPreliminary(configs)
        elif configs["wkfl_type"] == "data_load":
            return WkflDataLoad(configs)
        elif configs["wkfl_type"] == "data_clean":
            return WkflDataClean(configs)
        elif configs["wkfl_type"] == "address_validation":
            return WkflAddressValidation(configs)
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

    @classmethod
    def load_configs(cls) -> WorkflowConfigs:
        # return utils.load_configs(DATA_ROOT / configs.json)
        pass

    @abstractmethod
    def execute(self) -> None:
        pass


class WkflPreliminary(WorkflowBase):
    """Preliminary workflow stage. Merge class codes to taxpayer records, change colnames, etc."""
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        # set summary stats
        # update configuration file
        # set stage in config
        pass


class WkflDataLoad(WorkflowBase):
    """
    Initial data load

    INPUTS:
        - User inputs file path to their local raw data dir
    OUTPUTS:
        - Raw taxpayer, corporate and LLC data files
            - 'ROOT/raw/taxpayer_records[FileExt]'
            - 'ROOT/raw/corps[FileExt]'
            - 'ROOT/raw/llcs[FileExt]'
            - 'ROOT/raw/class_code_descriptions[FileExt]'
    """
    stage = WorkflowStage.DATA_LOAD
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        # prompt user for path to raw data
        origin_dir = ""
        origin = Path(origin_dir)
        # if user is running locally, ask where to generate new folders
        destination_dir = ""
        destination = Path(destination_dir)
        # set root path in configs
        self.configs["root"] = destination
        # copy raw data files to destination directory
        file_names = [
            r.TAXPAYER_RECORDS_RAW,
            r.CORPS_RAW,
            r.LLCS_RAW,
            r.CLASS_CODES_RAW
        ]
        for file_name in file_names:
            source_file = origin / file_name
            dest_file = destination / file_name
            try:
                if not source_file.exists():
                    # logging.warning(f"Source file not found: {source_file}")
                    continue
                shutil.copy2(source_file, dest_file)
                # logging.info(f"Successfully copied {file_name}")
            except PermissionError:
                # logging.error(f"Permission denied when copying {file_name}")
                raise
            except Exception as e:
                # logging.error(f"Error copying {file_name}: {str(e)}")
                raise
        # set summary stats
        # update configuration file


class WkflDataClean(WorkflowBase):
    """
    Initial data cleaning.

    INPUTS:
        - Raw taxpayer, corporate and LLC data
            - 'ROOT/raw/taxpayer_records[FileExt]'
            - 'ROOT/raw/corps[FileExt]'
            - 'ROOT/raw/llcs[FileExt]'
            - 'ROOT/raw/class_code_descriptions[FileExt]'
    OUTPUTS:
        - Cleaned taxpayer, corporate and LLC data
            - 'ROOT/processed/taxpayer_records[FileExt]'
            - 'ROOT/processed/corps[FileExt]'
            - 'ROOT/processed/llcs[FileExt]'
            - 'ROOT/processed/class_code_descriptions[FileExt]'
    """
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)
        self.dfs: dict[str, pd.DataFrame] = {
            "taxpayer_records": ops_df.load_df(
                utils.generate_path(
                    d.RAW,
                    r.get_raw_filename_ext(r.TAXPAYER_RECORDS_RAW, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
            "corps": ops_df.load_df(
                utils.generate_path(
                    d.RAW,
                    r.get_raw_filename_ext(r.CORPS_RAW, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
            "llcs": ops_df.load_df(
                utils.generate_path(
                    d.RAW,
                    r.get_raw_filename_ext(r.LLCS_RAW, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
            "class_codes": ops_df.load_df(
                utils.generate_path(
                    d.RAW,
                    r.get_raw_filename_ext(r.CLASS_CODES_RAW, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            )
        }
        self.cleaning_column_map: CleaningColumnMap = {
            "name" : {
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

    def execute(self):

        # todo: add validator that checks for required columns, throw error/failure immediately if not

        dfs = {
            key: df.copy()
            for key, df in self.dfs.items()
        }

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
                self.cleaning_column_map["name"][id]
            )

        # execute on the address columns only
        for id, df in self.dfs.items():
            # leave out additional class code cleaning for now
            # will have to be updated as class code format varies by municipality
            if id == "class_code_descriptions":
                continue
            dfs[id] = clean_df_addr.convert_nsew(
                dfs[id],
                self.cleaning_column_map["address"][id]
            )
            dfs[id] = clean_df_addr.remove_secondary_designators(
                dfs[id],
                self.cleaning_column_map["address"][id]
            )
            dfs[id] = clean_df_addr.convert_street_suffixes(
                dfs[id],
                self.cleaning_column_map["address"][id]
            )
            dfs[id] = clean_df_addr.fix_zip(
                dfs[id],
                self.cleaning_column_map["address"][id]
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
        # set summary stats
        # update configuration file
        # return?


class WkflAddressValidation(WorkflowBase):
    """
    Address validation

    INPUTS:
        - Cleaned taxpayer, corporate and LLC data
            - 'ROOT/raw/taxpayer_records[FileExt]'
            - 'ROOT/raw/corps[FileExt]'
            - 'ROOT/raw/llcs[FileExt]'
    OUTPUTS:
        - Master dataset for validated and unvalidated addresses
            - 'ROOT/processed/validated_addrs[FileExt]'
            - 'ROOT/processed/unvalidated_addrs[FileExt]'
        - Geocodio master files for validated and unvalidated addresses
            - 'ROOT/geocodio/gcd_validated[FileExt]'
            - 'ROOT/geocodio/gcd_unvalidated[FileExt]'
            - 'ROOT/geocodio/gcd_failed[FileExt]'

        - Geocodio partial files for all API call results, in 'ROOT/geocodio/partials'
    """
    # todo: specify to use clean_address to associate with validated addr object
    stage = WorkflowStage.ADDRESS_VALIDATION
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        while True:
            wkfl = WkflAddressBase.create_workflow(self.configs)
            # fetch list of all unique addresses from taxpayer and corp/llc datasets
            # pull out, standardize & save all PO box addresses, save to master validated address file
            # run remaining addresses through open addresses validator
            # save open address-validated addresses to master file
            # for all addresses that were NOT validated, send them through geocodio
            # prompt user for their geocodio API key, confirm the number of calls, include estimated cost, ask for their permission to continue
            # run remaining addresses through geocodio, filter, parse & string match, save validated addresses in master file
            # all remaining, unvalidated addresses get saved in master unvalidated address file
            # set summary stats
            # update configuration file


class WkflNameAnalysis(WorkflowBase):
    """
    Taxpayer name analysis

    INPUTS:
        - Clean taxpayer, corporate and LLC data
            - 'ROOT/processed/taxpayer_records[FileExt]'
            - 'ROOT/processed/corps[FileExt]'
            - 'ROOT/processed/llcs[FileExt]'
        - User inputs manual name research & standardizer data
    OUTPUTS:
        - is_bank, is_person, is_org boolean columns for name columns in clean taxpayer, corporate and LLC data
            - 'ROOT/processed/taxpayer_records[FileExt]'
            - 'ROOT/processed/corps[FileExt]'
            - 'ROOT/processed/llcs[FileExt]'
        - Name frequency dataset
            - 'ROOT/analysis/name_freq.ext'
        - Spelling standardizer dataset for banks and trusts
            - 'ROOT/analysis/bank_names[FileExt]'
            - 'ROOT/analysis/trust_names[FileExt]'
    """
    stage = WorkflowStage.NAME_ANALYSIS
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)
        self.dfs: dict[str, pd.DataFrame] = {
            "taxpayer_records": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.TAXPAYER_RECORDS_CLEAN, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
            p.CORPS_CLEAN: ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.CORPS_CLEAN, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
            p.LLCS_CLEAN: ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.LLCS_CLEAN, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            )
        }

    def execute(self):
        # create & save name frequency dataset
        # add boolean columns
        # set summary stats
        # update configuration file
        # set stage
        pass


class WkflAddressAnalysis(WorkflowBase):
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
    stage = WorkflowStage.ADDRESS_ANALYSIS
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)
        self.dfs: dict[str, pd.DataFrame] = {
            "validated_addrs": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.VALIDATED_ADDRS, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            )
        }

    def execute(self):
        # create & save dataframe with unique validated addresses & their count
        df_addr_counts: pd.DataFrame = ops_df.get_frequency_df(
            self.dfs["validated_addrs"],
            va.FORMATTED_ADDRESS
        )
        ops_df.save_df(df_addr_counts, utils.generate_path(
            d.ANALYSIS,
            a.ADDRESS_ANALYSIS,
            self.configs["stage"],
            self.configs["load_ext"]
        ))
        # set summary stats
        # update configuration file


class WkflRentalSubset(WorkflowBase):
    """
    Subset rental properties based on class code descriptions.

    INPUT:
        - Building class codes dataset
            - 'ROOT/processed/bldg_class_codes[FileExt]'
        - Taxpayer record data
            - 'ROOT/processed/taxpayer_records[FileExt]'
    OUTPUT:
        - Rental-subsetted taxpayer dataset
            - 'ROOT/processed/props_subsetted[FileExt]'
    """
    stage = WorkflowStage.RENTAL_SUBSET
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)
        self.dfs: dict[str, pd.DataFrame] = {
            "taxpayer_records": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.TAXPAYER_RECORDS_CLEAN, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
            "validated_addrs": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.VALIDATED_ADDRS, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
            "class_codes_clean": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.CLASS_CODES_CLEAN, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            )
        }

    def execute(self):
        # todo: add merge code for class codes
        dfs = {
            key: df.copy()
            for key, df in self.dfs.items()
        }
        # merge validated_addrs
        df_props_addrs: pd.DataFrame = merge_df.merge_validated_addrs(
            dfs["taxpayer_records"],
            dfs["validated_addrs"],
        )
        # add is_rental column
        df_props_rental_col: pd.DataFrame = cols_df.set_is_rental(
            df_props_addrs,
            dfs["class_codes"],
        )
        # execute initial subset based on is_rental
        df_rentals_initial: pd.DataFrame = df_props_rental_col[df_props_rental_col[tr.IS_RENTAL] == True]
        # fetch properties left out of initial subset with matching validated taxpayer addresses
        df_rentals_addrs: pd.DataFrame = subset_df.get_nonrentals_from_addrs(df_props_addrs, df_rentals_initial)
        # pull non-rentals with matching rental taxpayer addresses
        df_rentals_final: pd.DataFrame = pd.concat([df_rentals_initial, df_rentals_addrs], axis=1)
        # save
        ops_df.save_df(df_rentals_final, utils.generate_path(
            d.PROCESSED,
            p.PROPS_SUBSETTED,
            self.configs["stage"],
            self.configs["load_ext"],
        ))
        # set summary stats
        # update configuration file


class WkflCleanMerge(WorkflowBase):
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
    stage = WorkflowStage.CLEAN_MERGE
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)
        self.dfs: dict[str, pd.DataFrame] = {
            "taxpayer_records": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.PROPS_SUBSETTED, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
            "corps": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.CORPS_CLEAN, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
            "llcs": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.LLCS_CLEAN, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
            "validated_addrs": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.VALIDATED_ADDRS, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
        }
        self.bool_col_map: BooleanColumnMap = {
            "taxpayer_records": [tr.CLEAN_NAME],
            "corps": [c.PRESIDENT_NAME, c.SECRETARY_NAME],
            "llcs": [l.MANAGER_MEMBER_NAME, l.AGENT_NAME],
        }

    def execute(self):
        dfs = {
            key: df.copy()
            for key, df in self.dfs.items()
        }
        # fixing taxpayer addresses based on manual research - services/address.py
        # fixing taxpayer names based on manual research - services/base.py

        # add "core_name" columns
        dfs["taxpayer_records"]["core_name"] = cols_df.set_core_name(dfs["taxpayer_records"], tr.CLEAN_NAME)
        dfs["corps"]["core_name"] = cols_df.set_core_name(dfs["corps"], c.CLEAN_NAME)
        dfs["llcs"]["core_name"] = cols_df.set_core_name(dfs["llcs"], l.CLEAN_NAME)

        # add bool columns for is_common_name, is_org, is_llc, is_person, is_bank
        for id, cols in self.bool_col_map.items():
            for col in cols:
                dfs[id]["is_bank"] = cols_df.set_is_bank(dfs[id], col)
                dfs[id]["is_person"] = cols_df.set_is_person(dfs[id], col)
                dfs[id]["is_common_name"] = cols_df.set_is_common_name(dfs[id], col)
                dfs[id]["is_org"] = cols_df.set_is_org(dfs[id], col)
                dfs[id]["is_llc"] = cols_df.set_is_llc(dfs[id], col)

        # subset active corps/llcs
        dfs["corps"] = subset_df.get_active(dfs["corps"])
        dfs["llcs"] = subset_df.get_active(dfs["llcs"])

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
        df["taxpayer_records"] = merge_df.merge_orgs(df["taxpayer_records"], df["corps"])
        df["taxpayer_records"] = merge_df.merge_orgs(df["taxpayer_records"], df["corps"])

        # set summary stats
        # update configuration file


class WkflStringMatch(WorkflowBase):
    """
    String matching taxpayer records

    INPUTS:
        - Cleaned and merged property dataset with taxpayer records, corporations and LLCs
            - 'ROOT/processed/props_prepped[FileExt]'
        - User inputs matrix params
    OUTPUTS:
        - Inputted dataset with string matching result columns
    """
    stage = WorkflowStage.STRING_MATCH
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)


    def execute(self):
        # prompts user to create parameter matrix for string matching
        # outputs & saves string matched dataset to "processed" directory
        # set summary stats
        # update configuration file
        pass


class WkflNetworkGraph(WorkflowBase):
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

    def execute(self):
        # prompts user to create parameter matrix for network graph generation
        # outputs & saves networked properties datasets to "processed" directory
        # set summary stats
        # update configuration file
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
