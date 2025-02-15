from enum import IntEnum
from typing import ClassVar, Optional
from abc import ABC, abstractmethod

import pandas as pd

from opndb.constants.base import FileNames, DATA_ROOT
from opndb.df_ops import DataFrameOpsBase as df_ops, DataFrameBaseCleaners as clean_base, DataFrameNameCleaners as clean_name, DataFrameAddressCleaners as clean_addr, DataFrameCleanersAccuracy as clean_acc
from opndb.types.base import WorkflowConfigs, CleaningColumnMap
from opndb.utils import UtilsBase as utils
from opndb.workflows.addresses import WkflAddressBase


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


class WorkflowBase:
    """
    Base workflow class the controls execution of data processing tasks required for each stage of the opndb workflow.
    Each child class that inherits from WorkflowBase corresponds to the broader workflow stage.

    """
    dirs = FileNames.DataDirs

    def __init__(self, configs: WorkflowConfigs):
        self.configs = configs

    @classmethod
    def create_workflow(cls, configs: WorkflowConfigs) -> Optional['WorkflowBase']:
        """Instantiates workflow object based on last saved progress (config['wkfl_stage'])."""
        if configs["wkfl_type"] == "preliminary":
            return WkflPreliminary(configs)
        elif configs["wkfl_type"] == "data_load":
            return WkflDataLoad()
        elif configs["wkfl_type"] == "data_clean":
            return WkflDataClean()
        elif configs["wkfl_type"] == "address_validation":
            return WkflAddressValidation()
        elif configs["wkfl_type"] == "name_analysis":
            return WkflNameAnalysis()
        elif configs["wkfl_type"] == "address_analysis":
            return WkflAddressAnalysis()
        elif configs["wkfl_type"] == "rental_subset":
            return WkflRentalSubset()
        elif configs["wkfl_type"] == "clean_merge":
            return WkflCleanMerge()
        elif configs["wkfl_type"] == "string_match":
            return WkflStringMatch()
        elif configs["wkfl_type"] == "network_graph":
            return WkflNetworkGraph()
        elif configs["wkfl_type"] == "final_output":
            return WkflFinalOutput()
        return None

    @classmethod
    def load_configs(cls) -> WorkflowConfigs:
        # return utils.load_configs(DATA_ROOT / configs.json)
        pass

    @abstractmethod
    def execute(self) -> None:
        """Each workflow must implement an execute method"""
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
    raw = FileNames.Raw
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        # raw datasets converted to pandas dataframes
        # error is thrown if required columns are not present
        # if all required columns are present, validators are run on the dataframes
        # if there are validation errors, detailed error reports are shown
        # if there are no validation errors, files are saved into the "raw" directory
        # workflow should be for each individual dataset only and run validator and save only for single dataframe
        # def validate_load(self, raw_data, dataset_name):
        #     df: pd.DataFrame = pd.DataFrame(raw_data)
            # df, message = validate(df)
            # if message:
            # print failure message
            # store validation error logs to separate file, save & timestamp
            # else:
            # print success message
            # df_ops.save_df_csv(df, utils.generate_path(self.dirs.RAW, dataset_name, self.stage))
        # set summary stats
        # update configuration file
        # set stage
        pass


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
    raw = FileNames.Raw  # constant names
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)
        self.dfs: dict[str,[pd.DataFrame]] = {
            self.raw.TAXPAYER_RECORDS_RAW: df_ops.load_df_csv(
                utils.generate_path(
                    self.dirs.RAW,
                    self.raw.get_raw_filename_ext(self.raw.TAXPAYER_RECORDS_RAW, self.configs),
                    self.configs["prev_stage"],
                ), str
            ),
            "corps": df_ops.load_df_csv(
                utils.generate_path(
                    self.dirs.RAW,
                    self.raw.get_raw_filename_ext(self.raw.CORPS_RAW, self.configs),
                    self.configs["prev_stage"],
                ), str
            ),
            "llcs": df_ops.load_df_csv(
                utils.generate_path(
                    self.dirs.RAW,
                    self.raw.get_raw_filename_ext(self.raw.LLCS_RAW, self.configs),
                    self.configs["prev_stage"],
                ), str
            ),
            "class_code_descriptions": df_ops.load_df_csv(
                utils.generate_path(
                    self.dirs.RAW,
                    self.raw.get_raw_filename_ext(self.raw.CLASS_CODE_DESCRIPTIONS, self.configs),
                    self.configs["prev_stage"],
                ), str
            )
        }
        self.cleaning_column_map: CleaningColumnMap = {
            "name" : {
                "taxpayer_record": [
                    # string column names from raw data to have name cleaners run on them
                    # USE CONSTANTS
                ],
                "corps": [],
                "llcs": [],
            },
            "address": {
                "taxpayer_record": [
                    # string column names from raw data to have name cleaners run on them
                    # USE CONSTANTS
                ],
                "corps": [],
                "llcs": [],
            },
            "accuracy": {
                "name": {
                    "taxpayer_record": [
                        # string column names from raw data to have name cleaners run on them
                        # USE CONSTANTS
                    ],
                    "corps": [],
                    "llcs": [],
                },
                "address": {
                    "taxpayer_record": [
                        # string column names from raw data to have name cleaners run on them
                        # USE CONSTANTS
                    ],
                    "corps": [],
                    "llcs": [],
                }
            }
        }

    def execute(self):

        # todo: add boolean column for is_rental for

        # execute on all dataframes and columns
        for df in self.dfs:
            df = clean_base.make_upper(df)
            df = clean_base.remove_symbols_punctuation(df)
            df = clean_base.trim_whitespace(df)
            df = clean_base.remove_extra_spaces(df)
            df = clean_base.words_to_num(df)
            df = clean_base.deduplicate(df)
            df = clean_base.convert_ordinals(df)
            df = clean_base.take_first(df)
            df = clean_base.combine_numbers(df)

        # execute on name columns only
        for id, df in self.dfs.items():
            # leave out additional class code cleaning for now
            # will have to be updated as class code format varies by municipality
            if id == "class_code_descriptions":
                continue
            self.dfs[id] = clean_name.switch_the(
                self.dfs[id],
                self.cleaning_column_map["name"][id]
            )

        # execute on the address columns only
        for id, df in self.dfs.items():
            # leave out additional class code cleaning for now
            # will have to be updated as class code format varies by municipality
            if id == "class_code_descriptions":
                continue
            self.dfs[id] = clean_addr.convert_nsew(
                self.dfs[id],
                self.cleaning_column_map["address"][id]
            )
            self.dfs[id] = clean_addr.remove_secondary_designators(
                self.dfs[id],
                self.cleaning_column_map["address"][id]
            )
            self.dfs[id] = clean_addr.convert_street_suffixes(
                self.dfs[id],
                self.cleaning_column_map["address"][id]
            )
            self.dfs[id] = clean_addr.fix_zip(
                self.dfs[id],
                self.cleaning_column_map["address"][id]
            )

        # execute optional cleaners that reduce accuracy
        # can customize level of accuracy by including/excluding which string cleaning functions get called
        if self.configs["accuracy"] == "low":
            # execute on address columns only
            for id, df in self.dfs.items():
                self.dfs[id] = clean_acc.remove_secondary_component(
                    self.dfs[id],
                    self.cleaning_column_map["accuracy"]["address"]
                )
                self.dfs[id] = clean_acc.convert_mixed(
                    self.dfs[id],
                    self.cleaning_column_map["accuracy"]["address"]
                )
                self.dfs[id] = clean_acc.drop_letters(
                    self.dfs[id],
                    self.cleaning_column_map["accuracy"]["address"]
                )

        # run validators
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
        # set stage



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

    def execute(self):
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

    def execute(self):
        # set summary stats
        # update configuration file
        # set stage

        pass

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

    def execute(self):
        # set summary stats
        # update configuration file
        # set stage
        pass

class WkflCleanMerge(WorkflowBase):
    """
    Additional data cleaning & merging

    INPUTS:
        - Cleaned taxpayer, corporate and LLC data
            - 'ROOT/processed/taxpayer_records[FileExt]'
            - 'ROOT/processed/corps[FileExt]'
            - 'ROOT/processed/llcs[FileExt]'
    OUTPUTS:
        - Single property dataset containing all columns required for string matching & network graph generation
            - 'ROOT/processed/props_prepped[FileExt]'
    """
    stage = WorkflowStage.CLEAN_MERGE
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        # fixing taxpayer names based on manual research
        # merging corporate data & validated addresses into original property dataset (or creating new one entirely?)
        # adding boolean columns necessary for string matching & network graph generation
        # create new df with ONLY the columns required to run string matching - avoids large unwieldy datasets like in original chicago code
        # outputs & saves merged dataset to "processed" directory
        # set summary stats
        # update configuration file
        # set stage
        pass


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
        # set stage
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
        # set stage
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
        # set stage
        pass
