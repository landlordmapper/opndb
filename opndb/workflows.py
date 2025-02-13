from enum import IntEnum
from typing import ClassVar


import pandas as pd

from opndb.constants.base import FileNames, DATA_ROOT
from opndb.df_ops import DataFrameOpsBase as df_ops
from opndb.utils import UtilsBase as utils


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
    stage: ClassVar[WorkflowStage]
    dirs = FileNames.DataDirs

class WkflPreliminary(WorkflowBase):
    """Preliminary workflow stage. Merge class codes to taxpayer records, change colnames, etc."""
    stage = WorkflowStage.PRE

class WkflDataLoading(WorkflowBase):
    """Initial data load"""
    stage = WorkflowStage.DATA_LOAD
    raw = FileNames.Raw
    def __init__(self):
        super().__init__()

    # raw datasets converted to pandas dataframes
    # error is thrown if required columns are not present
    # if all required columns are present, validators are run on the dataframes
    # if there are validation errors, detailed error reports are shown
    # if there are no validation errors, files are saved into the "raw" directory

    # workflow should be for each individual dataset only and run validator and save only for single dataframe

    def validate_load(self, raw_data, dataset_name):
        df: pd.DataFrame = pd.DataFrame(raw_data)
        # df, message = validate(df)
        # if message:
            # print failure message
            # store validation error logs to separate file, save & timestamp
        # else:
            # print success message
            # df_ops.save_df_csv(df, utils.generate_path(self.dirs.RAW, dataset_name, self.stage))


class WkflDataCleaning(WorkflowBase):
    """Initial data cleaning"""
    prev_stage = WorkflowStage.DATA_LOAD
    stage = WorkflowStage.DATA_CLEANING
    raw = FileNames.Raw
    def __init__(self):
        super().__init__()
        self.df_taxpayer_records: pd.DataFrame = df_ops.load_df_csv(
            utils.generate_path(
                self.dirs.RAW,
                self.raw.TAXPAYER_RECORDS_RAW,
                self.prev_stage,
            ),
            "str"
        )
        self.df_corps: pd.DataFrame = df_ops.load_df_csv(
            utils.generate_path(
                self.dirs.RAW,
                self.raw.CORPS_RAW,
                self.prev_stage,
            ),
            "str"
        )
        self.df_llcs: pd.DataFrame = df_ops.load_df_csv(
            utils.generate_path(
                self.dirs.RAW,
                self.raw.LLCS_RAW,
                self.prev_stage,
            ),
            "str"
        )
        self.df_class_code_descriptions: pd.DataFrame = df_ops.load_df_csv(
            utils.generate_path(
                self.dirs.RAW,
                self.raw.CLASS_CODE_DESCRIPTIONS,
                self.prev_stage,
            ),
            "str"
        )

    # basic string cleaning operations performed on name and address columns for taxpayers and corps/llcs
    # trimming whitespace
    # removing symbols
    # subset corps for active only (?), drop dup corps, etc
    # cleaned files are saved to the "processed" directory

    def clean_taxpayer_records(self):
        pass

    def clean_corps(self):
        pass

    def clean_llcs(self):
        pass

    def clean_class_code_descriptions(self):
        pass


class WkflAddressValidation(WorkflowBase):
    """Address validation"""
    # fetch list of all unique addresses from taxpayer and corp/llc datasets
    # pull out, standardize & save all PO box addresses, save to master validated address file
    # run remaining addresses through open addresses validator
    # save open address-validated addresses to master file
    # for all addresses that were NOT validated, send them through geocodio
    # prompt user for their geocodio API key, confirm the number of calls, include estimated cost, ask for their permission to continue
    # run remaining addresses through geocodio, filter, parse & string match, save validated addresses in master file
    # all remaining, unvalidated addresses get saved in master unvalidated address file
    stage = WorkflowStage.ADDRESS_VALIDATION
    def __init__(self):
        super().__init__()

class WkflNameAnalysis(WorkflowBase):
    """Taxpayer name analysis"""
    stage = WorkflowStage.NAME_ANALYSIS
    def __init__(self):
        super().__init__()

class WkflAddressAnalysis(WorkflowBase):
    """Address analysis"""
    stage = WorkflowStage.ADDRESS_ANALYSIS
    def __init__(self):
        super().__init__()

class WkflRentalSubset(WorkflowBase):
    """Subset rental properties based on class code descriptions."""
    stage = WorkflowStage.RENTAL_SUBSET
    def __init__(self):
        super().__init__()

class WkflCleaningMerging(WorkflowBase):
    """Additional data cleaning & merging"""
    # fixing taxpayer names based on manual research
    # merging corporate data & validated addresses into original property dataset (or creating new one entirely?)
    # adding boolean columns necessary for string matching & network graph generation
    # create new df with ONLY the columns required to run string matching - avoids large unwieldy datasets like in original chicago code
    # outputs & saves merged dataset to "processed" directory
    stage = WorkflowStage.CLEAN_MERGE
    def __init__(self):
        super().__init__()


class WkflStringMatching(WorkflowBase):
    """String matching taxpayer records"""
    # prompts user to create parameter matrix for string matching
    # outputs & saves string matched dataset to "processed" directory
    stage = WorkflowStage.STRING_MATCH
    def __init__(self):
        super().__init__()

class WkflNetworkGraph(WorkflowBase):
    """Network graph generation"""
    # prompts user to create parameter matrix for network graph generation
    # outputs & saves networked properties datasets to "processed" directory
    stage = WorkflowStage.NETWORK_GRAPH


class WkflFinalOutput(WorkflowBase):
    """Produces final datasets to be converted into standardized format"""
    # saves final outputs to "final_outputs" directory
    stage = WorkflowStage.FINAL_OUTPUT
    def __init__(self):
        super().__init__()
