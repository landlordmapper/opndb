from enum import IntEnum
from typing import ClassVar


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


class WkflPreliminary(WorkflowBase):
    """Preliminary workflow stage. Merge class codes to taxpayer records, change colnames, etc."""
    stage = WorkflowStage.PRE

class WkflDataLoading(WorkflowBase):
    """Initial data load"""
    # raw datasets converted to pandas dataframes
    # error is thrown if required columns are not present
    # if all required columns are present, validators are run on the dataframes
    # if there are validation errors, detailed error reports are shown
    # if there are no validation errors, files are saved into the "raw" directory
    stage = WorkflowStage.DATA_LOAD


class WkflDataCleaning(WorkflowBase):
    """Initial data cleaning"""
    # basic string cleaning operations performed on name and address columns for taxpayers and corps/llcs
    # trimming whitespace
    # removing symbols
    # subset corps for active only (?), drop dup corps, etc
    # cleaned files are saved to the "processed" directory
    stage = WorkflowStage.DATA_CLEANING


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


class WkflNameAnalysis(WorkflowBase):
    """Taxpayer name analysis"""
    stage = WorkflowStage.NAME_ANALYSIS


class WkflAddressAnalysis(WorkflowBase):
    """Address analysis"""
    stage = WorkflowStage.ADDRESS_ANALYSIS


class WkflRentalSubset(WorkflowBase):
    """Subset rental properties based on class code descriptions."""
    stage = WorkflowStage.RENTAL_SUBSET


class WkflCleaningMerging(WorkflowBase):
    """Additional data cleaning & merging"""
    # fixing taxpayer names based on manual research
    # merging corporate data & validated addresses into original property dataset (or creating new one entirely?)
    # adding boolean columns necessary for string matching & network graph generation
    # create new df with ONLY the columns required to run string matching - avoids large unwieldy datasets like in original chicago code
    # outputs & saves merged dataset to "processed" directory
    stage = WorkflowStage.CLEAN_MERGE



class WkflStringMatching(WorkflowBase):
    """String matching taxpayer records"""
    # prompts user to create parameter matrix for string matching
    # outputs & saves string matched dataset to "processed" directory
    stage = WorkflowStage.STRING_MATCH


class WkflNetworkGraph(WorkflowBase):
    """Network graph generation"""
    # prompts user to create parameter matrix for network graph generation
    # outputs & saves networked properties datasets to "processed" directory
    stage = WorkflowStage.NETWORK_GRAPH


class WkflFinalOutput(WorkflowBase):
    """Produces final datasets to be converted into standardized format"""
    # saves final outputs to "final_outputs" directory
    stage = WorkflowStage.FINAL_OUTPUT
