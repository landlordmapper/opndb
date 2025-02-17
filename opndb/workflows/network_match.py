from abc import ABC, abstractmethod
from typing import Optional
import networkx as nx
import pandas as pd

from opndb.constants.files import Processed, Dirs, Analysis
from opndb.services.match import NetworkMatching
from opndb.types.base import WorkflowConfigs, NetworkMatchParams
from opndb.services.dataframe import DataFrameOpsBase as df_ops
from opndb.utils import UtilsBase as utils


# todo: pull out dataframe operation code and store in dataframe service

class WkflNetworkBase(ABC):

    """Base class for network graph generation workflow."""

    def __init__(self, configs: WorkflowConfigs):
        self.configs: WorkflowConfigs = configs
        self.params_matrix: list[NetworkMatchParams] | None = None
        self.dfs: dict[str, pd.DataFrame] = {
            Processed.PROPS_STRING_MATCHED: df_ops.load_df(
                utils.generate_path(
                    Dirs.PROCESSED,
                    Processed.get_raw_filename_ext(Processed.PROPS_STRING_MATCHED, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            ),
            Analysis.ADDRESS_ANALYSIS: df_ops.load_df(
                utils.generate_path(
                    Dirs.ANALYSIS,
                    Analysis.get_raw_filename_ext(Analysis.ADDRESS_ANALYSIS, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"],
                ), str
            )
        }
        self.df_input: pd.DataFrame | None = None
        self.df_process: pd.DataFrame | None = None
        self.df_output: pd.DataFrame | None = None

    @classmethod
    def create_workflow(cls, configs: WorkflowConfigs) -> Optional['WkflNetworkBase']:
        if configs["wkfl_type_ntwk"] == "params":
            return WkflNetworkParams(configs)
        elif configs["wkfl_type_ntwk"] == "prep":
            return WkflNetworkPrep(configs)
        elif configs["wkfl_type_ntwk"] == "process":
            return WkflNetworkProcess(configs)
        elif configs["wkfl_type_ntwk"] == "merge":
            return WkflNetworkMerge(configs)
        return None

    @abstractmethod
    def execute(self) -> None:
        pass


class WkflNetworkParams(WkflNetworkBase):
    """Workflow to set the parameters for network graph generation."""
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
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


class WkflNetworkPrep(WkflNetworkBase):
    """Dataframe preparation workflow. Generates df_input and stores in instance variable"""
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        df_input: pd.DataFrame = self.dfs[Processed.PROPS_STRING_MATCHED].copy()
        df_input.drop_duplicates(subset=["name_address_clean"], inplace=True)
        self.df_input = df_input


class WkflNetworkProcess(WkflNetworkBase):
    """Workflow to execute the network graph generation process."""
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):

        # raise exceptions is parameter matrix or input dataframe are empty
        if self.params_matrix is None:
            raise Exception("Parameter matrix is empty.")
        if self.df_process is None:
            raise Exception("Input dataframe is empty.")

        df_process: pd.DataFrame = self.df_input.copy()
        df_analysis: pd.DataFrame = self.dfs[Analysis.ADDRESS_ANALYSIS].copy()

        for i, params in enumerate(self.params_matrix):
            # generate network graph and dataframe with component ID column
            df_rentals_components, gMatches = NetworkMatching.rentals_network(
                df_process,
                df_analysis,
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
            df_process: pd.DataFrame = NetworkMatching.set_network_name(
                i+1,
                df_process,
                f"final_component_{i+1}",
                f"network_{i+1}"
            )
            # set text for node/edge data - should this be a separate dataset? probably
            df_process: pd.DataFrame = NetworkMatching.set_network_text(
                gMatches,
                df_process,
                f"final_component_{i+1}",
                f"network_{i+1}"
            )

        self.df_output: pd.DataFrame = df_process


class WkflNetworkMerge(WkflNetworkBase):
    """Workflow to merge string match results with rental dataset."""
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        pass
