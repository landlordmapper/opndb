from abc import abstractmethod, ABC
from typing import Optional
import nmslib
import pandas as pd
import networkx as nx

from opndb.constants.columns import TaxpayerRecords
from opndb.constants.files import Processed, Dirs, Analysis
from opndb.services.dataframe import DataFrameOpsBase as df_ops
from opndb.services.match import StringMatching, NetworkMatching, MatchingBase
from opndb.types.base import WorkflowConfigs, StringMatchParams, NmslibOptions
from opndb.utils import UtilsBase as utils


# todo: pull out dataframe operation code and store in dataframe service

class WkflStringMatchBase(ABC):

    """Base class for string matching workflow."""

    DEFAULT_NMSLIB: NmslibOptions = {
        "method": "hnsw",
        "space": "cosinesimil_sparse_fast",
        "data_type": nmslib.DataType.SPARSE_VECTOR
    }
    DEFAULT_QUERY_BATCH = {
        "num_threads": 8,
        "K": 1
    }

    def __init__(self, configs: WorkflowConfigs):
        self.configs: WorkflowConfigs = configs
        self.params_matrix: list[StringMatchParams] | None = None
        self.dfs: dict[str, pd.DataFrame] = {
            Processed.PROPS_PREPPED: df_ops.load_df(
                utils.generate_path(
                    Dirs.PROCESSED,
                    Processed.get_raw_filename_ext(Processed.PROPS_PREPPED, self.configs),
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
        self.df_process_input: pd.DataFrame | None = None
        self.df_process_results: pd.DataFrame | None = None
        self.df_process_output: pd.DataFrame | None = None

    @classmethod
    def create_workflow(cls, configs: WorkflowConfigs) -> Optional['WkflStringMatchBase']:
        if configs["wkfl_type_string_match"] == "params":
            return WkflStringMatchParams(configs)
        elif configs["wkfl_type_string_match"] == "prep":
            return WkflStringMatchPrep(configs)
        elif configs["wkfl_type_string_match"] == "process":
            return WkflStringMatchProcess(configs)
        elif configs["wkfl_type_string_match"] == "merge":
            return WkflStringMatchMerge(configs)
        return None

    @abstractmethod
    def execute(self) -> None:
        pass


class WkflStringMatchParams(WkflStringMatchBase):
    """Workflow to set parameters for string matching process."""
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
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


class WkflStringMatchPrep(WkflStringMatchBase):
    """Dataframe preparation workflow."""
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        # set address column so that if there is no validated address for a property, the raw address is used instead
        df_process_input: pd.DataFrame = self.dfs[Processed.PROPS_PREPPED].copy()
        df_process_input["matching_address"] = df_process_input.apply(
            lambda row: MatchingBase.set_matching_address(row), axis=1
        )
        # add name+address concatenated columns to use for matching
        df_process_input["name_address_clean"] = StringMatching.concatenate_name_addr(
            df_process_input,
            "name_address_clean",
            "clean_name",
            "matching_address"
        )
        df_process_input["name_address_core"] = StringMatching.concatenate_name_addr(
            df_process_input,
            "name_address_core",
            "core_name",
            "matching_address"
        )


class WkflStringMatchProcess(WkflStringMatchBase):
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):

        # raise exceptions is parameter matrix or input dataframe are empty
        if self.params_matrix is None:
            raise Exception("Parameter matrix is empty.")
        if self.df_process_input is None:
            raise Exception("Input dataframe is empty.")

        # execute param matrix loop for string matching workflow
        for i, params in enumerate(self.params_matrix):
            # set include_address column
            self.df_process_input["include_address"] = self.df_process_input["matching_address"].apply(
                lambda addr: MatchingBase.check_address(
                    addr,
                    self.dfs[Analysis.ADDRESS_ANALYSIS],
                    params["include_orgs"],
                    params["include_unresearched"],
                )
            )
            # filter out addresses
            df_filtered_addrs: pd.DataFrame = self.df_process_input[self.df_process_input["include_address"] == True]
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
            df_matches: pd.DataFrame = StringMatching.match_strings(
                ref_docs=ref_docs,
                query_docs=query_docs,
                params=params
            )
            # generate network graph to associated matches
            self.df_process_results: pd.DataFrame = NetworkMatching.string_match_network_graph(
                df_process_input=self.df_process_input,
                df_matches=df_matches,
                match_count=i,
                name_address_column=""  # todo: fix this
            )


class WkflStringMatchMerge(WkflStringMatchBase):
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        # merge output with original properties dataset
        df_process_output: pd.DataFrame = pd.merge(
            self.df_process_input,
            self.df_process_output,
            how="left",
            on=""  # todo: fix this
        )
        # clean up
        df_process_output: pd.DataFrame = df_ops.combine_columns_parallel(df_process_output)
        df_process_output.drop_duplicates(TaxpayerRecords.PIN, inplace=True)
        df_process_output.drop(columns=["original_doc"], inplace=True)  # todo: which other columns should be dropped? - NOT name_address_clean
        # done
        self.df_process_output: pd.DataFrame = df_process_output
        df_ops.save_df(
            self.df_process_output,
            utils.generate_path(
                Dirs.PROCESSED,
                Processed.PROPS_STRING_MATCHED,
                self.configs["stage"],
                self.configs["load_ext"],
            )
        )
