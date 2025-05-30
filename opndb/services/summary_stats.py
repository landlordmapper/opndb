from abc import abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from opndb.services.dataframe.base import DataFrameOpsBase as ops_df
from opndb.services.dataframe.ops import DataFrameSubsetters as subset_df
from opndb.types.base import WorkflowConfigs, StringMatchParams, NetworkMatchParams
from opndb.utils import UtilsBase as utils, PathGenerators as path_gen


console = Console()

class SummaryStatsBase(object):
    # todo: create excel file with tabs for different workflow stages and save summary stats in each tab

    def __init__(self):
        self.configs: WorkflowConfigs | None = None
        self.dfs: dict[str, pd.DataFrame] | None = None
        self.wkfl_name: str = ""
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    @classmethod
    def display_load_table(cls, dfs_in):
        """Prints out summary tables of dataframes loaded into workflow object."""
        table_data = []
        for id, df in dfs_in.items():  # todo: standardize this, enforce types
            if df is None:
                continue
            memory_usage = df.memory_usage(deep=True).sum()
            table_data.append({
                "dataset_name": id,
                "file_size": utils.sizeof_fmt(memory_usage),
                "record_count": len(df)
            })
        table = Table(title="Dataframes Loaded")
        table.add_column("Dataset Name", justify="right", style="bold yellow")
        table.add_column("File Size", justify="right", style="green")
        table.add_column("Number of Rows", justify="right", style="cyan")
        for td_obj in table_data:
            row_count = int(td_obj["record_count"])
            formatted_count = f"{row_count:,}"
            table.add_row(
                str(td_obj["dataset_name"]),
                str(td_obj["file_size"]),
                formatted_count,
            )
        console.print(table)
        console.print("\n")

    @classmethod
    def display_load_stats_table(cls, dfs_in):
        for id, df in dfs_in.items():
            if df is None:
                continue
            table_data = []
            for col in df.columns:
                # Count null values
                null_count = df[col].isna().sum()
                # Count empty strings (only for string/object columns)
                empty_string_count = 0
                if df[col].dtype == "object":
                    empty_string_count = (df[col] == "").sum()
                # Total empty values
                total_empty = null_count + empty_string_count
                table_data.append({
                    "column_name": col,
                    "null_count": total_empty
                })
            table = Table(title=f"{id} ({len(df):,} rows)")
            table.add_column("Column Name", justify="right", style="bold yellow")
            table.add_column("Empty/Null Values", justify="right", style="green")
            for td_obj in table_data:
                table.add_row(
                    str(td_obj["column_name"]),
                    str(f"{td_obj['null_count']:,}"),
                )
            console.print(table)
            console.print("\n")

    @abstractmethod
    def calculate(self) -> None:
        pass

    def print(self) -> None:
        # set table title and columns
        table = Table(title=f"Summary Stats: {self.wkfl_name}")
        table.add_column("Stat", justify="right", style="bold yellow")
        table.add_column("Count", justify="right", style="green")
        # set table data
        for stat, vals in self.stats.items():
            table.add_row(
                vals["display_name"],
                str(f"{vals['value']:,}"),
            )
        # print table
        console.print("\n")
        console.print(table)
        console.print("\n")

    def save(self) -> None:
        df_out = pd.DataFrame(self.data, columns=["Stat", "Count"])
        path: Path = path_gen.summary_stats(self.configs, self.wkfl_name)
        ops_df.save_df(df_out, path)


class SSDataClean(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs_out: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # store references to dataframes
        df_props = self.dfs_out["properties"]
        df_taxpayers = self.dfs_out["taxpayer_records"]
        df_corps = self.dfs_out["corps"]
        df_llcs = self.dfs_out["llcs"]
        df_addrs = self.dfs_out["unvalidated_addrs"]
        # calculate figures
        unique_props: int = len(df_props["pin"].unique())
        unique_taxpayers: int = len(df_taxpayers["raw_name_address"].unique())
        unique_corps_raw: int = len(df_corps["raw_name"].unique())
        unique_corps_clean: int = len(df_corps["clean_name"].unique())
        unique_llcs_raw: int = len(df_llcs["raw_name"].unique())
        unique_llcs_clean: int = len(df_llcs["clean_name"].unique())
        unique_addrs: int = len(df_addrs["raw_address"])
        # set stats
        self.stats = {
            "unique_props": {
                "display_name": "Unique Properties",
                "value": unique_props
            },
            "unique_taxpayers": {
                "display_name": "Unique Taxpayer Records",
                "value": unique_taxpayers
            },
            "unique_corps_raw": {
                "display_name": "Unique Corporations (Raw)",
                "value": unique_corps_raw
            },
            "unique_corps_clean": {
                "display_name": "Unique Corporations (Clean)",
                "value": unique_corps_clean
            },
            "unique_llcs_raw": {
                "display_name": "Unique LLCs (Raw)",
                "value": unique_llcs_raw
            },
            "unique_llcs_clean": {
                "display_name": "Unique LLCs (Clean)",
                "value": unique_llcs_clean
            },
            "unique_addrs": {
                "display_name": "Unique Addresses (Unvalidated)",
                "value": unique_addrs
            }
        }
        # save data for output
        for stat, vals in self.stats.items():
            self.data.append([vals["display_name"], vals["value"]])


class SSAddressClean(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs_out: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # store references to dataframes
        df_addrs = self.dfs_out["unvalidated_addrs"]
        # calculate figures
        pobox_count: int = len(df_addrs[df_addrs["is_pobox"] == True])
        # set stats
        self.stats = {
            "pobox_addrs": {
                "display_name": "PO Box Addresses",
                "value": pobox_count
            }
        }
        # save data for output
        for stat, vals in self.stats.items():
            self.data.append([vals["display_name"], vals["value"]])


class SSAddressGeocodio(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_in: dict[str, pd.DataFrame], dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs_in: dict[str, pd.DataFrame] = dfs_in
        self.dfs_out: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # store references to dataframes
        df_valid_before = self.dfs_in["gcd_validated"]
        df_unvalid_before = self.dfs_in["gcd_unvalidated"]
        df_valid_after = self.dfs_out["gcd_validated"]
        df_unvalid_after = self.dfs_out["gcd_unvalidated"]
        # calculate figures
        valid_before: int = len(df_valid_before["clean_address"].unique())
        unvalid_before: int = len(df_unvalid_before["clean_address"].unique())
        valid_after: int = len(df_valid_after["clean_address"].unique())
        unvalid_after: int = len(df_unvalid_after["clean_address"].unique())
        total_valid: int = valid_before + valid_after
        total_unvalid: int = unvalid_before + unvalid_after
        # set stats
        self.stats = {
            "valid_addrs": {
                "display_name": "Validated addresses",
                "value": f"{total_valid:,} (+{valid_after:,})"
            },
            "unvalid_addrs": {
                "display_name": "Unvalidated addresses",
                "value": f"{total_unvalid:,} (+{unvalid_after:,})"
            }
        }
        # save data for output
        for stat, vals in self.stats.items():
            self.data.append([vals["display_name"], vals["value"]])


class SSFixUnitsInitial(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # store references to dataframes
        # calculate figures
        # set stats
        pass


class SSFixUnitsFinal(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # store references to dataframes
        # calculate figures
        # set stats
        pass


class SSAddressMerge(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # store references to dataframes
        # calculate figures
        # set stats
        pass


class SSNameAnalysisInitial(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # store references to dataframes
        # calculate figures
        # set stats
        pass


class SSAddressAnalysisInitial(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # list of standardized names and counts of different variations that were standardized
        # ex:
        # CHICAGO LAND TRUST, 473
        # US BANK, 253
        # DEVON TRUST, 117

        # store references to dataframes
        # calculate figures
        # set stats
        pass


class SSAnalysisFinal(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # store references to dataframes
        # calculate figures
        # set stats
        pass


class SSRentalSubset(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # number of total properties in dataset
        # number of properties in initial subset
        # number of non-rental properties pulled in from taxpayer addrs after subsetting

        # store references to dataframes
        # calculate figures
        # set stats
        pass


class SSCleanMerge(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # store references to dataframes
        df_taxpayers: pd.DataFrame = self.dfs["taxpayers_prepped"]
        df_corps: pd.DataFrame = self.dfs["corps_subsetted"]
        df_llcs: pd.DataFrame = self.dfs["llcs_subsetted"]
        # calculate figures
        # number of taxpayer records associated with corps
        taxpayer_corps: int = len(df_taxpayers[df_taxpayers["origin"] == "corp"])
        # number of taxpayer records associated with llcs
        taxpayer_llcs: int = len(df_taxpayers[df_taxpayers["origin"] == "llc"])
        # number of unique corps pulled in
        unique_corps: int = len(df_corps)
        # number of unique LLCs pulled in
        unique_llcs: int = len(df_llcs)
        # number of taxpayer records identified as corp/entities but NOT assigned a corp/LLC
        df_missing: pd.DataFrame = df_taxpayers[df_taxpayers["origin"].isnull()]
        unidentified_llcs: int = len(df_missing[df_missing["is_llc"] == True])
        unidentified_orgs: int = len(df_missing[df_missing["is_org"] == True])
        # set stats
        self.stats: dict[str, Any] = {
            "taxpayer_corps": {
                "display_name": "Number of Linked Corporations in Taxpayer Records",
                "value": taxpayer_corps,
            },
            "taxpayer_llcs": {
                "display_name": "Number of Linked LLCs in Taxpayer Records",
                "value": taxpayer_llcs,
            },
            "unique_corps": {
                "display_name": "Unique Corporations Identified in Taxpayer Records",
                "value": unique_corps,
            },
            "unique_llcs": {
                "display_name": "Unique LLCs Identified in Taxpayer Records",
                "value": unique_llcs,
            },
            "unidentified_llcs": {
                "display_name": "Unlinked LLCs in Taxpayer Records",
                "value": unidentified_llcs,
            },
            "unidentified_orgs": {
                "display_name": "Unlinked Organizations in Taxpayer Records",
                "value": unidentified_orgs,
            },
        }
        # save data for output
        for stat, vals in self.stats.items():
            self.data.append([vals["display_name"], vals["value"]])


class SSStringMatch(SummaryStatsBase):

    def __init__(
        self,
        configs: WorkflowConfigs,
        wkfl_name: str,
        dfs_out: dict[str, pd.DataFrame],
        params_matrix: list[StringMatchParams]
    ):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[dict[str, Any]] = []
        self.params_matrix: list[StringMatchParams] = params_matrix

    def calculate(self) -> None:
        # use stats from methodology paper
        # store references to dataframes
        df_taxpayers: pd.DataFrame = self.dfs["taxpayers_string_matched"]
        # calculate figures
        data = []
        for i, params in enumerate(self.params_matrix):
            string_match_name: str = f"string_matched_name_{i+1}"
            matched_taxpayers: int = len(df_taxpayers[string_match_name].dropna())
            unique_matches: int = len(df_taxpayers[string_match_name].dropna().unique())
            percent_matched: float = round((matched_taxpayers / len(df_taxpayers) * 100), 2)
            data.append({
                "taxpayer_column": params["name_col"],
                "match_threshold": params["match_threshold"],
                "include_unvalidated?": params["include_unvalidated"],
                "include_unresearched?": params["include_unresearched"],
                "include_orgs?": params["include_orgs"],
                "include_missing_suites?": params["include_missing_suites"],
                "include_problem_suites?": params["include_problem_suites"],
                "address_column": params["address_suffix"],
                "string_match_name": string_match_name,
                "matched_taxpayers": matched_taxpayers,
                "unique_matches": unique_matches,
                "percent_matched": percent_matched,
            })
        # set stats
        self.data = data

    def print(self) -> None:
        table = Table(title=f"Summary Stats: {self.wkfl_name}")
        for col in self.data[0].keys():
            table.add_column(col, justify="right", style="green")
        for row in self.data:
            table.add_row(*[str(val) for val in row.values()])
        console.print("\n")
        console.print(table)
        console.print("\n")

    def save(self) -> None:
        df_out = pd.DataFrame(self.data)
        path: Path = path_gen.summary_stats(self.configs, self.wkfl_name)
        ops_df.save_df(df_out, path)


class SSNetworkGraph(SummaryStatsBase):

    def __init__(
        self,
        configs: WorkflowConfigs,
        wkfl_name: str,
        dfs_out: dict[str, pd.DataFrame],
        params_matrix: list[NetworkMatchParams]
    ):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[dict[str, Any]] = []
        self.params_matrix: list[NetworkMatchParams] = params_matrix

    def calculate(self) -> None:
        # use stats from methodology paper
        # store references to dataframes
        df_taxpayers: pd.DataFrame = self.dfs["taxpayers_networked"]
        # calculate figures
        data = []
        for i, params in enumerate(self.params_matrix):
            network_name: str = f"network_{i+1}"
            taxpayers_missing_networks: int = len(df_taxpayers[df_taxpayers[network_name].isnull()])
            unique_networks: int = len(df_taxpayers[network_name].dropna().unique())
            # calculating taxpayer record count in top 100
            df_freq: pd.DataFrame = subset_df.generate_frequency_df(df_taxpayers, network_name)
            top_networks = df_freq["value"][:100]
            taxpayers_in_top_100 = df_taxpayers[network_name].isin(top_networks).sum()
            data.append({
                "taxpayer_column": params["taxpayer_name_col"],
                "include_unvalidated?": params["include_unvalidated"],
                "include_unresearched?": params["include_unresearched"],
                "include_orgs?": params["include_orgs"],
                "include_missing_suites?": params["include_missing_suites"],
                "include_problem_suites?": params["include_problem_suites"],
                "address_column": params["address_suffix"],
                "string_match_name": params["string_match_name"],
                "network_name": network_name,
                "taxpayers_missing_networks": taxpayers_missing_networks,
                "unique_networks": unique_networks,
                "taxpayers_in_top_100": taxpayers_in_top_100,
            })
        # set stats
        self.data = data

    def print(self) -> None:
        table = Table(title=f"Summary Stats: {self.wkfl_name}")
        for col, val in self.data[0].items():
            table.add_column(col, justify="right", style="green")
        for row in self.data:
            table.add_row(*[str(value) for value in row.values()])
        console.print("\n")
        console.print(table)
        console.print("\n")

    def save(self) -> None:
        df_out = pd.DataFrame(self.data)
        path: Path = path_gen.summary_stats(self.configs, self.wkfl_name)
        ops_df.save_df(df_out, path)


class SSFinalOutput(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs_out: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, pd.DataFrame] = dfs_out
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {}

    def calculate(self) -> None:
        # use stats from methodology paper
        # store references to dataframes
        # calculate figures
        # set stats
        pass
