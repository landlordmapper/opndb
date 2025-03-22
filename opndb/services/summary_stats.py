from abc import abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from opndb.services.dataframe.base import DataFrameOpsBase as ops_df
from opndb.types.base import WorkflowConfigs
from opndb.utils import UtilsBase as utils, PathGenerators as path_gen

console = Console()

class SummaryStatsBase(object):
    # todo: create excel file with tabs for different workflow stages and save summary stats in each tab

    def __init__(self):
        pass

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

    @abstractmethod
    def print(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

class SSDataClean(SummaryStatsBase):

    def __init__(self, configs: WorkflowConfigs, wkfl_name: str, dfs: dict[str, pd.DataFrame]):
        super().__init__()
        self.configs: WorkflowConfigs = configs
        self.dfs: dict[str, dfs] = dfs
        self.wkfl_name: str = wkfl_name
        self.data: list[list[str | float | int]] = []
        self.stats: dict[str, Any] = {
            "unique_props": {
                "display_name": "Unique Properties", "value": None
            },
            "unique_taxpayers": {
                "display_name": "Unique Taxpayer Records", "value": None
            },
            "unique_corps_raw": {
                "display_name": "Unique Corporations (Raw)", "value": None
            },
            "unique_corps_clean": {
                "display_name": "Unique Corporations (Clean)", "value": None
            },
            "unique_llcs_raw": {
                "display_name": "Unique LLCs (Raw)", "value": None
            },
            "unique_llcs_clean": {
                "display_name": "Unique LLCs (Clean)", "value": None
            },
        }

    def calculate(self) -> None:
        # store references to dataframes
        df_props = self.dfs["properties"]
        df_taxpayers = self.dfs["taxpayer_records"]
        df_corps = self.dfs["corps"]
        df_llcs = self.dfs["llcs"]
        # calculate figures
        self.stats["unique_props"]["value"] = len(df_props["pin"].unique())
        self.stats["unique_taxpayers"]["value"] = len(df_taxpayers["raw_name_address"].unique())
        self.stats["unique_corps_raw"]["value"] = len(df_corps["raw_name"].unique())
        self.stats["unique_corps_clean"]["value"] = len(df_corps["clean_name"].unique())
        self.stats["unique_llcs_raw"]["value"] = len(df_llcs["raw_name"].unique())
        self.stats["unique_llcs_clean"]["value"] = len(df_llcs["clean_name"].unique())
        # save data for dataframe output
        for stat, vals in self.stats.items():
            self.data.append([vals["display_name"], vals["value"]])

    def print(self) -> None:
        table = Table(title=f"Summary Stats: {self.wkfl_name}")
        table.add_column("Stat", justify="right", style="bold yellow")
        table.add_column("Count", justify="right", style="green")
        for stat, vals in self.stats.items():
            table.add_row(
                vals["display_name"],
                str(f"{vals['value']:,}"),
            )
        console.print("\n")
        console.print(table)
        console.print("\n")

    def save(self) -> None:
        df_out = pd.DataFrame(self.data, columns=["Stat", "Count"])
        path: Path = path_gen.summary_stats_data_clean(self.configs)
        ops_df.save_df(df_out, path)

    # @classmethod
    # def summary_stats_address_initial(cls):
    #     # unique raw address count total
    #     # unique raw taxpayer address count
    #     # unique raw corp & LLC count, total AND per column (differentiate by manager/member addr, office addr, etc.)
    #     # po box addrs detected
    #     # po box addrs successfully validated
    #     # total validated & unvalidated after initial processing
    #     # number of unique raw name+addresses
    #     # number of unique cleaned name+addresses
    #     pass
    #
    # @classmethod
    # def summary_stats_address_open_addrs(cls):
    #     # number of addresses successfully validated by open addresses workflow
    #     # total validated & unvalidated after open addrs processing
    #     pass
    #
    # @classmethod
    # def summary_stats_address_geocodio(cls):
    #     # number of addresses passed into geocodio
    #     # number of addresses successfully validated by geocodio
    #     # number of addresses unsuccessfully validated by geocodio
    #     # number of failed geocodio calls
    #     # final total validated & unvalidated addresses after all validation processing
    #     pass
    #
    # @classmethod
    # def summary_stats_name_analysis(cls):
    #     # list of standardized names and counts of different variations that were standardized
    #     # ex:
    #     # CHICAGO LAND TRUST, 473
    #     # US BANK, 253
    #     # DEVON TRUST, 117
    #     pass
    #
    # @classmethod
    # def summary_stats_address_analysis(cls):
    #     # counts for ll_orgs, law firms, etc
    #     pass
    #
    # @classmethod
    # def summary_stats_rental_subset(cls):
    #     # number of total properties in dataset
    #     # number of properties in initial subset
    #     # number of non-rental properties pulled in from taxpayer addrs after subsetting
    #     pass
    #
    # @classmethod
    # def summary_stats_clean_merge(cls):
    #     # number of taxpayer records associated with corps
    #     # number of taxpayer records associated with llcs
    #     # number of unique corps pulled in
    #     # number of unique LLCs pulled in
    #     # number of taxpayer records identified as corp/entities but NOT assigned a corp/LLC
    #     # number of IS_LLC == True rows vs number of matched LLCs
    #     pass
    #
    # @classmethod
    # def summary_stats_string_matching(cls):
    #     # use stats from methodology paper
    #     pass
    #
    # @classmethod
    # def summary_stats_network_graph(cls):
    #     # use stats from methodology paper
    #     pass
    #
    # @classmethod
    # def summary_stats_final_output(cls):
    #     # final dataset stats
    #     pass
