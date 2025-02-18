from abc import abstractmethod, ABC
from typing import Optional
import pandas as pd

from opndb.constants.columns import UnvalidatedAddrs as u, ValidatedAddrs as va, UnvalidatedAddrs as ua
from opndb.constants.files import Dirs as d, Processed as p, Geocodio as g
from opndb.services.address import AddressBase as addr
from opndb.services.dataframe import DataFrameOpsBase as ops_df, DataFrameColumnGenerators as col_df, \
    DataFrameSubsetters as subset_df
from opndb.services.terminal_printers import TerminalBase as terminal
from opndb.types.base import WorkflowConfigs
from opndb.utils import UtilsBase as utils


class WkflAddressBase(ABC):

    def __init__(self, configs: WorkflowConfigs):
        self.configs = configs

    @classmethod
    def create_workflow(cls, configs: WorkflowConfigs) -> Optional['WkflAddressBase']:
        if configs["wkfl_type_addrs"] == "initial":
            return WkflAddressInitial(configs)
        elif configs["wkfl_type_addrs"] == "open_addrs":
            return WkflAddressOpenAddresses(configs)
        elif configs["wkfl_type_addrs"] == "geocodio":
            return WkflAddressGeocodio(configs)
        return None

    @abstractmethod
    def execute(self) -> None:
        pass

class WkflAddressInitial(WkflAddressBase):
    """
    Initial address workflow. Loads unique addresses from raw data, extracts PO boxes and saves master validated and
    unvalidated dataframes as instance variables.
    """
    # todo: create excel file with different tabs for stages and save summary stats in each tab
    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)
        self.dfs: dict[str, pd.DataFrame] = {
            "unvalidated_addrs": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.UNVALIDATED_ADDRS, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"]
                ), str
            ),
        }

    def execute(self):
        """Executes initial data loading workflow."""
        # todo: add conditional handling for whether or not street column is separate from full address column
        # todo: use pydantic/panderas model to created validated object address rows
        dfs = {
            key: df.copy()
            for key, df in self.dfs.items()
        }
        # add is_pobox
        dfs["unvalidated_addrs"]["is_pobox"] = col_df.set_is_pobox(dfs["unvalidated_addrs"], u.TAX_STREET)
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
        # set summary stats
        # update configuration file
        pass


class WkflAddressOpenAddresses(WkflAddressBase):

    # todo: figure out how to color code the summary stats so that people know how far off the mean their data is

    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        """Executes open address data workflow."""
        # pull unvalidated data into dataframes
        # prompt user for api key and display warning with number of calls and estimated cost
        # call geocodio or exit
        # add validated addrs to the master files and save to data dirs
        # remove validated raw address from unvalidated master file
        # set summary stats
        # update configuration file
        pass


class WkflAddressGeocodio(WkflAddressBase):

    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)
        self.dfs: dict[str, pd.DataFrame] = {
            "unvalidated_addrs": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.UNVALIDATED_ADDRS, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"]
                ), str
            ),
            "validated_addrs": ops_df.load_df(
                utils.generate_path(
                    d.PROCESSED,
                    p.get_raw_filename_ext(p.VALIDATED_ADDRS, self.configs),
                    self.configs["prev_stage"],
                    self.configs["load_ext"]
                ), str
            ),
        }
        self.api_key: str = ""

    def execute(self):
        """Executes geocodio data workflow."""
        dfs = {
            key: df.copy()
            for key, df in self.dfs.items()
        }
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
        # set summary stats
        # update configuration file
        pass
