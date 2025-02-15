from abc import abstractmethod, ABC
from typing import Optional

from opndb.types.base import WorkflowConfigs


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
        """Each workflow must implement an execute method"""
        pass

class WkflAddressInitial(WkflAddressBase):

    # todo: create excel file with different tabs for stages and save summary stats in each tab

    """
    Initial address workflow. Loads unique addresses from raw data, extracts PO boxes and saves master validated and
    unvalidated dataframes as instance variables.
    """

    def __init__(self, configs: WorkflowConfigs):
        super().__init__(configs)

    def execute(self):
        """Executes initial data loading workflow."""
        # load raw tax and corp/llc data
        # pull addresses out of them
        # get unique addresses out of all datasets
        # add is_pobox
        # create self.df_validated and add po boxes, save to data dir
        # remove them from the raw dataset
        # create self.df_unvalidated and save to data dir
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

    def execute(self):
        """Executes geocodio data workflow."""
        # pull unvalidated data into dataframes
        # prompt user for api key and display warning with number of calls and estimated cost
        # call geocodio or exit
        # add validated addrs to the master files and save to data dirs
        # remove validated raw address from unvalidated master file
        # set summary stats
        # update configuration file
        pass
