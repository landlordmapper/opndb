from opndb.services.config import ConfigManager
from opndb.workflows.mpls import WkflRawDataPrep, WkflDataClean, WkflBusinessMerge, WkflSetAddressColumns, \
    WkflAddressMerge, WkflAnalysisFinal, WkflRentalSubset, WkflMatchAddressCols, WkflStringMatch, WkflNetworkGraph, \
    WkflFinalOutput

# load configs
config_manager = ConfigManager()
if config_manager.exists:
    config_manager.load()
else:
    raise Exception("Config manager not found.")

# fetch raw data
taxpayers_city = ""
taxpayers_county = ""
mnsos_type1 = ""
mnsos_type2 = ""

# begin workflows

raw_data_prep = WkflRawDataPrep(config_manager)
data_clean = WkflDataClean(config_manager)
business_merge = WkflBusinessMerge(config_manager)
set_address_columns = WkflSetAddressColumns(config_manager)
address_merge = WkflAddressMerge(config_manager)
analysis = WkflAnalysisFinal(config_manager)
rental_subset = WkflRentalSubset(config_manager)
match_address_cols = WkflMatchAddressCols(config_manager)
string_match = WkflStringMatch(config_manager)
network_graph = WkflNetworkGraph(config_manager)
final_output = WkflFinalOutput(config_manager)