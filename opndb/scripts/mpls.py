from opndb.services.config import ConfigManager
from opndb.services.dataframe.base import DataFrameOpsBase as ops_df
from opndb.utils import UtilsBase, PathGenerators as path_gen
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

# fetch address dataframes
df_gcd_validated_formatted = ops_df.load_df(
    path_gen.geocodio_gcd_validated(config_manager.configs, "_formatted")
)

# fetch analysis dataframes
df_fixing_tax_names = ops_df.load_df(path_gen.analysis_fixing_tax_names(config_manager.configs))
df_frequent_tax_names = ops_df.load_df(path_gen.analysis_frequent_tax_names(config_manager.configs))
df_address_analysis = ops_df.load_df(path_gen.analysis_address_analysis(config_manager.configs))

# fetch rental license data
df_rental_licenses = ops_df.load_df(path_gen.pre_process_rental_licenses(config_manager.configs))


# begin workflows
raw_data_prep = WkflRawDataPrep(config_manager)
raw_data_prep.dfs_in = {
    "taxpayers_city": df_taxpayers_city,
    "taxpayers_county": df_taxpayers_county,
    "mnsos_type1": df_mnsos_type1,
    "mnsos_type3": df_mnsos_type3
}
raw_data_prep.validate()
raw_data_prep.process()

data_clean = WkflDataClean(config_manager)
data_clean.dfs_in = {
    "props_taxpayers": raw_data_prep.dfs_out["props_taxpayers"],
    "bus_filings": raw_data_prep.dfs_out["bus_filings"],
    "bus_names_addrs": raw_data_prep.dfs_out["bus_names_addrs"],
}
raw_data_prep.validate()
raw_data_prep.process()

business_merge = WkflBusinessMerge(config_manager)
business_merge.dfs_in = {
    "taxpayer_records": data_clean.dfs_out["taxpayer_records"],
    "bus_filings": data_clean.dfs_out["bus_filings"],
}
business_merge.validate()
business_merge.process()

address_merge = WkflAddressMerge(config_manager)
address_merge.dfs_in = {
    "gcd_validated_formatted": df_gcd_validated_formatted,
    "taxpayers_bus_merged": business_merge.dfs_out["taxpayers_bus_merged"],
    "bus_names_addrs": data_clean.dfs_out["bus_names_addrs"],
}

analysis = WkflAnalysisFinal(config_manager)
analysis.dfs_in = {
    "fixing_tax_names": df_fixing_tax_names,
    "frequent_tax_names": df_frequent_tax_names,
    "taxpayers_addr_merged": address_merge.dfs_out["taxpayers_addr_merged"],
}
analysis.validate()
analysis.process()

rental_subset = WkflRentalSubset(config_manager)
rental_subset.dfs_in = {
    "taxpayers_fixed": analysis.dfs_out["taxpayers_fixed"],
    "properties": data_clean.dfs_out["properties"],
    "rental_licenses": df_rental_licenses,
    "address_analysis": df_address_analysis
}

match_address_cols = WkflMatchAddressCols(config_manager)
match_address_cols.dfs_in = {
    "taxpayers_subsetted": rental_subset.dfs_out["taxpayers_subsetted"],
    "bus_names_addrs_merged": address_merge.dfs_out["bus_names_addrs_merged"],
    "address_analysis": df_address_analysis,
    "gcd_validated_formatted": df_gcd_validated_formatted,
}
match_address_cols.validate()
match_address_cols.process()

string_match = WkflStringMatch(config_manager)
string_match.dfs_in = {
    "taxpayers_prepped": match_address_cols.dfs_out["taxpayers_prepped"],
}
string_match.validate()
string_match.process()

network_graph = WkflNetworkGraph(config_manager)
network_graph.dfs_in = {
    "taxpayers_string_matched": string_match.dfs_out["taxpayers_string_matched"],
    "bus_names_addrs_subsetted": match_address_cols.dfs_out["bus_names_addrs_subsetted"],
}
network_graph.validate()
network_graph.process()

final_output = WkflFinalOutput(config_manager)
final_output.dfs_in = {
    "taxpayers_networked": network_graph.dfs_out["taxpayers_networked"],
    "bus_filings": data_clean.dfs_out["bus_filings"],
    "bus_names_addrs_subsetted": match_address_cols.dfs_out["bus_names_addrs_subsetted"],
    "address_analysis": df_address_analysis,
    "gcd_validated_formatted": df_gcd_validated_formatted,
}
final_output.validate()
final_output.process()

# save final output
final_output.save()

