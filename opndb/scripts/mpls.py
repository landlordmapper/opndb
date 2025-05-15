import csv

import pandas as pd
import gc

from opndb.schema.base.process import AddressAnalysis, FrequentTaxNames, FixingTaxNames
from opndb.services.config import ConfigManager
from opndb.services.dataframe.base import DataFrameOpsBase as ops_df
from opndb.services.terminal_printers import TerminalBase
from opndb.utils import UtilsBase, PathGenerators as path_gen
from opndb.workflows.mpls import WkflRawDataPrep, WkflDataClean, WkflBusinessMerge, WkflSetAddressColumns, \
    WkflAddressMerge, WkflAnalysisFinal, WkflRentalSubset, WkflMatchAddressCols, WkflStringMatch, WkflNetworkGraph, \
    WkflFinalOutput

def execute_mpls():

    # load configs
    config_manager = ConfigManager()
    if config_manager.exists:
        config_manager.load()
    else:
        raise Exception("Config manager not found.")

    # fetch raw data from assessors sites
    TerminalBase.print_with_dots("Loading property taxpayer data for the city of Minneapolis")
    taxpayers_city_url = "https://hub.arcgis.com/api/v3/datasets/5f4b033b92724341810b832385d4f7c2_0/downloads/data?format=csv&spatialRefId=4326&where=1%3D1"
    df_taxpayers_city = ops_df.load_df(taxpayers_city_url)

    TerminalBase.print_with_dots("Loading property taxpayer data for Hennepin County")
    taxpayers_county_url = "https://hub.arcgis.com/api/v3/datasets/7975aabf6e1e42998a40a4b085ffefdf_1/downloads/data?format=csv&spatialRefId=26915&where=1%3D1"
    df_taxpayers_county = ops_df.load_df(taxpayers_county_url)

    TerminalBase.print_with_dots("Loading rental license data for the city of Minneapolis")
    rental_licenses = "https://hub.arcgis.com/api/v3/datasets/baf5f14d67704668884275686e3db867_0/downloads/data?format=csv&spatialRefId=3857&where=1%3D1"
    df_rental_licenses = ops_df.load_df(rental_licenses)

    # load business filings
    df_mnsos_type1 = pd.read_csv(
        str(path_gen.pre_process_business_filings_1(config_manager.configs)),
        dtype=str,
        encoding="windows-1252",
        encoding_errors="replace",
        delimiter=",",
        quoting=csv.QUOTE_NONE,  # disables quote logic entirely
        engine="python",  # fallback parser that handles irregular rows better
        on_bad_lines="skip",  # logs but keeps going
    )
    df_mnsos_type3 = pd.read_csv(
        str(path_gen.pre_process_business_filings_3(config_manager.configs)),
        dtype=str,
        encoding="windows-1252",
        encoding_errors="replace",
        delimiter=",",
        quoting=csv.QUOTE_NONE,  # disables quote logic entirely
        engine="python",  # fallback parser that handles irregular rows better
        on_bad_lines="skip",  # logs but keeps going
    )

    # fetch address dataframes
    df_gcd_validated_formatted = ops_df.load_df(
        path_gen.geocodio_gcd_validated(config_manager.configs, "_formatted")
    )

    # fetch analysis dataframes
    df_fixing_tax_names = ops_df.load_df(
        path_gen.analysis_fixing_tax_names(config_manager.configs),
        FixingTaxNames
    )
    df_frequent_tax_names = ops_df.load_df(
        path_gen.analysis_frequent_tax_names(config_manager.configs),
        FrequentTaxNames,
        True
    )
    df_address_analysis = ops_df.load_df(
        path_gen.analysis_address_analysis(config_manager.configs),
        AddressAnalysis,
        True
    )

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

    # clean up memory
    del df_mnsos_type1, df_mnsos_type3, df_taxpayers_city, df_taxpayers_county
    gc.collect()

    data_clean = WkflDataClean(config_manager)
    data_clean.dfs_in = {
        "props_taxpayers": raw_data_prep.dfs_out["props_taxpayers"],
        "bus_filings": raw_data_prep.dfs_out["bus_filings"],
        "bus_names_addrs": raw_data_prep.dfs_out["bus_names_addrs"],
    }
    data_clean.validate()
    data_clean.process()

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
    address_merge.validate()
    address_merge.process()

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
    rental_subset.validate()
    rental_subset.process()

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

