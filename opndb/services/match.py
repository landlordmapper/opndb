import nmslib
from collections import Counter
import json
from typing import List
import Levenshtein as lev

import networkx as nx
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn, TimeElapsedColumn
)
from rich.console import Console
from opndb.constants.columns import AddressAnalysis
from opndb.services.terminal_printers import TerminalBase as t
from opndb.types.base import StringMatchParams, NetworkMatchParams
from opndb.services.dataframe.base import DataFrameOpsBase as df_ops
from opndb.utils import UtilsBase as utils

# todo: add type hints and docstrings where missing
console = Console()

class MatchBase:

    @classmethod
    def check_name(cls, row: pd.Series) -> bool:
        """
        Returns "True" if the row SHOULD be included in the network analysis, and False if it should be ignored.
        """
        return pd.isna(row["IS_COMMON_NAME"]) or row["IS_COMMON_NAME"] is False

    @classmethod
    def check_entity_name(cls, row: pd.Series) -> bool:
        """
        Returns "True" if the row SHOULD be included in the network analysis, and False if it should be ignored.
        """
        return pd.isna(row["IS_COMMON_NAME"]) or row["IS_COMMON_NAME"] is False

    @classmethod
    def check_address(
            cls,
            address: str | None,
            df_researched: pd.DataFrame,
            include_orgs: bool,
            include_unresearched: bool
    ) -> bool:
        """
        Returns "True" if the address SHOULD be included in the network analysis, and False if it should be ignored.
        """
        if not address:
            return False
        # the address has NOT already been analyzed
        if address not in list(df_researched["value"].dropna().unique()):
            return include_unresearched
        # address has been analyzed - check if it should be ignored
        else:
            df_addr_analysis = df_researched[df_researched["value"] == address]
            if df_addr_analysis["is_landlord_org"].eq("t").any():
                return include_orgs
            elif df_addr_analysis["is_lawfirm"].eq("t").any():
                return False
            elif df_addr_analysis["is_missing_suite"].eq("t").any():
                return False
            elif df_addr_analysis["is_financial_services"].eq("t").any():
                return False
            elif df_addr_analysis["is_virtual_office_agent"].eq("t").any():
                return False
            elif df_addr_analysis["fix_address"].eq("t").any():
                return False
            elif df_addr_analysis["is_ignore_misc"].eq("t").any():
                return False
            else:
                return True

    @classmethod
    def set_matching_address(cls, row):
        if pd.isna(row["merge_address_1"]) or row["merge_address_1"].strip() == "":
            return row["raw_address_v"]  # todo: run cleaned addresses through validator and change this to "clean_address_v" throughout the code where needed
        else:
            return row["merge_address_1"]


class StringMatch(MatchBase):

    @classmethod
    def ngrams(cls, string, n=3):

        # convert string into ascii encoding
        string = string.encode("ascii", errors="ignore").decode()

        # converts letters in string to lower case
        string = string.lower()

        # removes unwanted characters
        chars_to_remove = [')', '(', '.', '|', '[', ']', '{', '}', "'"]
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        string = re.sub(rx, '', string) # remove the list of chars defined above

        # replace various symbols with non-symbols
        string = string.replace('&', 'and')
        string = string.replace(',', ' ').replace('-', ' ')

        string = string.title() # Capital at start of each word
        string = re.sub(' +',' ',string).strip() # combine whitespace
        string = ' ' + string + ' ' # pad
        #string = re.sub(r'[,-./]', r'', string)

        # core N-gram generation
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    @classmethod
    def match_entities(cls):
        return

    @classmethod
    def match_strings(cls, ref_docs: list[str], query_docs: list[str], params: StringMatchParams):

        # set up vectorizer and index for string matching
        t.print_with_dots("Initializing vectorizer")
        vectorizer = TfidfVectorizer(min_df=1, analyzer=cls.ngrams)
        t.print_with_dots("Generating matrix")
        tf_idf_matrix = vectorizer.fit_transform(ref_docs)
        messy_tf_idf_matrix = vectorizer.transform(query_docs)
        data_matrix = tf_idf_matrix
        t.print_with_dots("Initializing HNSW index")
        index = nmslib.init(
            method=params["nmslib_opts"]["method"],
            space=params["nmslib_opts"]["space"],
            data_type=params["nmslib_opts"]["data_type"]
        )
        t.print_with_dots("Adding data to index")
        index.addDataPointBatch(data_matrix)
        index.createIndex()

        # execute query and store good matches
        t.print_with_dots("Executing query")
        query_matrix = messy_tf_idf_matrix
        nbrs = index.knnQueryBatch(
            query_matrix,
            k=params["query_batch_opts"]["K"],
            num_threads=params["query_batch_opts"]["num_threads"]
        )
        mts =[]

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            "•",
            TextColumn("[bold cyan]{task.fields[processed]}/{task.total} records"),
        ) as progress:
            task = progress.add_task(
                f"[yellow]Processing query results records...",
                total=len(nbrs),
                processed=0,
            )
            processed_count = 0
            for i in range(len(nbrs)):  # todo: add progress bar
                original_nm = query_docs[i]
                for row in list(range(len(nbrs[i][0]))):
                    try:
                        matched_nm = ref_docs[nbrs[i][0][row]]
                        conf = abs(nbrs[i][1][row])
                    except:
                        matched_nm = "no match found"
                        conf = None
                    mts.append([original_nm, matched_nm, conf])
                processed_count += 1
                progress.update(
                    task,
                    advance=1,
                    processed=processed_count,
                    description=f"[yellow]Processing record {processed_count}/{len(nbrs)}"
                )

        df_matches = pd.DataFrame(mts,columns=["original_doc", "matched_doc", "conf"])
        df_matches["ldist"] = df_matches[["matched_doc", "original_doc"]].apply(lambda x: lev.distance(x[0], x[1]), axis=1)
        df_matches["conf1"] = 1 - df_matches["conf"]

        if params["match_threshold"] is not None:
            df_good_matches = df_matches[(df_matches["ldist"] > 0) & (df_matches["conf1"] > params["match_threshold"]) & (df_matches["conf1"] < 1)].sort_values(by=["conf1"])
            console.print("String matching complete ✅")
            return df_good_matches  # does this return duplicates? like if there are multiple matches above the threshhold it needs to pick the highest one, NOT include all of them
        else:
            console.print("String matching complete ✅")
            return df_matches


class NetworkMatchBase(MatchBase):
    """
    Network matching service class. Contains all functions related to building network graph objects and connected
    component maps. Methods are defined roughly in their order of execution in generalized network graph workflows.
    """
    # todo: standardize pattern of what these functions return (Ex: set of functions that return nx.Graph objects, dataframe objects, other network graph-related objects, etc.)
    @classmethod
    def build_edge(cls, g, node_a, node_b, common_names=None, common_addrs=None):
        if (common_names is None or node_a not in common_names) and (common_addrs is None or node_b not in common_addrs):
            g.add_edge(node_a, node_b)

    @classmethod
    def process_row_network(
            cls,
            g: nx.Graph,
            row: pd.Series,
            df_analysis: pd.DataFrame,
            clean_core_column: str,
            include_orgs: bool,
            include_unresearched: bool
    ) -> None:
        """
        Adds nodes and edges for taxpayer name and taxpayer address. Uses clean address to check for inclusion
        detection and matching_address for the network graph

        Names that are NOT indicated as being common names are included. Addresses that pass the check_address test are
        included. If both the name and address pass the name and address checks, add them as nodes AND edges

        If the name passes the check_name test but the address does not pass the check_address test, add ONLY the name
        as a node.
        """

        name = row[clean_core_column]
        clean_address = row["GCD_FORMATTED_ADDRESS"]
        matching_address = row["GCD_FORMATTED_MATCH"]

        if cls.check_address(clean_address, df_analysis, include_orgs, include_unresearched) and cls.check_name(row):
            if pd.notnull(name) and not utils.is_encoded_empty(name) and pd.notnull(
                    clean_address) and not utils.is_encoded_empty(clean_address):
                g.add_edge(name, matching_address)
        elif cls.check_name(row) and pd.notnull(name) and not utils.is_encoded_empty(name):
            g.add_node(name)

    @classmethod
    def process_row_network_string_match(
            cls,
            g: nx.Graph,
            row: pd.Series,
            string_match_column: str,
            df_analysis: pd.DataFrame,
            clean_core_column: str,
            include_orgs: bool,
            include_unresearched: bool
    ) -> None:

        # todo: fix these column names
        fuzzy_match_combo = row[string_match_column]
        clean_address = row["GCD_FORMATTED_ADDRESS"]
        name = row[clean_core_column]
        matching_address = row["GCD_FORMATTED_MATCH"]

        if cls.check_address(clean_address, df_analysis, include_orgs, include_unresearched) and cls.check_name(row):
            if pd.notnull(name) and pd.notnull(clean_address):
                g.add_edge(name, fuzzy_match_combo)
                g.add_edge(matching_address, fuzzy_match_combo)
        elif cls.check_name(row) and pd.notnull(name):
            g.add_edge(name, fuzzy_match_combo)
        else:
            g.add_node(fuzzy_match_combo)

    @classmethod
    def process_row_network_entity(
            cls,
            g: nx.Graph,
            row: pd.Series,
            df_analysis: pd.DataFrame,
            string_match_column: str,
            clean_core_column: str,
            clean_core_column_entity: str,
            include_orgs: bool,
            include_unresearched: bool
    ) -> None:
        """Processes nodes and edges related EXCLUSIVELY to entity_name and entity_address"""
        # todo: fix these column names
        entities_to_ignore: List[str] = ["CHICAGO TITLE LAND TRUST COMPANY", "CHICAGO TITLE LAND"]
        taxpayer_name = row[clean_core_column]
        entity_name = row[clean_core_column_entity]
        fuzzy_match_combo = row[string_match_column]
        entity_addresses = [
            row["GCD_FORMATTED_ADDRESS_ADDRESS_1"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_2"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_3"],
        ]
        entity_matching_addresses = [
            row["GCD_FORMATTED_ADDRESS_ADDRESS_1_MATCH"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_2_MATCH"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_3_MATCH"],
        ]
        for i, address in enumerate(entity_addresses):
            if cls.check_address(address, df_analysis, include_orgs, include_unresearched):
                if pd.notnull(address) and entity_name not in entities_to_ignore:
                    g.add_edge(entity_name, entity_matching_addresses[i])
                    g.add_edge(taxpayer_name, entity_matching_addresses[i])
                    if pd.notnull(fuzzy_match_combo):
                        g.add_edge(entity_name, fuzzy_match_combo)
                        g.add_edge(entity_matching_addresses[i], fuzzy_match_combo)
                elif pd.notnull(address):
                    g.add_edge(taxpayer_name, entity_matching_addresses[i])
                    if pd.notnull(fuzzy_match_combo):
                        g.add_edge(entity_matching_addresses[i], fuzzy_match_combo)

    @classmethod
    def rentals_network(
            cls,
            network_id: int,
            df_input: pd.DataFrame,
            df_analysis: pd.DataFrame,
            params: NetworkMatchParams
    ):
        # todo: break this into smaller pieces
        """
        Generates and NetworkX graph object containing nodes and edges for rental dataset based on parameters. Returns
        graph object and dataframe with associated component IDs.
        """
        gMatches = nx.Graph()
        for i, row in df_input.iterrows():
            # 1. Add nodes & edges for taxpayer name and address
            cls.process_row_network(
                gMatches,
                row,
                df_analysis,
                params["taxpayer_name_col"],
                params["include_orgs"],
                params["include_unresearched"]
            )
            # 2. Add nodes & edges for fuzzy match combo name (if present)
            if pd.notnull(row[params["string_match_name"]]):
                cls.process_row_network_string_match(
                    gMatches,
                    row,
                    params["string_match_name"],
                    df_analysis,
                    params["taxpayer_name_col"],
                    params["include_orgs"],
                    params["include_unresearched"]
                )
            # 3. Add nodes & edges for entity name and entity address (if present)
            if pd.notnull(row[params["entity_name_col"]]):
                cls.process_row_network_entity(
                    g=gMatches,
                    row=row,
                    df_analysis=df_analysis,
                    string_match_column=params["string_match_name"],
                    clean_core_column=params["taxpayer_name_col"],
                    clean_core_column_entity=params["entity_name_col"],
                    include_orgs=params["include_orgs"],
                    include_unresearched=params["include_unresearched"]
                )
        # get unique values for each column used to generate nodes and edges
        taxpayer_names_set = list(set(df_input[params["taxpayer_name_col"]].dropna().unique()))
        fuzzy_matches_set = list(set(df_input[params["string_match_name"]].dropna().unique()))
        clean_addresses_set = list(set(
                list(set(df_input["GCD_FORMATTED_MATCH"].dropna().unique())) +
                list(set(df_input["GCD_FORMATTED_ADDRESS_ADDRESS_1_MATCH"].dropna().unique()))  #+
                # list(set(df_unique["GCD_FORMATTED_ADDRESS_ADDRESS_2_MATCH"].dropna().unique())) +
                # list(set(df_unique["GCD_FORMATTED_ADDRESS_ADDRESS_3_MATCH"].dropna().unique()))
        ))
        entity_names_set = list(set(df_input[params["entity_name_col"]].dropna().unique()))
        # loop through connected to components to associate component IDs
        # assign components to unique values from each column used to generate nodes and edges
        component_map: dict[str, int] = {}
        for i, connections in enumerate(list(nx.connected_components(gMatches))):
            for component in connections:
                if component in taxpayer_names_set:
                    component_map[component] = i
                elif component in fuzzy_matches_set:
                    component_map[component] = i
                elif component in clean_addresses_set:
                    component_map[component] = i
                elif component in entity_names_set:
                    component_map[component] = i

        df_input[f"final_component_{network_id+1}"] = df_input.apply(
            lambda row: cls.set_component(row, component_map, params), axis=1
        )
        return df_input, gMatches

    @classmethod
    def set_component(
            cls,
            row: pd.Series,
            component_map: dict,
            params: NetworkMatchParams
    ):
        """
        Assigns connected component to rental property row. Uses connected component map generated by
        build_connected_component_map() to associate a property with a network based on the association of taxpayer
        names, entity names, and mailing addresses.
        """
        keys_to_check = [
            row[params["taxpayer_name_col"]],
            row["GCD_FORMATTED_MATCH"],
            row[params["string_match_name"]],
            row[params["entity_name_col"]],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_1_MATCH"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_2_MATCH"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_3_MATCH"]
        ]
        for key in keys_to_check:
            if key in component_map.keys():
                return component_map[key]
        # If no match is found, print debug info and return np.nan
        # print(f"KeyError for CleanName: {row['CLEAN_NAME']} and ADDRESS: {row['CLEAN_ADDRESS']}")
        return np.nan

    @classmethod
    def set_network_name(
            cls,
            network_id: int,
            df_process_results: pd.DataFrame,
            component_col_name: str,
            network_col_name: str,
    ) -> pd.DataFrame:

        df_networked = df_process_results.copy()
        df_networked[network_col_name] = np.nan
        unique_networks = df_networked[component_col_name].unique()

        for ntwk in unique_networks:
            df_subset = df_networked[df_networked[component_col_name] == ntwk]
            unique_names = list(df_subset["clean_name"].dropna())
            name_counts = Counter(unique_names)
            sorted_names = [name for name, count in name_counts.most_common()]

            if sorted_names:
                network_name_short = f"{sorted_names[0]} Etc."
            else:
                network_name_short = f"Network {network_id} - {ntwk}"

            concatenated_names = " -- ".join(sorted_names[:3])
            concatenated_names += f" -- {ntwk} -- ({network_id})"

            # Handle case where there are no clean names
            df_networked.loc[
                df_networked[component_col_name] == ntwk, network_col_name
            ] = concatenated_names

            df_networked.loc[
                df_networked[component_col_name] == ntwk, f"{network_col_name}_short"
            ] = network_name_short

        return df_networked

    @classmethod
    def set_network_text(
            cls,
            gMatches: nx.Graph,
            df_process_results: pd.DataFrame,
            network_col_name: str,
            component_col_name: str
    ):
        df_networked = df_process_results.copy()
        df_networked[f"{network_col_name}_text"] = np.nan
        unique_networks = df_networked[component_col_name].unique()
        components = list(nx.connected_components(gMatches))

        for ntwk in unique_networks:
            if ntwk == None or np.isnan(ntwk): continue
            nodes = list(gMatches.subgraph(components[int(ntwk)]).nodes())
            edges = list(gMatches.subgraph(components[int(ntwk)]).edges())
            if len(edges) == 0:
                network_text = json.dumps(nodes)
            else:
                network_text = json.dumps(edges)
            df_networked.loc[
                df_networked[component_col_name] == ntwk, f"{network_col_name}_text"
            ] = network_text

        return df_networked

    # @classmethod
    # def build_connected_component_map(
    #         cls,
    #         gMatches: nx.Graph,
    #         df_process_input: pd.DataFrame,
    #         params: NetworkMatchParams
    # ) -> dict[str, int]:
    #     """Builds and returns dictionary mapping connected components to names and addresses."""
    #     # get unique values for each column used to generate nodes and edges
    #     taxpayer_names_set = list(set(df_process_input[params["taxpayer_name_col"]].dropna().unique()))
    #     fuzzy_matches_set = list(set(df_process_input[params["string_match_name"]].dropna().unique()))
    #     clean_addresses_set = list(set(
    #             list(set(df_process_input["GCD_FORMATTED_MATCH"].dropna().unique())) +
    #             list(set(df_process_input["GCD_FORMATTED_ADDRESS_ADDRESS_1_MATCH"].dropna().unique()))  #+
    #             # list(set(df_unique["GCD_FORMATTED_ADDRESS_ADDRESS_2_MATCH"].dropna().unique())) +
    #             # list(set(df_unique["GCD_FORMATTED_ADDRESS_ADDRESS_3_MATCH"].dropna().unique()))
    #     ))
    #     entity_names_set = list(set(df_process_input[params["entity_name_col"]].dropna().unique()))
    #     # loop through connected to components to associate component IDs
    #     # assign components to unique values from each column used to generate nodes and edges
    #     component_map: dict[str, int] = {}
    #     for i, connections in enumerate(list(nx.connected_components(gMatches))):
    #         for component in connections:
    #             if component in taxpayer_names_set:
    #                 component_map[component] = i
    #             elif component in fuzzy_matches_set:
    #                 component_map[component] = i
    #             elif component in clean_addresses_set:
    #                 component_map[component] = i
    #             elif component in entity_names_set:
    #                 component_map[component] = i
    #
    #     return component_map

    @classmethod
    def string_match_network_graph(
            cls,
            df_process_input: pd.DataFrame,
            df_matches: pd.DataFrame,
            match_count: int,
            name_address_column: str
    ) -> pd.DataFrame:
        """Generates network graph for string match results. Outputs dataframe containing network graph results."""
        # generate network graph
        t.print_with_dots("Building network graph for string match results")
        gMatches = nx.Graph()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            "•",
            TextColumn("[bold cyan]{task.fields[processed]}/{task.total} records"),
        ) as progress:
            task = progress.add_task(
                f"[yellow]Building nodes and edges...",
                total=len(df_matches),
                processed=0,
            )
            processed_count = 0
            for i, row in df_matches.iterrows():
                cls.build_edge(gMatches, row["original_doc"], row["matched_doc"])
                processed_count += 1
                progress.update(
                    task,
                    advance=1,
                    processed=processed_count,
                    description=f"[yellow]Processing record {processed_count}/{len(df_matches)}"
                )

        # loop through each connected component
        t.print_with_dots("Building dictionary to map taxpayer records to connected components")
        component_map = {}
        component_map_names = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            "•",
            TextColumn("[bold cyan]{task.fields[processed]}/{task.total} records"),
        ) as progress:
            task = progress.add_task(
                f"[yellow]Building dictionary...",
                total=len(df_matches),
                processed=0,
            )
            processed_count = 0
            for i, connections in enumerate(list(nx.connected_components(gMatches))):
                # pull out name with the shortest length as representative "canonical" name for network
                shortest = min(connections, key=len)
                # store key/value pair for original name and new name in dictionary
                for component in connections:
                    component_map[component] = shortest
                shortest_two = sorted(connections, key=len)[:3]
                shortest_names = []
                for name in shortest_two:
                    name_addr_split = name.split("-")
                    shortest_names.append(name_addr_split[0].strip())
                # concatenate the two shortest names with " -- " as the separator
                canonical_name = ' -- '.join(shortest_names)
                # store key/value pair for original name and new name in dictionary
                for component in connections:
                    component_map_names[component] = f"{canonical_name} -- {i}"
            processed_count += 1
            progress.update(
                task,
                advance=1,
                processed=processed_count,
                description=f"[yellow]Processing record {processed_count}/{len(df_matches)}"
            )

        t.print_with_dots("Adding components to taxpayer records")
        # add new column for landlord network name
        df_matches["fuzzy_match_name"] = df_matches["original_doc"].apply(lambda x: component_map[x])  # this is likely the redundant column
        df_matches["fuzzy_match_combo"] = df_matches["original_doc"].apply(lambda x: component_map_names[x])

        # merge clean name and clean address columns based on the simplified, calculated network name
        # df_matches = pd.merge(df_matches, df_filtered[['CLEAN_NAME', 'CLEAN_ADDRESS']], how='left', left_on='FUZZY_MATCH_NAME', right_on='CLEAN_NAME')

        # merge clean name and clean address columns based on the raw NameAddress string concatenation
        # df_matches = pd.merge(df_matches, df_filtered[['CLEAN_NAME', 'CLEAN_ADDRESS', 'NAME_ADDRESS']], how='left', left_on='FUZZY_MATCH_NAME', right_on='NAME_ADDRESS')

        # remove redundant columns
        df_matches = df_ops.combine_columns_parallel(df_matches)

        # the "clean names" here are the
        # df_matches.rename(columns={'CLEAN_NAME':'FUZZY_NAME', 'CLEAN_ADDRESS':'FUZZY_ADDRESS'}, inplace=True)

        # Keep good matches and join back to data
        # df_matches.drop_duplicates(subset=['FUZZY_NAME', 'FUZZY_ADDRESS', 'ORIGINAL_DOC'], inplace=True)
        t.print_with_dots("Merging string matching components into taxpayer records")
        df_filtered = pd.merge(
            df_process_input,
            df_matches[["original_doc", "fuzzy_match_combo"]],
            how="left",
            left_on=name_address_column,
            right_on="original_doc"
        )

        # fill in empty rows with the name of their corresponding CleanName and CleanAddress values from the new dataframe
        # df_filtered['FUZZY_NAME'].fillna(df_filtered['CLEAN_NAME'], inplace=True)
        # df_filtered['FUZZY_ADDRESS'].fillna(df_filtered['CLEAN_ADDRESS'], inplace=True)
        df_filtered = df_filtered.rename(columns={"fuzzy_match_combo": f"string_matched_name_{match_count+1}"})
        console.print("String match data merged ✅")
        return df_filtered


class NetworkMatchGraph(NetworkMatchBase):
    """Class for all network-related functions that return graph objects."""
    pass


class NetworkMatchDF(NetworkMatchBase):
    """Class for all network-related functions that return dataframes."""
    pass


class NetworkMatchNodesEdges(NetworkMatchBase):
    """Class for all network-related functions that create, manipulate and return nodes and edges of graph objects."""
    pass
