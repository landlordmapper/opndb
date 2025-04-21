import gc
from pprint import pprint

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
    def check_address(
        cls,
        address: str | None,
        is_validated: bool,  # COLUMN
        is_researched: bool,  # COLUMN
        exclude_address: bool,  # COLUMN
        is_org_address: bool,  # COLUMN
        include_unvalidated: bool,  # PARAM
        include_unresearched: bool,  # PARAM
        include_orgs: bool,  # PARAM
    ) -> bool:
        """
        Returns "True" if the address SHOULD be included in the network analysis, and False if it should be ignored.
        """
        if not address or pd.isna(address):
            return False

        if not include_unvalidated and not is_validated:
            return False

        if exclude_address:
            return False
        else:
            if include_unresearched:
                return True
            else:
                if not is_researched:
                    return False
                else:
                    if not include_orgs:
                        return True
                    else:
                        if is_org_address:
                            return False
                        else:
                            return True

        # if not address or pd.isna(address):
        #     return False
        # if is_validated:
        #     if is_researched:
        #         if not exclude_address:
        #             if not is_org_address:
        #                 return True
        #             else:
        #                 return include_orgs
        #         else:
        #             return False  # exclude address from network graph by default
        #     else:
        #         return include_unresearched
        # else:
        #     if include_unvalidated:
        #         return include_unresearched  # if it's not validated, is_researched will ALWAYS be false
        #     else:
        #         return False

    @classmethod
    def set_matching_address(cls, row):
        if pd.isna(row["raw_address_v"]) or row["raw_address_v"].strip() == "":
            return row["raw_address"]  # todo: run cleaned addresses through validator and change this to "clean_address_v" throughout the code where needed
        else:
            return row["raw_address_v"]


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
        """
        Executes string matching for documents passed as parameters. Returns dataframe of 'good matches', i.e. matches
        that fall within the threshold set in the params object.
        """
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
        t.print_with_dots("Executing query")
        query_matrix = messy_tf_idf_matrix
        nbrs = index.knnQueryBatch(
            query_matrix,
            k=params["query_batch_opts"]["K"],
            num_threads=params["query_batch_opts"]["num_threads"]
        )
        mts =[]
        for i in range(len(nbrs)):
            original_nm = query_docs[i]
            for row in list(range(len(nbrs[i][0]))):
                try:
                    matched_nm = ref_docs[nbrs[i][0][row]]
                    conf = abs(nbrs[i][1][row])
                except:
                    matched_nm = "no match found"
                    conf = None
                mts.append([original_nm, matched_nm, conf])
        df_matches = pd.DataFrame(mts,columns=["original_doc", "matched_doc", "conf"])
        df_matches["ldist"] = df_matches[["matched_doc", "original_doc"]].apply(
            lambda x: lev.distance(x[0], x[1]), axis=1
        )
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
    @classmethod
    def process_row_network(
        cls,
        g: nx.Graph,
        row: pd.Series,
        params: NetworkMatchParams,
    ) -> None:
        """
        Adds nodes and edges for taxpayer name and taxpayer address. If both add_name and add_address evaluate to true,
        add them as an edge. If address
        """

        name: str = row[params["taxpayer_name_col"]]
        address: str = row["match_address_t"]
        add_name: bool = not row["exclude_name"]
        add_address: bool = cls.check_address(
            address,
            row["is_validated_t"],
            row["is_researched_t"],
            row["exclude_address_t"],
            row["is_org_address_t"],
            params["include_unvalidated"],
            params["include_unresearched"],
            params["include_orgs"],
        )

        if add_name and add_address:
            g.add_edge(name, address)
        elif add_name:
            g.add_node(name)
        elif add_address:
            g.add_node(address)
        else:
            return

    @classmethod
    def process_row_network_string_match(
        cls,
        g: nx.Graph,
        row: pd.Series,
        params: NetworkMatchParams,
    ) -> None:
        """
        Addes nodes and edges for string match results. Adds string match name to name and address depending on whether
        add_name and add_address evaluate to True.
        """

        name: str = row[params["taxpayer_name_col"]]
        address: str = row["match_address_t"]
        string_match: str = row[params["string_match_name"]]

        add_name: bool = not row["exclude_name"]
        add_address: bool = cls.check_address(
            address,
            row["is_validated_t"],
            row["is_researched_t"],
            row["exclude_address_t"],
            row["is_org_address_t"],
            params["include_unvalidated"],
            params["include_unresearched"],
            params["include_orgs"],
        )

        if add_name:
            g.add_edge(name, string_match)
        if add_address:
            g.add_edge(address, string_match)

    @classmethod
    def process_row_network_entity(
        cls,
        g: nx.Graph,
        row: pd.Series,
        params: NetworkMatchParams
    ) -> None:
        """
        Processes nodes and edges for entity addresses.
        """

        # todo: deal with this elsewhere
        entities_to_ignore: List[str] = ["CHICAGO TITLE LAND TRUST COMPANY", "CHICAGO TITLE LAND"]

        # add_name boolean not necessary - entities will NEVER be exclude_name == True
        entity_name: str = row[params["taxpayer_name_col"]]
        entity_addresses = [
            row["entity_address_1"],
            row["entity_address_2"],
            row["entity_address_3"],
        ]

        for i, address in enumerate(entity_addresses):
            add_address: bool = cls.check_address(
                address,
                row[f"is_validated_e{i+1}"],
                row[f"is_researched_e{i + 1}"],
                row[f"exclude_address_e{i+1}"],
                row[f"is_org_address_e{i+1}"],
                params["include_unvalidated"],
                params["include_unresearched"],
                params["include_orgs"],
            )
            if add_address and entity_name not in entities_to_ignore:
                g.add_edge(entity_name, address)

    @classmethod
    def taxpayers_network(
        cls,
        df_taxpayers: pd.DataFrame,
        params: NetworkMatchParams
    ) -> nx.Graph:
        """
        Generates and NetworkX graph object containing nodes and edges for rental dataset based on parameters. Returns
        graph object and dataframe with associated component IDs.
        """
        gMatches = nx.Graph()

        t.print_with_dots("Adding taxpayer names and addresses to network graph object")
        with t.create_progress_bar(
            "[yellow]Processing taxpayer records...", len(df_taxpayers)
        )[0] as progress:
            task = progress.tasks[0]
            processed_count = 0
            for i, row in df_taxpayers.iterrows():
                cls.process_row_network(gMatches, row, params)
                processed_count += 1
                progress.update(
                    task.id,
                    advance=1,
                    processed=processed_count,
                    description=f"[yellow]Processing taxpayer record {processed_count}/{len(df_taxpayers)}",
                )

        t.print_with_dots("Adding string matching results to network graph object")
        with t.create_progress_bar(
            "[yellow]Processing taxpayer records...", len(df_taxpayers)
        )[0] as progress:
            task = progress.tasks[0]
            processed_count = 0
            for i, row in df_taxpayers.iterrows():
                if pd.notnull(row[params["string_match_name"]]):
                    cls.process_row_network_string_match(gMatches, row, params)
                processed_count += 1
                progress.update(
                    task.id,
                    advance=1,
                    processed=processed_count,
                    description=f"[yellow]Processing taxpayer record {processed_count}/{len(df_taxpayers)}",
                )

        t.print_with_dots("Adding corporations and LLCs from taxpayer records to network graph object")
        with t.create_progress_bar(
            "[yellow]Processing taxpayer records...", len(df_taxpayers)
        )[0] as progress:
            task = progress.tasks[0]
            processed_count = 0
            for i, row in df_taxpayers.iterrows():
                if pd.notnull(row["entity_clean_name"]):
                    cls.process_row_network_entity(gMatches, row, params)
                processed_count += 1
                progress.update(
                    task.id,
                    advance=1,
                    processed=processed_count,
                    description=f"[yellow]Processing taxpayer record {processed_count}/{len(df_taxpayers)}",
                )

        # log progress
        console.print(
            {
                "CONNECTED COMPONENT COUNT (ENTITIES):": nx.number_connected_components(gMatches),
                "NODE COUNT:": nx.number_of_nodes(gMatches),
                "EDGE COUNT:": nx.number_of_edges(gMatches),
            }
        )

        return gMatches

    @classmethod
    def set_taxpayer_component(
        cls,
        network_id: int,
        df_taxpayers: pd.DataFrame,
        gMatches: nx.Graph,
        params: NetworkMatchParams
    ) -> pd.DataFrame:
        """
        Assigns an ID value to each taxpayer record representing the associated connected component from the network
        graph object generated from the taxpayers_network() function.
        """
        # get unique values for each column used to generate nodes and edges
        taxpayer_names_set = list(set(df_taxpayers[params["taxpayer_name_col"]].dropna().unique()))
        fuzzy_matches_set = list(set(df_taxpayers[params["string_match_name"]].dropna().unique()))
        clean_addresses_set = list(set(
            list(df_taxpayers["match_address_t"].dropna()) +
            list(df_taxpayers["match_address_e1"].dropna()) +
            list(df_taxpayers["match_address_e2"].dropna()) +
            list(df_taxpayers["match_address_e3"].dropna())
        ))
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
        df_taxpayers[f"final_component_{network_id}"] = df_taxpayers.apply(
            lambda row: cls.set_component(row, component_map, params), axis=1
        )
        return df_taxpayers

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
            row["match_address_t"],
            row[params["string_match_name"]],
            row["match_address_e1"],
            row["match_address_e2"],
            row["match_address_e3"],
        ]
        for key in keys_to_check:
            if key in component_map.keys():
                return component_map[key]
        # If no match is found, print debug info and return np.nan
        # print(f"KeyError for CleanName: {row['CLEAN_NAME']} and ADDRESS: {row['CLEAN_ADDRESS']}")
        return np.nan

    @classmethod
    def set_network_name(cls, network_id: int, df_taxpayers: pd.DataFrame) -> pd.DataFrame:
        """
        Sets unique name string for landlord network. Fetches the 3 most common taxpayer names associated with the
        network, concatenates them with ' -- ', and adds the network ID in parentheses at the end.
        """
        df_taxpayers[f"network_{network_id}"] = np.nan
        unique_networks = df_taxpayers[f"final_component_{network_id}"].unique()

        with t.create_progress_bar(
            "[yellow]Setting network names for taxpayer records...", len(unique_networks)
        )[0] as progress:
            task = progress.tasks[0]
            processed_count = 0
            for ntwk in unique_networks:
                df_subset = df_taxpayers[df_taxpayers[f"final_component_{network_id}"] == ntwk]
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
                df_taxpayers.loc[
                    df_taxpayers[f"final_component_{network_id}"] == ntwk, f"network_{network_id}"
                ] = concatenated_names

                df_taxpayers.loc[
                    df_taxpayers[f"final_component_{network_id}"] == ntwk, f"network_{network_id}_short"
                ] = network_name_short
                processed_count += 1
                progress.update(
                    task.id,
                    advance=1,
                    processed=processed_count,
                    description=f"[yellow]Processing network {processed_count}/{len(unique_networks)}",
                )

        return df_taxpayers

    @classmethod
    def set_network_text(cls, network_id: int, gMatches: nx.Graph, df_taxpayers: pd.DataFrame) -> pd.DataFrame:
        df_taxpayers[f"network_{network_id}_text"] = np.nan
        unique_networks = df_taxpayers[f"final_component_{network_id}"].dropna().unique()
        components = list(nx.connected_components(gMatches))
        for ntwk in unique_networks:
            nodes = list(gMatches.subgraph(components[int(ntwk)]).nodes())
            edges = list(gMatches.subgraph(components[int(ntwk)]).edges())
            if len(edges) == 0:
                network_text = json.dumps(nodes)
            else:
                network_text = json.dumps(edges)
            df_taxpayers.loc[
                df_taxpayers[f"final_component_{network_id}"] == ntwk, f"network_{network_id}_text"
            ] = network_text
        return df_taxpayers

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
    def string_match_network_graph(cls, df_matches: pd.DataFrame) -> pd.DataFrame:
        """Generates network graph for string match results. Outputs dataframe containing network graph results."""
        # generate network graph
        t.print_with_dots("Building network graph for string match results")
        gMatches = nx.Graph()
        for i, row in df_matches.iterrows():
            gMatches.add_edge(row["original_doc"], row["matched_doc"])

        # loop through each connected component
        t.print_with_dots("Building dictionary to map taxpayer records to connected components")
        component_map = {}
        component_map_names = {}
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

        t.print_with_dots("Adding components to taxpayer records")
        # add new column for landlord network name
        df_matches["fuzzy_match_name"] = df_matches["original_doc"].apply(lambda x: component_map[x])  # this is likely the redundant column
        df_matches["fuzzy_match_combo"] = df_matches["original_doc"].apply(lambda x: component_map_names[x])
        df_matches = df_ops.combine_columns_parallel(df_matches)
        return df_matches


class NetworkMatchGraph(NetworkMatchBase):
    """Class for all network-related functions that return graph objects."""
    pass


class NetworkMatchDF(NetworkMatchBase):
    """Class for all network-related functions that return dataframes."""
    pass


class NetworkMatchNodesEdges(NetworkMatchBase):
    """Class for all network-related functions that create, manipulate and return nodes and edges of graph objects."""
    pass
