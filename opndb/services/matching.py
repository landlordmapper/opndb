import json
import string
from datetime import datetime
from typing import List
from xml.etree.ElementInclude import include

import numpy as np
import word2number as w2n
import pandas as pd
import re
import time
from collections import Counter

import Levenshtein as lev
import networkx as nx
import nmslib
from sklearn.feature_extraction.text import TfidfVectorizer

from opndb.constants.columns import AddressAnalysis
from opndb.services.dataframe import DataFrameOpsBase as df_ops
from opndb.types.base import StringMatchParams


class MatchingBase:

    @classmethod
    def check_address(
            cls,
            address: str | None,
            df_analysis: pd.DataFrame,
            include_orgs: bool,
            include_unresearched: bool
    ) -> bool:
        """
        Returns "True" if the address SHOULD be included in the network analysis, and False if it should be ignored.
        """
        aa = AddressAnalysis
        if not address:
            return False
        # the address has NOT already been analyzed
        if address not in list(df_analysis[aa.ADDRESS].dropna().unique()):
            return include_unresearched
        # address has been analyzed - check if it should be ignored
        else:
            df_addr_analysis = df_analysis[df_analysis[aa.ADDRESS] == address]
            if df_addr_analysis[aa.IS_LANDLORD_ORG].eq("t").any():
                return include_orgs
            elif df_addr_analysis[aa.IS_LAWFIRM].eq("t").any():
                return False
            elif df_addr_analysis[aa.IS_MISSING_SUITE].eq("t").any():
                return False
            elif df_addr_analysis[aa.IS_FINANCIAL_SERVICES].eq("t").any():
                return False
            elif df_addr_analysis[aa.IS_VIRTUAL_OFFICE_AGENT].eq("t").any():
                return False
            elif df_addr_analysis[aa.FIX_ADDRESS].eq("t").any():
                return False
            elif df_addr_analysis[aa.IS_IGNORE_MISC].eq("t").any():
                return False
            else:
                return True

    @classmethod
    def set_matching_address(cls, row):
        # todo: fix these column names
        if pd.isna(row["GCD_FORMATTED_MATCH"]) or row["GCD_FORMATTED_MATCH"].strip() == "":
            return row["RAW_ADDRESS"]
        else:
            return row["GCD_FORMATTED_MATCH"]


class StringMatching(MatchingBase):

    @classmethod
    def concatenate_name_addr(cls, df: pd.DataFrame, name_addr_col: str, name_col: str, addr_col: str) -> pd.DataFrame:
        """Generates column with name and address concatenated. Used for string matching workflow."""
        df[name_addr_col] = df[name_col] + " - " + df[addr_col]
        return df

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
    def match_strings(cls, ref_docs: list[str], query_docs: list[str], params: StringMatchParams):

        # set up vectorizer and index for string matching
        vectorizer = TfidfVectorizer(min_df=1, analyzer=cls.ngrams)
        tf_idf_matrix = vectorizer.fit_transform(ref_docs)
        messy_tf_idf_matrix = vectorizer.transform(query_docs)
        data_matrix = tf_idf_matrix
        index = nmslib.init(
            method=params["nmslib_opts"]["method"],
            space=params["nmslib_opts"]["space"],
            data_type=params["nmslib_opts"]["data_type"]
        )
        index.addDataPointBatch(data_matrix)
        index.createIndex()

        # execute query and store good matches
        query_matrix = messy_tf_idf_matrix
        nbrs = index.knnQueryBatch(
            query_matrix,
            k=params["query_batch_opts"]["K"],
            num_threads=params["query_batch_opts"]["num_threads"]
        )
        mts =[]
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

        df_matches = pd.DataFrame(mts,columns=["original_doc", "matched_doc", "cont"])
        df_matches["ldist"] = df_matches[["matched_doc", "original_doc"]].apply(lambda x: lev.distance(x[0], x[1]), axis=1)
        df_matches["conf1"] = 1 - df_matches["conf"]

        if params["match_threshold"] is not None:
            df_good_matches = df_matches[(df_matches["ldist"] > 0) & (df_matches["conf1"] > params["match_threshold"]) & (df_matches["conf1"] < 1)].sort_values(by=["conf1"])
            return df_good_matches  # does this return duplicates? like if there are multiple matches above the threshhold it needs to pick the highest one, NOT include all of them
        else:
            return df_matches


class NetworkMatching(MatchingBase):

    @classmethod
    def build_edge(cls, g, node_a, node_b, common_names=None, common_addrs=None):
        if (common_names is None or node_a not in common_names) and (common_addrs is None or node_b not in common_addrs):
            g.add_edge(node_a, node_b)

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
        gMatches = nx.Graph()
        for i, row in df_matches.iterrows():  # todo: add progress bar
            cls.build_edge(gMatches, row["original_doc"], row["matched_doc"])

        # loop through each connected component
        combosgMatches = {}
        combosgMatchesNames = {}
        for i, connections in enumerate(list(nx.connected_components(gMatches))):
            # pull out name with the shortest length as representative "canonical" name for network
            shortest = min(connections, key=len)
            # store key/value pair for original name and new name in dictionary
            for component in connections:
                combosgMatches[component] = shortest
            shortest_two = sorted(connections, key=len)[:3]
            shortest_names = []
            for name in shortest_two:
                name_addr_split = name.split("-")
                shortest_names.append(name_addr_split[0].strip())
            # concatenate the two shortest names with " -- " as the separator
            canonical_name = ' -- '.join(shortest_names)
            # store key/value pair for original name and new name in dictionary
            for component in connections:
                combosgMatchesNames[component] = f"{canonical_name} -- {i}"

        # add new column for landlord network name
        df_matches["fuzzy_match_name"] = df_matches["original_doc"].apply(lambda x: combosgMatches[x])  # this is likely the redundant column
        df_matches["fuzzy_match_combo"] = df_matches["original_doc"].apply(lambda x: combosgMatchesNames[x])

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
        df_filtered = pd.merge(df_process_input, df_matches[["original_doc", "fuzzy_match_combo"]], how="left", left_on=name_address_column, right_on="original_doc")

        # fill in empty rows with the name of their corresponding CleanName and CleanAddress values from the new dataframe
        # df_filtered['FUZZY_NAME'].fillna(df_filtered['CLEAN_NAME'], inplace=True)
        # df_filtered['FUZZY_ADDRESS'].fillna(df_filtered['CLEAN_ADDRESS'], inplace=True)
        df_filtered = df_filtered.rename(columns={"fuzzy_match_combo": f"string_matched_name_{match_count+1}"})

        return df_filtered

