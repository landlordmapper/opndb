import numpy as np
import pandas as pd
import networkx as nx
from opndb.services.match import MatchBase, NetworkMatchBase
from opndb.types.base import NetworkMatchParams


def test_check_address():

    f = {
        "address": "123 Oak St, Springfield IL",
        "exclude_address": False,
        "include_unresearched": True,
        "is_researched": False,
        "include_orgs": True,
        "is_org_address": False,
    }

    result = MatchBase.check_address(
        f["address"],
        f["exclude_address"],
        f["include_unresearched"],
        f["is_researched"],
        f["include_orgs"],
        f["is_org_address"],
    )

    assert result == True


def test_network_generator():

    data = [
        {
            "clean_name": "Erick Larson",
            "entity_clean_name": np.nan,
            "string_matched_name_1": np.nan,
            "exclude_name": False,
            "match_address": "123 Oak St",
            "exclude_address_t": False,
            "is_validated_t": True,
            "is_researched_t": True,
            "is_org_address_t": False,
            "entity_address_1": np.nan,
            "exclude_address_e1": np.nan,
            "is_validated_e1": np.nan,
            "is_researched_e1": np.nan,
            "is_org_address_e1": np.nan,
            "entity_address_2": np.nan,
            "exclude_address_e2": np.nan,
            "is_validated_e2": np.nan,
            "is_researched_e2": np.nan,
            "is_org_address_e2": np.nan,
            "entity_address_3": np.nan,
            "exclude_address_e3": np.nan,
            "is_validated_e3": np.nan,
            "is_researched_e3": np.nan,
            "is_org_address_e3": np.nan
        },
        {
            "clean_name": "Top Props LLC",
            "entity_clean_name": "Top Props LLC",
            "string_matched_name_1": np.nan,
            "exclude_name": False,
            "match_address": "456 Maple St",
            "exclude_address_t": False,
            "is_validated_t": True,
            "is_researched_t": True,
            "is_org_address_t": True,
            "entity_address_1": "555 State St",
            "exclude_address_e1": False,
            "is_validated_e1": True,
            "is_researched_e1": True,
            "is_org_address_e1": True,
            "entity_address_2": "998 Jameston St",
            "exclude_address_e2": False,
            "is_validated_e2": True,
            "is_researched_e2": True,
            "is_org_address_e2": True,
            "entity_address_3": "322 Staple St",
            "exclude_address_e3": False,
            "is_validated_e3": True,
            "is_researched_e3": True,
            "is_org_address_e3": True
        },
        {
            "clean_name": "Real Management Corp",
            "entity_clean_name": "Real Management Corp",
            "string_matched_name_1": np.nan,
            "exclude_name": False,
            "match_address": "789 Diamond St",
            "exclude_address_t": False,
            "is_validated_t": True,
            "is_researched_t": True,
            "is_org_address_t": True,
            "entity_address_1": "928 River Rd",
            "exclude_address_e1": False,
            "is_validated_e1": True,
            "is_researched_e1": True,
            "is_org_address_e1": True,
            "entity_address_2": "523 Taylor Ave",
            "exclude_address_e2": False,
            "is_validated_e2": True,
            "is_researched_e2": True,
            "is_org_address_e2": True,
            "entity_address_3": "423 Angeles Ave",
            "exclude_address_e3": False,
            "is_validated_e3": True,
            "is_researched_e3": True,
            "is_org_address_e3": True
        }
    ]

    params: NetworkMatchParams = {
        "taxpayer_name_col": "clean_name",
        "include_orgs": False,
        "include_unresearched": False,
        "string_match_name": "string_matched_name_1",
    }

    df_taxpayers: pd.DataFrame = pd.DataFrame(data)
    g: nx.Graph = NetworkMatchBase.taxpayers_network(
        df_taxpayers,
        params
    )
    assert g is not None