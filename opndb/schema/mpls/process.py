import re
from typing import Final
import pandera as pa
from opndb.validator.df_model import OPNDFModel

# todo: add pandas dtype objects & handle boolean columns properly

VALID_ZIP_CODE_REGEX: Final[re] = r"^\d{5}(-\d{4})?$"

class Properties(OPNDFModel):

    _OUT: list[str] = [
        "pin",
        "land_use",
        "building_use",
        "prop_type",
        "is_exempt",
        "is_homestead",
        "raw_name_address",
        "clean_name_address",
        "num_units"
    ]

    @classmethod
    def out(cls) -> list[str]:
        return cls._OUT


class PropertiesRentals(Properties):
    pass


class TaxpayerRecords(OPNDFModel):

    _OUT: list[str] = [
        "raw_name",
        "raw_name_2",
        "raw_street",
        "raw_city_state_zip",
        "raw_address",
        "raw_name_address",
        "clean_name",
        "clean_name_2",
        "clean_street",
        "clean_city",
        "clean_state",
        "clean_zip_code",
        "clean_city_state_zip",
        "clean_address",
        "clean_name_address",
    ]

    @classmethod
    def out(cls) -> list[str]:
        return cls._OUT


class TaxpayersBusMerged(TaxpayerRecords):
    pass


class TaxpayersAddrMerged(TaxpayersBusMerged):
    pass


class TaxpayersFixed(TaxpayersAddrMerged):
    pass


class TaxpayersSubsetted(OPNDFModel):
    pass


class TaxpayersPrepped(TaxpayersSubsetted):

    is_trust: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Trust?",
        description="Boolean representing whether or not a trust string pattern is identified in the cleaned taxpayer name"
    )
    is_person: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Person?",
        description="Boolean representing whether or not a person string pattern is identified in the cleaned taxpayer name"
    )
    is_org: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Org?",
        description="Boolean representing whether or not an organization name string pattern is identified in the cleaned taxpayer name"
    )
    is_llc: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is LLC?",
        description="Boolean representing whether or not an LLC string pattern is identified in the cleaned taxpayer name"
    )
    is_clean_match: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Clean Match?",
        description="Boolean representing whether or not the match was made on based on the clean taxpayer name"
    )
    is_core_match: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Core Match?",
        description="Boolean representing whether or not the match was made on based on the core taxpayer name"
    )
    is_string_match: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is String Match?",
        description="Boolean representing whether or not the match was made as a result of string matching workflow"
    )
    is_validated: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Validated Address? (Taxpayer)",
    )
    exclude_address: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Exclude Address (Taxpayer Mailing)",
    )
    is_researched: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Researched? (Taxpayer Mailing)",
    )
    is_org_address: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Landlord Org Address? (Taxpayer Mailing)",
    )
    is_missing_suite: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Missing Suite?",
    )
    is_problem_suite: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Problem Suite?",
    )
    is_realtor: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Realtor?",
    )


class TaxpayersStringMatched(OPNDFModel):
    pass


class TaxpayersNetworked(TaxpayersStringMatched):
    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    network_1: str = pa.Field(
        nullable=True,
        unique=True,
        title="Network 1",
        description="Unique identifier for network 1 results. Concatenation of 3 most commonly appearing taxpayer names in network, followed by the value of the connected component originating from the NetworkX graph object and the network match parameter combination in parenthesis."
    )
    network_1_short: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 1 Short Name",
        description="Short name for the network 1 results. Most commonly appearing taxpayer name followed by 'Etc.'"
    )
    network_1_text: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 1 Text",
        description="Nodes and edges data in text format for connected component calculated from network 1 results."
    )
    network_2: str = pa.Field(
        nullable=True,
        unique=True,
        title="Network 2",
        description="Unique identifier for network 2 results. Concatenation of 3 most commonly appearing taxpayer names in network, followed by the value of the connected component originating from the NetworkX graph object and the network match parameter combination in parenthesis."
    )
    network_2_short: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 2 Short Name",
        description="Short name for the network 2 results. Most commonly appearing taxpayer name followed by 'Etc.'"
    )
    network_2_text: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 2 Text",
        description="Nodes and edges data in text format for connected component calculated from network 2 results."
    )
    network_3: str = pa.Field(
        nullable=True,
        unique=True,
        title="Network 3",
        description="Unique identifier for network 3 results. Concatenation of 3 most commonly appearing taxpayer names in network, followed by the value of the connected component originating from the NetworkX graph object and the network match parameter combination in parenthesis."
    )
    network_3_short: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 3 Short Name",
        description="Short name for the network 3 results. Most commonly appearing taxpayer name followed by 'Etc.'"
    )
    network_3_text: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 3 Text",
        description="Nodes and edges data in text format for connected component calculated from network 3 results."
    )
    network_4: str = pa.Field(
        nullable=True,
        unique=True,
        title="Network 4",
        description="Unique identifier for network 4 results. Concatenation of 3 most commonly appearing taxpayer names in network, followed by the value of the connected component originating from the NetworkX graph object and the network match parameter combination in parenthesis."
    )
    network_4_short: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 4 Short Name",
        description="Short name for the network 4 results. Most commonly appearing taxpayer name followed by 'Etc.'"
    )
    network_4_text: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 4 Text",
        description="Nodes and edges data in text format for connected component calculated from network 4 results."
    )
    network_5: str = pa.Field(
        nullable=True,
        unique=True,
        title="Network 5",
        description="Unique identifier for network 5 results. Concatenation of 3 most commonly appearing taxpayer names in network, followed by the value of the connected component originating from the NetworkX graph object and the network match parameter combination in parenthesis."
    )
    network_5_short: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 5 Short Name",
        description="Short name for the network 5 results. Most commonly appearing taxpayer name followed by 'Etc.'"
    )
    network_5_text: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 5 Text",
        description="Nodes and edges data in text format for connected component calculated from network 5 results."
    )
    network_6: str = pa.Field(
        nullable=True,
        unique=True,
        title="Network 6",
        description="Unique identifier for network 6 results. Concatenation of 3 most commonly appearing taxpayer names in network, followed by the value of the connected component originating from the NetworkX graph object and the network match parameter combination in parenthesis."
    )
    network_6_short: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 6 Short Name",
        description="Short name for the network 6 results. Most commonly appearing taxpayer name followed by 'Etc.'"
    )
    network_6_text: str = pa.Field(
        nullable=True,
        unique=False,
        title="Network 6 Text",
        description="Nodes and edges data in text format for connected component calculated from network 6 results."
    )


class BusinessFilings(OPNDFModel):
    pass


class BusinessNamesAddrs(OPNDFModel):
    pass


class BusinessNamesAddrsMerged(BusinessNamesAddrs):
    pass


class BusinessNamesAddrsSubsetted(BusinessNamesAddrsMerged):
    is_validated: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Validated Address? (Taxpayer)",
    )
    exclude_address: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Exclude Address (Taxpayer Mailing)",
    )
    is_researched: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Researched? (Taxpayer Mailing)",
    )
    is_org_address: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Landlord Org Address? (Taxpayer Mailing)",
    )
    is_missing_suite: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Missing Suite?",
    )
    is_problem_suite: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Problem Suite?",
    )
    is_realtor: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Realtor?",
    )
