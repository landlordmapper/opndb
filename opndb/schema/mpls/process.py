import re
from typing import Final
import pandera as pa
from opndb.validator.df_model import OPNDFModel


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

    pin: str = pa.Field(
        nullable=False,
        unique=True,
        title="PIN",
        description="Unique tax identifier for property",
    )
    land_use: str = pa.Field(
        nullable=True,
        unique=False,
        title="Land Use",
        description="",
    )
    building_use: str = pa.Field(
        nullable=True,
        unique=False,
        title="Buildings Use",
        description="",
    )
    prop_type: str = pa.Field(
        nullable=True,
        unique=False,
        title="Property Type",
        description="",
    )
    is_exempt: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Exempt?",
        description="",
    )
    is_homestead: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Homestead?",
        description="",
    )
    raw_name_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Name Address",
        description="Concatenation of raw taxpayer name and full address, used for linking to taxpayer records",
    )
    clean_name_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Name Address",
        description="Concatenation of clean taxpayer name and full address, used for linking to taxpayer records",
    )
    num_units: str = pa.Field(
        nullable=True,
        unique=False,
        title="Number of Units",
        description="",
    )


class PropertiesRentals(Properties):
    is_unit_gte_1: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Unit Greater Than 1?",
        description="Boolean column used to subset rental properties",
    )


class PropertyAddresses(OPNDFModel):
    pin: str = pa.Field()
    number: str = pa.Field()
    number_suffix: str = pa.Field()
    predirectional: str = pa.Field()
    street: str = pa.Field()
    suffix: str = pa.Field()
    postdirectional: str = pa.Field()
    secondaryunit: str = pa.Field()
    secondarynumber: str = pa.Field()
    city: str = pa.Field()
    state: str = pa.Field()
    zip_code: str = pa.Field()
    formatted_address: str = pa.Field()


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

    raw_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Taxpayer Name (Raw)",
        description="",
    )
    raw_name_2: str = pa.Field(
        nullable=True,
        unique=False,
        title="Secondary Taxpayer Name (Raw)",
        description="",
    )
    raw_street: str = pa.Field(
        nullable=False,
        unique=False,
        title="Taxpayer Street (Raw)",
        description="",
    )
    raw_city_state_zip: str = pa.Field(
        nullable=False,
        unique=False,
        title="Taxpayer City State Zip (Raw)",
        description="",
    )
    raw_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Full Taxpayer Address (Raw)",
        description="",
    )
    raw_name_address: str = pa.Field(
        nullable=False,
        unique=True,
        title="Taxpayer Name + Address (Raw)",
        description="Concatenation of raw taxpayer name and full address, used for linking to taxpayer records",
    )
    clean_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Taxpayer Name (Clean)",
        description="",
    )
    clean_name_2: str = pa.Field(
        nullable=True,
        unique=False,
        title="Secondary Taxpayer Name (Clean)",
        description="",
    )
    clean_street: str = pa.Field(
        nullable=False,
        unique=False,
        title="Taxpayer Street (Clean)",
        description="",
    )
    clean_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="Taxpayer City (Clean)",
        description="",
    )
    clean_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="Taxpayer State (Clean)",
        description="",
    )
    clean_zip_code: str = pa.Field(
        nullable=True,
        unique=False,
        title="Taxpayer Zip Code (Clean)",
        description="",
    )
    clean_city_state_zip: str = pa.Field(
        nullable=False,
        unique=False,
        title="Taxpayer City State Zip (Clean)",
        description="",
    )
    clean_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Full Taxpayer Address (Clean)",
        description="",
    )
    clean_name_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Taxpayer Name + Address (Clean)",
        description="",
    )


class TaxpayersBusMerged(TaxpayerRecords):
    core_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Core Name",
        description="Clean taxpayer name after removing commonly appearing keywords to increase matches"
    )
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
    uid: str = pa.Field(
        nullable=True,
        unique=False,
        title="UID",
        description="Unique identifier for MNSOS business record matched with taxpayer name",
    )
    entity_clean_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Entity Clean Name",
        description="Clean name of corporation or LLC that was matched with the taxpayer record. Will be null if no match was found."
    )
    entity_core_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Entity Core Name",
        description="Core name of corporation or LLC that was matched with the taxpayer record. Will be null if no match was found."
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


class TaxpayersAddrMerged(TaxpayersBusMerged):
    clean_address_v1: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Address v1",
        description="Complete validated address",
    )
    clean_address_v2: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Address v2",
        description="Validated address with suite numbers removed",
    )
    clean_address_v3: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Address v3",
        description="Validated address with pre- and post-directionals removed",
    )
    clean_address_v4: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Address v4",
        description="Validated addresses with suite numbers, pre- and post-directionals removed",
    )


class TaxpayersFixed(TaxpayersAddrMerged):
    exclude_name: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Exclude Name?",
        description="Boolean representing whether the taxpayer name should be left out of matching processes"
    )


class TaxpayersSubsetted(TaxpayersFixed):
    pass


class TaxpayersPrepped(TaxpayersSubsetted):
    match_address_v1: str = pa.Field(
        nullable=False,
        unique=False,
        title="Match Address v1",
        description="v1 address to be used in matching processes",
    )
    match_address_v2: str = pa.Field(
        nullable=False,
        unique=False,
        title="Match Address v2",
        description="v2 address to be used in matching processes",
    )
    match_address_v3: str = pa.Field(
        nullable=False,
        unique=False,
        title="Match Address v3",
        description="v3 address to be used in matching processes",
    )
    match_address_v4: str = pa.Field(
        nullable=False,
        unique=False,
        title="Match Address v4",
        description="v4 address to be used in matching processes",
    )
    clean_name_address_v1: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Name + Address v1",
        description="Concatenation of clean taxpayer name and v1 address for use in string matching process",
    )
    core_name_address_v1: str = pa.Field(
        nullable=True,
        unique=False,
        title="Core Name + Address v1",
        description="Concatenation of core taxpayer name and v1 address for use in string matching process",
    )
    clean_name_address_v2: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Name + Address v1",
        description="Concatenation of clean taxpayer name and v2 address for use in string matching process",
    )
    core_name_address_v2: str = pa.Field(
        nullable=True,
        unique=False,
        title="Core Name + Address v2",
        description="Concatenation of core taxpayer name and v2 address for use in string matching process",
    )
    clean_name_address_v3: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Name + Address v3",
        description="Concatenation of clean taxpayer name and v3 address for use in string matching process",
    )
    core_name_address_v3: str = pa.Field(
        nullable=True,
        unique=False,
        title="Core Name + Address v3",
        description="Concatenation of core taxpayer name and v3 address for use in string matching process",
    )
    clean_name_address_v4: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Name + Address v4",
        description="Concatenation of clean taxpayer name and v4 address for use in string matching process",
    )
    core_name_address_v4: str = pa.Field(
        nullable=True,
        unique=False,
        title="Core Name + Address v4",
        description="Concatenation of core taxpayer name and v4 address for use in string matching process",
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


class TaxpayersStringMatched(TaxpayersPrepped):
    # string_matched_name_1: str = pa.Field(
    #     nullable=True,
    #     unique=False,
    #     title="String Matched Name 1",
    #     description="Unique identifier for string match 1 results"
    # )
    pass


class TaxpayersNetworked(TaxpayersStringMatched):
    # network_1: str = pa.Field(
    #     nullable=True,
    #     unique=True,
    #     title="Network 1",
    #     description="Unique identifier for network 1 results. Concatenation of 3 most commonly appearing taxpayer names in network, followed by the value of the connected component originating from the NetworkX graph object and the network match parameter combination in parenthesis."
    # )
    # network_1_short: str = pa.Field(
    #     nullable=True,
    #     unique=False,
    #     title="Network 1 Short Name",
    #     description="Short name for the network 1 results. Most commonly appearing taxpayer name followed by 'Etc.'"
    # )
    # network_1_text: str = pa.Field(
    #     nullable=True,
    #     unique=False,
    #     title="Network 1 Text",
    #     description="Nodes and edges data in text format for connected component calculated from network 1 results."
    # )
    pass


class BusinessFilings(OPNDFModel):
    uid: str = pa.Field(
        nullable=False,
        unique=True,
        title="UID",
        description="Unique identifier for MNSOS business records.",
    )
    status: str = pa.Field(
        nullable=False,
        unique=False,
        title="Status",
    )
    raw_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Name",
        description="Name of business entity as registered with the state of Minnesota before cleaning",
    )
    clean_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Name",
        description="Name of business entity as registered with the state of Minnesota after cleaning",
    )
    filing_date: str = pa.Field(
        nullable=True,
        unique=False,
        title="Filing Date",
    )
    expiration_date: str = pa.Field(
        nullable=True,
        unique=False,
        title="Expiration Date",
        description="",
    )
    home_jurisdiction: str = pa.Field(
        nullable=True,
        unique=False,
        title="Home Jurisdiction",
        description="State in which the business entity is based",
    )
    home_business_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Home Business Name",
        description="Name of the business as registered in its home jurisdiction",
    )
    is_llc_non_profit: bool = pa.Field(
        nullable=False,
        title="Is LLC Non-Profit?",
    )
    is_lllp: bool = pa.Field(
        nullable=False,
        title="Is LLLP?",
    )
    is_professional: bool = pa.Field(
        nullable=False,
        title="Is Professional?",
        description="",
    )


class BusinessNamesAddrs(OPNDFModel):
    uid: str = pa.Field(
        nullable=False,
        unique=False,
        title="UID",
        description="Unique identifier for MNSOS business records.",
    )
    name_type: str = pa.Field(
        nullable=True,
        unique=False,
        title="Name Type",
        description="Party name classification according to the MN Secretary of State",
    )
    address_type: str = pa.Field(
        nullable=True,
        unique=False,
        title="Address Type",
        description="Address classification according to the MN Secretary of State",
    )
    raw_party_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Party Name (Raw)",
        description=""
    )
    raw_street_1: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street 1 (Raw)",
        description=""
    )
    raw_street_2: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street 2 (Raw)",
        description=""
    )
    raw_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="City (Raw)",
        description=""
    )
    raw_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="State (Raw)",
        description=""
    )
    raw_zip_code: str = pa.Field(
        nullable=True,
        unique=False,
        title="Zip Code (Raw)",
        description=""
    )
    raw_zip_code_ext: str = pa.Field(
        nullable=True,
        unique=False,
        title="Zip Code Extension (Raw)",
        description=""
    )
    raw_country: str = pa.Field(
        nullable=True,
        unique=False,
        title="Country (Raw)",
        description=""
    )
    raw_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Full Address (Raw)",
        description=""
    )
    clean_party_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Party Name (Clean)",
        description=""
    )
    clean_street_1: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street 1 (Clean)",
        description=""
    )
    clean_street_2: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street 2 (Clean)",
        description=""
    )
    clean_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="City (Clean)",
        description=""
    )
    clean_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="State (Clean)",
        description=""
    )
    clean_zip_code: str = pa.Field(
        nullable=True,
        unique=False,
        title="Zip Code (Clean)",
        description=""
    )
    clean_zip_code_ext: str = pa.Field(
        nullable=True,
        unique=False,
        title="Zip Code Extension (Clean)",
        description=""
    )
    clean_country: str = pa.Field(
        nullable=True,
        unique=False,
        title="Country (Clean)",
        description=""
    )
    clean_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Full Address (Clean)",
        description=""
    )
    is_incomplete_address: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Incomplete Address?",
        description="Boolean indicating whether or not the address is missing key components (street & zip code)",
    )


class BusinessNamesAddrsMerged(BusinessNamesAddrs):
    clean_address_v1: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Address v1",
        description="Complete validated address",
    )
    clean_address_v2: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Address v2",
        description="Validated address with suite numbers removed",
    )
    clean_address_v3: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Address v3",
        description="Validated address with pre- and post-directionals removed",
    )
    clean_address_v4: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Address v4",
        description="Validated addresses with suite numbers, pre- and post-directionals removed",
    )


class BusinessNamesAddrsSubsetted(BusinessNamesAddrsMerged):
    match_address_v1: str = pa.Field(
        nullable=False,
        unique=False,
        title="Match Address v1",
        description="v1 address to be used in matching processes",
    )
    match_address_v2: str = pa.Field(
        nullable=False,
        unique=False,
        title="Match Address v2",
        description="v2 address to be used in matching processes",
    )
    match_address_v3: str = pa.Field(
        nullable=False,
        unique=False,
        title="Match Address v3",
        description="v3 address to be used in matching processes",
    )
    match_address_v4: str = pa.Field(
        nullable=False,
        unique=False,
        title="Match Address v4",
        description="v4 address to be used in matching processes",
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
