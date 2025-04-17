import re
from typing import Final
import pandera as pa
from opndb.validator.df_model import OPNDFModel

# todo: add pandas dtype objects & handle boolean columns properly

VALID_ZIP_CODE_REGEX: Final[re] = r"^\d{5}(-\d{4})?$"

class Properties(OPNDFModel):
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _OUT: list[str] = ["pin", "raw_name_address", "class_code"]

    @classmethod
    def out(cls) -> list[str]:
        return cls._OUT

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    pin: str = pa.Field(
        nullable=False,
        unique=True,
        title="PIN",
        description="Unique tax identifier for the property",
    )
    raw_name_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Name Address",
        description="Concatenation of raw taxpayer name and address. Used as unique identifier for taxpayer records."
    )
    clean_name_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Name Address",
        description="Concatenation of clean taxpayer name and address. Can be used as unique identifier for taxpayer records."
    )
    class_code: str = pa.Field(
        nullable=False,
        title="Class Code",
        description="Municipal code indicating land use for the property, required for subsetting rental properties."
    )
    num_units: int | None = pa.Field(
        nullable=True,
        title="Number of Units",
        description="Number of rental apartment units in the property."
    )


class Corps(OPNDFModel):
    """
    Cleaned dataset for state-level registered corporations.
    """
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _VALIDATED_ADDRESS_MERGE: list[str] = [
        "raw_president_address",
        "raw_secretary_address",
    ]

    @classmethod
    def validated_address_merge(cls) -> list[str]:
        return cls._VALIDATED_ADDRESS_MERGE

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    file_number: str = pa.Field(
        nullable=False,
        unique=True,
        title="Corporation File Number",
        description="Unique identifier number assigned to corporation by secretary of state upon incorporation.",
    )
    status: str = pa.Field(
        nullable=False,
        unique=False,
        title="Status",
        description="Organization's status (active, inactive, involuntarily dissolved, etc.)",
    )
    raw_name: str = pa.Field(
        nullable=False,
        unique=True,
        title="Raw Raw Name",
        description="Corporation name exactly how it appears in the raw data"
    )
    raw_president_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw President Name",
        description="Corporation president name exactly how it appears in the raw data"
    )
    raw_president_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw President Address",
        description="Concatenated president address exactly how it appears in the raw data"
    )
    raw_secretary_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Secretary Name",
        description="Secretary name exactly how it appears in the raw data"
    )
    raw_secretary_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Secretary Address",
        description="Concatenated secretary address exactly how it appears in the raw data"
    )
    clean_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Name",
        description="Corporation name after strings cleaners have been applied"
    )
    clean_president_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean President Name",
        description="Corporation president name after strings cleaners have been applied"
    )
    clean_president_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean President Address",
        description="Concatenated president address after strings cleaners have been applied"
    )
    clean_secretary_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Secretary Name",
        description="Secretary name after strings cleaners have been applied"
    )
    clean_secretary_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Secretary Address",
        description="Concatenated secretary address after strings cleaners have been applied"
    )


class CorpsMerged(Corps):
    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    raw_president_address_v: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw President Address Validated",
        description="Complete validated mailing address for corporation president. Merged into dataset from validated address dataset."
    )  # todo: these will need to be clean_addresses in the future
    raw_secretary_address_v: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Secretary Address Validated",
        description="Complete validated mailing address for corporation secretary. Merged into dataset from validated address dataset."
    )  # these are only raw addresses because of Chicago's data


class LLCs(OPNDFModel):
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _VALIDATED_ADDRESS_MERGE: list[str] = [
        "raw_office_address",
        "raw_agent_address",
        "raw_manager_member_address",
    ]

    @classmethod
    def validated_address_merge(cls) -> list[str]:
        return cls._VALIDATED_ADDRESS_MERGE

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    file_number: str = pa.Field(
        nullable=False,
        unique=True,
        title="LLC File Number",
        description="Unique identifier number assigned to an LLC by secretary of state.",
    )
    status: str = pa.Field(
        nullable=False,
        unique=False,
        title="Status",
        description="Organization's status (active, inactive, involuntarily dissolved, etc.)",
    )
    raw_name: str = pa.Field(
        nullable=False,
        unique=True,
        title="LLC Raw Name",
        description="LLC name exactly how it appears in the raw data",
    )
    raw_manager_member_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Manager Member Name",
    )
    raw_manager_member_street: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Manager Member Street",
    )
    raw_manager_member_city: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Manager Member City",
    )
    raw_manager_member_zip: str = pa.Field(
        nullable=True,  # todo: thous should be False, only set it like this because of the Chicago raw data
        unique=False,
        title="Raw Manager Member Zip",
    )
    raw_manager_member_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Manager Member Address",
    )
    raw_agent_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Agent Name",
    )
    raw_agent_street: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Agent Street",
    )
    raw_agent_zip: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Agent Zip",
    )
    raw_agent_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Agent Address",
    )
    raw_office_street: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Office Street",
    )
    raw_office_city: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Office City",
    )
    raw_office_zip: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Office Zip",
    )
    raw_office_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Office Address",
    )
    clean_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="LLC Clean Name",
    )
    clean_manager_member_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Manager Member Name",
    )
    clean_manager_member_street: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Manager Member Street",
    )
    clean_manager_member_city: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Manager Member City",
    )
    clean_manager_member_zip: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Manager Member Zip",
    )
    clean_manager_member_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Manager Member Address",
    )
    clean_agent_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Agent Name",
    )
    clean_agent_street: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Agent Street",
    )
    clean_agent_zip: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Agent Zip",
    )
    clean_agent_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Agent Address",
    )
    clean_office_street: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Office Street",
    )
    clean_office_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Office City",
    )
    clean_office_zip: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Office Zip",
    )
    clean_office_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Office Address",
    )


class LLCsMerged(LLCs):
    raw_office_address_v: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Office Address Validated",
        description="Complete validated mailing address for LLC office. Merged into dataset from validated address dataset."
    )  # todo: these will need to be clean_addresses in the future
    raw_agent_address_v: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Agent Address Validated",
        description="Complete validated mailing address for LLC agent. Merged into dataset from validated address dataset."
    )
    raw_manager_member_address_v: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Manager Member Address Validated",
        description="Complete validated mailing address for LLC manager/member. Merged into dataset from validated address dataset."
    )


class TaxpayerRecords(OPNDFModel):
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _OUT: list[str] = [
        "raw_name",
        "raw_street",
        "raw_city",
        "raw_state",
        "raw_zip",
        "raw_address",
        "raw_name_address",
        "clean_name",
        "clean_street",
        "clean_city",
        "clean_state",
        "clean_zip",
        "clean_address",
    ]
    _UNVALIDATED_ADDR_COLS: list[str] = [
        "raw_street",
        "raw_city",
        "raw_state",
        "raw_zip",
        "raw_address",
        "clean_street",
        "clean_city",
        "clean_state",
        "clean_zip",
        "clean_address",
    ]
    _VALIDATED_ADDRESS_MERGE: list[str] = ["raw_address"]

    @classmethod
    def out(cls) -> list[str]:
        return cls._OUT

    @classmethod
    def unvalidated_addr_cols(cls) -> list[str]:
        return cls._UNVALIDATED_ADDR_COLS

    @classmethod
    def validated_address_merge(cls) -> list[str]:
        return cls._VALIDATED_ADDRESS_MERGE

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    raw_name: str = pa.Field(
        nullable=False,
        title="Raw Taxpayer Name",
        description="Taxpayer name EXACTLY how it appears in the raw data.",
    )
    raw_street: str = pa.Field(
        nullable=False,
        title="Raw Taxpayer Street",
        description="Taxpayer street address EXACTLY how it appears in the raw data."
    )
    raw_city: str = pa.Field(
        nullable=False,
        title="Raw Taxpayer City",
        description="Taxpayer city EXACTLY how it appears in the raw data."
    )
    raw_state: str = pa.Field(
        nullable=False,
        title="Raw Taxpayer State",
        description="Taxpayer state EXACTLY how it appears in the raw data."
    )
    raw_zip: str = pa.Field(
        nullable=False,
        title="Raw Taxpayer Zip",
        description="Taxpayer zip code EXACTLY how it appears in the raw data."
    )
    raw_address: str = pa.Field(
        nullable=False,
        title="Raw Taxpayer Address",
        description="Concatenation of raw taxpayer address components."
    )
    raw_name_address: str = pa.Field(
        nullable=False,
        unique=True,
        title="Raw Taxpayer Name+Address",
        description="Concatenation of raw taxpayer name and full address, to be used for identifying unique taxpayer records."
    )
    clean_name: str = pa.Field(
        nullable=False,
        title="Clean Taxpayer Name",
        description="Taxpayer name AFTER running it through the string cleaners."
    )
    clean_street: str = pa.Field(
        nullable=False,
        title="Clean Taxpayer Street",
        description="Taxpayer street address AFTER running it through the string cleaners."
    )
    clean_city: str = pa.Field(
        nullable=False,
        title="Clean Taxpayer City",
        description="Taxpayer city AFTER running it through the string cleaners."
    )
    clean_state: str = pa.Field(
        nullable=False,
        title="Clean Taxpayer State",
        description="Taxpayer state AFTER running it through the string cleaners."
    )
    clean_zip: str = pa.Field(
        nullable=False,
        title="Clean Taxpayer Zip",
        description="Taxpayer zip code AFTER running it through the string cleaners."
    )
    clean_address: str = pa.Field(
        nullable=False,
        title="Clean Taxpayer Address",
        description="Concatenation of clean taxpayer address components."
    )


class TaxpayersMerged(TaxpayerRecords):
    """
    Outputted taxpayer record dataset resulting from the AddressMerge workflow
    """
    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    clean_address_v: str = pa.Field(
        nullable=True,
        title="Validated Taxpayer Address",
        description="Validated clean taxpayer mailing address."
    )


class TaxpayersFixed(TaxpayersMerged):
    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    is_common_name: bool = pa.Field(
        nullable=False,
        title="Is Common Name?",
        description="Boolean column indicating whether or not the taxpayer name is identified as a 'common name' (ex: John Smith, Juan Garcia, etc.)."
    )
    is_landlord_org: bool = pa.Field(
        nullable=False,
        title="Is Landlord Org?",
        description="Boolean column indicating whether or not the validated taxpayer address is associated with a 'landlord organization' (property management company, wealth management company, realtor, etc.)."
    )


class TaxpayersSubsetted(TaxpayersFixed):
    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    is_rental: bool = pa.Field(
        nullable=False,
        title="Is Rental?",
        description="Boolean column indicating whether or not the taxpayer record is associated with a rental property."
    )


class TaxpayersPrepped(TaxpayersSubsetted):
    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    core_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Core Name",
        description="Clean taxpayer name after removing commonly appearing keywords to increase matches"
    )
    is_trust: str = pa.Field(
        nullable=False,
        unique=False,
        title="Is Trust?",
        description="Boolean representing whether or not a trust string pattern is identified in the cleaned taxpayer name"
    )
    is_person: str = pa.Field(
        nullable=False,
        unique=False,
        title="Is Person?",
        description="Boolean representing whether or not a person string pattern is identified in the cleaned taxpayer name"
    )
    is_org: str = pa.Field(
        nullable=False,
        unique=False,
        title="Is Org?",
        description="Boolean representing whether or not an organization name string pattern is identified in the cleaned taxpayer name"
    )
    is_llc: str = pa.Field(
        nullable=False,
        unique=False,
        title="Is LLC?",
        description="Boolean representing whether or not an LLC string pattern is identified in the cleaned taxpayer name"
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
    merge_address_1: str = pa.Field(
        nullable=True,
        unique=False,
        title="Merge Address 1",
        description="President address (corps) or office address (LLCs) to be used in matching. Either validated or unvalidated address depending on whether or not the raw address was successfully validated."
    )
    merge_address_2: str = pa.Field(
        nullable=True,
        unique=False,
        title="Merge Address 2",
        description="Secretary address (corps) or manager/member address (LLCs) to be used in matching. Either validated or unvalidated address depending on whether or not the raw address was successfully validated."
    )
    merge_address_3: str = pa.Field(
        nullable=True,
        unique=False,
        title="Merge Address 3",
        description="Agent address (LLCs) to be used in matching. Either validated or unvalidated address depending on whether or not the raw address was successfully validated."
    )
    is_clean_match: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Clean Match?",
        description="Boolean representing whether or not the match was made on based on the clean taxpayer name"
    )
    is_core_match: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Core Match?",
        description="Boolean representing whether or not the match was made on based on the core taxpayer name"
    )
    is_string_match: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is String Match?",
        description="Boolean representing whether or not the match was made as a result of string matching workflow"
    )


class TaxpayersStringMatched(TaxpayersPrepped):
    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    match_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Match Address",
        description="Taxpayer address to be used for string matching. Either the validated address (if successfully validated) or unvalidated cleaned address (if unsuccessfully validated)."
    )
    clean_name_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Clean Name Address",
        description="Concatenation of cleaned taxpayer name and address"
    )
    core_name_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Core Name Address",
        description="Concatenation of core taxpayer name and cleaned address"
    )
    include_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Include Address?",
        description="Boolean representing whether or not the address should be included in the matching process. Determined based on manual address research."
    )
    string_matched_name_1: str = pa.Field(
        nullable=True,
        unique=False,
        title="String Matched Name 1",
        description="Unique identifier for string match 1 results"
    )
    string_matched_name_2: str = pa.Field(
        nullable=True,
        unique=False,
        title="String Matched Name 2",
        description="Unique identifier for string match 2 results"
    )
    string_matched_name_3: str = pa.Field(
        nullable=True,
        unique=False,
        title="String Matched Name 3",
        description="Unique identifier for string match 3 results"
    )


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

class UnvalidatedAddrs(OPNDFModel):
    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    raw_street: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Street",
        description="Raw street address (street number, street name and secondary address information)",
    )
    raw_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw City",
    )
    raw_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw State",
    )
    raw_zip: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Zip Code",
    )
    raw_address: str = pa.Field(
        nullable=False,
        unique=True,
        title="Raw Address",
        description="Concatenated mailing address as appears in the raw data. Required regardless of whether the address is split into separate components (street, city, state, zip)."
    )
    clean_street: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Street",
        description="Clean street address (street number, street name and secondary address information)",
    )
    clean_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean City",
    )
    clean_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean State",
    )
    clean_zip: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Zip Code",
    )
    clean_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Address",
        description="Concatenated and cleaned full mailing address. Required regardless of whether the address is split into separate components (street, city, state, zip)."
    )
    addr_type: str = pa.Field(
        nullable=True,
        unique=False,
        title="Address Type",
    )
    origin: str = pa.Field(
        nullable=True,
        unique=False,
        title="Origin",
    )
    active: str = pa.Field(
        nullable=True,
        unique=False,
        title="Active?",
        description="Boolean column representing whether or not the corporate or LLC record from which the address originates is active or inactive."
    )


class UnvalidatedAddrsClean(UnvalidatedAddrs):
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _GEOCODIO_COLUMNS: list[str] = [
        "raw_address",
        "clean_address",
        "clean_street",
        "clean_city",
        "clean_state",
        "clean_zip",
        "is_pobox",
    ]

    @classmethod
    def geocodio_columns(cls) -> list[str]:
        return cls._GEOCODIO_COLUMNS

    is_pobox: str = pa.Field(
        nullable=False,
        unique=False,
        title="Is PO Box?",
        description="Boolean indicating whether the street address has been identified as a PO Box."
    )


class Geocodio(OPNDFModel):
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _VALIDATED_COLUMNS: list[str] = [
        "raw_address",
        "clean_address",
        "number",
        "predirectional",
        "prefix",
        "street",
        "suffix",
        "postdirectional",
        "secondaryunit",
        "secondarynumber",
        "city",
        "county",
        "state",
        "zip",
        "country",
        "lng",
        "lat",
        "accuracy",
        "formatted_address",
    ]

    @classmethod
    def validated_columns(cls) -> list[str]:
        return cls._VALIDATED_COLUMNS

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    raw_address: str = pa.Field(
        nullable=False,
        unique=True,
        title="Raw Address",
        description="Concatenated mailing address as appears in the raw data. Required regardless of whether the address is split into separate components (street, city, state, zip)."
    )
    clean_address: str = pa.Field(
        nullable=False,
        unique=False,  # todo: this should be true
        title="Clean Address",
        description="Concatenated and cleaned full mailing address. Required regardless of whether the address is split into separate components (street, city, state, zip)."
    )
    number: str = pa.Field(
        nullable=False,
        unique=False,
        title="Street Number",
    )
    predirectional: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Predirectional",
        description="Predirectional value for street address (Ex: 'N' in '123 N OAK ST')"
    )
    prefix: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Prefix",
        description="Prefix for street address (Ex: 'AVE' in '4754 AVE H')"
    )
    street: str = pa.Field(
        nullable=False,
        unique=False,
        title="Street Name",
    )
    suffix: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Suffix",
        description="Suffix for street address (Ex: 'ST' in '123 OAK ST')"
    )
    postdirectional: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Postdirectional",
        description="Postdirectional value for street address (Ex: 'N' in '123 OAK ST N')"
    )
    secondaryunit: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Secondary Unit",
        description="Descriptor for secondary address (Ex: 'SUITE' in 'SUITE 101')"
    )
    secondarynumber: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Secondary Number",
        description="Number component of the secondary address (Ex: '101' in 'SUITE 101')"
    )
    city: str = pa.Field(
        nullable=False,
        unique=False,
        title="City",
    )
    county: str = pa.Field(
        nullable=True,
        unique=False,
        title="County",
    )
    state: str = pa.Field(
        nullable=False,
        unique=False,
        title="State",
    )
    zip: str = pa.Field(
        nullable=False,
        unique=False,
        title="Zip Code",
    )
    country: str = pa.Field(
        nullable=False,
        unique=False,
        title="Country",
    )
    lng: str = pa.Field(
        nullable=True,
        unique=False,
        title="Longitude",
    )
    lat: str = pa.Field(
        nullable=True,
        unique=False,
        title="Latitude",
    )
    accuracy: str = pa.Field(
        nullable=True,
        unique=False,
        title="Accuracy",
        description="Accuracy score provided by Geocodio"
    )
    formatted_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Formatted Address",
        description="Concatenation of all address components into a single string."
    )
    is_pobox: str = pa.Field(
        nullable=False,
        unique=False,
        title="Is PO Box?",
        description="Boolean indicating whether the street address has been identified as a PO Box."
    )


class FixingAddrs(Geocodio):
    check_sec_num: str = pa.Field(
        nullable=False,
        unique=False,
        title="Check Secondary Number",
        description="Contains secondary number value detected by regex run on street address for rows whose validated address does NOT contain a secondary number. Used to manually fix validated addresses that incorrectly excluded secondary address information."
    )


class FixingTaxNames(OPNDFModel):
    raw_value: str = pa.Field(
        nullable=False,
        unique=True,
        title="Raw Value",
        description="Value of taxpayer name to be changed"
    )
    standardized_value: str = pa.Field(
        nullable=False,
        unique=False,
        title="Standardized Value",
        description="New value of taxpayer name to be changed"
    )


class AddressAnalysis(OPNDFModel):
    address: str = pa.Field(
        nullable=False,
        unique=True,
        title="Address",
        description="Validated address to be analyzed"
    )
    frequency: str = pa.Field(
        nullable=False,
        unique=False,
        title="Frequency",
        description="Number of times the validated address appears in either the taxpayer, corporate or LLC datasets"
    )
    value: str | None = pa.Field(
        title="Value",
        description="Column used to match researched addresses for Chicago. Should NOT be present in future iterations"
    )
    name: str = pa.Field(
        nullable=True,\
        unique=False,
        title="Name",
        description="Name of organization/entity associated with the address"
    )
    urls: str = pa.Field(
        nullable=True,
        unique=False,
        title="URLs",
        description="URLs associated with organization/entity associated with the address"
    )
    notes: str = pa.Field(
        nullable=True,
        unique=False,
        title="Notes",
        description="Any notes or additional considerations observed from researching address"
    )
    is_landlord_org: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Landlord Organization?",
        description="Boolean representing whether or not the entity/organization associated with the address is a property management company, real estate developer, real estate agency, investment or wealth management firm, or any other organization that can be held accountable for the conditions of a rental property. "
    )
    is_govt_agency: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Government Agency?",
        description="Boolean representing whether or not the address is associated with a government office"
    )
    is_lawfirm: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Lawfirm?",
        description="Boolean representing whether or not the entity/organization associated with the address is a lawfirm"
    )
    is_missing_suite: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Missing Suite?",
        description="Boolean representing whether or not the address is missing a suite number. Applies to addresses that point to office buildings, UPS stores, lock box services or other virtual mail services but that do NOT include a secondary address identifier."
    )
    is_financial_services: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Financial Services?",
        description="Boolean representing whether or not the entity/organization associated with the address is a financial services company (tax services, mortgage services, etc.)"
    )
    is_assoc_bus: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Associated Business?",
        description="Boolean representing whether or not the entity/organization associated with the address is a business that is unrelated to property management. This could either be the business of the property's 'true owner', or the owner of the property containing the business."
    )
    fix_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Fix Address?",
        description="Indicates whether or not the address needs to be fixed"
    )
    is_virtual_office_agent: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Virtual Office / Agent?",
        description="Boolean representing whether or not the entity/organization associated with the address is a virtual office or registered agent service."
    )
    yelp_urls: str = pa.Field(
        nullable=True,
        unique=False,
        title="Yelp URLs",
        description="Yelp URLs associated with organization/entity associated with the address"
    )
    is_nonprofit: str = pa.Field(
        nullable=True,
        unique=False,
        title="Boolean representing whether or not the entity/organization associated with the address is a nonprofit organization",
        description=""
    )
    google_urls: str = pa.Field(
        nullable=True,
        unique=False,
        title="Google URLs",
        description="Google URLs associated with organization/entity associated with the address"
    )
    is_ignore_misc: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Ignore Misc?",
        description=""
    )
    google_place_id: str = pa.Field(
        nullable=True,
        unique=False,
        title="Google Place ID",
    )
    is_researched: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Researched?",
        description="Boolean representing whether or not the address has been manually researched"
    )


class FrequentTaxNames(OPNDFModel):
    value: str = pa.Field(
        nullable=False,
        unique=True,
        title="Value",
        description="Value of taxpayer name to be analyzed"
    )
    frequency: str = pa.Field(
        nullable=False,
        unique=False,
        title="Frequency",
        description="Number of times the name appears in the taxpayer dataset"
    )
    is_common_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Common Name?",
        description="Boolean indicating whether or not the name is a common name and should therefore be excluded from the network graph generation."
    )
