import re
from typing import Final
import pandera as pa
from opndb.validator.df_model import OPNDFModel

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


class TaxpayerRecordsMerged(TaxpayerRecords):
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


class TaxpayersFixed(TaxpayerRecordsMerged):
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
    core_name: str = pa.Field()
    is_trust: bool = pa.Field()
    is_person: bool = pa.Field()
    is_org: bool = pa.Field()
    is_llc: bool = pa.Field()
    entity_clean_name: str = pa.Field()
    entity_core_name: str = pa.Field()
    merge_address_1: str = pa.Field()
    merge_address_2: str = pa.Field()
    merge_address_3: str = pa.Field()
    is_clean_match: bool = pa.Field()
    is_core_match: bool = pa.Field()
    is_string_match: bool = pa.Field()


class TaxpayersStringMatched(TaxpayersPrepped):
    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    match_address: str = pa.Field()
    clean_name_address: str = pa.Field()
    core_name_address: str = pa.Field()
    include_address: bool = pa.Field()
    string_matched_name_1: str = pa.Field()
    string_matched_name_2: str = pa.Field()
    string_matched_name_3: str = pa.Field()


class TaxpayersNetworked(TaxpayersStringMatched):
    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    network_1: str = pa.Field()
    network_1_short: str = pa.Field()
    network_2: str = pa.Field()
    network_2_short: str = pa.Field()
    network_3: str = pa.Field()
    network_3_short: str = pa.Field()
    network_4: str = pa.Field()
    network_4_short: str = pa.Field()
    network_5: str = pa.Field()
    network_5_short: str = pa.Field()
    network_6: str = pa.Field()
    network_6_short: str = pa.Field()


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


# --------------------------
# ----OUTPUTTED DATASETS----
# --------------------------
# these will be the final outputted datasets to be stored in the s3 bucket and available to be pulled down and used
