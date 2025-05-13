import pandera as pa
from opndb.validator.df_model import OPNDFModel


class NetworkCalcs(OPNDFModel):
    network_id: str = pa.Field()
    taxpayer_name: str = pa.Field()
    entity_name: str = pa.Field()
    string_match: str = pa.Field()
    include_orgs: bool = pa.Field()
    include_orgs_string: bool = pa.Field()
    include_unresearched: bool = pa.Field()
    include_missing_suite: bool = pa.Field()
    include_problem_suite: bool = pa.Field()
    address_suffix: str = pa.Field()
    include_unresearched_string: bool = pa.Field()
    include_missing_suite_string: bool = pa.Field()
    include_problem_suite_string: bool = pa.Field()
    address_suffix_string: str = pa.Field()


class EntityTypes(OPNDFModel):
    name: str = pa.Field()
    description: str = pa.Field()


class Entities(OPNDFModel):
    name: str = pa.Field()
    urls: str = pa.Field()
    yelp_urls: str = pa.Field()
    google_urls: str = pa.Field()
    google_place_id: str = pa.Field()
    entity_type: str = pa.Field()


class ValidatedAddresses(OPNDFModel):
    number: str = pa.Field()
    predirectional: str = pa.Field()
    prefix: str = pa.Field()
    street: str = pa.Field()
    suffix: str = pa.Field()
    postdirectional: str = pa.Field()
    secondaryunit: str = pa.Field()
    secondarynumber: str = pa.Field()
    city: str = pa.Field()
    county: str = pa.Field()
    state: str = pa.Field()
    zip: str = pa.Field()
    country: str = pa.Field()
    lng: str = pa.Field()
    lat: str = pa.Field()
    accuracy: str = pa.Field()
    formatted_address: str = pa.Field()
    landlord_entity: str = pa.Field()


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
    raw_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Party Name (Raw)",
        description=""
    )
    clean_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Party Name (Clean)",
        description=""
    )
    address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Full Address",
        description=""
    )
    address_v: str = pa.Field(
        nullable=True,
        unique=False,
        title="Full Address (Validated)",
        description=""
    )


class Networks(OPNDFModel):
    name: str = pa.Field()
    short_name: str = pa.Field()
    network_calc: str = pa.Field()
    nodes_edges: str = pa.Field()


class TaxpayerRecords(OPNDFModel):
    raw_name: str = pa.Field()
    raw_name_2: str = pa.Field()
    clean_name: str = pa.Field()
    clean_name_2: str = pa.Field()
    address: str = pa.Field()
    address_v: str = pa.Field()
    entity_uid: str = pa.Field()
    network_1: str = pa.Field()
    network_2: str = pa.Field()
    network_3: str = pa.Field()
    network_4: str = pa.Field()
    network_5: str = pa.Field()
    network_6: str = pa.Field()
