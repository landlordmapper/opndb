import pandera as pa
from opndb.validator.df_model import OPNDFModel


class NetworkCalcs(OPNDFModel):
    network_id: str = pa.Field()
    taxpayer_name: str = pa.Field()
    entity_name: str = pa.Field()
    string_match: str = pa.Field()
    include_orgs: str = pa.Field()
    include_orgs_string: str = pa.Field()
    include_unresearched: str = pa.Field()
    include_unresearched_string: str = pa.Field()


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
    formatted_address_v: str = pa.Field()
    landlord_entity: str = pa.Field()


class Corps(OPNDFModel):
    file_number: str = pa.Field()
    status: str = pa.Field()
    date_incorporated: str | None = pa.Field()
    date_dissolved: str | None = pa.Field()
    raw_name: str = pa.Field()
    clean_name: str = pa.Field()
    president: str = pa.Field()
    president_address: str = pa.Field()
    president_address_v: str = pa.Field()
    secretary: str = pa.Field()
    secretary_address: str = pa.Field()
    secretary_address_v: str = pa.Field()


class LLCs(OPNDFModel):
    file_number: str = pa.Field()
    status: str = pa.Field()
    date_incorporated: str | None = pa.Field()
    date_dissolved: str | None = pa.Field()
    raw_name: str = pa.Field()
    clean_name: str = pa.Field()
    manager_member: str = pa.Field()
    manager_member_address: str = pa.Field()
    manager_member_address_v: str = pa.Field()
    agent: str = pa.Field()
    agent_address: str = pa.Field()
    agent_address_v: str = pa.Field()
    office_address: str = pa.Field()
    office_address_v: str = pa.Field()


class Networks(OPNDFModel):
    name: str = pa.Field()
    short_name: str = pa.Field()
    network_calc: str = pa.Field()
    nodes_edges: str = pa.Field()


class TaxpayerRecords(OPNDFModel):
    raw_name: str = pa.Field()
    clean_name: str = pa.Field()
    address: str = pa.Field()
    address_v: str = pa.Field()
    corp_llc_name: str = pa.Field()
    network_1: str = pa.Field()
    network_2: str = pa.Field()
    network_3: str = pa.Field()
    network_4: str = pa.Field()
    network_5: str = pa.Field()
    network_6: str = pa.Field()
