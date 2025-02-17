from pathlib import Path
from typing import TypedDict, Literal, Any

from opndb.workflows.base import WorkflowStage


FileExt = Literal[".csv", ".json", ".geojson"]

class WorkflowConfigs(TypedDict):
    root: Path
    load_ext: FileExt
    prev_stage: WorkflowStage | None
    stage: WorkflowStage  # might be redundant if we have wkfl type
    wkfl_type: str  # todo: create mapper that maps human readable wkfl type to WorkflowStage number
    wkfl_type_addrs: str
    wkfl_type_string_match: str
    wkfl_type_ntwk: str
    accuracy: str

class CleaningColumnMap(TypedDict):
    name: dict[str, list[str]]
    address: dict[str, list[str]]
    accuracy: {
        str, dict[str, [list[str]]],
        str, dict[str, [list[str]]]
    }

class BooleanColumnMap(TypedDict):
    taxpayer_records: list[str]
    corps: list[str]
    llcs: list[str]

class RawAddress(TypedDict, total=False):
    """
    Used to generate list of unique addresses from raw data. Handles situations in which there's only a single field
    for complete address AND in which the address is broken into street, city, state and zip columns
    """
    complete_addr: str
    street: str | None
    city: str | None
    state: str | None
    zip: str | None


class GeocodioAddressComponents(TypedDict, total=False):
    number: str
    predirectional: str
    prefix: str
    street: str
    suffix: str
    postdirectional: str
    secondaryunit: str
    secondarynumber: str
    city: str
    county: str
    state: str
    zip: str
    country: str


class GeocodioLocation(TypedDict):
    lat: str
    long: str


class GeocodioResult(TypedDict, total=False):
    address_components: GeocodioAddressComponents
    formatted_address: str
    location: GeocodioLocation
    accuracy: int | float
    accuracy_type: str
    source: str


class GeocodioResponse(TypedDict):
    input: GeocodioAddressComponents
    results: list[GeocodioResult]


class GeocodioResultFlat(TypedDict):
    """Single object flattening geocodio result object by including moving lat, lng, accuracy and formatted_address as keys."""
    number: str
    predirectional: str
    prefix: str
    street: str
    suffix: str
    postdirectional: str
    secondaryunit: str
    secondarynumber: str
    city: str
    county: str
    state: str
    zip: str
    country: str
    lng: str
    lat: str
    accuracy: int | float
    formatted_address: str

class GeocodioResultProcessed(TypedDict):
    raw_addr: RawAddress
    results: list[GeocodioResultFlat]
    results_parsed: list[GeocodioResultFlat] | None

class NmslibOptions(TypedDict):
    method: str
    space: str
    data_type: Any

class QueryBatchOptions(TypedDict):
    num_threads: int
    K: int

class StringMatchParams(TypedDict):
    name_col: str
    match_threshold: int | float
    include_orgs: bool
    include_unresearched: bool
    nmslib_opts: NmslibOptions
    query_batch_opts: QueryBatchOptions

class NetworkMatchParams(TypedDict):
    taxpayer_name_col: str
    entity_name_col: str
    include_orgs: bool
    include_unresearched: bool
    string_match_name: str
