from enum import IntEnum
from pathlib import Path
from typing import TypedDict, Literal, Any


FileExt = Literal["csv", "json", "geojson"]


class WorkflowStage(IntEnum):
    """Keeps track of workflow stages."""
    PRE = 0
    DATA_LOAD = 1
    DATA_CLEANING = 2
    ADDRESS_VALIDATION = 3
    NAME_ANALYSIS = 4
    ADDRESS_ANALYSIS = 5
    RENTAL_SUBSET = 6
    CLEAN_MERGE = 7
    STRING_MATCH = 8
    NETWORK_GRAPH = 9
    FINAL_OUTPUT = 10


class WorkflowConfigs(TypedDict, total=False):
    data_root: Path
    load_ext: FileExt
    prev_stage: WorkflowStage | None
    stage: WorkflowStage  # might be redundant if we have wkfl type
    wkfl_type: str  # todo: create mapper that maps human readable wkfl type to WorkflowStage number
    wkfl_type_addrs: str
    wkfl_type_string_match: str
    wkfl_type_ntwk: str
    accuracy: str
    geocodio_api_key: str


class CleaningColumnMap(TypedDict):
    name: dict[str, list[str]]
    address: dict[str, list[str]]
    accuracy: {
        str, dict[str, list[str]],
        str, dict[str, list[str]]
    }

class BooleanColumnMap(TypedDict):
    taxpayer_records: list[str]
    corps: list[str]
    llcs: list[str]

class CleanAddress(TypedDict, total=False):
    """
    Used to generate list of unique addresses from raw data. Handles situations in which there's only a single field
    for complete address AND in which the address is broken into street, city, state and zip columns
    """
    raw_address: str
    clean_address: str
    street: str | None
    city: str | None
    state: str | None
    zip: str | None
    is_pobox: bool


class GeocodioQueryObject(TypedDict):
    """Used to construct URL query parameters for geocodio API calls"""
    street: str
    street2: str | None
    city: str | None
    state: str | None
    postal_code: str | None
    country: str | None
    complete_address: str


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


class GeocodioResultFlat(TypedDict, total=False):
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
    clean_address: str
    is_pobox: bool

class GeocodioResultProcessed(TypedDict):
    clean_address: CleanAddress
    results: list[GeocodioResultFlat]
    results_parsed: list[GeocodioResultFlat] | None

class GeocodioResultFinal(TypedDict):
    clean_address: str
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
    is_pobox: bool

class GeocodioReturnObject(TypedDict, total=False):
    validated: list[GeocodioResultFinal]
    unvalidated: list[GeocodioResultFinal]
    failed: list[GeocodioResultFinal]

class NmslibOptions(TypedDict):
    method: str
    space: str
    data_type: Any

class QueryBatchOptions(TypedDict):
    num_threads: int
    K: int

class StringMatchParams(TypedDict):
    name_col: str | None
    match_threshold: int | float
    include_unvalidated: bool
    include_unresearched: bool
    include_orgs: bool
    nmslib_opts: NmslibOptions
    query_batch_opts: QueryBatchOptions

class StringMatchParamsMN(TypedDict):
    name_col: str | None
    match_threshold: int | float
    include_unvalidated: bool
    include_unresearched: bool
    include_orgs: bool
    include_missing_suites: bool
    include_problem_suites: bool
    address: str
    nmslib_opts: NmslibOptions
    query_batch_opts: QueryBatchOptions

class NetworkMatchParams(TypedDict):
    taxpayer_name_col: str
    include_unvalidated: bool
    include_unresearched: bool
    include_orgs: bool
    string_match_name: str
