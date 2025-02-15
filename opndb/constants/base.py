from abc import abstractmethod
from pathlib import Path

from opndb.types.base import WorkflowConfigs, FileExt

DATA_ROOT: Path = Path("")  # todo: change all DATA_ROOT references to configs["root"]
GEOCODIO_URL = "https://api.geocod.io/v1.7/geocode?api_key="

DIRECTIONS: dict[str, str] = {
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
    "NORTHEAST": "NE",
    "SOUTHEAST": "SE",
    "NORTHWEST": "NW",
    "SOUTHWEST": "SW",
}

STREET_SUFFIXES: dict[str, str] = {
    "STREET": "ST",
    "STREE": "ST",
    "AVENUE": "AVE",
    "AVENU": "AVE",
    "AV":"AVE",
    "LANE": "LN",
    "DRIVE": "DR",
    "BOULEVARD": "BLVD",
    "BULEVARD": "BLVD",
    "BOULEVAR": "BLVD",
    "ROAD":"RD",
    "COURT":"CT",
    "PLACE": "PL",
    "WAY": "WAY"
}

SECONDARY_KEYWORDS: set[str] = {
    "#",
    "UNIT",
    "FLOOR",
    "FL",
    "SUITE",
    "STE",
    "APT",
    "ROOM"
}

UNIQUE_KEYS = [
    'CIR ', 'APARTMENTS ', 'SERVICES ', 'INVESTMENTS ', 'HOLDINGS ',
    'LN ', 'COMPANY ', 'AUTHORITY ', 'INC ', 'FORECLOSURE ',
    'ESTABLISHED ', 'CONDO TRUST ', 'COOPERATIVE ', 'PARTNERS ', 'CR ',
    'PARTNERSHIP ', 'GROUP ', 'ASSOCIATION ', 'TRUSTEES ', 'TRUST ',
    'PROPERTIES ', 'MANAGEMENT ', 'SQUARE ', 'MANAGERS ', 'EXCHANGE ',
    'REAL ESTATE ', 'DEVELOPMENT ', 'REDEVELOPMENT ', 'MORTGAGE ',
    'RESIDENTIAL ', 'REALTY TRUST ', 'CORPORATION ', 'LIMITED ', 'LLC ',
    'ORGANIZATION ', 'REALTY ', 'PRT ', 'VENTURE ', 'RENTAL ', 'UNION ',
    'CONDO '
]