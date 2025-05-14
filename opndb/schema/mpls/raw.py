from typing import Any

import numpy as np
import pandera as pa
from opndb.validator.df_model import OPNDFModel


class PropsTaxpayers(OPNDFModel):

    _RAW: list[str] = [
        "tax_name",
        "tax_name_2",
        "tax_street",
        "tax_city_state_zip",
        "tax_address",
    ]
    _CLEAN_RENAME_MAP: dict[str, Any] = {
        "tax_name": "clean_name",
        "tax_name_2": "clean_name_2",
        "tax_street": "clean_street",
        "tax_city_state_zip": "clean_city_state_zip",
    }
    _NAME_ADDRESS_CONCAT_MAP: dict[str, Any] = {  # same as base props taxpayers
        "raw": {
            "name_addr": "raw_name_address",
            "name": "raw_name",
            "name_2": "raw_name_2",
            "addr": "raw_address"
        },
        "clean": {
            "name_addr": "clean_name_address",
            "name": "clean_name",
            "name_2": "clean_name_2",
            "addr": "clean_address"
        }
    }
    _BASIC_CLEAN: list[str] = [
        "clean_name",
        "clean_name_2",
        "clean_street",
        "clean_city_state_zip",
    ]
    _NAME_CLEAN: list[str] = ["clean_name", "clean_name_2"]
    _ADDRESS_CLEAN: dict[str, Any] = {
        "street": ["clean_street"],
        "zip": ["clean_zip"],
    }

    @classmethod
    def raw(cls) -> list[str]:
        return cls._RAW

    @classmethod
    def clean_rename_map(cls) -> dict[str, Any]:
        return cls._CLEAN_RENAME_MAP

    @classmethod
    def name_address_concat_map(cls) -> dict[str, Any]:
        return cls._NAME_ADDRESS_CONCAT_MAP

    @classmethod
    def basic_clean(cls) -> list[str]:
        return cls._BASIC_CLEAN

    @classmethod
    def name_clean(cls) -> list[str]:
        return cls._NAME_CLEAN

    @classmethod
    def address_clean(cls) -> dict[str, Any]:
        return cls._ADDRESS_CLEAN

    pin: str = pa.Field(
        nullable=False,
        unique=True,
        title="PIN",
        description="Unique tax identifier for property",
    )
    tax_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Tax Name",
        description="Taxpayer name",
    )
    tax_name_2: str = pa.Field(
        nullable=True,
        unique=False,
        title="Tax Name 2",
        description="Secondary taxpayer name",
    )
    tax_street: str = pa.Field(
        nullable=False,
        unique=False,
        title="Tax Street",
        description="Taxpayer street address (street name, number, secondary unit and pre/post directionals)",
    )
    tax_city_state_zip: str = pa.Field(
        nullable=False,
        unique=False,
        title="Tax City State Zip",
        description="City, state and zip code of taxpayer mailing address",
    )
    municipality: str = pa.Field(
        nullable=False,
        unique=False,
        title="Municipality",
        description="Name of municipality in Hennepin county where the property is located",
    )
    tax_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Tax Address",
        description="Concatenated raw taxpayer address",
    )
    land_use: str = pa.Field(
        nullable=True,
        unique=False,
        title="Land Use",
        description="Land use categorization assigned by Hennepin county authorities",
    )
    building_use: str = pa.Field(
        nullable=True,
        unique=False,
        title="Building Use",
        description="Building use categorization assigned by Hennepin county authorities",
    )
    prop_type: str = pa.Field(
        nullable=True,
        unique=False,
        title="Property Type",
        description="Property type assigned by the city of Minneapolis",
    )
    is_exempt: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Exempt?",
    )
    is_homestead: str = pa.Field(
        nullable=True,
        unique=False,
        title="Is Homestead?",
    )
    num_units: str = pa.Field(
        nullable=True,
        unique=False,
        title="Number of Units",
        description="Number of units associated with the property",
    )


class BusinessRecordsBase(OPNDFModel):

    _MN_STATE_FIXER: dict[str, str] = {
        "": np.nan,
        "CHOOSE ONE": np.nan,
        "NONE": np.nan,
        "SELECT ONE": np.nan,
        "SELECT STATE": np.nan,
        "STATE": np.nan,
        "CHOOSE A STATE": np.nan,
        "SELECT": np.nan,
        "CLICK HERE": np.nan,
        "/TX": "TX",
        "0": np.nan,
        "00": np.nan,
        "000": np.nan,
        "0000": np.nan,
        "00000": np.nan,
        "00000MN": "MN",
        "0000MN": "MN",
        "00MN": "MN",
        "0HIOMN": "OH",
        "0MN": "MN",
        "ALABAMA MN": "AL",
        "ALABAMA REPUBLIC": "AL",
        "ALABAMA STATE": "AL",
        "ALASKA AK": "AK",
        "ALEXANDRA CT": "",
        "GEORGIA": "GA",
        "TEXAS": "TX",
        "ARIZON": "AZ",
        "ARIZONA AZ": "AZ",
        "ARIZONA REPUBLIC": "AZ",
        "ARIZONA STATE": "AZ",
        "ARIZONA STATE REPUBLIC": "AZ",
        "ARK": "AR",
        "ARKANSA": "AR",
        "ARKANSAS AR": "AR",
        "ARKANSAS N": "AR",
        "ARKANSAS REPUBLIC": "AR",
        "ARKASNAS": "AR",
        "AZ - ARIZONA": "AZ",
        "AZ-ARIZONA": "AZ",
        "CA - CALIFORNIA": "CA",
        "CAIFORNIA": "CA",
        "CAILFORINA": "CA",
        "CAILFORNIA": "CA",
        "CALFIORNIA": "CA",
        "CALFORNIA": "CA",
        "CALI": "CA",
        "CALIF": "CA",
        "CALIFO": "CA",
        "CALIFON": "CA",
        "CALIFONIA": "CA",
        "CALIFORINA": "CA",
        "CALIFORNA": "CA",
        "CALIFORNAI": "CA",
        "CALIFORNI": "CA",
        "CALIFORNIA CA": "CA",
        "CALIFORNIA MN": "CA",
        "CALIFORNIA REPUBLIC": "CA",
        "CALIFORNIA REPUBLIC /UNITED STATES OF AMERICA": "CA",
        "CALIFORNIA REPUBLIC REPUBLIC": "CA",
        "CALIFORNIA REPUBLIC TERRITORY": "CA",
        "CALIFORNIA STATE": "CA",
        "CALIFORNIA TERRITORY": "CA",
        "CALIFORNIAMN": "CA",
        "CALIFORNIAN": "CA",
        "CALIFORRNIA": "CA",
        "CALIRFORNIA": "CA",
        "CALIUFORNIA": "CA",
        "CHOOSE": np.nan,
        "CO COLORADO": "CO",
        "CO - COLORADO": "CO",
        "COLORADO CO": "CO",
        "COLORADO REPUBLIC NON-DOMESTIC W/O THE U S": "CO",
        "COLORADO STATE": "CO",
        "COLORADO STATE THE REPUBLIC": "CO",
        "CONECTICUT": "CT",
        "CONN": "CT",
        "CONNECT": "CT",
        "CONNECTICUT CT": "CT",
        "CONNECTICUT REPUBLIC": "CT",
        "CONNECTTICUT REPUBLIC": "CT",
        "CONNTICUTMN": "CT",
        "COUNTRY CODE": np.nan,
        "DE - DELAWARE": "DE",
        "DEFAULT": np.nan,
        "DELALWARE": "DE",
        "DELAWARE DE": "DE",
        "DELAWARE REPUBLIC": "DE",
        "DELAWAREW": "DE",
        "DISTRICT OF COLOMBIA": "DC",
        "DISTRICT OF COLUMBIA": "DC",
        "DISTRICT OF COLUMBIA DC": "DC",
        "E290C817-5237-415E-99AF-ACC644F8361C": np.nan,
        "ENNESSEE": "TN",
        "FL - FLORIDA": "FL",
        "FL STATES": "FL",
        "FLMN": "FL",
        "FLO": "FL",
        "FLOIDA": "FL",
        "FLORDA": "FL",
        "FLORDIA": "FL",
        "FLORI DA": "FL",
        "FLORIA": "FL",
        "FLORID": "FL",
        "FLORIDA FL": "FL",
        "FLORIDA REPUBLIC": "FL",
        "FLORIDA REPUBLIC POSTAL CODE EXCEPTED NEAR": "FL",
        "FLORIDA REPUBLIC WITHOUT THE UNITED STATES": "FL",
        "FLORIDA REPUBLICMN": "FL",
        "FLORIDA RPUBLIC": "FL",
        "FLORIDA STATE": "FL",
        "FLORIDA UNION MEMBER STATE": "FL",
        "FLORIDA ZIP CODE EXCEPTED NEAR POSTAL CODE": "FL",
        "FLORIDAMN": "FL",
        "FLORIDAN": "FL",
        "FLROIDA": "FL",
        "FLÓRIDA": "FL",
        "FORIDA": "FL",
        "GA - GEORGIA": "GA",
        "GA MN": "GA",
        "GAMN": "GA",
        "GEAORGIA": "GA",
        "GEORGA": "GA",
        "GEORGAI": "GA",
        "GEORGIA GA": "GA",
        "GEORGIA MN": "GA",
        "GEORGIA N": "GA",
        "GEORGIA STATE": "GA",
        "GEORGIA STATE REPUBLIC": "GA",
        "GEORGIAMN": "GA",
        "GEORGIE": "GA",
        "GEORIGA": "GA",
        "GEROGIA": "GA",
        "HAWAII HI": "",
        "HENNEPIN": "MN",
        "HENNEPIN COUNTY": "MN",
        "HENNEPIN-MINNESOTA": "MN",
        "HENNIPEN": "MN",
        "I A": "IA",
        "IA - IOWA": "IA",
        "IA MN": "IA",
        "IA-IOWA": "IA",
        "IAMN": "IA",
        "ID - IDAHO": "ID",
        "IIL": "IL",
        "IIN": "IN",
        "IL - ILLINOIS": "IL",
        "ILIINOIS": "IL",
        "ILINOIS": "IL",
        "ILL": "IL",
        "ILLINIOS": "IL",
        "ILLINOI": "IL",
        "ILLINOI NEAR": "IL",
        "ILLINOIS IL": "IL",
        "ILLINOIS REPUBLIC": "IL",
        "ILLIONIS": "IL",
        "ILLIONOIS": "IL",
        "ILLLINOIS": "IL",
        "ILLMILLN": "IL",
        "ILLNOIS": "IL",
        "ILMN": "IL",
        "IN - INDIANA": "IN",
        "IN MINNESOTA REPUBLIC": "MN",
        "IND": "IN",
        "INDIANA IN": "IN",
        "INDIANA REPUBLIC": "IN",
        "INDIANIA": "IN",
        "INMN": "IN",
        "INN": "IN",
        "INNESOTA": "MN",
        "IOWA IA": "IA",
        "IOWA 51103MN": "IA",
        "IOWAMN": "IA",
        "KANASAS": "KS",
        "KANSAS KS": "KS",
        "KANSAS REPUBLIC": "KS",
        "KENTUCKEY": "KY",
        "KENTUCKY KY": "KY",
        "KENTUCY": "KY",
        "LLINOIS": "IL",
        "LOUISANA": "LA",
        "LOUISIANA LA": "LA",
        "LOUISIANA REPUBLIC": "LA",
        "LOUISIANA STATE": "LA",
        "LOUISISNA": "LA",
        "LOUSIANA": "LA",
        "M MINNESOTAN": "MN",
        "M MMNM": "MN",
        "M MN": "MN",
        "M MNN": "MN",
        "M N": "MN",
        "M NCN": "MN",
        "M0": "MO",
        "M56N": "MN",
        "MA - MASSACHUSETTS": "MA",
        "MAARYLAND": "MD",
        "MARYLAND MD": "MD",
        "MARYLAND MN": "MD",
        "MARYLAND REPUBLIC": "MD",
        "MARYLAND REUBLIC": "MD",
        "MASS": "MA",
        "MASSACHUCETTES": "MA",
        "MASSACHUESETTS STATE REPUBLIC": "MA",
        "MASSACHUSETS REPUBLIC": "MA",
        "MASSACHUSETTS DE JURE": "MA",
        "MASSACHUSETTS MA": "MA",
        "MASSACHUSETTS REPUBLIC": "MA",
        "MASSACHUSETTSMN": "MA",
        "MASSACHUSSETS": "MA",
        "MASSACHUSSETTS": "MA",
        "MASSACHUTTS": "MA",
        "MCALIFORNIA": "CA",
        "MD - MARYLAND": "MD",
        "MI - MICHIGAN": "MI",
        "MI MICHIGAN": "MI",
        "MI MN": "MI",
        "MICH": "MI",
        "MICHGAN": "MI",
        "MICHIGAN MI": "MI",
        "MICHIGAN REPUBLIC": "MI",
        "MICHIGAN/REPUBLIC": "MI",
        "MICHIGANMN": "MI",
        "MIMN": "MN",
        "MIMN MN": "MN",
        "MINEESOTA": "MN",
        "MINESOTA": "MN",
        "MINMESOTA": "MN",
        "MINMN": "MN",
        "MINN": "MN",
        "MINN MN": "MN",
        "MINNE": "MN",
        "MINNEAOTA": "MN",
        "MINNEAOTS": "MN",
        "MINNEAPOLIS": "MN",
        "MINNEAPOLIS MN": "MN",
        "MINNEAPOLIS N": "MN",
        "MINNEASOTA": "MN",
        "MINNEASOTA REPUBLIC": "MN",
        "MINNEDOTA": "MN",
        "MINNEMNMN": "MN",
        "MINNEOSTA": "MN",
        "MINNEOTA": "MN",
        "MINNES": "MN",
        "MINNES MN": "MN",
        "MINNESO": "MN",
        "MINNESOA": "MN",
        "MINNESOAT": "MN",
        "MINNESOT": "MN",
        "MINNESOTA MN": "MN",
        "MINNESOTA MN MN": "MN",
        "MINNESOTA REPUBLIC": "MN",
        "MINNESOTA USA": "MN",
        "MINNESOTA - MN": "MN",
        "MINNESOTA - SALESPERSON": "MN",
        "MINNESOTA - USA": "MN",
        "MINNESOTA INNESOTA": "MN",
        "MINNESOTA MN US": "MN",
        "MINNESOTA MNMN": "MN",
        "MINNESOTA N": "MN",
        "MINNESOTA REPUBLIC MEAR": "MN",
        "MINNESOTA REPUBLIC NEAR": "MN",
        "MINNESOTA REPUBLIC NEAR ZIP": "MN",
        "MINNESOTA REPUBLIC STATE OF THE UNION": "MN",
        "MINNESOTA REPUBLIC UNION STATE": "MN",
        "MINNESOTA TERRITORY": "MN",
        "MINNESOTA – MN": "MN",
        "MINNESOTA-MN": "MN",
        "MINNESOTAA": "MN",
        "MINNESOTAMINNESOTA": "MN",
        "MINNESOTAMMN": "MN",
        "MINNESOTAMN": "MN",
        "MINNESOTAN": "MN",
        "MINNESOTMEAR REPUBLIC": "MN",
        "MINNESOTO": "MN",
        "MINNESOTS": "MN",
        "MINNESPTA": "MN",
        "MINNESSOTA": "MN",
        "MINNESTOA": "MN",
        "MINNESTOTA": "MN",
        "MINNMN": "MN",
        "MINNNESOTA": "MN",
        "MINNSOTA": "MN",
        "MISISSIPPI": "MS",
        "MISSISSI": "MS",
        "MISSISSIPPI STATE": "MS",
        "MISSOUIR": "MO",
        "MISSOURI MO": "MO",
        "MISSOURI REPUBLIC": "MO",
        "MISSOURI STATE": "MO",
        "MISSOURI STATEMN": "MO",
        "MISSOURI-REPUBLIC": "MO",
        "MISSOURIMN": "MO",
        "MISSOUURI": "MO",
        "MISSSISSIPPI": "MS",
        "MISSSOURI": "MO",
        "MMASSACHUSETTS": "MA",
        "MMINNESOTA": "MN",
        "MMINNESOTA N": "MN",
        "MMINNESOTAN": "MN",
        "MMNINNESOTA": "MN",
        "MMNMMN": "MN",
        "MMNMN": "MN",
        "MMNN": "MN",
        "MN MN": "MN",
        "MN MINNESOTA": "MN",
        "MN - MINNESOTA": "MN",
        "MN EXT COUNTRY": "MN",
        "MN PA": "PA",
        "MN - US": "MN",
        "MN -- MINNESOTA": "MN",
        "MN DON T ASK ME AGAIN ON THIS DEVICE DON T ASK ME": "MN",
        "MN I MN": "MN",
        "MN MMN": "MN",
        "MN MN MN": "MN",
        "MN MNMN": "MN",
        "MN N": "MN",
        "MN UNITED STATES TRUE": "MN",
        "MN US": "MN",
        "MN USA": "MN",
        "MN- MINNESOTA": "MN",
        "MN-MINNESOTA": "MN",
        "MN253": "MN",
        "MN4": "MN",
        "MN5": "MN",
        "MNCOLORADO": "CO",
        "MNEVADAN": "NV",
        "MNFLORIDA": "FL",
        "MNFLROIDA": "FL",
        "MNGEORGIA": "GA",
        "MNINNESOTA": "MN",
        "MNMI": "MI",
        "MNMICHIGAN": "MI",
        "MNMICHIGAN REPUBLIC": "MI",
        "MNMINNESOTA": "MN",
        "MNMM": "MN",
        "MNMMN": "MN",
        "MNMMN MN": "MN",
        "MNMMNN": "MN",
        "MNMN": "MN",
        "MNMN MN": "MN",
        "MNMNMN": "MN",
        "MNMNN": "MN",
        "MNN": "MN",
        "MNN55": "MN",
        "MNNEW JERSEY": "NJ",
        "MNNM": "MN",
        "MNNMN": "MN",
        "MNNN": "MN",
        "MNNNESOTA MN": "MN",
        "MNNV": "NV",
        "MNTX": "TX",
        "MNWASHINGTON": "WA",
        "MNWI": "WI",
        "MNWY": "WY",
        "MONTANA MT": "MT",
        "MONTANTA": "MT",
        "MPENNSYLVANIAN": "PA",
        "MPLS": "MN",
        "MRAYLAND": "MD",
        "N JERSEY": "NJ",
        "N MEX": "NM",
        "N C": "NC",
        "N D": "ND",
        "N J": "NJ",
        "N Y": "NY",
        "NATION CALIFORNIA": "CA",
        "NATION COLORADO": "CO",
        "NATION DELAWARE": "DE",
        "NATION FLORIDA": "FL",
        "NATION INDIANA": "IN",
        "NATION IOWA": "IA",
        "NATION NEW YORK": "NY",
        "NATION OHIO": "OH",
        "NATION STATE OHIO": "OH",
        "NATION TEXAS": "TX",
        "NATION-STATE LOUISIANA": "LA",
        "NATION-STATE WISCONSIN": "WI",
        "NATIONS-STATE LOUISIANA": "LA",
        "NC - NORTH CAROLINA": "NC",
        "NC REPUBLIC": "NC",
        "NCMN": "NC",
        "ND - NORTH DAKOTA": "ND",
        "ND DAKOTA": "ND",
        "ND N": "ND",
        "ND1": "ND",
        "NDMN": "ND",
        "NDN": "ND",
        "NEAR FLORIDA": "FL",
        "NEBRASKA NE": "NE",
        "NEVADA NATION-STATE": "NV",
        "NEVADA NV": "NV",
        "NEVADA REPUBLIC": "NV",
        "NEVADA THE REPUBLIC": "NV",
        "NEVEDA": "NV",
        "NEW YORK": "NY",
        "NEW ADDRESS": np.nan,
        "NEW HAMPSHIER": "NH",
        "NEW HAMPSHIRE REPUBLIC": "NH",
        "NEW JEESEY": "NJ",
        "NEW JERSEY NJ": "NJ",
        "NEW JERSEY REPUBLIC": "NJ",
        "NEW JERSEY STATE": "NJ",
        "NEW JERSEYEY": "NJ",
        "NEW JERSY": "NJ",
        "NEW YORK NEAR": "NY",
        "NEW YORK NY": "NY",
        "NEW YORK REPUBLIC": "NY",
        "NEW YORK STATE": "NY",
        "NEW YORK STATE REPUBLIC": "NY",
        "NEW YORKMN": "NY",
        "NEW-YORK": "NY",
        "NEWYORK": "NY",
        "NEWYORK REPUBLIC": "NY",
        "NEY YORK": "NY",
        "NJ - NEW JERSEY": "NJ",
        "NJMN": "NJ",
        "NONE - PLEASE SELECT A STATE": np.nan,
        "NONE PROVIDED": np.nan,
        "NORTH CAROLINA": "NC",
        "NORTH CAROLIKNA": "NC",
        "NORTH CAROLIN": "NC",
        "NORTH CAROLINA NC": "NC",
        "NORTH CAROLINA ZIP EXEMPT NEAR": "NC",
        "NORTH CAROLINA NA": "NC",
        "NORTH CAROLINA NON DOMESTIC NEAR": "NC",
        "NORTH CAROLINA REPUBLIC": "NC",
        "NORTH CAROLINA REPUBLIC EXEMPT": "NC",
        "NORTH CAROLINA REPUBLLIC": "NC",
        "NORTH CAROLINAMN": "NC",
        "NORTH DAKOTA ND": "ND",
        "NORTHCAROLINA": "NC",
        "NOT APPLICABLE": np.nan,
        "NOT SELECTED": np.nan,
        "NV - NEVADA": "NV",
        "NV MN": "NV",
        "NVMN": "NV",
        "NY - NEW YORK": "NY",
        "NY-NEW YORK": "NY",
        "OBJECT 1080": np.nan,
        "OBJECT 1686": np.nan,
        "OBJECT 183": np.nan,
        "OBJECT 1996": np.nan,
        "OBJECT 2309": np.nan,
        "OBJECT 253": np.nan,
        "OBJECT 26": np.nan,
        "OBJECT 3327": np.nan,
        "OBJECT 50": np.nan,
        "OBJECT 7725": np.nan,
        "OHIO OH": "OH",
        "OHIO A REPUBLIC": "OH",
        "OHIO MOHIIN": "OH",
        "OHIO REPUBLIC": "OH",
        "OHIO REPUBLIC USA": "OH",
        "OHIO/REPUBLIC": "OH",
        "OHIOMN": "OH",
        "OHMN": "OH",
        "OHO": "OH",
        "OKLAHOMA OK": "OK",
        "OKLAHOMA REPUBLIC": "OK",
        "OLAHOMA": "OK",
        "OR REPUBLIC": "OR",
        "ORGON": "OR",
        "OTHER": np.nan,
        "OTHER NON US": np.nan,
        "OUT OF STATE": np.nan,
        "OUT OF THE COUNTRY": np.nan,
        "P A": "PA",
        "PA - PENNSYLVANIA": "PA",
        "PENN": "PA",
        "PENNS YVANIA": "PA",
        "PENNSLVANIA": "PA",
        "PENNSLYVANIA": "PA",
        "PENNSYLAVANIA": "PA",
        "PENNSYLVAIA": "PA",
        "PENNSYLVAINA": "PA",
        "PENNSYLVANI REPUBLIC": "PA",
        "PENNSYLVANIA PA": "PA",
        "PENNSYLVANIA A REPUBLIC": "PA",
        "PENNSYLVANIA REPBULIC": "PA",
        "PENNSYLVANIA REPUBLIC": "PA",
        "PENNSYLVANIAMN": "PA",
        "PENNSYLVANIE": "PA",
        "PENNSYLVANIEA": "PA",
        "PENNSYLVIA": "PA",
        "PENNSYVANIA": "PA",
        "PLEASE SELECT": np.nan,
        "PLEASE SELECT A STATE": np.nan,
        "PLEASE SELECT YOUR STATE": np.nan,
        "REPUBLIC FOR GEORGIA": "GA",
        "REPUBLIC NEAR ARIZONA": "AZ",
        "REPUBLIC NEAR OKLAHOMA": "OK",
        "REPUBLIC NEBRASKA": "NE",
        "REPUBLIC OF CALIFORNIA": "CA",
        "REPUBLIC OF GEORGIA": "GA",
        "REPUBLIC OF MISSOURI POSTAL CODE EXCEPTED": "MO",
        "REPUBLIC OF TENNESSEE": "TN",
        "REPUBLIC VIRGINIA": "VA",
        "REPUBLIC-NEBRASKA": "NE",
        "REPULIC OF NEVADA": "NV",
        "RHODE ISLAND AND PROVIDENCE PLANTATIONS": "RI",
        "SD - SOUTH DAKOTA": "SD",
        "SDMN": "SD",
        "SELECT A STATE": np.nan,
        "SELECT A STATE OR PROVINCE": np.nan,
        "SELECT A STATE/REGION": np.nan,
        "SELECT AN OPTION": np.nan,
        "SELECT AN OPTION…": np.nan,
        "SELECT FROM LIST": np.nan,
        "SELECT HERE": np.nan,
        "SELECT STATE/PROVINCE": np.nan,
        "SELECT US STATE": np.nan,
        "SELECT YOUR STATE": np.nan,
        "SOUTH CAROLINA SC": "SC",
        "SOUTH CAROLINA REPUBLIC": "SC",
        "SOUTH CAROLNA": "SC",
        "SOUTH DAKOTA SD": "SD",
        "SOUTH DAKOTA STATE": "SD",
        "SOUTH-CAROLINA-TERRITORY": "SC",
        "SOUTH-DAKOTA": "SD",
        "SOUTHCAROLINA": "SC",
        "STATE US RESIDENTS": np.nan,
        "STATE OF ARIZONA": "AZ",
        "STATE OF FLOR": "FL",
        "STATE OF FLORIDA": "FL",
        "STATE OF ILLINOIS": "IL",
        "STATE OF ILLINOIS A UNION MEMBER": "IL",
        "STATE OF KANSAS": "KS",
        "STATE OF SOUTH CAROLINA": "SC",
        "STATE/PROVINCE": np.nan,
        "STATES": np.nan,
        "STRING MINNESOTA": "MN",
        "STRING MN": "MN",
        "TEAXS": "TX",
        "TENN": "TN",
        "TENNEESSEE": "TN",
        "TENNESEE": "TN",
        "TENNESSE": "TN",
        "TENNESSEE TN": "TN",
        "TENNESSEE A REPUBLIC": "TN",
        "TENNESSEE MN": "TN",
        "TENNESSEE REPLUPIC": "TN",
        "TENNESSEE REPUBLIC": "TN",
        "TENNESSEE STATE": "TN",
        "TENNESSEE STATE REPUBLIC": "TN",
        "TEX": "TX",
        "TEXA": "TX",
        "TEXAD": "TX",
        "TEXAS TX": "TX",
        "TEXAS ZIP": "TX",
        "TEXAS N": "TX",
        "TEXAS NATION-STATE": "TX",
        "TEXAS REPUBLIC": "TX",
        "TEXAS REPUBLIC NEAR": "TX",
        "TEXAS STATE": "TX",
        "TEXAS STATE REPUBLIC": "TX",
        "TEXAS-TERRITORY": "TX",
        "TX - TEXAS": "TX",
        "TX1": "TX",
        "TXAS": "TX",
        "TXMN": "TX",
        "UNION STATE OF TEXAS": "TX",
        "UNION STATE OF ILLINOIS": "IL",
        "UNITED STATES US — MINNESOTA": "MN",
        "UNITED STATES - MINNESOTA": "MN",
        "UNITED STATES STATE OF PENNSYLVANIA": "PA",
        "US-MINNESOTA": "MN",
        "US-MN": "MN",
        "USA MINNESOTA": "MN",
        "USA MN": "MN",
        "USA-MN": "MN",
        "USAZ": "AZ",
        "UTAH UT": "UT",
        "UTAH REPUBLIC": "UT",
        "VA - VIRGINIA": "VA",
        "VIRG": "VA",
        "VIRGINA": "VA",
        "VIRGINIA REPUBLIC": "VA",
        "VIRGINIA VA": "VA",
        "VIRGINIA MN": "VA",
        "VIRGINNIA": "VA",
        "WA - WASHINGTON": "WA",
        "WAHINGTON": "WA",
        "WAMN": "WA",
        "WASGINGTON": "WA",
        "WASH": "WA",
        "WASH D C": "DC",
        "WASHINGTOIN": "WA",
        "WASHINGTON WA": "WA",
        "WASHINGTON DC": "DC",
        "WASHINGTON REPUBLIC": "WA",
        "WASHINGTON STATE": "WA",
        "WASHINGTONMN": "WA",
        "WEST VIRGINIA REPUBLIC": "WV",
        "WEST VIRGINIS": "WV",
        "WI - WISCON": "WI",
        "WI - WISCONSIN": "WI",
        "WI WI": "WI",
        "WI WISCONSIN": "WI",
        "WI-WISCONSIN": "WI",
        "WI455MN": "WI",
        "WICSONSIN": "WI",
        "WIS": "WI",
        "WISC": "WI",
        "WISCCONSIN": "WI",
        "WISCONSIN WI": "WI",
        "WISCONSIN MN": "WI",
        "WISCONSIN REPUBLIC": "WI",
        "WISCONSIN REPUBLIC TERRITORY": "WI",
        "WISCONSINITE": "WI",
        "WISCONSON": "WI",
        "WWW KAMRA-JOOSTEN COM": np.nan,
        "WYOMING WY": "WY",
        "YOUR RESIDENT STATE": np.nan,
        "Æ˜ŽÅ°¼È‹�È¾¾Å·Ž": np.nan,
        "–": np.nan
    }

    @classmethod
    def mn_state_fixer(cls) -> dict[str, str]:
        return cls._MN_STATE_FIXER


class BusinessFilings(BusinessRecordsBase):

    _RAW: list[str] = ["name"]
    _CLEAN_RENAME_MAP: dict[str, str] = {"name": "clean_name"}
    _BASIC_CLEAN: list[str] = ["clean_name"]
    _NAME_CLEAN: list[str] = ["clean_name"]
    _ADDRESS_CLEAN: list[str] = []
    _OUT: list[str] = [
        "uid",
        "status",
        "raw_name",
        "clean_name",
        "filing_date",
        "expiration_date",
        "home_jurisdiction",
        "home_business_name",
        "is_llc_non_profit",
        "is_lllp",
        "is_professional",
    ]

    @classmethod
    def raw(cls) -> list[str]:
        return cls._RAW

    @classmethod
    def clean_rename_map(cls) -> dict[str, str]:
        return cls._CLEAN_RENAME_MAP

    @classmethod
    def basic_clean(cls) -> list[str]:
        return cls._BASIC_CLEAN

    @classmethod
    def name_clean(cls) -> list[str]:
        return cls._NAME_CLEAN

    @classmethod
    def out(cls) -> list[str]:
        return cls._OUT

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
    name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Name",
        description="Name of business entity as registered with the state of Minnesota.",
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
        nullable=True,
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


class BusinessNamesAddrs(BusinessRecordsBase):

    _RAW: list[str] = [
        "party_name",
        "street_1",
        "street_2",
        "city",
        "state",
        "zip_code",
        "zip_code_ext",
        "country",
        "address"
    ]
    _CLEAN_RENAME_MAP: dict[str, str] = {
        "party_name": "clean_party_name",
        "street_1": "clean_street_1",
        "street_2": "clean_street_2",
        "city": "clean_city",
        "state": "clean_state",
        "zip_code": "clean_zip_code",
        "zip_code_ext": "clean_zip_code_ext",
        "country": "clean_country",
        "address": "clean_address"
    }
    _BASIC_CLEAN: list[str] = [
        "clean_party_name",
        "clean_street_1",
        "clean_street_2",
        "clean_city",
        "clean_state",
        "clean_zip_code",
        "clean_zip_code_ext",
        "clean_country",
    ]
    _NAME_CLEAN: list[str] = ["clean_party_name"]
    _ADDRESS_CLEAN: dict[str, list[str]] = {
        "street": [
            "clean_street_1",
            "clean_street_2"
        ],
        "zip": ["clean_zip_code"]
    }
    _OUT: list[str] = [
        "uid",
        "name_type",
        "address_type",
        "raw_party_name",
        "raw_street_1",
        "raw_street_2",
        "raw_city",
        "raw_state",
        "raw_zip_code",
        "raw_zip_code_ext",
        "raw_country",
        "raw_address",
        "clean_party_name",
        "clean_street_1",
        "clean_street_2",
        "clean_city",
        "clean_state",
        "clean_zip_code",
        "clean_zip_code_ext",
        "clean_country",
        "clean_address",
        "is_incomplete_address"
    ]

    @classmethod
    def raw(cls) -> list[str]:
        return cls._RAW

    @classmethod
    def clean_rename_map(cls) -> dict[str, str]:
        return cls._CLEAN_RENAME_MAP

    @classmethod
    def basic_clean(cls) -> list[str]:
        return cls._BASIC_CLEAN

    @classmethod
    def name_clean(cls) -> list[str]:
        return cls._NAME_CLEAN

    @classmethod
    def address_clean(cls) -> dict[str, list[str]]:
        return cls._ADDRESS_CLEAN

    @classmethod
    def out(cls) -> list[str]:
        return cls._OUT

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
    party_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Party Name",
        description="Name of the party to the business filing",
    )
    street_1: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street 1",
        description="Street address (line 2)",
    )
    street_2: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street 2",
        description="Street address (line 2)",
    )
    city: str = pa.Field(
        nullable=True,
        unique=False,
        title="City",
    )
    state: str = pa.Field(
        nullable=True,
        unique=False,
        title="State",
    )
    zip_code: str = pa.Field(
        nullable=True,
        unique=False,
        title="Zip Code",
    )
    zip_code_ext: str = pa.Field(
        nullable=True,
        unique=False,
        title="Zip Code Extension",
    )
    country: str = pa.Field(
        nullable=True,
        unique=False,
        title="Country",
    )
    is_incomplete_address: bool = pa.Field(
        nullable=False,
        unique=False,
        title="Is Incomplete Address?",
        description="Boolean indicating whether or not the address is missing key components (street & zip code)",
    )
