from typing import Any

import numpy as np
import pandera as pa
from opndb.validator.df_model import OPNDFModel


class PropsTaxpayersMN(OPNDFModel):

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
        "clean_name"
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
        unique=True,
        title="UID",
        description="Unique identifier for MNSOS business records.",
    )
    name_type: str = pa.Field()
    address_type: str = pa.Field()
    party_name: str = pa.Field()
    street_1: str = pa.Field()
    street_2: str = pa.Field()
    city: str = pa.Field()
    state: str = pa.Field()
    zip_code: str = pa.Field()
    zip_code_ext: str = pa.Field()
    country: str = pa.Field()
    raw_address: str = pa.Field()
    is_incomplete_address: bool = pa.Field()


class PropsTaxpayers(OPNDFModel):
    """
    Raw dataset containing both property and taxpayer record data. The opndb workflow will split up this dataset into
    separate datasets: one for taxpayer records, the other for properties.

    Constants defined at top of class are used for data processing, specifically to indicate which columns should be
    renamed, iterated over, etc. Each constant corresponds to a class method returning the value of the constant.
    """
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _RAW: list[str] = [
        "tax_name",
        "tax_name_2",
        "tax_street",
        "tax_city",
        "tax_state",
        "tax_zip",
    ]
    _CLEAN_RENAME_MAP: dict[str, Any] = {
        "tax_name": "clean_name",
        "tax_name_2": "clean_name_2",
        "tax_street": "clean_street",
        "tax_city": "clean_city",
        "tax_state": "clean_state",
        "tax_zip": "clean_zip",
    }
    _RAW_ADDRESS_MAP: list[dict[str, Any]] = [
        {
            "full_address": "raw_address",
            "address_cols": [
                "raw_street",
                "raw_city",
                "raw_state",
                "raw_zip",
            ]
        }
    ]
    _NAME_ADDRESS_CONCAT_MAP: dict[str, Any] = {
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
        "pin",
        "class_code",
        "clean_name",
        "clean_name_2"
        "clean_street",
        "clean_city",
        "clean_state",
        "clean_zip",
    ]
    _NAME_CLEAN: list[str] = ["clean_name", "clean_name_2"]
    _ADDRESS_CLEAN: dict[str, Any] = {
        "street": ["clean_street"],
        "zip": ["clean_zip"],
    }
    _CLEAN_ADDRESS_MAP: list[dict[str, Any]] = [
        {
            "full_address": "clean_address",
            "address_cols": [
                "clean_street",
                "clean_city",
                "clean_state",
                "clean_zip",
            ]
        }
    ]

    @classmethod
    def raw(cls) -> list[str]:
        return cls._RAW

    @classmethod
    def clean_rename_map(cls) -> dict[str, Any]:
        return cls._CLEAN_RENAME_MAP

    @classmethod
    def raw_address_map(cls) -> list[dict[str, Any]]:
        return cls._RAW_ADDRESS_MAP

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

    @classmethod
    def clean_address_map(cls) -> list[dict[str, Any]]:
        return cls._CLEAN_ADDRESS_MAP

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    pin: str = pa.Field(
        nullable=False,
        unique=True,
        title="PIN",
        description="Unique tax identifier for the property",
    )
    tax_name: str = pa.Field(
        nullable=False,
        title="Taxpayer Name (Primary)",
        description="Primary taxpayer name indicated for the property",
    )
    tax_name_2: str  = pa.Field(
        nullable=True,
        title="Taxpayer Name (Secondary)",
        description="Secondary taxpayer name indicated for the property",
    )
    tax_street: str = pa.Field(
        nullable=False,
        title="Taxpayer Street",
        description="Street address for the taxpayer of the property, including street number, street name, secondary/unit number and all prefixes and suffixes.",
    )
    tax_city: str = pa.Field(
        nullable=False,
        title="Taxpayer City",
        description="City associated with property taxpayer's street address."
    )
    tax_state: str = pa.Field(
        nullable=False,
        title="Taxpayer State",
        description="State associated with property taxpayer's street address."
    )
    tax_zip: str = pa.Field(
        nullable=False,
        title="Taxpayer Zip",
        description="Zip code associated with property taxpayer's street address."
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
    Raw dataset for state-level corporate records. Note that the availability of address-related columns is subject to
    the quality of the original data and the ability to parse complete address strings.
    """
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _RAW: list[str] = [
        "name",
        "president_name",
        "president_address",
        "secretary_name",
        "secretary_address",
    ]
    _CLEAN_RENAME_MAP: dict[str, str] = {
        "name": "clean_name",
        "president_name": "clean_president_name",
        "president_address": "clean_president_address",
        "secretary_name": "clean_secretary_name",
        "secretary_address": "clean_secretary_address",
    }
    _RAW_ADDRESS_MAP: None = None
    _BASIC_CLEAN: list[str] = [
        "file_number",
        "status",
        "clean_name",
        "clean_president_name",
        "clean_president_address",
        "clean_secretary_name",
        "clean_secretary_address",
    ]
    _NAME_CLEAN: list[str] = [
        "clean_name",
        "clean_president_name",
        "clean_secretary_name",
    ]
    _ADDRESS_CLEAN: dict[str, list[str]] = {
        "street": [
            "clean_president_address",
            "clean_secretary_address",
        ],
        "zip": []
    }
    _CLEAN_ADDRESS_MAP: None = None
    _OUT: list[str] = [
        "file_number",
        "status",
        "raw_name",
        "raw_president_name",
        "raw_president_address",
        "raw_secretary_name",
        "raw_secretary_address",
        "clean_name",
        "clean_president_name",
        "clean_president_address",
        "clean_secretary_name",
        "clean_secretary_address",
    ]
    _UNVALIDATED_COL_OBJS: list[dict[str, str]] = [
        {
            "raw_president_address": "raw_address",
            "clean_president_address": "clean_address",
            "status": "status",
        },
        {
            "raw_secretary_address": "raw_address",
            "clean_secretary_address": "clean_address",
            "status": "status",
        }
    ]

    @classmethod
    def raw(cls) -> list[str]:
        return cls._RAW

    @classmethod
    def clean_rename_map(cls) -> dict[str, str]:
        return cls._CLEAN_RENAME_MAP

    @classmethod
    def raw_address_map(cls) -> None:
        return cls._RAW_ADDRESS_MAP

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
    def clean_address_map(cls) -> None:
        return cls._CLEAN_ADDRESS_MAP

    @classmethod
    def out(cls) -> list[str]:
        return cls._OUT

    @classmethod
    def unvalidated_col_objs(cls) -> list[dict[str, str]]:
        return cls._UNVALIDATED_COL_OBJS

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    name: str = pa.Field(
        nullable=False,
        unique=True,
        title="Corporation Name",
        description="Corporation name",
    )
    file_number: str = pa.Field(
        nullable=False,
        unique=True,
        title="Corporation File Number",
        description="Unique identifier number assigned to corporation by secretary of state upon incorporation.",
    )
    date_incorporated: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Date Incorporated",
        description="Date Incorporated",
    )
    date_dissolved: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Date Dissolved",
        description="Date Dissolved",
    )
    status: str | None = pa.Field(
        nullable=False,
        unique=False,
        title="Status",
        description="Organization's status (active, inactive, involuntarily dissolved, etc.)",
    )
    president_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="President Name",
    )
    president_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="President Address",
        description="Complete concatenated mailing address for corporation president",
    )
    president_street: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="President Street",
        description="President street",
    )
    president_city: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="President City",
        description="President city",
    )
    president_state: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="President State",
        description="President state",
    )
    president_zip: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="President Zip Code",
        description="President zip code",
    )
    secretary_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary Name",
        description="Secretary name",
    )
    secretary_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary Address",
        description="Secretary address",
    )
    secretary_street: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary Street",
        description="Secretary street",
    )
    secretary_city: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary City",
        description="Secretary city",
    )
    secretary_state: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary State",
        description="Secretary state",
    )
    secretary_zip: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary Zip Code",
        description="Secretary zip code",
    )

class LLCs(OPNDFModel):
    """
    Raw dataset for state-level LLC records. Note that the availability of address-related columns is subject to
    the quality of the original data and the ability to parse complete address strings.
    """
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _RAW: list[str] = [
        "name",
        "manager_member_name",
        "manager_member_street",
        "manager_member_city",
        "manager_member_zip",
        "agent_name",
        "agent_street",
        "agent_zip",
        "office_street",
        "office_city",
        "office_zip",
    ]
    _CLEAN_RENAME_MAP: dict[str, str] = {
        "name": "clean_name",
        "manager_member_name": "clean_manager_member_name",
        "manager_member_street": "clean_manager_member_street",
        "manager_member_city": "clean_manager_member_city",
        "manager_member_zip": "clean_manager_member_zip",
        "agent_name": "clean_agent_name",
        "agent_street": "clean_agent_street",
        "agent_zip": "clean_agent_zip",
        "office_street": "clean_office_street",
        "office_city": "clean_office_city",
        "office_zip": "clean_office_zip",
    }
    _RAW_ADDRESS_MAP: list[dict[str, list[str]]] = [
        {
            "full_address": "raw_manager_member_address",
            "address_cols": [
                "raw_manager_member_street",
                "raw_manager_member_city",
                "raw_manager_member_zip",
            ],
        },
        {
            "full_address": "raw_agent_address",
            "address_cols": [
                "raw_agent_street",
                "raw_agent_zip",
            ],
        },
        {
            "full_address": "raw_office_address",
            "address_cols": [
                "raw_office_street",
                "raw_office_city",
                "raw_office_zip",
            ],
        },
    ]
    _BASIC_CLEAN: list[str] = [
        "file_number",
        "status",
        "clean_name",
        "clean_manager_member_name",
        "clean_manager_member_street",
        "clean_manager_member_city",
        "clean_manager_member_zip",
        "clean_agent_name",
        "clean_agent_street",
        "clean_agent_zip",
        "clean_office_street",
        "clean_office_city",
        "clean_office_zip",
    ]
    _NAME_CLEAN: list[str] = [
        "clean_name",
        "clean_manager_member_name",
        "clean_agent_name",
    ]
    _ADDRESS_CLEAN: dict[str, list[str]] = {
        "street": [
            "clean_manager_member_street",
            "clean_agent_street",
            "clean_office_street",
        ],
        "zip": [
            "clean_manager_member_zip",
            "clean_agent_zip",
            "clean_office_zip",
        ],
    }
    _CLEAN_ADDRESS_MAP: list[dict[str, list[str]]] = [
        {
            "full_address": "clean_manager_member_address",
            "address_cols": [
                "clean_manager_member_street",
                "clean_manager_member_city",
                "clean_manager_member_zip",
            ],
        },
        {
            "full_address": "clean_agent_address",
            "address_cols": [
                "clean_agent_street",
                "clean_agent_zip",
            ],
        },
        {
            "full_address": "clean_office_address",
            "address_cols": [
                "clean_office_street",
                "clean_office_city",
                "clean_office_zip",
            ],
        },
    ]
    _OUT: list[str] = [
        "file_number",
        "status",
        "raw_name",
        "raw_manager_member_name",
        "raw_manager_member_street",
        "raw_manager_member_city",
        "raw_manager_member_zip",
        "raw_manager_member_address",
        "raw_agent_name",
        "raw_agent_street",
        "raw_agent_zip",
        "raw_agent_address",
        "raw_office_street",
        "raw_office_city",
        "raw_office_zip",
        "raw_office_address",
        "clean_name",
        "clean_manager_member_name",
        "clean_manager_member_street",
        "clean_manager_member_city",
        "clean_manager_member_zip",
        "clean_manager_member_address",
        "clean_agent_name",
        "clean_agent_street",
        "clean_agent_zip",
        "clean_agent_address",
        "clean_office_street",
        "clean_office_city",
        "clean_office_zip",
        "clean_office_address",
    ]
    _UNVALIDATED_COL_OBJS: list[dict[str, str]] = [
        {
            "raw_manager_member_street": "raw_street",
            "raw_manager_member_city": "raw_city",
            "raw_manager_member_zip": "raw_zip",
            "raw_manager_member_address": "raw_address",
            "clean_manager_member_street": "clean_street",
            "clean_manager_member_city": "clean_city",
            "clean_manager_member_zip": "clean_zip",
            "clean_manager_member_address": "clean_address",
            "status": "status",
        },
        {
            "raw_agent_street": "raw_street",
            "raw_agent_zip": "raw_zip",
            "raw_agent_address": "raw_address",
            "clean_agent_street": "clean_street",
            "clean_agent_zip": "clean_zip",
            "clean_agent_address": "clean_address",
            "status": "status",
        },
        {
            "raw_office_street": "raw_street",
            "raw_office_city": "raw_city",
            "raw_office_zip": "raw_zip",
            "raw_office_address": "raw_address",
            "clean_office_street": "clean_street",
            "clean_office_city": "clean_city",
            "clean_office_zip": "clean_zip",
            "clean_office_address": "clean_address",
            "status": "status",
        },
    ]

    @classmethod
    def raw(cls) -> list[str]:
        return cls._RAW

    @classmethod
    def clean_rename_map(cls) -> dict[str, str]:
        return cls._CLEAN_RENAME_MAP

    @classmethod
    def raw_address_map(cls) -> list[dict[str, list[str]]]:
        return cls._RAW_ADDRESS_MAP

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
    def clean_address_map(cls) -> list[dict[str, list[str]]]:
        return cls._CLEAN_ADDRESS_MAP

    @classmethod
    def out(cls) -> list[str]:
        return cls._OUT

    @classmethod
    def unvalidated_col_objs(cls) -> list[dict[str, str]]:
        return cls._UNVALIDATED_COL_OBJS

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    name: str = pa.Field(
        nullable=False,
        unique=True,
        title="LLC Name",
        description="LLC name",
    )
    file_number: str = pa.Field(
        nullable=False,
        unique=True,
        title="LLC File Number",
        description="Unique identifier number assigned to LLC by secretary of state upon incorporation.",
    )
    date_incorporated: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Date Incorporated",
        description="Date Incorporated",
    )
    date_dissolved: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Date Dissolved",
        description="Date Dissolved",
    )
    status: str = pa.Field(
        nullable=False,
        unique=False,
        title="Status",
        description="Organization's status (active, inactive, involuntarily dissolved, etc.)",
    )
    manager_member_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Manager/Member Name",
        description="Manager/member name",
    )
    manager_member_address: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member Address",
        description="Manager/member address",
    )
    manager_member_street: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member Street",
        description="Manager/member street",
    )
    manager_member_city: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member City",
        description="Manager/member city",
    )
    manager_member_state: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member State",
        description="Manager/member state",
    )
    manager_member_zip: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member Zip Code",
        description="Manager/member zip code",
    )
    agent_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Agent Name",
        description="Agent name",
    )
    agent_address: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Agent Address",
        description="Agent address",
    )
    agent_street: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Agent Street",
        description="Agent street",
    )
    agent_city: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Agent City",
        description="Agent city",
    )
    agent_state: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Agent State",
        description="Agent state",
    )
    agent_zip: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Agent Zip Code",
        description="Agent zip code",
    )
    office_address: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Office Address",
        description="Office address",
    )
    office_street: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Office Street",
        description="Office street",
    )
    office_city: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Office City",
        description="Office city",
    )
    office_state: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Office State",
        description="Office state",
    )
    office_zip: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Office Zip Code",
        description="Office zip code",
    )

class ClassCodes(OPNDFModel):
    """
    Dataset containing building class codes and their meaning. Usually set by the municipality to dictate zoning. Used
    to subset rental properties.
    """
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _BASIC_CLEAN: list[str] = [
        "code",
        "category",
        "description"
    ]

    @classmethod
    def basic_clean(cls) -> list[str]:
        return cls._BASIC_CLEAN

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    code: str = pa.Field(
        nullable=False,
        unique=True,
        title="Class Code",
        description="Code provided by municipal or county government designating land use. Used to subset rental properties.",
    )
    category: str = pa.Field(
        nullable=True,
        unique=False,
        title="Class Category",
        description="Optional categorization/descriptor associated with class code.",
    )
    description: str = pa.Field(
        nullable=True,
        title="Class Description",
        description="Detailed description defining land use for class code.",
    )
    is_rental: bool = pa.Field(
        nullable=False,
        title="Is Rental?",
        description="Boolean column indicating whether or not the class code is associated with rental properties."
    )
