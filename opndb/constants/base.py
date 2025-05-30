from pathlib import Path


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
    "ROOM",
    "AAYYYYYYYEEE"
}

CORE_NAME_KEYS = [
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

TRUSTS = [
    "ATRUS", "TRUSTLAN", "TRUSTEE", "TRUSTE", "TRUSTTRU", "TRUSTU", "TRUSTAS", "TRUST", "TRUS", "TRU", "TRST", "TR",
    "TTEE", "TT"
]
TRUSTS_STRINGS = [
    "TRUSTEE", "TRUSTE", "TRUSTTRU", "TRUSTU", "TRUSTAS", "TRUST"
]
TRUST_COMPANIES_IDS = [
    "CHICAGO TITLE LAND TRUST COMPANY",
    "ATG TRUST COMPANY",
    "BMO HARRIS TRUST",
    "FIRST MIDWEST BANK TRUST",
    "PARKWAY BANK TRUST",
    "OLD NATIONAL TRUST",
    "MARQUETTE BANK TRUST"
]

CORP_WORDS = [
    'LLC', 'PROPERTIES', 'CHEMICAL', 'PC', 'MD', 'SALON', 'GOOD', 'INSTITUTE', 'HOSPITAL', 'WELLESLEY', 'PROGRAM',
    'SHORE', 'TRUSTEES', 'APPLIED',
    'COASTAL', 'WORLD', 'VENTURES', 'PLYMOUTH', 'HARVARD', 'END', 'BUILDING', 'DELIVERY', 'TOWN', 'YOUTH',
    'FOODS', 'BLUE', 'GOVERNMENT', 'POST', 'SERVICE', 'EXPORT', 'PACKAGING', 'ISLAND', 'WEALTH', 'ALPHA', '80TH',
    'CATERING', 'COUNSELING', '50TH', 'ADVISORS', 'ESSEX', 'INDUSTRIAL', 'SOCIAL', 'WESTERN', 'CENTERS', 'FENWAY',
    'CHESTNUT', 'HOME', 'LINE', 'UNITED', 'SAFETY', 'BACK', 'MATERIALS', 'WEST', 'SUN', 'SCIENCE', 'HOLDINGS',
    'UNION', '40TH', 'TRANSPORT', 'FLOOR', '90TH', 'BUSINESS', 'ENTERTAINMENT', 'ST', 'FOR', 'STORAGE',
    'SYSTEMS', 'INVESTORS', 'ENGINEERS', '30TH', 'HOTEL', 'FINE', 'CONSULTANTS', 'PATRIOT', 'SPECIALTY', 'HOSPITALITY',
    'TRANSPORTATION', 'SUPPLY', 'TRAINING', 'NEW ENGLAND', 'PUBLIC', 'BIG', 'HIGHWAY', 'AGENCY', 'REAL', 'MERRIMACK',
    'NORTH END', 'HEALTH', 'STATES', 'PARTNERSHIP', 'PLAZA', 'MASSACHUSETTS', 'MUNICIPAL', 'COFFEE', 'COMMONWEALTH',
    'RESTAURANT', 'WORLDWIDE', 'EYE', 'WINE', 'PLUMBING', 'STRATEGIES', 'PIONEER', 'TIME', 'ALL', 'VINEYARD', 'ROCK',
    'TRINITY', 'COMPANY', 'CENTRAL', 'COUNCIL', 'CLEAN', 'PARK', 'CABLE', 'EXCHANGE', 'TECHNOLOGIES', 'EDUCATIONAL',
    'APARTMENTS', 'MATTAPAN', 'SOCCER', 'ELITE', 'GOLF', 'PREMIER', 'ART', 'OCEAN', 'EASTERN', 'STAR', 'FRANKLIN',
    'ACADEMY',
    'GREATER', 'SPECIALISTS', 'CLEANING', 'SMART', 'CAB', 'LOGISTICS', 'CONTRACTORS', 'BROKERAGE', 'LANDSCAPE',
    'HIGHLAND', 'BEAUTY', 'DIGITAL', 'ADVISORY', 'NETWORKS', 'SILVER', 'MEDICAL', 'CALIFORNIA', 'SALEM', 'TECHNOLOGY',
    'SALES', 'SYSTEM', 'PASS', 'CONVENIENCE', 'FASHION', 'RESOURCE', 'ESTATE', 'MANUFACTURING', 'CHARITABLE',
    'PERFORMANCE', 'COMMUNITY', 'MAINTENANCE', 'STUDIO', 'BRIDGE', 'LABS', 'POWER', 'PRODUCTS', 'COUNTRY', 'CAR',
    'PAINTING', 'AND', 'VALLEY', 'PLLC', 'COLLABORATIVE', 'OAK', 'GOLD', 'BEACON', 'REVOCABLE', 'SOURCE',
    'BROCKTON', 'PHOTOGRAPHY', 'SOUTH', 'TRI', '3RD', 'WATER', 'LEARNING', 'COM', 'NATIONAL', 'SUPPORT', 'CITY',
    'DRIVE', 'UNLIMITED', 'LIGHT', 'SOLUTIONS', 'VETERANS', 'DRYWALL', 'POND', 'TAX', 'INVESTMENTS',
    'RECOVERY', 'ORGANIZATION', 'CONNECTION', 'EXPRESS', 'DISTRIBUTION', 'ARCHDIOCESE', 'GARAGE', 'THERAPEUTICS',
    'COAST', 'BUILDERS', 'PRECISION', 'RENTALS', 'TELECOMMUNICATIONS', 'KITCHEN', 'RETAIL', 'LABORATORIES',
    'COMMUNICATIONS', 'MINISTRIES', 'RESIDENTIAL', 'INTERACTIVE', 'AIR', 'NORTHEAST', 'DATA', 'THROUGH',
    'FOOD', 'BENEFITS', 'NURSING', 'DORCHESTER', 'SCIENTIFIC', 'PLASTERING', 'PET', 'CARPENTRY', 'MDPC',
    'METAL', 'DRIVEWAY', 'STATE', 'PARTS', 'BOSTON HOUSING AUTHORITY', 'BOULEVARD', 'ASSOCIATES', 'PLEASANT',
    'SPRINGFIELD', 'TOURS', 'MOTORS', 'PACIFIC', 'FUND', 'AVIATION', 'STRATEGIC', 'PHYSICAL', 'INNOVATIVE',
    'ENGINEERING', 'NANTUCKET', 'LLC', 'CORNER', 'ATLANTIC', 'CHARLESTOWN', 'CENTER', 'SEAFOOD', 'ELECTRIC', 'GRILL',
    'ENTERPRISE', 'WALTHAM', 'ENGLAND', 'QUALITY', 'NEWTON', 'CORPORATE', 'PLUS', 'IMPORTS', 'INFORMATION', 'CLASSIC',
    'EAGLE', 'NET', 'TITLE', 'CREDIT', 'RESOURCES', 'SCHOLARSHIP', 'HILL', 'GRANITE', 'FARMS', 'REAL ESTATE',
    'FAMILY', 'FITNESS', 'FLOORING', 'COMPANIES', 'EQUIPMENT', 'CONCORD', 'VENTURE', 'GENERAL', 'ROYAL', 'COLLEGE',
    'DONUTS', 'MEMORIAL', 'SECURITY', 'MIDDLESEX', 'REPAIR', 'GREAT', 'NEWBURY', '70TH', 'NORTH', 'FUNDS', 'INCOME',
    'THERAPY', 'PRESS', 'NATURAL', 'TERRACE', 'YOUR', 'OIL', 'CHURCH', 'AUTHORITY', 'PRODUCTIONS', 'CROSSWAY',
    'CONTINENTAL', 'ADVANCED', 'TECHNICAL', 'NEW', 'TRUCK', 'ARCHITECTS', 'CONCEPTS', 'SERIES', 'LEASING', 'CAFE',
    'BAY',
    'HOUSING', '1ST', 'EDGE', 'YORK', 'GLOBAL', 'CONTRACTOR', 'PRINTING', 'FURNITURE', 'HAIR', 'IGLESIA', 'WHOLESALE',
    'AUTO', 'GARDEN', 'CR', 'CIR', 'SON', 'CARE', 'FRIENDS', 'WORCESTER', 'PROJECT', 'WAY', 'FORECLOSURE', 'BAR',
    'MOBILE', 'PUBLISHING', 'PRIME', 'FIRE', 'FUNDING', 'PIZZA', 'MANAGEMENT', 'NORTHERN', 'DENTAL', 'NETWORK',
    'LIBERTY', 'FINANCE', 'STEEL', 'ENVIRONMENTAL', 'REMODELING', 'BOSTON', 'STOP', 'STUDIOS', 'CONDO', 'THEATRE',
    'MECHANICAL', 'TRADING', 'CONDO TRUST', 'UNIVERSAL', 'HIGH', 'BEST', 'INSURANCE', 'MOTOR', 'GOD', 'METRO',
    'COLONIAL', 'CONSTRUCTION', 'GAS', 'PHARMACY', 'CHIROPRACTIC', 'VILLAGE', '100TH', 'PRIVATE', 'INC', 'MOUNTAIN',
    'WOOD', 'MARINE', 'ASSOCIATION', 'SOUTH END', 'EQUITY', 'ACQUISITION', 'CHAPTER', 'PINE', 'IMPROVEMENT', 'BAKERY',
    'BROADWAY', 'MUSIC', 'LENDING', 'INDEPENDENT', 'PARKWAY', 'CHILDREN', 'CORPORATION', 'LIVING', 'BROTHERS',
    'SONS', 'SUB', 'REALTY TRUST', 'TRAVEL', 'INTERNATIONAL', 'RECORDS', 'REDEVELOPMENT', 'AVE', 'PLACE', 'CAPE',
    'DMD', 'WARF', 'ANDOVER', 'INDUSTRIES', 'COMPUTER', 'DIRECT', 'CONTROL', 'COMMERCIAL', 'HALL', 'RESEARCH',
    'COD', 'SHOP', 'PACKAGE', 'EXECUTIVE', 'PARTNERS', 'COMMITTEE', 'JEWELRY', 'LEAGUE', 'TRADE',
    'FISHERIES', 'ATHLETIC', 'CLEANERS', 'HOUSE', 'SOLAR', 'HEATING', 'INN', 'ARTS', 'HOCKEY', 'SUMMIT',
    'DESIGNS', 'TRANS', 'LN', 'SOFTWARE', 'OFFICE', 'HARBOR', 'ENERGY', 'WOODS', 'UNIVERSITY',
    'WORKS', 'CAMBRIDGE', 'FARM', 'MAIN', '60TH', 'INTERIORS', 'TOP', 'SHOE', 'FISHING', 'PAPER',
    'FOUNDATION', 'FALL', 'MANAGERS', '20TH', 'TIRE', 'LIFE', 'HOMES', 'IMAGING', 'ROMAN CATHOLIC',
    '5TH', 'CHOICE', 'TRUCKING', 'ADVERTISING', 'STORES', 'SPORTS', 'STORE', 'DANCE', 'ROSLINDALE',
    'BACK BAY', 'KIDS', 'RESTORATION', 'DAY', 'GROUP', 'GLASS', 'LIABILITY', 'PROPERTIES', 'BEACH',
    'WASHINGTON', 'MARKETING', 'LOWELL', 'TOTAL', 'BODY', 'LAND', 'SOCIETY', 'SECURITIES', 'PLANNING',
    'ROXBURY', 'AMERICAN', 'LIQUORS', 'LANDSCAPING', 'WIRELESS', 'CONSULTING', 'TEAM', 'ICE', 'ACTION',
    'LLP', 'LIMOUSINE', 'CAPITAL', 'GRAPHICS', 'COVE', 'MASONRY', 'GALLERY', 'CARPET', 'GRACE', 'RD',
    'ENTERPRISES', 'DESIGN', 'ALLIANCE', 'ADVANTAGE', 'AMERICA', 'SEA', 'EAST', 'PHOENIX', 'DISTRIBUTORS',
    'MEDIA', 'TOOL', 'TAXI', 'ESTABLISHED', 'DEVELOPMENT', 'BEDFORD', 'CREATIVE', 'LEGAL', 'ELECTRICAL',
    'EXTENSION', 'COURT', 'DELI', 'BROKERS', 'SQUARE', 'FISH', 'PROFESSIONAL', 'FUEL', 'COLONY', 'PROTECTION',
    'LIMITED', 'ACCESS', 'CUSTOM', 'FOREST', 'HERITAGE', 'BURLINGTON', 'VIEW', 'INTEGRATED', 'COOPERATIVE',
    'FRAMINGHAM', 'SPA', 'AUTOMOTIVE', 'SCHOOL', 'OWNER', 'PHARMACEUTICALS', 'OLD', 'COUNTY', 'CONCRETE',
    'REALTY', 'CLUB', 'WOMEN', 'BERKSHIRE', 'GOLDEN', 'FINANCIAL', 'ALLEY', 'TREE', 'PETROLEUM', 'TECH',
    'TELECOM', 'RIVER', 'DOG', 'VALUE', 'OFFICES', 'USA', 'PRESIDENT', 'VIDEO', 'RECYCLING', 'RENTAL',
    'ALLSTON', 'EDUCATION', 'WASTE', 'BRIGHTON', 'SUMMER', 'STAFFING', 'ELECTRONICS', 'ROOFING', 'ASSET',
    'VISION', 'SERVICES', 'CONTRACTING', 'DMDPC', 'MACHINE', 'FREE', 'WELLNESS', 'PRO', 'ESTATES', 'STATION',
    'HEALTHCARE', 'MORTGAGE', 'MARKET', 'TOWING', '2ND', 'TILE', 'PORTFOLIOS', 'ASSETS', 'RESERVES'
]

PO_BOXES = [
    "POSTOFFICEBOX", "POSTBOX", "POBBOX", "POBOC", "POBOX", "PBOX", "POST", "POBX", "BOX", "POB", "PO"
]
PO_BOXES_DEPT = ["TAXDEPT", "TAXDEP", "TXDEPT", "DEPT"]
PO_BOXES_REMOVE = [
    "LOUISAVE", "SSWANTST", "CHICAGO", "6890S2300E", "REDEPT", "RAVSTN", "RAVNIASTA", "RAVINIASTA", "RAVINIAST",
    "RAVINIA", "REVINIAST"
]