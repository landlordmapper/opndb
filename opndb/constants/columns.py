from typing import Any


class ClassCodes:
    CODE: str = "code"
    CATEGORY: str = "category"
    DESCRIPTION: str = "description"
    IS_RENTAL: str = "is_rental"

class TaxpayerRecords:
    PIN: str = "pin"
    TAX_NAME: str = "tax_name"
    TAX_ADDR: str = "tax_addr_full"
    TAX_STREET: str = "tax_street"
    TAX_CITY: str = "tax_city"
    TAX_STATE: str = "tax_state"
    TAX_ZIP: str = "tax_zip"
    BLDG_CLASS: str = "bldg_class"
    IS_RENTAL: str = "is_rental"
    IS_BANK: str = "is_bank"
    IS_PERSON: str = "is_person"
    IS_ORG: str = "is_org"

    # merged address data
    # todo: determine whether it's necessary to include all these? or only include ones that are absolutely necessary for whatever is being done
    NUMBER: str = "number"
    PREDIRECTIONAL: str = "predirectional"
    PREFIX: str = "prefix"
    STREET: str = "street"
    SUFFIX: str = "suffix"
    POSTDIRECTIONAL: str = "postdirectional"
    SECONDARYUNIT: str = "secondaryunit"
    SECONDARYNUMBER: str = "secondarynumber"
    CITY: str = "city"
    COUNTY: str = "county"
    STATE: str = "state"
    ZIP: str = "zip"
    COUNTRY: str = "country"
    LNG: str = "lng"
    LAT: str = "lat"
    ACCURACY: str = "accuracy"
    FORMATTED_ADDRESS: str = "formatted_address"


class Corps:

    NAME: str = "name"
    FILE_NUMBER: str = "file_number"
    DATE_INCORPORATED: str = "date_incorporated"

    PRESIDENT_NAME: str = "president_name"
    PRESIDENT_ADDR: str = "president_addr"
    PRESIDENT_ADDR_STREET: str = "president_addr_street"
    PRESIDENT_ADDR_CITY: str = "president_addr_city"
    PRESIDENT_ADDR_STATE: str = "president_addr_state"
    PRESIDENT_ADDR_ZIP: str = "president_addr_zip"
    IS_BANK_PRESIDENT: str = "is_bank_president"
    IS_PERSON_PRESIDENT: str = "is_person_president"
    IS_ORG_PRESIDENT: str = "is_org_president"

    SECRETARY_NAME: str = "secretary_name"
    SECRETARY_ADDR: str = "secretary_addr"
    SECRETARY_ADDR_STREET: str = "secretary_addr_street"
    SECRETARY_ADDR_CITY: str = "secretary_addr_city"
    SECRETARY_ADDR_STATE: str = "secretary_addr_state"
    SECRETARY_ADDR_ZIP: str = "secretary_addr_zip"
    IS_BANK_SECRETARY: str = "is_bank_secretary"
    IS_PERSON_SECRETARY: str = "is_person_secretary"
    IS_ORG_SECRETARY: str = "is_org_secretary"


class LLCs:

    NAME: str = "name"
    FILE_NUMBER: str = "file_number"
    DATE_INCORPORATED: str = "date_incorporated"

    MANAGER_MEMBER_NAME: str = "manager_member_name"
    MANAGER_MEMBER_ADDR: str = "manager_member_addr"
    MANAGER_MEMBER_ADDR_STREET: str = "manager_member_addr"
    MANAGER_MEMBER_ADDR_CITY: str = "manager_member_addr"
    MANAGER_MEMBER_ADDR_STATE: str = "manager_member_addr"
    MANAGER_MEMBER_ADDR_ZIP: str = "manager_member_addr"
    IS_BANK_MANAGER_MEMBER: str = "is_bank_secretary"
    IS_PERSON_MANAGER_MEMBER: str = "is_person_secretary"
    IS_ORG_MANAGER_MEMBER: str = "is_org_secretary"

    AGENT_NAME: str = "agent_name"
    AGENT_ADDR: str = "agent_addr"
    AGENT_ADDR_STREET: str = "agent_addr_street"
    AGENT_ADDR_CITY: str = "agent_addr_city"
    AGENT_ADDR_STATE: str = "agent_addr_state"
    AGENT_ADDR_ZIP: str = "agent_addr_zip"
    IS_BANK_AGENT: str = "is_bank_agent"
    IS_PERSON_AGENT: str = "is_person_agent"
    IS_ORG_AGENT: str = "is_org_agent"

    OFFICE_ADDR: str = "office_addr"
    OFFICE_ADDR_STREET: str = "office_addr_street"
    OFFICE_ADDR_CITY: str = "office_addr_city"
    OFFICE_ADDR_STATE: str = "office_addr_state"
    OFFICE_ADDR_ZIP: str = "office_addr_zip"


class ValidatedAddrs:
    TAX_ADDR: str = "tax_addr_full"
    TAX_STREET: str = "tax_street"
    TAX_CITY: str = "tax_city"
    TAX_STATE: str = "tax_state"
    TAX_ZIP: str = "tax_zip"
    NUMBER: str = "number"
    PREDIRECTIONAL: str = "predirectional"
    PREFIX: str = "prefix"
    STREET: str = "street"
    SUFFIX: str = "suffix"
    POSTDIRECTIONAL: str = "postdirectional"
    SECONDARYUNIT: str = "secondaryunit"
    SECONDARYNUMBER: str = "secondarynumber"
    CITY: str = "city"
    COUNTY: str = "county"
    STATE: str = "state"
    ZIP: str = "zip"
    COUNTRY: str = "country"
    LNG: str = "lng"
    LAT: str = "lat"
    ACCURACY: str = "accuracy"
    FORMATTED_ADDRESS: str = "formatted_address"


class UnvalidatedAddrs:
    TAX_FULL_ADDRESS: str = "tax_addr_full"
    TAX_STREET: str = "tax_street"
    TAX_CITY: str = "tax_city"
    TAX_STATE: str = "tax_state"
    TAX_ZIP: str = "tax_zip"


class AddressAnalysis:
    ADDRESS: str = "ADDRESS"
    COUNT: str = "COUNT"
    NAME: str = "NAME"
    URL: str = "URL"
    NOTES: str = "NOTES"
    IS_LANDLORD_ORG: str = "IS_LANDLORD_ORG"
    IS_GOVT_AGENCY: str = "IS_GOVT_AGENCY"
    IS_LAWFIRM: str = "IS_LAWFIRM"
    IS_MISSING_SUITE: str = "IS_MISSING_SUITE"
    IS_FINANCIAL_SERVICES: str = "IS_FINANCIAL_SERVICES"
    IS_ASSOC_BUS: str = "IS_ASSOC_BUS"
    FIX_ADDRESS: str = "FIX_ADDRESS"
    IS_FIXED: str = "IS_FIXED"
    IS_VIRTUAL_OFFICE_AGENT: str = "IS_VIRTUAL_OFFICE_AGENT"
    IS_NONPROFIT: str = "IS_NONPROFIT"
    IS_IGNORE_MISC: str = "IS_IGNORE_MISC"
    YELP_URL: str = "YELP_URL"
    GOOGLE_URL: str = "GOOGLE_URL"
