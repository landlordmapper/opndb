class ClassCodes:
    CODE: str = "code"
    CATEGORY: str = "category"
    DESCRIPTION: str = "description"
    IS_RENTAL: str = "is_rental"

class Properties:
    PIN: str = "pin"
    TAX_NAME: str = "tax_name"
    TAX_ADDRESS: str = "tax_address"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    TAX_STREET: str = "tax_street"  # todo: add documentation: optional - addresses may come in a single string format and need to be parsed
    TAX_CITY: str = "tax_city"
    TAX_STATE: str = "tax_state"
    TAX_ZIP: str = "tax_zip"
    CLASS_CODE: str = "class_code"
    NUM_UNITS: str = "num_units"  # todo: add documentation: optional - ideal but will work without it

class TaxpayerRecords:
    RAW_NAME: str = "raw_name"
    RAW_ADDRESS: str = "raw_addr_full"
    RAW_STREET: str = "raw_street"
    RAW_CITY: str = "raw_city"
    RAW_STATE: str = "raw_state"
    RAW_ZIP: str = "raw_zip"
    CLEAN_NAME: str = "clean_name"
    CLEAN_ADDRESS: str = "clean_address"
    CLEAN_STREET: str = "clean_street"
    CLEAN_CITY: str = "clean_city"
    CLEAN_STATE: str = "clean_state"
    CLEAN_ZIP: str = "clean_zip"
    CORE_NAME: str = "core_name"
    IS_RENTAL: str = "is_rental"
    IS_BANK: str = "is_bank"
    IS_PERSON: str = "is_person"
    IS_COMMON_NAME: str = "is_common_name"
    IS_ORG: str = "is_org"
    IS_LLC: str = "is_llc"

    # merged address data
    # todo: determine whether it's necessary to include all these? or only include ones that are absolutely necessary for whatever is being done
    FORMATTED_ADDRESS: str = "formatted_address"

class Corps:

    # raw data fields
    NAME: str = "raw_name"
    FILE_NUMBER: str = "file_number"
    DATE_INCORPORATED: str = "date_incorporated"
    DATE_DISSOLVED: str = "date_dissolved"  # todo: add documentation: optional may be active
    STATUS: str = "status"
    PRESIDENT_NAME: str = "president_name"
    PRESIDENT_ADDRESS: str = "president_address"
    PRESIDENT_STREET: str = "president_street"
    PRESIDENT_CITY: str = "president_city"
    PRESIDENT_STATE: str = "president_state"
    PRESIDENT_ZIP: str = "president_zip"
    SECRETARY_NAME: str = "secretary_name"
    SECRETARY_ADDRESS: str = "secretary_address"
    SECRETARY_STREET: str = "secretary_street"
    SECRETARY_CITY: str = "secretary_city"
    SECRETARY_STATE: str = "secretary_state"
    SECRETARY_ZIP: str = "secretary_zip"

    CLEAN_NAME: str = "clean_name"
    CORE_NAME: str = "core_name"
    IS_BANK_PRESIDENT: str = "is_bank_president"
    IS_PERSON_PRESIDENT: str = "is_person_president"
    IS_COMMON_NAME_PRESIDENT: str = "is_common_name_president"
    IS_ORG_PRESIDENT: str = "is_org_president"
    IS_LLC_PRESIDENT: str = "is_llc_president"
    IS_BANK_SECRETARY: str = "is_bank_secretary"
    IS_PERSON_SECRETARY: str = "is_person_secretary"
    IS_COMMON_NAME_SECRETARY: str = "is_common_name_secretary"
    IS_ORG_SECRETARY: str = "is_org_secretary"
    IS_LLC_SECRETARY: str = "is_llc_secretary"


class LLCs:

    # raw data fields
    NAME: str = "name"
    FILE_NUMBER: str = "file_number"
    DATE_INCORPORATED: str = "date_incorporated"
    DATE_DISSOLVED: str = "date_dissolved"  # todo: add documentation: optional may be active
    STATUS: str = "status"  # todo: add documentation: optional - may be codes whose meanings will determine how the active filters are applied
    MANAGER_MEMBER_NAME: str = "manager_member_name"
    MANAGER_MEMBER_ADDRESS: str = "manager_member_address"
    MANAGER_MEMBER_STREET: str = "manager_member_street"
    MANAGER_MEMBER_CITY: str = "manager_member_city"
    MANAGER_MEMBER_STATE: str = "manager_member_state"  # todo: add documentation: optional - sometimes missing, not ideal
    MANAGER_MEMBER_ZIP: str = "manager_member_zip"
    AGENT_NAME: str = "agent_name"
    AGENT_ADDRESS: str = "agent_address"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    AGENT_STREET: str = "agent_street"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    AGENT_CITY: str = "agent_city"
    AGENT_STATE: str = "agent_state"  # todo: add documentation: optional - sometimes missing, not ideal
    AGENT_ZIP: str = "agent_zip"
    OFFICE_ADDRESS: str = "office_address"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    OFFICE_STREET: str = "office_street"
    OFFICE_CITY: str = "office_city"
    OFFICE_STATE: str = "office_state"  # todo: add documentation: optional - sometimes missing, not ideal
    OFFICE_ZIP: str = "office_addr_zip"

    CLEAN_NAME: str = "clean_name"
    CORE_NAME: str = "core_name"
    IS_BANK_MANAGER_MEMBER: str = "is_bank_manager_member"
    IS_PERSON_MANAGER_MEMBER: str = "is_person_manager_member"
    IS_COMMON_NAME_MANAGER_MEMBER: str = "is_common_name_manager_member"
    IS_ORG_MANAGER_MEMBER: str = "is_org_secretary"
    IS_LLC_MANAGER_MEMBER: str = "is_llc_secretary"
    IS_BANK_AGENT: str = "is_bank_agent"
    IS_PERSON_AGENT: str = "is_person_agent"
    IS_COMMON_NAME_AGENT: str = "is_common_name_agent"
    IS_ORG_AGENT: str = "is_org_agent"
    IS_LLC_AGENT: str = "is_llc_agent"


class ValidatedAddrs:
    CLEAN_ADDRESS: str = "tax_addr_full"
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
